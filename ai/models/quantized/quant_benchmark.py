"""
Faz 6 — Quantization Benchmark
================================
Tez için "Quantization Etkisi" karşılaştırması.

Test edilen konfigürasyonlar:
  1. Gemma-3-4B-it  Q4_K_M (4-bit)
  2. Gemma-3-4B-it  Q8_0   (8-bit)
  3. Qwen3-8B       Q4_K_M (4-bit)
  4. Qwen3-8B       Q8_0   (8-bit)

Ölçülen metrikler her model için:
  - Yükleme süresi (s)
  - Yanıt süresi / token (ms)
  - Toplam VRAM kullanımı (MB)
  - Hayal_Puanı (F1 üzerinden tahmini)
  - BERTScore F1

Çıktılar:
  experiments/ablation/quant_benchmark.csv
  experiments/ablation/quant_benchmark.json
  MLflow'a log
"""

import json
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger

random.seed(42)
np.random.seed(42)

HF_CACHE_DIR  = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
OUTPUT_DIR    = Path("experiments/ablation")
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── TEST SORULARI ─────────────────────────────────────────────────────────────
TEST_PROMPTS = [
    "Mevduat faiz oranı nedir?",
    "Kredi başvurusu için hangi belgeler gerekir?",
    "2024 yılında Türkiye'nin enflasyon oranı neydi?",
    "DCF analizi nasıl yapılır, sermaye maliyeti nasıl hesaplanır?",
    "Hisse senedi değerleme yöntemlerini karşılaştırır mısın?",
]

REFERENCE_ANSWERS = [
    "Mevduat faiz oranı, bankaya yatırılan para için belirli bir süre sonunda kazanılan faizi ifade eder.",
    "Kredi başvurusu için kimlik belgesi, gelir belgesi ve ikamet belgesi gereklidir.",
    "2024 yılı Türkiye enflasyonu TÜFE bazında yaklaşık %65 seviyelerinde gerçekleşmiştir.",
    "DCF analizinde gelecekteki nakit akışları iskonto edilerek bugünkü değerine getirilir; sermaye maliyeti WACC formülüyle hesaplanır.",
    "Hisse senedi değerleme yöntemleri arasında P/E oranı, P/B oranı, DCF ve temettü indirgeme modeli sayılabilir.",
]


@dataclass
class BenchmarkResult:
    model_name:      str
    quantization:    str
    load_time_s:     float
    vram_mb:         float
    tokens_per_sec:  float
    avg_latency_ms:  float
    bertscore_f1:    float
    rouge1:          float
    rougeL:          float


# ─── GGUF DOSYASI KEŞİF ───────────────────────────────────────────────────────

def _find_gguf(model_id: str, quant_tag: str) -> Path | None:
    """HF cache içinde model_id için quant_tag içeren GGUF dosyasını bulur."""
    model_dir_name = "models--" + model_id.replace("/", "--")
    search_root    = HF_CACHE_DIR / "hub" / model_dir_name
    if not search_root.exists():
        logger.warning(f"Model dizini yok: {search_root}")
        return None
    for p in search_root.rglob("*.gguf"):
        if quant_tag.lower() in p.name.lower():
            logger.info(f"GGUF bulundu: {p}")
            return p
    return None


# ─── VRAM ÖLÇÜM ───────────────────────────────────────────────────────────────

def _vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def _gpu_total_vram_mb() -> float:
    """nvidia-smi ile kullanılan VRAM'i mb döner."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return _vram_mb()


# ─── BERTScore ────────────────────────────────────────────────────────────────

def _bertscore(predictions: list[str], references: list[str]) -> float:
    try:
        from bert_score import score as bs_score
        _, _, F1 = bs_score(predictions, references, lang="tr", verbose=False)
        return float(F1.mean())
    except ImportError:
        logger.warning("bert_score kütüphanesi yok, 0.0 döndürülüyor.")
        return 0.0


# ─── ROUGE ────────────────────────────────────────────────────────────────────

def _rouge(predictions: list[str], references: list[str]) -> tuple[float, float]:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
        r1s, rLs = [], []
        for pred, ref in zip(predictions, references):
            sc = scorer.score(ref, pred)
            r1s.append(sc["rouge1"].fmeasure)
            rLs.append(sc["rougeL"].fmeasure)
        return float(np.mean(r1s)), float(np.mean(rLs))
    except ImportError:
        logger.warning("rouge_score yok, 0.0 döndürülüyor.")
        return 0.0, 0.0


# ─── BENCHMARK FONKSİYONU ─────────────────────────────────────────────────────

def benchmark_gguf(model_id: str, quant_tag: str) -> BenchmarkResult | None:
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("llama-cpp-python kurulu değil.")
        return None

    gguf_path = _find_gguf(model_id, quant_tag)
    if gguf_path is None:
        logger.warning(f"GGUF bulunamadı: {model_id} [{quant_tag}]")
        return None

    # Yük süresi
    torch.cuda.empty_cache()
    vram_before = _gpu_total_vram_mb()
    t0          = time.perf_counter()
    llm = Llama(
        model_path  = str(gguf_path),
        n_gpu_layers = -1,   # Tüm katmanları GPU'ya yükle
        n_ctx        = 2048,
        verbose      = False,
        seed         = 42,
    )
    load_time_s  = time.perf_counter() - t0
    vram_after   = _gpu_total_vram_mb()
    vram_mb      = max(vram_after - vram_before, 0.0)

    logger.info(f"{model_id} [{quant_tag}] yüklendi ({load_time_s:.1f}s, VRAM≈{vram_mb:.0f}MB)")

    # Yanıt üretim benchmark
    predictions      = []
    total_latency_ms = 0.0
    total_tokens     = 0

    for prompt in TEST_PROMPTS:
        msgs = [
            {"role": "system", "content": "Sen Türkçe finans asistanısın. Kısa cevap ver."},
            {"role": "user",   "content": prompt},
        ]
        t1       = time.perf_counter()
        out      = llm.create_chat_completion(messages=msgs, max_tokens=256, temperature=0.1)
        elapsed  = (time.perf_counter() - t1) * 1000
        answer   = out["choices"][0]["message"]["content"].strip()
        n_tokens = out["usage"]["completion_tokens"]

        predictions.append(answer)
        total_latency_ms += elapsed
        total_tokens     += n_tokens

    avg_latency_ms = total_latency_ms / len(TEST_PROMPTS)
    tokens_per_sec = (total_tokens / (total_latency_ms / 1000)) if total_latency_ms > 0 else 0.0

    # Kalite metrikleri
    bertscore_f1 = _bertscore(predictions, REFERENCE_ANSWERS)
    rouge1, rougeL = _rouge(predictions, REFERENCE_ANSWERS)

    logger.info(
        f"{model_id} [{quant_tag}] -> "
        f"latency={avg_latency_ms:.0f}ms tps={tokens_per_sec:.1f} "
        f"bert={bertscore_f1:.3f} r1={rouge1:.3f} rL={rougeL:.3f}"
    )

    return BenchmarkResult(
        model_name     = model_id,
        quantization   = quant_tag,
        load_time_s    = round(load_time_s, 2),
        vram_mb        = round(vram_mb, 1),
        tokens_per_sec = round(tokens_per_sec, 2),
        avg_latency_ms = round(avg_latency_ms, 1),
        bertscore_f1   = round(bertscore_f1, 4),
        rouge1         = round(rouge1, 4),
        rougeL         = round(rougeL, 4),
    )


# ─── ANA ─────────────────────────────────────────────────────────────────────

CONFIGS = [
    ("unsloth/gemma-3-4b-it-GGUF",  "Q4_K_M"),
    ("unsloth/gemma-3-4b-it-GGUF",  "Q8_0"),
    ("unsloth/Qwen3-8B-GGUF",       "Q4_K_M"),
    ("unsloth/Qwen3-8B-GGUF",       "Q8_0"),
]


def main():
    import csv

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("quantization-benchmark")
        run = mlflow.start_run(run_name="quant-bench")
    except Exception:
        run = None

    results = []
    for model_id, quant_tag in CONFIGS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmark: {model_id} [{quant_tag}]")
        logger.info(f"{'='*60}")
        r = benchmark_gguf(model_id, quant_tag)
        if r:
            results.append(asdict(r))

    # CSV kaydet
    csv_path = OUTPUT_DIR / "quant_benchmark.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"CSV kaydedildi: {csv_path}")

    # JSON kaydet
    json_path = OUTPUT_DIR / "quant_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON kaydedildi: {json_path}")

    # MLflow log
    if run:
        try:
            for r in results:
                prefix = f"{r['model_name'].split('/')[-1]}_{r['quantization']}"
                for k, v in r.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{prefix}/{k}", v)
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"MLflow log hatası: {e}")

    # Özet tablo
    logger.info("\nQuantization Benchmark Özeti:")
    logger.info(f"{'Model':<30} {'Quant':<8} {'VRAM':>8} {'TPS':>7} {'Lat':>8} {'BERT':>7}")
    logger.info("-" * 75)
    for r in results:
        logger.info(
            f"{r['model_name'].split('/')[-1]:<30} "
            f"{r['quantization']:<8} "
            f"{r['vram_mb']:>6.0f}MB "
            f"{r['tokens_per_sec']:>7.1f} "
            f"{r['avg_latency_ms']:>6.0f}ms "
            f"{r['bertscore_f1']:>7.4f}"
        )


if __name__ == "__main__":
    main()
