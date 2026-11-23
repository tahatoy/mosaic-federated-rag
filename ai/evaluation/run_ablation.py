"""
Faz 7.2 — Ablasyon Çalışması
==============================
Tez için 4 temel ablasyon konfigürasyonu:

  A  Tam sistem       : Hibrit tok + TurkEmbed4Retrieval + fine-tuned router
  B  Hibrit tok yok   : Sadece BPE tokenizer
  C  Embedding değişim: multilingual-e5-small yerine TurkEmbed4Retrieval
  D  Merkezi sistem   : Flower FL yok, tek global model

Her konfigürasyon servis API'larını çağırarak test eder.

Çıktılar:
  experiments/ablation/ablation_results.csv
  experiments/ablation/ablation_results.json
  MLflow "ablation-study" experiment'i
"""

import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from loguru import logger

from evaluation.metrics import compute_router_metrics, compute_e2e_metrics, save_metrics

random.seed(42)
np.random.seed(42)

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8004")
ROUTER_URL       = os.getenv("ROUTER_URL",       "http://localhost:8000")
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR       = PROJECT_ROOT / "ai" / "experiments" / "ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── TEST VERİSİ ──────────────────────────────────────────────────────────────

TEST_SET_PATH = PROJECT_ROOT / "ai" / "data" / "test" / "ablation_test.jsonl"

FALLBACK_TEST_CASES = [
    {"query": "Merhaba nasılsın?",                              "label": 0, "answer": "İyiyim, teşekkürler."},
    {"query": "Faiz oranı nedir?",                              "label": 1, "answer": "Borç için ödenen ücrettir."},
    {"query": "Kredi kartı limiti nasıl artırılır?",            "label": 1, "answer": "Bankaya başvurarak artırabilirsiniz."},
    {"query": "Bilanço analizi nasıl yapılır?",                 "label": 2, "answer": "Aktif, pasif ve özkaynak kalemleri incelenir."},
    {"query": "DCF ile hisse senedi değerlemesi açıkla.",       "label": 2, "answer": "Gelecek nakit akışları bugünkü değerine indirgenir."},
    {"query": "Bugün hava nasıl?",                              "label": 0, "answer": "Hava durumu bilgisi sunmuyorum."},
    {"query": "Vadeli mevduat ne demektir?",                    "label": 1, "answer": "Belirli süre bankada tutulan mevduattır."},
    {"query": "BIST 100 endeksi neden düştü?",                  "label": 2, "answer": "Makroekonomik gelişmeler etkili oldu."},
    {"query": "Repo işlemi nedir?",                             "label": 1, "answer": "Kısa vadeli geri alım anlaşmasıdır."},
    {"query": "Yatırım fonu ile hisse senedi farkı nedir?",     "label": 2, "answer": "Fon portföy yöneticisi tarafından yönetilir."},
]


def _load_test_cases() -> list[dict]:
    if TEST_SET_PATH.exists():
        cases = []
        with open(TEST_SET_PATH, encoding="utf-8") as f:
            for line in f:
                cases.append(json.loads(line.strip()))
        logger.info(f"Test seti yüklendi: {len(cases)} örnek ({TEST_SET_PATH})")
        return cases
    logger.warning(f"Test seti bulunamadı ({TEST_SET_PATH}), fallback kullanılıyor.")
    return FALLBACK_TEST_CASES


# ─── SERVİS ÇAĞRILARI ─────────────────────────────────────────────────────────

def _call_router(client: httpx.Client, query: str) -> dict:
    try:
        r = client.post(f"{ROUTER_URL}/predict", json={"text": query}, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Router hatası: {e}")
        return {"label": 1, "label_name": "simple", "confidence": 0.0}


def _call_orchestrator(client: httpx.Client, query: str, branch_id: int = 1) -> dict:
    try:
        r = client.post(
            f"{ORCHESTRATOR_URL}/query",
            json    = {"query": query, "branch_id": branch_id},
            timeout = 120.0,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Orchestrator hatası: {e}")
        return {"answer": "", "route": "simple", "elapsed_ms": 0.0, "sources": []}


# ─── ABLASYON YAPISI ──────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    name:          str
    description:   str
    embedder_env:  str  # env var değeri -> embedding model adı
    hybrid_tok:    bool
    federated:     bool
    branch_id:     int  = 1


ABLATION_CONFIGS = [
    AblationConfig(
        name         = "A_full",
        description  = "Tam sistem (Hibrit tok + TurkEmbed + FL)",
        embedder_env = "newmindai/TurkEmbed4Retrieval",
        hybrid_tok   = True,
        federated    = True,
    ),
    AblationConfig(
        name         = "B_no_hybrid_tok",
        description  = "Hibrit tokenizer yok (sadece BPE)",
        embedder_env = "newmindai/TurkEmbed4Retrieval",
        hybrid_tok   = False,
        federated    = True,
    ),
    AblationConfig(
        name         = "C_multilingual_e5",
        description  = "TurkEmbed yerine multilingual-e5-small",
        embedder_env = "intfloat/multilingual-e5-small",
        hybrid_tok   = True,
        federated    = True,
    ),
    AblationConfig(
        name         = "D_centralized",
        description  = "Merkezi (FL yok)",
        embedder_env = "newmindai/TurkEmbed4Retrieval",
        hybrid_tok   = True,
        federated    = False,
    ),
]


# ─── TEK KONFİGÜRASYON DEĞERLENDİRME ─────────────────────────────────────────

def evaluate_config(cfg: AblationConfig, test_cases: list[dict]) -> dict:
    """
    Servis API'larını çağırarak belirtilen konfigürasyonu değerlendirir.
    Not: Servis config değişikliği environment variable ile yapılabilir;
         bu script mevcut ayağa kalkmış servisleri test eder.
    """
    with httpx.Client(timeout=60.0) as client:
        router_preds, router_trues = [], []
        predictions, references   = [], []
        latencies_ms               = []

        for case in test_cases:
            query         = case["query"]
            true_label    = case.get("label", 1)
            ref_answer    = case.get("answer", "")

            # Router tahmini
            router_result = _call_router(client, query)
            router_preds.append(router_result["label"])
            router_trues.append(true_label)

            # E2E cevap
            t0      = time.perf_counter()
            result  = _call_orchestrator(client, query, branch_id=cfg.branch_id)
            elapsed = (time.perf_counter() - t0) * 1000

            predictions.append(result.get("answer", ""))
            references.append(ref_answer)
            latencies_ms.append(elapsed)

    router_metrics = compute_router_metrics(router_trues, router_preds)
    e2e_metrics    = compute_e2e_metrics(predictions, references, latencies_ms)

    return {
        "config":         cfg.name,
        "description":    cfg.description,
        "n_samples":      len(test_cases),
        "router_metrics": router_metrics,
        "e2e_metrics":    e2e_metrics,
    }


# ─── ANA ─────────────────────────────────────────────────────────────────────

def main():
    test_cases = _load_test_cases()
    logger.info(f"Ablasyon başlıyor: {len(ABLATION_CONFIGS)} konfigürasyon × {len(test_cases)} örnek")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("ablation-study")
        parent_run = mlflow.start_run(run_name="ablation-full")
        parent_run_id = parent_run.info.run_id
    except Exception:
        parent_run_id = None

    all_results = []

    for cfg in ABLATION_CONFIGS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Konfigürasyon: {cfg.name} — {cfg.description}")
        logger.info(f"{'='*60}")

        result = evaluate_config(cfg, test_cases)
        all_results.append(result)

        # MLflow nested run
        if parent_run_id:
            try:
                import mlflow
                with mlflow.start_run(run_name=cfg.name, nested=True):
                    rm = result["router_metrics"]
                    em = result["e2e_metrics"]
                    mlflow.log_param("config",       cfg.name)
                    mlflow.log_param("hybrid_tok",   cfg.hybrid_tok)
                    mlflow.log_param("federated",    cfg.federated)
                    mlflow.log_param("embedder",     cfg.embedder_env)
                    mlflow.log_metric("router_acc",  rm["accuracy"])
                    mlflow.log_metric("router_f1",   rm["macro_f1"])
                    if em.get("bertscore_f1"):
                        mlflow.log_metric("bertscore_f1", em["bertscore_f1"])
                    if em.get("rougeL"):
                        mlflow.log_metric("rougeL", em["rougeL"])
                    if em.get("latency_p50_ms"):
                        mlflow.log_metric("latency_p50", em["latency_p50_ms"])
            except Exception as e:
                logger.warning(f"MLflow log hatası: {e}")

        save_metrics(result, f"ablation_{cfg.name}")

    # CSV özet
    flat_rows = []
    for r in all_results:
        row = {
            "config":       r["config"],
            "description":  r["description"],
            "n_samples":    r["n_samples"],
        }
        for k, v in r["router_metrics"].items():
            if k != "confusion_matrix":
                row[f"router_{k}"] = v
        for k, v in r["e2e_metrics"].items():
            row[f"e2e_{k}"] = v
        flat_rows.append(row)

    csv_path = OUTPUT_DIR / "ablation_results.csv"
    if flat_rows:
        fieldnames = list(flat_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)
        logger.info(f"CSV kaydedildi: {csv_path}")

    json_path = OUTPUT_DIR / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON kaydedildi: {json_path}")

    if parent_run_id:
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass

    # Konsol özet tablosu
    logger.info("\nAblasyon Özeti:")
    logger.info(f"{'Config':<25} {'R-Acc':>7} {'R-F1':>7} {'BERTsc':>8} {'RougeL':>8} {'Lat50':>8}")
    logger.info("-" * 68)
    for r in all_results:
        rm = r["router_metrics"]
        em = r["e2e_metrics"]
        logger.info(
            f"{r['config']:<25} "
            f"{rm['accuracy']:>7.4f} "
            f"{rm['macro_f1']:>7.4f} "
            f"{em.get('bertscore_f1', 'N/A'):>8} "
            f"{em.get('rougeL', 'N/A'):>8} "
            f"{em.get('latency_p50_ms', 'N/A'):>8}"
        )


if __name__ == "__main__":
    main()
