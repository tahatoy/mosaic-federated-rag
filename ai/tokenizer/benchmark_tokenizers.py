"""
Faz 2.3 — Tokenizer Kıyaslama Benchmark'ı
==========================================
5 tokenizer'ı karşılaştırır:
  1. SentencePiece baseline (ham corpus, morfem dönüşümsüz)
  2. BPE baseline     (ham corpus, morfem dönüşümsüz)
  3. Hibrit (Zemberek + BPE) — önerimiz
  4. newmindai/TurkEmbed4Retrieval tokenizer'ı
  5. intfloat/multilingual-e5-small tokenizer'ı

Metrikler:
  - Fertility score (token/kelime oranı — düşük = iyi)
  - OOV oranı (finans terminolojisi test seti üzerinde)
  - Retrieval Recall@5 (mini retrieval deneyi)

Çıktı:
  experiments/ablation/tokenizer_benchmark.json
  experiments/ablation/tokenizer_benchmark.csv

Kullanım:
  python tokenizer/benchmark_tokenizers.py
"""

import random
import sys
import json
import csv
from pathlib import Path

import numpy as np

random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).parent.parent
ABLATION_DIR = ROOT / "experiments" / "ablation"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

PROC_DIR  = ROOT / "data" / "processed"
TOKEN_DIR = ROOT / "models" / "tokenizer"

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

# ─── FINANS TERMİNOLOJİSİ TEST SETİ ─────────────────────────────────────────

FINANCE_TERMS = [
    "akreditif", "alivre", "amortismanlar", "arbitraj", "borsaistanbul",
    "BIST100", "TCMB", "BDDK", "CMB", "SPK",
    "büyümeoranlari", "cariaçık", "dövizrezer", "tahvilgetiri",
    "opsiyon", "endeksleme", "faizdışıgelir", "kredirisktaşıma",
    "likidite", "menkul", "mevduat", "naktakasi", "portföy",
    "prepayment", "refinansman", "repofaiz", "securitization", "swap",
    "türev", "volatilite", "yuklenme",
]

FINANCE_SENTENCES = [
    "TCMB politika faizini 250 baz puan artırdı.",
    "Şirketin EBITDA marjı %18'e geriledi.",
    "BIST100 endeksi günlük yüzde 1,4 değer kazandı.",
    "Cari açık beklentilerin üzerinde geldi.",
    "Kısa vadeli döviz borçları 12 milyar dolara ulaştı.",
    "Bankacılık sektörünün takipteki kredi oranı arttı.",
    "Tahvil faizleri para politikasındaki belirsizlikle yükseldi.",
    "Yatırım fonlarının portföy büyüklüğü rekor kırdı.",
    "Şirket, halka arzdan 500 milyon TL topladı.",
    "Konut kredisi faizleri tekrar düşüşe geçti.",
]


# ─── TOKENİZER YÜKLEYİCİLER ──────────────────────────────────────────────────

def load_sentencepiece_baseline() -> object:
    """Ham corpus üzerinde eğitilmiş SP modeli (morfem dönüşümsüz)."""
    sp_path = TOKEN_DIR / "bpe_baseline.model"
    if not sp_path.exists():
        logger.warning(f"SP baseline modeli bulunamadı: {sp_path}")
        return None
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(str(sp_path))
        return sp
    except Exception as e:
        logger.error(f"SP yükleme hatası: {e}")
        return None


def load_bpe_baseline() -> object:
    """BPE baseline modeli."""
    bpe_path = TOKEN_DIR / "bpe.model"
    if not bpe_path.exists():
        return None
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(str(bpe_path))
        return sp
    except Exception:
        return None


def load_hybrid() -> object | None:
    """Zemberek + BPE hibrit tokenizer."""
    bpe_path = TOKEN_DIR / "bpe.model"
    if not bpe_path.exists():
        return None
    try:
        from tokenizer.hybrid_tokenizer import HybridTokenizer
        return HybridTokenizer(bpe_model_path=bpe_path)
    except Exception as e:
        logger.error(f"Hibrit tokenizer yükleme hatası: {e}")
        return None


def load_hf_tokenizer(model_name: str) -> object | None:
    """HuggingFace tokenizer yükle."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        logger.error(f"{model_name} tokenizer yükleme hatası: {e}")
        return None


# ─── METRİK HESAPLAMA ─────────────────────────────────────────────────────────

def tokenize_for_benchmark(tokenizer, text: str) -> list[str]:
    """Tokenizer tipine göre tokenize eder."""
    try:
        if hasattr(tokenizer, "tokenize"):
            # HuggingFace tokenizer
            return tokenizer.tokenize(text)
        elif hasattr(tokenizer, "EncodeAsPieces"):
            # SentencePiece
            return tokenizer.EncodeAsPieces(text)
        else:
            # HybridTokenizer
            return tokenizer.tokenize(text)
    except Exception:
        return text.split()


def fertility_score(tokenizer, sentences: list[str]) -> float:
    """
    Fertility = ortalama token/kelime oranı.
    Düşük = daha iyi sıkıştırma (finans terimlerini daha az token'a böler).
    """
    ratios = []
    for sent in sentences:
        words = sent.split()
        tokens = tokenize_for_benchmark(tokenizer, sent)
        if words:
            ratios.append(len(tokens) / len(words))
    return float(np.mean(ratios)) if ratios else 0.0


def oov_rate(tokenizer, terms: list[str]) -> float:
    """
    OOV oranı = kelime hazinesinde olmayan finans terimi yüzdesi.
    SentencePiece için <unk> token sayısı ile hesaplanır.
    """
    unk_count = 0
    total = 0

    for term in terms:
        tokens = tokenize_for_benchmark(tokenizer, term)
        total += len(tokens)
        for tok in tokens:
            # SentencePiece: <unk> veya ⁇ oov göstergesi
            if tok in ("<unk>", "[UNK]", "⁇") or tok.startswith("▁▁"):
                unk_count += 1

    return unk_count / total if total > 0 else 0.0


def recall_at_k(tokenizer, queries: list[str], docs: list[str], k: int = 5) -> float:
    """
    Basit token overlap Recall@K (embedding olmadan).
    Tezde tam retrieval F1 hesabı Faz 4'te yapılır — bu sadece tokenizer etkisini gösterir.
    """
    from rank_bm25 import BM25Okapi

    # BM25 corpus oluştur
    tokenized_docs = [tokenize_for_benchmark(tokenizer, d) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)

    hits = 0
    for i, query in enumerate(queries):
        q_tokens = tokenize_for_benchmark(tokenizer, query)
        scores = bm25.get_scores(q_tokens)
        top_k = np.argsort(scores)[::-1][:k]
        # Basit eşleşme: sorguyla aynı indeks belgede mi var?
        if i < len(docs) and i in top_k:
            hits += 1
    return hits / len(queries) if queries else 0.0


# ─── BENCHMARK ANA DÖNGÜSÜ ────────────────────────────────────────────────────

def run_benchmark() -> dict:
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"

    tokenizers = {
        "sentencepiece_baseline": load_sentencepiece_baseline(),
        "bpe_morfem":             load_bpe_baseline(),
        "hybrid_zemberek_bpe":    load_hybrid(),
        "TurkEmbed4Retrieval":    load_hf_tokenizer("newmindai/TurkEmbed4Retrieval"),
        "multilingual_e5_small":  load_hf_tokenizer("intfloat/multilingual-e5-small"),
    }

    results = {}
    for name, tok in tokenizers.items():
        if tok is None:
            logger.warning(f"{name}: yüklenemedi, atlanıyor")
            results[name] = {"error": "yüklenemedi"}
            continue

        logger.info(f"Benchmark: {name}")
        try:
            fs   = fertility_score(tok, FINANCE_SENTENCES)
            oov  = oov_rate(tok, FINANCE_TERMS)
            rec5 = recall_at_k(tok, FINANCE_SENTENCES, FINANCE_SENTENCES, k=5)

            results[name] = {
                "fertility_score": round(fs, 4),
                "oov_rate_finance": round(oov, 4),
                "recall_at_5": round(rec5, 4),
            }
            logger.info(f"  fertility={fs:.3f} | OOV={oov:.3f} | Recall@5={rec5:.3f}")
        except Exception as e:
            logger.error(f"{name} benchmark hatası: {e}")
            results[name] = {"error": str(e)}

    return results


def save_results(results: dict) -> None:
    # JSON
    json_path = ABLATION_DIR / "tokenizer_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON sonuçlar: {json_path}")

    # CSV
    csv_path = ABLATION_DIR / "tokenizer_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tokenizer", "fertility_score", "oov_rate_finance", "recall_at_5"])
        for name, metrics in results.items():
            if "error" not in metrics:
                writer.writerow([
                    name,
                    metrics.get("fertility_score", ""),
                    metrics.get("oov_rate_finance", ""),
                    metrics.get("recall_at_5", ""),
                ])
    logger.info(f"CSV sonuçlar: {csv_path}")


# ─── MLflow Entegrasyonu ──────────────────────────────────────────────────────

def log_to_mlflow(results: dict) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("tokenizer_benchmark")
        with mlflow.start_run(run_name="faz2_tokenizer_karsilastirma"):
            for tokenizer_name, metrics in results.items():
                if "error" not in metrics:
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{tokenizer_name}_{metric_name}", value)
        logger.info("MLflow'a loglandı.")
    except Exception as e:
        logger.warning(f"MLflow loglama hatası: {e}")


def main():
    logger.info("=" * 60)
    logger.info("Faz 2.3 — Tokenizer Benchmark")
    logger.info("=" * 60)

    results = run_benchmark()
    save_results(results)
    log_to_mlflow(results)

    # Özet tablo
    logger.info("\n" + "─" * 60)
    logger.info(f"{'Tokenizer':<35} {'Fertility':>10} {'OOV':>8} {'R@5':>8}")
    logger.info("─" * 60)
    for name, m in results.items():
        if "error" not in m:
            logger.info(
                f"{name:<35} {m['fertility_score']:>10.3f} "
                f"{m['oov_rate_finance']:>8.3f} {m['recall_at_5']:>8.3f}"
            )
    logger.info("─" * 60)


if __name__ == "__main__":
    main()
