"""
Faz 7.1 — Değerlendirme Metrikleri
=====================================
Tüm tez deneyleri için ortak metrik fonksiyonları.

Desteklenen metrikler:
  - Router: accuracy, macro-F1, confusion_matrix
  - RAG:    MRR@K, Recall@K, NDCG@K, answer_faithfulness (BERTScore)
  - E2E:    BERTScore F1, ROUGE-1/L, latency_p50/p95

Çıktılar experiments/results/ dizinine JSON olarak kaydedilir.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "ai" / "experiments" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── ROUTER METRİKLERİ ────────────────────────────────────────────────────────

def compute_router_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class F1 ve confusion matrix döner."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    labels      = [0, 1, 2]
    label_names = ["chitchat", "simple", "complex"]

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(
        y_true, y_pred,
        labels        = labels,
        target_names  = label_names,
        output_dict   = True,
        zero_division = 0,
    )
    return {
        "accuracy":       round(acc, 4),
        "macro_f1":       round(f1,  4),
        "confusion_matrix": cm,
        "per_class":      {
            name: {
                "precision": round(report[name]["precision"], 4),
                "recall":    round(report[name]["recall"],    4),
                "f1":        round(report[name]["f1-score"],  4),
                "support":   report[name]["support"],
            }
            for name in label_names
        },
    }


# ─── RAG METRİKLERİ ───────────────────────────────────────────────────────────

def compute_rag_metrics(
    retrieved_ids:  list[list[str]],
    relevant_ids:   list[list[str]],
    k_values:       list[int] = None,
) -> dict[str, Any]:
    """MRR@K, Recall@K ve NDCG@K hesaplar."""
    if k_values is None:
        k_values = [1, 3, 5]

    def _mrr_at_k(ret: list[str], rel: set[str], k: int) -> float:
        for rank, doc_id in enumerate(ret[:k], start=1):
            if doc_id in rel:
                return 1.0 / rank
        return 0.0

    def _recall_at_k(ret: list[str], rel: set[str], k: int) -> float:
        if not rel:
            return 0.0
        return len(set(ret[:k]) & rel) / len(rel)

    def _ndcg_at_k(ret: list[str], rel: set[str], k: int) -> float:
        dcg, idcg = 0.0, 0.0
        for rank, doc_id in enumerate(ret[:k], start=1):
            relevance = 1.0 if doc_id in rel else 0.0
            dcg      += relevance / np.log2(rank + 1)
        for rank in range(1, min(len(rel), k) + 1):
            idcg += 1.0 / np.log2(rank + 1)
        return dcg / idcg if idcg > 0 else 0.0

    metrics: dict[str, Any] = {}
    for k in k_values:
        mrr_scores    = []
        recall_scores = []
        ndcg_scores   = []
        for ret, rel_list in zip(retrieved_ids, relevant_ids):
            rel = set(rel_list)
            mrr_scores.append(_mrr_at_k(ret, rel, k))
            recall_scores.append(_recall_at_k(ret, rel, k))
            ndcg_scores.append(_ndcg_at_k(ret, rel, k))
        metrics[f"MRR@{k}"]    = round(float(np.mean(mrr_scores)),    4)
        metrics[f"Recall@{k}"] = round(float(np.mean(recall_scores)), 4)
        metrics[f"NDCG@{k}"]   = round(float(np.mean(ndcg_scores)),   4)

    return metrics


# ─── E2E METRİKLERİ ───────────────────────────────────────────────────────────

def compute_e2e_metrics(
    predictions: list[str],
    references:  list[str],
    latencies_ms: list[float] = None,
) -> dict[str, Any]:
    """BERTScore F1, ROUGE ve isteğe bağlı gecikme istatistikleri döner."""
    metrics: dict[str, Any] = {}

    # BERTScore
    try:
        from bert_score import score as bs_score
        _, _, F1 = bs_score(predictions, references, lang="tr", verbose=False)
        metrics["bertscore_f1"] = round(float(F1.mean()), 4)
    except ImportError:
        logger.warning("bert_score yok; metric atlandı.")
        metrics["bertscore_f1"] = None

    # ROUGE
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        r1s, r2s, rLs = [], [], []
        for pred, ref in zip(predictions, references):
            sc = scorer.score(ref, pred)
            r1s.append(sc["rouge1"].fmeasure)
            r2s.append(sc["rouge2"].fmeasure)
            rLs.append(sc["rougeL"].fmeasure)
        metrics["rouge1"] = round(float(np.mean(r1s)), 4)
        metrics["rouge2"] = round(float(np.mean(r2s)), 4)
        metrics["rougeL"] = round(float(np.mean(rLs)), 4)
    except ImportError:
        logger.warning("rouge_score yok; metric atlandı.")

    # Gecikme
    if latencies_ms:
        arr = np.array(latencies_ms)
        metrics["latency_p50_ms"] = round(float(np.percentile(arr, 50)),  1)
        metrics["latency_p95_ms"] = round(float(np.percentile(arr, 95)),  1)
        metrics["latency_mean_ms"] = round(float(arr.mean()),             1)

    return metrics


# ─── KAYDETME ─────────────────────────────────────────────────────────────────

def save_metrics(metrics: dict, name: str):
    path = OUTPUT_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Metrikler kaydedildi: {path}")
    return path
