"""
Faz 3.1 — Router Dataset Builder
===================================
MASSIVE + CLINC + COMPLEX kaynaklarını birleştirerek dengeli bir
3-sınıf router eğitim seti oluşturur.

Sınıflar:
  0 = chitchat  (günlük konuşma, finans dışı)
  1 = simple    (basit finans sorusu, lokal RAG yeterli)
  2 = complex   (analitik finans sorusu, LLM + global RAG gerekli)

Çıktı:
  data/processed/router_train.jsonl
  data/processed/router_val.jsonl
  data/processed/router_test.jsonl
  experiments/ablation/router_class_dist.json

Kullanım:
  python models/router/build_router_dataset.py [--target-per-class 2500]
"""

import argparse
import random
import sys
import json
from collections import Counter
from pathlib import Path

import numpy as np

random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).parent.parent.parent
PROC_DIR    = ROOT / "data" / "processed"
RAW_DIR     = ROOT / "data" / "raw"
ABLATION    = ROOT / "experiments" / "ablation"
ABLATION.mkdir(parents=True, exist_ok=True)

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level} | {message}")


SIMPLE_FINANCE_TEMPLATES = [
    "Kredi faizi ne kadar?",
    "Hesabımın bakiyesi nedir?",
    "Döviz kurları ne durumda?",
    "Altın fiyatı kaç TL?",
    "Şubenin çalışma saatleri neler?",
    "IBAN numaram nedir?",
    "Mevduat faizleri nasıl?",
    "ATM nerede?",
    "EFT ücreti nedir?",
    "Kartımı nasıl aktive ederim?",
    "Kredi notu sorgulama nasıl yapılır?",
    "Bireysel emeklilik başlatmak istiyorum.",
    "Konut kredisi başvurusu nasıl yapılır?",
    "Otomatik ödeme talimatı verebilir miyim?",
    "Hesabıma yabancı para transferi gelebilir mi?",
    "Taşıt kredisi faiz oranı ne kadardır?",
    "Vadeli mevduat açmak istiyorum.",
    "Kredi kartı limitimi artırabilir miyim?",
    "Havale işlemi nasıl yapılır?",
    "İnternet bankacılığı şifremi unuttum.",
]


def load_source(path: Path) -> list[dict]:
    """JSONL dosyasını yükler."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def build_dataset(target_per_class: int = 2500) -> dict[str, list[dict]]:
    """Kaynak dosyaları birleştirerek dengeli dataset oluşturur."""

    # ── Sınıf 0: CHITCHAT ────────────────────────────────────────────
    chitchat_sources = [
        load_source(PROC_DIR / "router_train.jsonl"),
        load_source(RAW_DIR / "router_clinc_tr.jsonl"),
        load_source(RAW_DIR / "router_massive.jsonl"),
    ]
    chitchat = []
    for src in chitchat_sources:
        chitchat.extend([r for r in src if r.get("label") == 0])

    # ── Sınıf 1: SIMPLE ──────────────────────────────────────────────
    simple_sources = [
        load_source(PROC_DIR / "router_train.jsonl"),
        load_source(RAW_DIR / "router_massive.jsonl"),
    ]
    simple = []
    for src in simple_sources:
        simple.extend([r for r in src if r.get("label") == 1])

    # Şablon tabanlı basit örnekler ekle (yetersizse)
    if len(simple) < target_per_class:
        for tmpl in SIMPLE_FINANCE_TEMPLATES:
            simple.append({
                "source": "template_simple",
                "text": tmpl,
                "label": 1,
                "label_name": "simple",
            })

    # ── Sınıf 2: COMPLEX ─────────────────────────────────────────────
    complex_sources = [
        load_source(RAW_DIR / "router_complex.jsonl"),
        load_source(RAW_DIR / "router_synth_complex.jsonl"),
        load_source(PROC_DIR / "router_train.jsonl"),
    ]
    complex_recs = []
    for src in complex_sources:
        complex_recs.extend([r for r in src if r.get("label") == 2])

    raw_counts = {
        "chitchat": len(chitchat),
        "simple": len(simple),
        "complex": len(complex_recs),
    }
    logger.info(f"Ham sınıf sayıları: {raw_counts}")

    # Dengele: her sınıftan target_per_class örnek al
    random.shuffle(chitchat)
    random.shuffle(simple)
    random.shuffle(complex_recs)

    chitchat    = chitchat[:target_per_class]
    simple      = simple[:target_per_class]
    complex_recs = complex_recs[:target_per_class]

    # Birleştir ve karıştır
    all_records = chitchat + simple + complex_recs
    random.shuffle(all_records)

    # 80/10/10 split
    n = len(all_records)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)

    splits = {
        "train": all_records[:n_train],
        "val":   all_records[n_train:n_train + n_val],
        "test":  all_records[n_train + n_val:],
    }

    # Dağılım raporu
    for split_name, recs in splits.items():
        dist = Counter(r.get("label") for r in recs)
        logger.info(f"{split_name}: {len(recs)} -> {dict(dist)}")

    return splits


def save_splits(splits: dict) -> None:
    for split_name, records in splits.items():
        out_path = PROC_DIR / f"router_{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Kaydedildi: {out_path} ({len(records)} kayıt)")

    # Sınıf dağılımı istatistikleri
    dist_info = {}
    for split_name, records in splits.items():
        dist_info[split_name] = dict(Counter(r.get("label") for r in records))
    stats_path = ABLATION / "router_class_dist.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(dist_info, f, indent=2)
    logger.info(f"Dağılım istatistikleri: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Router Dataset Builder")
    parser.add_argument("--target-per-class", type=int, default=2500,
                        help="Her sınıftan kaç örnek (varsayılan: 2500)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Faz 3.1 — Router Dataset Hazırlama")
    logger.info("=" * 60)

    splits = build_dataset(target_per_class=args.target_per_class)
    save_splits(splits)
    logger.info("Tamamlandı.")


if __name__ == "__main__":
    main()
