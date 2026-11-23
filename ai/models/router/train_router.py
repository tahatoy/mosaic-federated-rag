"""
Faz 3.2 — BERT Router Fine-tuning
====================================
dbmdz/bert-base-turkish-cased modelini 3-sınıf sınıflandırma için
fine-tune eder.

Sınıflar:
  0 = chitchat  -> SLM'e yönlendir, RAG yok
  1 = simple    -> SLM'e yönlendir, lokal RAG
  2 = complex   -> LLM'e yönlendir, global RAG

Eğitim detayları:
  - Model: dbmdz/bert-base-turkish-cased
  - Görev: SequenceClassification (3 sınıf)
  - Batch: 32, LR: 2e-5, Epoch: 5
  - GPU: RTX 5060 Ti — ~2 saat
  - Seed: 42

Çıktı:
  models/router/checkpoint/   -> fine-tuned model
  experiments/ablation/router_metrics.json

Kullanım:
  python models/router/train_router.py [--epochs 5] [--batch-size 32]
"""

import argparse
import random
import sys
import json
from pathlib import Path

import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

ROOT = Path(__file__).parent.parent.parent
PROC_DIR    = ROOT / "data" / "processed"
ROUTER_DIR  = ROOT / "models" / "router"
CKPT_DIR    = ROUTER_DIR / "checkpoint"
ABLATION    = ROOT / "experiments" / "ablation"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
ABLATION.mkdir(parents=True, exist_ok=True)

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

BASE_MODEL  = "dbmdz/bert-base-turkish-cased"
NUM_LABELS  = 3
LABEL_NAMES = ["chitchat", "simple", "complex"]


# ─── DATASET ──────────────────────────────────────────────────────────────────

class RouterDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int = 128):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec   = self.records[idx]
        text  = rec.get("text", "")
        label = int(rec.get("label", 0))

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),
        }


def load_split(split_name: str) -> list[dict]:
    path = PROC_DIR / f"router_{split_name}.jsonl"
    if not path.exists():
        logger.warning(f"Split dosyası bulunamadı: {path}")
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ─── EĞİTİM ───────────────────────────────────────────────────────────────────

def train(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_length: int = 128,
):
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    logger.info(f"Cihaz: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Model: {BASE_MODEL}")
    logger.info(f"Epoch: {epochs} | Batch: {batch_size} | LR: {lr}")

    # Tokenizer ve model yükle
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label={i: lbl for i, lbl in enumerate(LABEL_NAMES)},
        label2id={lbl: i for i, lbl in enumerate(LABEL_NAMES)},
    )

    # Dataset yükle
    train_recs = load_split("train")
    val_recs   = load_split("val")

    if not train_recs:
        logger.error("Eğitim seti boş — önce build_router_dataset.py çalıştırın.")
        return

    logger.info(f"Eğitim: {len(train_recs)} | Doğrulama: {len(val_recs)}")

    train_ds = RouterDataset(train_recs, tokenizer, max_length=max_length)
    val_ds   = RouterDataset(val_recs,   tokenizer, max_length=max_length)

    # Metrik fonksiyonu
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc  = accuracy_score(labels, preds)
        f1   = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1}

    # Training arguments
    total_steps = (len(train_ds) // batch_size) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    training_args = TrainingArguments(
        output_dir=str(CKPT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",  # MLflow manuel loglanacak
        seed=42,
        dataloader_pin_memory=True,
        bf16=torch.cuda.is_available(),  # Blackwell BF16 destekli
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Eğitim başlıyor...")
    trainer.train()
    logger.info("Eğitim tamamlandı.")

    # Test seti değerlendirmesi
    test_recs = load_split("test")
    if test_recs:
        test_ds = RouterDataset(test_recs, tokenizer, max_length=max_length)
        test_results = trainer.predict(test_ds)
        test_preds = np.argmax(test_results.predictions, axis=-1)
        test_labels = [r.get("label") for r in test_recs]

        acc  = accuracy_score(test_labels, test_preds)
        f1   = f1_score(test_labels, test_preds, average="macro")
        cm   = confusion_matrix(test_labels, test_preds).tolist()

        metrics = {
            "test_accuracy": round(acc, 4),
            "test_f1_macro": round(f1, 4),
            "confusion_matrix": cm,
            "label_names": LABEL_NAMES,
        }

        metrics_path = ABLATION / "router_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as mf:
            json.dump(metrics, mf, indent=2)

        logger.info(f"\nTest Accuracy: {acc:.4f}")
        logger.info(f"Test F1 Macro: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{np.array(cm)}")

        # MLflow
        try:
            import mlflow
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("router_finetuning")
            with mlflow.start_run(run_name="faz3_bert_router"):
                mlflow.log_params({
                    "base_model": BASE_MODEL,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "max_length": max_length,
                })
                mlflow.log_metrics({"test_accuracy": acc, "test_f1_macro": f1})
                mlflow.log_artifact(str(metrics_path))
            logger.info("MLflow'a loglandı.")
        except Exception as e:
            logger.warning(f"MLflow hatası: {e}")

    # Modeli kaydet
    model.save_pretrained(str(CKPT_DIR))
    tokenizer.save_pretrained(str(CKPT_DIR))
    logger.info(f"Model kaydedildi: {CKPT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="BERT Router Fine-tuning")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max-length", type=int,   default=128)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Faz 3.2 — BERT Router Fine-tuning")
    logger.info("=" * 60)

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
