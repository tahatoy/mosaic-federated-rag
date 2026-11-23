"""
Faz 5.2 — Flower Federated Learning İstemci (Şube)
====================================================
Router modelinin (BERT) ağırlıklarını federe öğrenir.
Her şube kendi split'i üzerinde fit/evaluate yapar.

BRANCH_ID=1  ->  Bireysel Bankacılık Şubesi
BRANCH_ID=2  ->  Kurumsal Bankacılık Şubesi
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from loguru import logger

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ─── ORTAM DEĞİŞKENLERİ ───────────────────────────────────────────────────────
BRANCH_ID        = int(os.getenv("BRANCH_ID",           "1"))
SERVER_ADDRESS   = os.getenv("FL_SERVER_ADDRESS",        "127.0.0.1:9080")

# Docker hostname kontrolü
if "flower-server" in SERVER_ADDRESS.lower() and not os.path.exists("/.dockerenv"):
    logger.warning(
        f"Docker-dahili Flower sunucu adresi tespit edildi: {SERVER_ADDRESS}. "
        "Yerel çalışmada bu adres çözülemeyebilir. 127.0.0.1:9080 deneniyor..."
    )
    # Eğer varsayılan değer ise veya çözülemez görünüyorsa yerel adrese çek
    if SERVER_ADDRESS == "flower-server:9080":
        SERVER_ADDRESS = "127.0.0.1:9080"
DATA_DIR         = Path(os.getenv("DATA_DIR",            "/app/data/splits"))
MODEL_DIR        = Path(os.getenv("MODEL_DIR",           "/app/models/router"))
LOCAL_EPOCHS     = int(os.getenv("FL_LOCAL_EPOCHS",      "3"))
BATCH_SIZE       = int(os.getenv("FL_BATCH_SIZE",        "32"))
LR               = float(os.getenv("FL_LR",             "2e-5"))
MAX_LENGTH       = int(os.getenv("FL_MAX_LENGTH",        "128"))
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI",      "http://mlflow:5000")
# FedProx proximal regularization — µ=0 degerleri standart FedAvg'ye esittir
FEDPROX_MU       = float(os.getenv("FEDPROX_MU",         "0.01"))

LABEL_NAMES      = ["chitchat", "simple", "complex"]


# ─── VERİ YÜKLEME ─────────────────────────────────────────────────────────────

def _load_branch_data(split: str) -> Tuple[List[str], List[int]]:
    """
    data/splits/branch_{BRANCH_ID}/{split}.jsonl dosyasını yükler.
    Her satır: {"text": "...", "label": 0|1|2}
    """
    import json
    path = DATA_DIR / f"branch_{BRANCH_ID}" / f"{split}.jsonl"
    texts, labels = [], []
    if not path.exists():
        logger.warning(f"Şube verisi bulunamadı: {path}")
        return texts, labels
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            texts.append(item["text"])
            labels.append(int(item["label"]))
    logger.info(f"[Branch {BRANCH_ID}] {split}: {len(texts)} örnek yüklendi.")
    return texts, labels


class RouterDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer):
        self.encodings = tokenizer(
            texts,
            max_length  = MAX_LENGTH,
            padding     = "max_length",
            truncation  = True,
            return_tensors = "pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ─── MODEL YARDIMCILARI ───────────────────────────────────────────────────────

def _get_weights(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_weights(model: torch.nn.Module, weights: List[np.ndarray]):
    state_dict_keys = list(model.state_dict().keys())
    new_state = {k: torch.tensor(v) for k, v in zip(state_dict_keys, weights)}
    model.load_state_dict(new_state, strict=True)


# ─── FLOWER İSTEMCİSİ ─────────────────────────────────────────────────────────

class RouterClient(fl.client.NumPyClient):
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Branch {BRANCH_ID}] Cihaz: {self.device}")

        checkpoint = str(MODEL_DIR)
        fallback   = "dbmdz/bert-base-turkish-cased"
        src        = checkpoint if Path(checkpoint).exists() else fallback
        logger.info(f"Model yükleniyor: {src}")

        self.tokenizer = AutoTokenizer.from_pretrained(src)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            src, num_labels=3
        ).to(self.device)

        train_texts, train_labels = _load_branch_data("train")
        val_texts,   val_labels   = _load_branch_data("dev")

        self.train_ds = RouterDataset(train_texts, train_labels, self.tokenizer)
        self.val_ds   = RouterDataset(val_texts,   val_labels,   self.tokenizer)

    # ─── Flower API ───────────────────────────────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return _get_weights(self.model)

    def set_parameters(self, parameters: List[np.ndarray]):
        _set_weights(self.model, parameters)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)

        # FedProx: küresel ağırlıkları sabit referans olarak kaydet (Li et al. 2020)
        mu = config.get("fedprox_mu", FEDPROX_MU)
        global_params = [
            torch.tensor(p, device=self.device, requires_grad=False)
            for p in parameters
        ]

        loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size = BATCH_SIZE,
            shuffle    = True,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for _ in range(LOCAL_EPOCHS):
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                ).logits
                loss = criterion(logits, labels)

                # FedProx proximal terimi: (µ/2) * ||w_local - w_global||²
                if mu > 0.0:
                    prox = torch.tensor(0.0, device=self.device)
                    for local_p, global_p in zip(self.model.parameters(), global_params):
                        prox = prox + torch.sum((local_p - global_p) ** 2)
                    loss = loss + (mu / 2.0) * prox

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss    += loss.item() * labels.size(0)
                preds          = logits.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss     = total_loss    / total_samples
        avg_accuracy = total_correct / total_samples

        logger.info(
            f"[Branch {BRANCH_ID}] fit -> loss={avg_loss:.4f} acc={avg_accuracy:.4f} "
            f"n={total_samples} mu={mu}"
        )

        return (
            _get_weights(self.model),
            total_samples,
            {"loss": avg_loss, "accuracy": avg_accuracy, "branch_id": BRANCH_ID,
             "fedprox_mu": mu},
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)

        loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size = BATCH_SIZE,
            shuffle    = False,
        )
        criterion = torch.nn.CrossEntropyLoss()

        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                logits = self.model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                ).logits
                loss           = criterion(logits, labels)

                total_loss    += loss.item() * labels.size(0)
                preds          = logits.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss     = total_loss    / total_samples
        avg_accuracy = total_correct / total_samples

        logger.info(
            f"[Branch {BRANCH_ID}] eval -> loss={avg_loss:.4f} acc={avg_accuracy:.4f} "
            f"n={total_samples}"
        )

        return (
            avg_loss,
            total_samples,
            {"accuracy": avg_accuracy, "branch_id": BRANCH_ID},
        )


# ─── GİRİŞ NOKTASI ────────────────────────────────────────────────────────────

def main():
    logger.info(f"Flower istemci başlatılıyor | Branch {BRANCH_ID} -> {SERVER_ADDRESS}")
    fl.client.start_numpy_client(
        server_address = SERVER_ADDRESS,
        client         = RouterClient(),
    )
    logger.info(f"[Branch {BRANCH_ID}] Federated eğitim tamamlandı.")


if __name__ == "__main__":
    main()
