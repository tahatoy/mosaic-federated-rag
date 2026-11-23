"""
Faz 4.1 — Embedding Modeli Fine-tuning
========================================
TurkEmbed4Retrieval ve multilingual-e5-small modellerini
Türkçe finans domain'ine ince ayar yapar.

Yöntem: Contrastive Learning (MultipleNegativesRankingLoss)
Veri: Finans sentence pair'ları (pozitif/negatif)
Tokenizer: Opsiyonel hibrit tokenizer swap

Çıktı:
  models/embeddings/turkembed_finetuned/
  models/embeddings/e5small_finetuned/

Kullanım:
  python models/embeddings/train_embeddings.py [--model turkembed|e5small|both]
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

ROOT = Path(__file__).parent.parent.parent
PROC_DIR  = ROOT / "data" / "processed"
RAW_DIR   = ROOT / "data" / "raw"
EMBED_DIR = ROOT / "models" / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

MODEL_CONFIGS = {
    "turkembed": {
        "name": "newmindai/TurkEmbed4Retrieval",
        "output_dir": EMBED_DIR / "turkembed_finetuned",
    },
    "e5small": {
        "name": "intfloat/multilingual-e5-small",
        "output_dir": EMBED_DIR / "e5small_finetuned",
    },
}


# ─── SENTENCE PAIR VERİSİ ─────────────────────────────────────────────────────

def build_sentence_pairs(max_pairs: int = 5000) -> list[tuple[str, str]]:
    """
    Finans corpus'undan pozitif soru-cevap çiftleri oluşturur.
    Negatif örnekler framework tarafından in-batch oluşturulur.
    """
    pairs: list[tuple[str, str]] = []

    # 1. Sentetik Q&A'dan çiftler
    qa_path = RAW_DIR / "synthetic_qa.jsonl"
    if qa_path.exists():
        with open(qa_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    q = rec.get("question", "")
                    a = rec.get("answer", "")
                    if q and a:
                        pairs.append((q, a[:256]))
                except json.JSONDecodeError:
                    continue
        logger.info(f"Sentetik Q&A'dan: {len(pairs)} çift")

    # 2. Finance instructions'dan çiftler
    inst_path = PROC_DIR / "finance_rag_train.jsonl"
    if inst_path.exists():
        with open(inst_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    instruction = rec.get("instruction", "")
                    response    = rec.get("response", "")
                    if instruction and response:
                        pairs.append((instruction[:256], response[:256]))
                except json.JSONDecodeError:
                    continue

    # 3. Finans haberi başlık -> ilk paragraf çiftleri
    news_path = PROC_DIR / "fin_news_train.jsonl"
    if news_path.exists():
        with open(news_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    title = rec.get("title", "")
                    text  = rec.get("text", "")
                    if title and text and len(text) > 100:
                        first_para = text.split("\n")[0][:300]
                        if len(first_para) > 50:
                            pairs.append((title, first_para))
                except json.JSONDecodeError:
                    continue

    random.shuffle(pairs)
    logger.info(f"Toplam sentence pair: {len(pairs)}")
    return pairs[:max_pairs]


# ─── EĞİTİM ───────────────────────────────────────────────────────────────────

def train_embedding_model(
    model_config: dict,
    pairs: list[tuple[str, str]],
    epochs: int = 3,
    batch_size: int = 32,
    use_hybrid_tokenizer: bool = False,
) -> None:
    from sentence_transformers import (
        SentenceTransformer,
        InputExample,
        losses,
    )
    from torch.utils.data import DataLoader

    model_name = model_config["name"]
    output_dir = model_config["output_dir"]

    logger.info(f"Model yükleniyor: {model_name}")
    model = SentenceTransformer(model_name)

    # Opsiyonel: hibrit tokenizer swap
    if use_hybrid_tokenizer:
        try:
            from tokenizer.hybrid_tokenizer import HybridTokenizer
            token_dir = ROOT / "models" / "tokenizer"
            if (token_dir / "bpe.model").exists():
                # sentence-transformers tokenizer'ı doğrudan değiştirme desteklenmiyor
                # Bu yüzden hibrit ön işlemeyi girişte uyguluyoruz
                hybrid_tok = HybridTokenizer.from_pretrained(str(token_dir))
                logger.info("Hibrit tokenizer ön işleme aktif.")
                # Pair'leri hibrit tokenizer ile ön işle
                processed_pairs = []
                for q, a in pairs:
                    q_tokens = " ".join(hybrid_tok.tokenize(q))
                    a_tokens = " ".join(hybrid_tok.tokenize(a))
                    processed_pairs.append((q_tokens, a_tokens))
                pairs = processed_pairs
        except Exception as e:
            logger.warning(f"Hibrit tokenizer swap atlandı: {e}")

    # InputExample formatına dönüştür
    train_examples = [InputExample(texts=[q, a]) for q, a in pairs]

    # Val split (%10)
    n_val = max(1, len(train_examples) // 10)
    val_examples   = train_examples[:n_val]
    train_examples = train_examples[n_val:]

    logger.info(f"Eğitim: {len(train_examples)} | Val: {len(val_examples)}")

    # DataLoader
    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
    )

    # MultipleNegativesRankingLoss — in-batch negatives
    loss = losses.MultipleNegativesRankingLoss(model)

    # Warmup adımları
    warmup_steps = int(len(train_loader) * epochs * 0.1)

    logger.info(f"Fine-tuning başlıyor: {model_name}")
    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True,
        use_amp=torch.cuda.is_available(),  # Mixed precision
    )

    logger.info(f"Model kaydedildi: {output_dir}")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("embedding_finetuning")
        with mlflow.start_run(run_name=f"faz4_{model_config['name'].split('/')[-1]}"):
            mlflow.log_params({
                "base_model": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "train_pairs": len(train_examples),
                "use_hybrid_tokenizer": use_hybrid_tokenizer,
            })
        logger.info("MLflow'a loglandı.")
    except Exception as e:
        logger.warning(f"MLflow hatası: {e}")


def main():
    parser = argparse.ArgumentParser(description="Embedding Fine-tuning")
    parser.add_argument("--model", choices=["turkembed", "e5small", "both"],
                        default="both", help="Hangi model eğitilsin")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-pairs",  type=int, default=5000)
    parser.add_argument("--use-hybrid-tokenizer", action="store_true",
                        help="Hibrit tokenizer ön işleme uygula")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Faz 4.1 — Embedding Modeli Fine-tuning")
    logger.info("=" * 60)

    pairs = build_sentence_pairs(max_pairs=args.max_pairs)
    if not pairs:
        logger.error("Sentence pair bulunamadı — önce veri toplama scriptlerini çalıştırın.")
        return

    models_to_train = (
        list(MODEL_CONFIGS.keys())
        if args.model == "both"
        else [args.model]
    )

    for model_key in models_to_train:
        config = MODEL_CONFIGS[model_key]
        logger.info(f"\nEğitim: {config['name']}")
        train_embedding_model(
            model_config=config,
            pairs=pairs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_hybrid_tokenizer=args.use_hybrid_tokenizer,
        )

    logger.info("Tüm embedding modelleri eğitimi tamamlandı.")


if __name__ == "__main__":
    main()
