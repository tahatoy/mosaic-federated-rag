"""
Faz 2.2 — BPE Trainer
======================
Finans corpus'u üzerinde SentencePiece BPE modeli eğitir.
Girdi: Zemberek morfem çıktıları (veya ham corpus)
Çıktı: models/tokenizer/bpe.model + bpe.vocab

Kullanım:
  python tokenizer/train_bpe.py [--vocab-size 8000]
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).parent.parent
PROC_DIR  = ROOT / "data" / "processed"
TOKEN_DIR = ROOT / "models" / "tokenizer"
TOKEN_DIR.mkdir(parents=True, exist_ok=True)

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level} | {message}")


def build_morpheme_corpus(max_lines: int = 500_000) -> Path:
    """
    Pretraining corpus + finance corpus'unu Zemberek morfemlerine çevirerek
    BPE eğitimi için geçici bir dosya oluşturur.
    """
    from tokenizer.zemberek_wrapper import get_analyzer
    import json

    analyzer = get_analyzer()

    sources = [
        PROC_DIR / "pretraining_corpus.txt",
        PROC_DIR / "finance_rag_train.jsonl",
    ]

    tmp_path = TOKEN_DIR / "morpheme_corpus.tmp.txt"
    logger.info(f"Morfem corpus oluşturuluyor (max {max_lines:,} satır)...")

    line_count = 0
    with open(tmp_path, "w", encoding="utf-8") as out:
        for src in sources:
            if not src.exists():
                logger.warning(f"Bulunamadı: {src}")
                continue

            if src.suffix == ".txt":
                with open(src, encoding="utf-8") as f:
                    for line in f:
                        if line_count >= max_lines:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        # Her kelimeyi morfemlerine ayır
                        if analyzer.is_available:
                            morphemized = analyzer.tokenize_sentence(line)
                            out_line = " ".join(morphemized)
                        else:
                            out_line = line
                        out.write(out_line + "\n")
                        line_count += 1

            elif src.suffix == ".jsonl":
                with open(src, encoding="utf-8") as f:
                    for raw_line in f:
                        if line_count >= max_lines:
                            break
                        try:
                            rec = json.loads(raw_line)
                            text = rec.get("response") or rec.get("text") or ""
                            if not text:
                                continue
                            if analyzer.is_available:
                                morphemized = analyzer.tokenize_sentence(text)
                                out_line = " ".join(morphemized)
                            else:
                                out_line = text
                            out.write(out_line + "\n")
                            line_count += 1
                        except json.JSONDecodeError:
                            continue

    logger.info(f"Morfem corpus hazır: {line_count:,} satır -> {tmp_path}")
    return tmp_path


def train_bpe(
    corpus_path: Path,
    model_prefix: str = "bpe",
    vocab_size: int = 8000,
    character_coverage: float = 0.9995,
) -> Path:
    """
    SentencePiece BPE modeli eğitir.

    Returns:
        Eğitilen modelin .model dosyası yolu
    """
    try:
        import sentencepiece as spm
    except ImportError:
        logger.error("sentencepiece kurulu değil: pip install sentencepiece")
        raise

    model_path = TOKEN_DIR / f"{model_prefix}.model"
    logger.info(f"BPE eğitimi başlıyor: vocab_size={vocab_size}, corpus={corpus_path}")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(TOKEN_DIR / model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="[PAD]",
        unk_piece="[UNK]",
        bos_piece="[BOS]",
        eos_piece="[EOS]",
        # Türkçe'de önemli karakterleri koru
        user_defined_symbols=["[SEP]", "[CLS]", "[MASK]"],
        byte_fallback=True,  # OOV karakterler için byte-level fallback
        seed_sentencepiece_size=100_000,
        shuffle_input_sentence=True,
        num_threads=4,
        # Türkçeye özel: noktalama işaretlerini ayrı token yap
        split_by_unicode_script=True,
        split_by_number=True,
        split_digits=True,
    )

    logger.info(f"BPE modeli kaydedildi: {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="BPE Tokenizer Eğitici")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Kelime hazinesi boyutu")
    parser.add_argument("--skip-morpheme", action="store_true",
                        help="Zemberek morfem dönüşümünü atla (ham corpus kullan)")
    parser.add_argument("--max-lines", type=int, default=500_000,
                        help="Corpus'tan kaç satır kullanılsın")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Faz 2.2 — BPE Tokenizer Eğitimi")
    logger.info("=" * 60)

    if args.skip_morpheme:
        # Ham corpus kullan
        corpus_path = PROC_DIR / "pretraining_corpus.txt"
        if not corpus_path.exists():
            logger.error(f"Corpus bulunamadı: {corpus_path}")
            return
    else:
        corpus_path = build_morpheme_corpus(max_lines=args.max_lines)

    model_path = train_bpe(corpus_path, vocab_size=args.vocab_size)
    logger.info(f"Eğitim tamamlandı: {model_path}")


if __name__ == "__main__":
    main()
