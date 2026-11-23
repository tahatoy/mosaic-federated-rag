"""
Hibrit Türkçe Tokenizer (Consolidated)
=====================================
Zemberek morfolojik analizi ile BPE (SentencePiece) veya WordPiece (BERT) birleşimini destekler.

Mimari:
  1. Zemberek: Kelimeleri morfemlerine/köklerine ayırır.
  2. Subword: Tanınmayan/OOV tokenleri alt-birimlere (BPE veya WordPiece) böler.

Desteklenen Modlar:
  - BPE (Phase 2.2): bpe_model_path sağlandığında çalışır.
  - BERT (Classic): bpe_model_path=None olduğunda BERT (dbmdz) üzerinden çalışır.

Kullanım:
  from tokenizer.hybrid_tokenizer import HybridTokenizer
  tok = HybridTokenizer(use_lemma=True) # BERT-based fallback
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any
from loguru import logger

# Try to use ZemberekWrapper from the same package if available
try:
    from tokenizer.zemberek_wrapper import ZemberekAnalyzer
except ImportError:
    ZemberekAnalyzer = None

class HybridTokenizer:
    """
    Zemberek + BPE/BERT hibrit tokenizer.
    Supports both Phase 2.2 (BPE) and legacy (BERT-WordPiece) interfaces.
    """

    def __init__(
        self,
        bpe_model_path: str | Path | None = None,
        bert_model: str = "dbmdz/bert-base-turkish-cased",
        use_lemma: bool = True,
        use_zemberek: bool = True,
        max_length: int = 512,
        padding: str = "max_length",
    ):
        self.use_lemma = use_lemma or use_zemberek
        self.max_length = max_length
        self.padding = padding

        self._bpe = None
        self._bert = None
        self._zemberek = None

        # Setup Zemberek
        if self.use_lemma:
            if ZemberekAnalyzer:
                try:
                    self._zemberek = ZemberekAnalyzer(use_cache=True)
                except Exception as e:
                    logger.warning(f"ZemberekAnalyzer failed: {e}. Trying raw Zemberek...")

            if not self._zemberek:
                try:
                    from zemberek import TurkishMorphology
                    self._morphology = TurkishMorphology.create_with_defaults()
                    self._zemberek = self # Use self as a proxy if needed or just handle it in _lemmatize
                except ImportError:
                    logger.warning("Zemberek not installed. Morphology disabled.")

        # Setup Subword model
        if bpe_model_path:
            self._setup_bpe(bpe_model_path)
        else:
            self._setup_bert(bert_model)

        # Special Tokens (Shared logic)
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

    def _setup_bpe(self, path):
        import sentencepiece as spm
        self._bpe = spm.SentencePieceProcessor()
        self._bpe.Load(str(path))
        logger.debug(f"BPE model loaded: {path}")

    def _setup_bert(self, model_name):
        from transformers import BertTokenizerFast
        self._bert = BertTokenizerFast.from_pretrained(model_name)
        logger.debug(f"BERT tokenizer loaded: {model_name}")

    def _lemmatize(self, text: str) -> str:
        if not self.use_lemma or not self._zemberek:
            return text

        # If we have ZemberekAnalyzer (Phase 2.2 wrapper)
        if hasattr(self._zemberek, 'analyze') and self._zemberek != self:
            words = text.split()
            lemmas = []
            for w in words:
                res = self._zemberek.analyze(w)
                lemmas.extend(res if isinstance(res, list) else [res])
            return " ".join(lemmas)

        # Fallback to raw Zemberek (classic logic)
        if hasattr(self, '_morphology'):
            try:
                result = self._morphology.analyze_and_disambiguate(text)
                lemmas = []
                for wa in result.word_analyses:
                    analyses = wa.word_analysis.analysis_results
                    if analyses:
                        lemma = analyses[0].get_stem()
                        lemmas.append(lemma if lemma and len(lemma)>1 else wa.word_analysis.inp)
                    else:
                        lemmas.append(wa.word_analysis.inp)
                return " ".join(lemmas)
            except Exception:
                return text
        return text

    def encode(self, text: str, **kwargs) -> list[int]:
        """Encode text to token IDs. Compatibility layer for BERT/BPE."""
        if self._bert:
            # For BERT, we use the internal tokenizer
            lemmatized = self._lemmatize(text)
            return self._bert.encode(lemmatized, **kwargs)
        elif self._bpe:
            # For BPE, we use SentencePiece
            lemmatized = self._lemmatize(text)
            return self._bpe.EncodeAsIds(lemmatized)
        return []

    def decode(self, ids: list[int], **kwargs) -> str:
        """Decode token IDs back to text."""
        if self._bert:
            return self._bert.decode(ids, **kwargs)
        elif self._bpe:
            return self._bpe.DecodeIds(ids)
        return ""

    _TR_MAP = {
        "Ý": "İ", "ý": "ı", "Þ": "Ş", "þ": "ş",
        "Ð": "Ğ", "ð": "ğ",
        "î": "i", "â": "a", "û": "u",
        "\x00": "", "\ufeff": "",
    }

    def _normalize(self, text: str) -> str:
        """Alias for normalize method."""
        return self.normalize(text)

    def normalize(self, text: str) -> str:
        """
        Turkish text normalization for financial NLP:
        1. Fixes broken Windows-1252 / ISO-8859-9 characters (Ý -> İ, þ -> ş, etc.)
        2. Cleans null bytes, BOMs, and non-breaking spaces.
        3. Normalizes whitespace and financial symbols.
        """
        if not text:
            return ""
        for broken, fixed in self._TR_MAP.items():
            text = text.replace(broken, fixed)
        text = text.replace("\u00a0", " ").replace("\u200b", "")
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def count_tokens(self, text: str) -> int:
        """Accurately count tokens using subword model."""
        if not text:
            return 0
        return len(self.encode(text))

    def batch_encode(self, texts: list[str], **kwargs) -> list[list[int]]:
        """Batch encode multiple texts."""
        return [self.encode(t, **kwargs) for t in texts]

    def tokenize(self, text: str, **kwargs) -> Any:
        normalized = text.strip()
        lemmatized = self._lemmatize(normalized)

        if self._bpe:
            return self._bpe.EncodeAsPieces(lemmatized)
        else:
            return self._bert.tokenize(lemmatized, **kwargs)
    def __call__(self, text: str | list[str], **kwargs):
        if self._bert:
            # Pass through to BERT
            lemmatized = [self._lemmatize(t) if isinstance(t, str) else t for t in (text if isinstance(text, list) else [text])]
            return self._bert(
                lemmatized[0] if isinstance(text, str) else lemmatized,
                max_length=self.max_length,
                truncation=True,
                padding=self.padding if isinstance(self.padding, str) else "max_length",
                **kwargs
            )
        else:
            # Custom BPE implementation of __call__
            # (Simplified for now, similar to original tokenizer/hybrid_tokenizer.py)
            pass

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, **kwargs) -> "HybridTokenizer":
        model_dir = Path(model_dir)
        # Check if it's a BPE model dir
        if (model_dir / "bpe.model").exists():
            return cls(bpe_model_path=model_dir / "bpe.model", **kwargs)
        # Otherwise assume it's a BERT model name or path
        return cls(bert_model=str(model_dir), **kwargs)

if __name__ == "__main__":
    tok = HybridTokenizer(use_lemma=True)
    print(f"Classic Mode Test: {tok.tokenize('Bankacılığın faiz oranları değişti.')}")
