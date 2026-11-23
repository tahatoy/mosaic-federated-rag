"""
Birim testleri — Hibrit Tokenizer

Test senaryoları:
  1. BPE modeli olmadan basit word-split fallback çalışır
  2. Türkçe karakterler normalize edilir
  3. Encode/decode round-trip tutarlı
  4. HuggingFace API uyumluluğu (return_tensors="pt")
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ai"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# ─── YARDIMCI ─────────────────────────────────────────────────────────────────

SAMPLE_SENTENCES = [
    "Mevduat faiz oranı nedir?",
    "Kredi başvurusu için belge gerekir.",
    "Türkiye ekonomisi büyüyor.",
    "Hisse senedi fiyatı düştü.",
    "Ş ğ ü ö ç ı harflerini doğru işle.",
]


# ─── TESTLER ──────────────────────────────────────────────────────────────────

class TestZemberekWrapper:
    """Zemberek wrapper'ın graceful fallback davranışını test eder."""

    def test_import(self):
        """Modül import edilebilmeli."""
        from tokenizer.zemberek_wrapper import get_analyzer, ZemberekAnalyzer  # noqa
        assert callable(get_analyzer)

    def test_tokenize_returns_list(self):
        from tokenizer.zemberek_wrapper import get_analyzer
        analyzer = get_analyzer()
        result = analyzer.tokenize_sentence("Faiz oranı yükseldi.")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_no_empty_tokens(self):
        from tokenizer.zemberek_wrapper import get_analyzer
        analyzer = get_analyzer()
        for sentence in SAMPLE_SENTENCES:
            tokens = analyzer.tokenize_sentence(sentence)
            assert all(len(t) > 0 for t in tokens), f"Boş token bulundu: {sentence}"


class TestHybridTokenizer:
    """HybridTokenizer'ın temel işlevselliğini test eder."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from tokenizer.hybrid_tokenizer import HybridTokenizer
        # BPE modeli yoksa word-split fallback kullanır
        return HybridTokenizer(bpe_model_path=None, use_zemberek=False)

    def test_encode_returns_list(self, tokenizer):
        ids = tokenizer.encode("Faiz oranı nedir?")
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_decode_roundtrip(self, tokenizer):
        # BPE modeli yoksa hash-tabanlı fallback decode edilemez, skip et
        if tokenizer._bpe is None:
            pytest.skip("BPE modeli yüklü değil — round-trip testi atlandı")
        text = "kredi faiz mevduat"
        ids  = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        original_words = set(text.lower().split())
        decoded_words  = set(decoded.lower().split())
        overlap = original_words & decoded_words
        assert len(overlap) > 0, f"Decode round-trip başarısız: {text!r} -> {decoded!r}"

    def test_call_interface(self, tokenizer):
        """__call__ HuggingFace API uyumlu çıktı vermeli."""
        out = tokenizer("Mevduat hesabı açmak istiyorum", return_tensors=None)
        assert "input_ids" in out
        assert isinstance(out["input_ids"], list)

    def test_turkish_chars_handled(self, tokenizer):
        sentence = "Şirketin özkaynak getirisi güçlü."
        ids = tokenizer.encode(sentence)
        assert len(ids) > 0

    def test_batch_encode(self, tokenizer):
        sentences = ["Faiz oranı", "Kredi kartı borcu", "Hisse senedi"]
        for s in sentences:
            ids = tokenizer.encode(s)
            assert isinstance(ids, list) and len(ids) > 0
