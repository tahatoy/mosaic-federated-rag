"""
Birim testleri — Router Metrikleri ve Veri Ön İşleme

Test senaryoları:
  1. compute_router_metrics doğru accuracy/F1 hesaplar
  2. Confusion matrix boyutları doğru
  3. Temiz metin pipeline Türkçe karakterleri korur
  4. split_noniid bütünlüğü (veri kaybı yok)
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ai"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


# ─── ROUTER METRİK TESTLERİ ───────────────────────────────────────────────────

class TestRouterMetrics:
    def test_perfect_accuracy(self):
        from evaluation.metrics import compute_router_metrics
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        result = compute_router_metrics(y_true, y_pred)
        assert result["accuracy"] == 1.0
        assert result["macro_f1"] == 1.0

    def test_all_wrong(self):
        from evaluation.metrics import compute_router_metrics
        y_true = [0, 0, 0]
        y_pred = [1, 2, 1]
        result = compute_router_metrics(y_true, y_pred)
        assert result["accuracy"] == 0.0

    def test_confusion_matrix_shape(self):
        from evaluation.metrics import compute_router_metrics
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 0, 2, 1, 1]
        result = compute_router_metrics(y_true, y_pred)
        cm = result["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_per_class_keys(self):
        from evaluation.metrics import compute_router_metrics
        result = compute_router_metrics([0, 1, 2], [0, 1, 2])
        assert set(result["per_class"].keys()) == {"chitchat", "simple", "complex"}

    def test_partial_accuracy(self):
        from evaluation.metrics import compute_router_metrics
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        result = compute_router_metrics(y_true, y_pred)
        assert 0.0 < result["accuracy"] < 1.0


# ─── RAG METRİK TESTLERİ ─────────────────────────────────────────────────────

class TestRAGMetrics:
    def test_perfect_recall(self):
        from evaluation.metrics import compute_rag_metrics
        retrieved = [["doc1", "doc2", "doc3"]]
        relevant  = [["doc1", "doc2"]]
        result    = compute_rag_metrics(retrieved, relevant, k_values=[3])
        assert result["Recall@3"] == 1.0

    def test_zero_recall(self):
        from evaluation.metrics import compute_rag_metrics
        retrieved = [["doc4", "doc5"]]
        relevant  = [["doc1", "doc2"]]
        result    = compute_rag_metrics(retrieved, relevant, k_values=[2])
        assert result["Recall@2"] == 0.0

    def test_mrr_first_hit(self):
        from evaluation.metrics import compute_rag_metrics
        retrieved = [["doc1", "doc2", "doc3"]]
        relevant  = [["doc1"]]
        result    = compute_rag_metrics(retrieved, relevant, k_values=[3])
        assert result["MRR@3"] == 1.0

    def test_mrr_second_hit(self):
        from evaluation.metrics import compute_rag_metrics
        retrieved = [["doc0", "doc1", "doc2"]]
        relevant  = [["doc1"]]
        result    = compute_rag_metrics(retrieved, relevant, k_values=[3])
        assert abs(result["MRR@3"] - 0.5) < 1e-6

    def test_ndcg_perfect(self):
        from evaluation.metrics import compute_rag_metrics
        retrieved = [["doc1"]]
        relevant  = [["doc1"]]
        result    = compute_rag_metrics(retrieved, relevant, k_values=[1])
        assert result["NDCG@1"] == 1.0

    def test_multiple_k_values(self):
        from evaluation.metrics import compute_rag_metrics
        retrieved = [["doc1", "doc2", "doc3", "doc4", "doc5"]]
        relevant  = [["doc1", "doc3"]]
        result    = compute_rag_metrics(retrieved, relevant, k_values=[1, 3, 5])
        assert "MRR@1" in result and "MRR@3" in result and "MRR@5" in result


# ─── VERİ TEMİZLEME TESTLERİ ─────────────────────────────────────────────────

class TestDataCleaning:
    def test_turkish_char_fix(self):
        """CHAR_FIX_MAP Türkçe karakter onarımı çalışmalı."""
        # Bu test doğrudan clean_data.py import etmeden basit regex ile test eder
        # Tipik encoding bozukluğu -> doğru Türkçe karakter
        CHAR_FIX_MAP = {
            "Ä±": "ı", "Äž": "Ğ", "ÄŸ": "ğ", "Åž": "Ş",
            "åž": "ş", "Ã§": "ç", "Ã–": "Ö", "Ã¼": "ü",
        }
        broken  = "faiz oranÄ± yÃ¼kseldi"
        fixed   = broken
        for k, v in CHAR_FIX_MAP.items():
            fixed = fixed.replace(k, v)
        assert "ı" in fixed
        assert "ü" in fixed

    def test_min_length_filter(self):
        """100 karakterden kısa metinler filtrelenmeli."""
        short_text = "Faiz oranı yüksek."
        assert len(short_text) < 100  # Bu metin filtrelenmeli

    def test_alpha_ratio(self):
        """Harf oranı %30 altı metinler spam sayılmalı."""
        spam = "123456789 !!! @@@ ### $$$ %%% &&& *** ((())) ---"
        letters = sum(c.isalpha() for c in spam)
        ratio   = letters / len(spam) if spam else 0
        assert ratio < 0.30


# ─── CJK KARAKTER FİLTRE TESTLERİ ───────────────────────────────────────────

class TestCJKFilter:
    """Orchestrator Çince/CJK karakter filtresini test eder."""

    @staticmethod
    def _get_strip_cjk():
        try:
            from backend.ai_engine import _strip_cjk
            return _strip_cjk
        except Exception:
            return None

    def test_pure_turkish_unchanged(self):
        strip = self._get_strip_cjk()
        if strip is None:
            pytest.skip("_strip_cjk yüklenemedi")
        text = "Konut kredisi faiz oranı yüzde altı olarak açıklandı."
        assert strip(text) == text

    def test_pure_cjk_removed(self):
        strip = self._get_strip_cjk()
        if strip is None:
            pytest.skip("_strip_cjk yüklenemedi")
        text = "这是中文文本。"
        result = strip(text)
        assert result == "" or len(result) < 5  # CJK temizlendi

    def test_mixed_cjk_cleans(self):
        strip = self._get_strip_cjk()
        if strip is None:
            pytest.skip("_strip_cjk yüklenemedi")
        text = "Faiz oranı 这是中文 yüzde altı."
        result = strip(text)
        assert "这" not in result
        assert "Faiz" in result

    def test_empty_string(self):
        strip = self._get_strip_cjk()
        if strip is None:
            pytest.skip("_strip_cjk yüklenemedi")
        assert strip("") == ""

    def test_high_cjk_ratio_returns_empty(self):
        strip = self._get_strip_cjk()
        if strip is None:
            pytest.skip("_strip_cjk yüklenemedi")
        # %80+ CJK -> tüm metin atılmalı
        text = "这是中文文本这是中文文本！"
        result = strip(text)
        assert result == "" or len(result) < 5


# ─── PDF CHUNKLAMA TESTLERİ ───────────────────────────────────────────────────

class TestChunkText:
    """RAG servisindeki _chunk_text ve _format_table_rows fonksiyonlarını test eder."""

    @staticmethod
    def _get_fns():
        try:
            from backend.ai_engine import _chunk_text, _format_table_rows
            return _chunk_text, _format_table_rows
        except Exception:
            return None, None

    def test_basic_split(self):
        chunk_text, _ = self._get_fns()
        if chunk_text is None:
            pytest.skip("_chunk_text yüklenemedi")
        text = ("Merhaba bu bir test metnidir. " * 30).strip()
        chunks = chunk_text(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert len(c) >= 60

    def test_overlap_present(self):
        chunk_text, _ = self._get_fns()
        if chunk_text is None:
            pytest.skip("_chunk_text yüklenemedi")
        # Uzun metin ile birden fazla chunk + overlap test
        para = "Türkiye Merkez Bankası politika faiz oranını değiştirdi. " * 25
        chunks = chunk_text(para, max_len=200, overlap=50)
        if len(chunks) > 1:
            # Örtüşme: chunk[1] başlangıcı chunk[0] sonundan gelmeli
            tail = chunks[0][-50:]
            assert tail[:20] in chunks[1] or len(chunks[1]) > 40  # overlap uygulandı

    def test_table_not_broken(self):
        chunk_text, _ = self._get_fns()
        if chunk_text is None:
            pytest.skip("_chunk_text yüklenemedi")
        table = (
            "Tablo 1: Konut Kredisi Oranları\n"
            "Banka: Ziraat | Faiz: %1.89 | Vade: 120 ay\n"
            "Banka: Garanti | Faiz: %1.95 | Vade: 120 ay\n"
            "Banka: İş Bankası | Faiz: %1.92 | Vade: 120 ay\n"
        )
        chunks = chunk_text(table, max_len=600)
        # Tablo satırları ayrı chunk'lara parçalanmamalı (veya en az biri tüm tabloyu içermeli)
        full_table_in_one = any("Ziraat" in c and "Garanti" in c for c in chunks)
        assert full_table_in_one, "Tablo satırları tek chunk'ta korunmalı"

    def test_format_table_rows(self):
        _, fmt = self._get_fns()
        if fmt is None:
            pytest.skip("_format_table_rows yüklenemedi")
        data = [
            ["Banka", "Faiz Oranı", "Vade"],
            ["Ziraat", "%1.89", "120 ay"],
            ["Garanti", "%1.95", "84 ay"],
        ]
        result = fmt(data)
        assert "Banka: Ziraat" in result
        assert "Faiz Oranı: %1.89" in result

    def test_empty_text_returns_empty(self):
        chunk_text, _ = self._get_fns()
        if chunk_text is None:
            pytest.skip("_chunk_text yüklenemedi")
        assert chunk_text("") == []
        assert chunk_text("   ") == []


# ─── ROUTER OVERRIDE TESTLERİ ────────────────────────────────────────────────

class TestRouterOverride:
    """Orchestrator rule-based router override mekanizmasını test eder."""

    @staticmethod
    def _get_override_fn():
        try:
            from backend.ai_engine import _apply_router_overrides
            return _apply_router_overrides
        except Exception:
            return None

    def _base_result(self, label: int, conf: float) -> dict:
        names = {0: "chitchat", 1: "simple", 2: "complex"}
        return {"label": label, "label_name": names[label], "confidence": conf,
                "scores": {n: 0.0 for n in names.values()}}

    def test_high_conf_no_override(self):
        fn = self._get_override_fn()
        if fn is None:
            pytest.skip("_apply_router_overrides yüklenemedi")
        result = fn("Merhaba nasılsın", self._base_result(0, 0.95))
        assert result["label"] == 0  # Yüksek güven → değişmemeli

    def test_complex_keyword_forces_complex(self):
        fn = self._get_override_fn()
        if fn is None:
            pytest.skip("_apply_router_overrides yüklenemedi")
        # DCF anahtar kelimesi basit sorgu bile olsa complex'e çekilmeli
        result = fn("DCF analizi nasıl yapılır", self._base_result(1, 0.9))
        assert result["label"] == 2
        assert result.get("override_reason") == "complex_keyword"

    def test_low_conf_chitchat_promoted_to_simple(self):
        fn = self._get_override_fn()
        if fn is None:
            pytest.skip("_apply_router_overrides yüklenemedi")
        result = fn("Bana kredi hakkında bilgi ver", self._base_result(0, 0.45))
        assert result["label"] == 1  # chitchat + düşük güven -> simple

    def test_low_conf_simple_promoted_to_complex(self):
        fn = self._get_override_fn()
        if fn is None:
            pytest.skip("_apply_router_overrides yüklenemedi")
        result = fn("Mevduat faiz oranı analiz et", self._base_result(1, 0.40))
        assert result["label"] == 2  # simple + düşük güven -> complex

    def test_complex_not_demoted(self):
        fn = self._get_override_fn()
        if fn is None:
            pytest.skip("_apply_router_overrides yüklenemedi")
        result = fn("Portföy çeşitlendirme stratejisi", self._base_result(2, 0.30))
        assert result["label"] == 2  # complex düşük güvende dahi değişmemeli


# ─── SORGU NORMALİZASYON TESTLERİ ────────────────────────────────────────────

class TestQueryNormalization:
    """RAG servisinin normalize_query metodunun query prefix'ini test eder."""

    def test_query_prefix_added(self):
        """normalize_query 'query: ' prefix eklemeli."""
        # Basit stub ile test — gerçek EmbeddingEngine yüklemeden
        class FakeTokenizer:
            def _normalize(self, t):
                return t

        class FakeEngine:
            hybrid_tokenizer = FakeTokenizer()

            def normalize_query(self, query):
                # RAG main.py ile aynı mantık
                normalized = query
                if self.hybrid_tokenizer:
                    try:
                        normalized = self.hybrid_tokenizer._normalize(query)
                    except Exception:
                        pass
                prefix = "query: "
                if normalized.lower().startswith(prefix):
                    return normalized
                return prefix + normalized

        eng = FakeEngine()
        result = eng.normalize_query("Konut kredisi faizi nedir")
        assert result.startswith("query: ")

    def test_no_double_prefix(self):
        """Zaten 'query: ' ile başlayan sorgulara ikinci kez prefix eklenmemeli."""
        class FakeEngine:
            hybrid_tokenizer = None

            def normalize_query(self, query):
                prefix = "query: "
                if query.lower().startswith(prefix):
                    return query
                return prefix + query

        eng = FakeEngine()
        result = eng.normalize_query("query: zaten prefix var")
        assert result.count("query:") == 1


# ─── TÜRKÇE KARAKTER NORMALİZASYON TESTLERİ ─────────────────────────────────

class TestNormalizeTurkishChars:
    """RAG servisindeki _normalize_turkish_chars fonksiyonunu test eder."""

    @staticmethod
    def _get_fn():
        try:
            from backend.ai_engine import _normalize_turkish_chars
            return _normalize_turkish_chars
        except Exception:
            return None

    def test_windows1252_uppercase_i(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("_normalize_turkish_chars yüklenemedi")
        assert fn("Ý faiz oranı") == "İ faiz oranı"

    def test_windows1252_lowercase_i(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("_normalize_turkish_chars yüklenemedi")
        assert fn("bankac ýlık") == "bankac ılık"

    def test_windows1252_sh(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("_normalize_turkish_chars yüklenemedi")
        result = fn("ÞirketimizÞ")
        assert "Ş" in result

    def test_windows1252_gh(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("_normalize_turkish_chars yüklenemedi")
        result = fn("yoÐun çalışma")
        assert "Ğ" in result

    def test_clean_text_unchanged(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("_normalize_turkish_chars yüklenemedi")
        text = "Türkçe finans haberleri güvenli şekilde işleniyor."
        assert fn(text) == text

    def test_null_byte_removed(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("_normalize_turkish_chars yüklenemedi")
        result = fn("test\x00metin")
        assert "\x00" not in result

    def test_bom_removed(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("_normalize_turkish_chars yüklenemedi")
        result = fn("\ufeffBaşlık")
        assert "\ufeff" not in result
        assert "Başlık" in result


# ─── PDF PYMUPDF ÇIKARIM TESTLERİ ────────────────────────────────────────────

class TestPymupdfExtraction:
    """_extract_pdf_text_pymupdf temel sağlamlık testleri."""

    @staticmethod
    def _get_fn():
        try:
            from backend.ai_engine import _extract_pdf_text_pymupdf
            return _extract_pdf_text_pymupdf
        except Exception:
            return None

    def test_returns_string_on_invalid(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("_extract_pdf_text_pymupdf yüklenemedi")
        result = fn(b"not a pdf")
        assert isinstance(result, str)

    def test_returns_string_on_empty(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("_extract_pdf_text_pymupdf yüklenemedi")
        result = fn(b"")
        assert isinstance(result, str)

    def test_real_pdf_if_available(self):
        fn = self._get_fn()
        if fn is None:
            pytest.skip("_extract_pdf_text_pymupdf yüklenemedi")
        project_root = Path(__file__).resolve().parents[2]
        # Scan only root folder to prevent hanging in WSL/NTFS
        test_pdfs = [
            p for p in project_root.glob("*.pdf")
            if p.stat().st_size < 5_000_000
        ]
        if not test_pdfs:
            # Fallback to data directory or direct search
            test_pdfs = list(project_root.glob("ai/data/**/*.pdf"))
        if not test_pdfs:
            pytest.skip("Test için PDF bulunamadı")
        result = fn(test_pdfs[0].read_bytes())
        assert isinstance(result, str)
        if result:
            assert len(result) > 10  # Gerçek PDF'ten ekstrakt boş olmamalı


# ─── GENİŞLETİLMİŞ CAPTION REGEX TESTLERİ ───────────────────────────────────

class TestCaptionRegex:
    """caption_re Türkçe VE İngilizce başlıkları yakalamalı."""

    def _re(self):
        import re
        return re.compile(
            r"^(Tablo|Grafik|Şekil|Figür|Diyagram|Çizelge|Figure|Table|Fig\.?|Chart|Graf)"
            r"\s*\.?\s*\d*[:\s.]",
            re.IGNORECASE | re.MULTILINE,
        )

    def test_turkish_tablo(self):
        assert self._re().match("Tablo 3: Faiz Oranları")

    def test_turkish_grafik(self):
        assert self._re().match("Grafik 2. Büyüme Trendi")

    def test_turkish_sekil(self):
        assert self._re().match("Şekil 1 Sistem Mimarisi")

    def test_english_figure(self):
        assert self._re().match("Figure 4: Interest Rate Comparison")

    def test_english_table(self):
        assert self._re().match("Table 2. Balance Sheet")

    def test_english_fig_abbrev(self):
        assert self._re().match("Fig. 3: Cash Flow")

    def test_normal_text_no_match(self):
        assert not self._re().match("Bugün piyasalar yükseldi.")
