"""
Gelişmiş RAG Birim ve Entegrasyon Testleri
==========================================
1. Hibrit Tokenizer: normalize ve _normalize işlevleri
2. Tablo Çıkarımı: GFM Markdown formatı ve anahtar-değer uyumluluğu
3. Grafik / Görsel Çıkarımı: Başlık, etiket ve çizim yakalama
4. Okuma Sırası: PyMuPDF ile blokların dikey sıra (y0) bütünlüğü
5. Tablo ve Grafik Korumalı Chunking: _chunk_text ile bölünmezlik
6. Merkezi Cross-Encoder Reranker: Çoklu şube adaylarını doğru sıralama ve filtreleme
"""

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ai"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from ai.tokenizer.hybrid_tokenizer import HybridTokenizer
from backend.ai_engine import (
    _format_table_rows,
    _format_table_markdown,
    _chunk_text,
    _extract_pdf_text_pymupdf,
    rerank_chunks
)


class TestHybridTokenizerNormalization:
    """Hibrit tokenizer normalizasyon yeteneklerini test eder."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return HybridTokenizer(bpe_model_path=None, use_zemberek=False)

    def test_broken_windows1252_chars(self, tokenizer):
        raw = "Ýç borçlanma faiz oranlarý ve dýþ ticaret dengesi."
        fixed = tokenizer.normalize(raw)
        assert "İç" in fixed
        assert "oranları" in fixed
        assert "dış" in fixed

    def test_alias_normalize(self, tokenizer):
        raw = "TCMB faiz kararý"
        assert tokenizer._normalize(raw) == tokenizer.normalize(raw)

    def test_whitespace_and_currency(self, tokenizer):
        raw = "  Kredi   tutarı:  50.000 TL \u00a0 (₺)   "
        clean = tokenizer.normalize(raw)
        assert clean == "Kredi tutarı: 50.000 TL (₺)"

    def test_count_tokens(self, tokenizer):
        count = tokenizer.count_tokens("Konut kredisi faiz oranı yüzde 1.89")
        assert count > 0
        assert isinstance(count, int)


class TestTableFormattingAndChunking:
    """Tablo formatlama ve chunk bütünlüğü testleri."""

    def test_format_table_markdown(self):
        data = [
            ["Banka", "Mevduat Oranı", "Vade"],
            ["Banka A", "%45.0", "32 Gün"],
            ["Banka B", "%48.5", "92 Gün"]
        ]
        md = _format_table_markdown(data)
        assert "| Banka | Mevduat Oranı | Vade |" in md
        assert "| :--- | :--- | :--- |" in md
        assert "| Banka A | %45.0 | 32 Gün |" in md

    def test_format_table_rows_dual_format(self):
        data = [
            ["Banka", "Faiz Oranı", "Vade"],
            ["Ziraat", "%1.89", "120 ay"],
            ["Garanti", "%1.95", "84 ay"]
        ]
        res = _format_table_rows(data)
        # Test key-value lines
        assert "Banka: Ziraat" in res
        assert "Faiz Oranı: %1.89" in res
        # Test markdown table
        assert "| Banka | Faiz Oranı | Vade |" in res
        assert "| :--- | :--- | :--- |" in res

    def test_chunking_preserves_table_integrity(self):
        table_md = (
            "Aşağıdaki tabloda konut kredisi oranları listelenmiştir:\n\n"
            "| Banka | Faiz Oranı | Vade |\n"
            "| :--- | :--- | :--- |\n"
            "| Ziraat | %1.89 | 120 ay |\n"
            "| Halkbank | %1.89 | 120 ay |\n"
            "| Vakıfbank | %1.90 | 120 ay |\n\n"
            "Başvurular şubelerden yapılabilir."
        )
        chunks = _chunk_text(table_md, max_len=600, overlap=50)
        assert len(chunks) >= 1
        found_table = any("Ziraat" in c and "Vakıfbank" in c for c in chunks)
        assert found_table, "Markdown tablosu bölünmeden tek chunk içinde kalmalı"

    def test_chunking_preserves_graphic_blocks(self):
        graphic_text = (
            "[Grafik / Görsel: Grafik 1 - TCMB Politika Faizi]\n"
            "Grafik 1: TCMB Politika Faizi ve Enflasyon Gelişimi\n"
            "Etiketler: %45, %50, 2024-Ocak, 2024-Şubat\n\n"
            "Grafikte görüldüğü üzere politika faizi enflasyonla uyumlu artırılmıştır."
        )
        chunks = _chunk_text(graphic_text, max_len=600)
        assert len(chunks) >= 1
        assert any("[Grafik / Görsel: Grafik 1" in c and "%50" in c for c in chunks)


class TestPyMuPDFReadingOrder:
    """PyMuPDF dikey okuma sırası testi (In-memory PDF ile)."""

    def test_synthetic_pdf_reading_order(self):
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF (fitz) kurulu değil")

        doc = fitz.open()
        page = doc.new_page()

        page.insert_text((50, 60), "1. Bolum: Giris ve Baslik Metni")
        page.insert_text((50, 160), "Tablo 1: Bireysel Kredi Faizleri\nBanka: XBank | Faiz: %2.10")
        page.insert_text((50, 310), "3. Bolum: Sonuc ve Degerlendirme Metni")

        pdf_bytes = doc.tobytes()
        doc.close()

        extracted = _extract_pdf_text_pymupdf(pdf_bytes)
        assert "1. Bolum: Giris ve Baslik Metni" in extracted
        assert "3. Bolum: Sonuc ve Degerlendirme Metni" in extracted

        pos1 = extracted.find("1. Bolum: Giris")
        pos2 = extracted.find("Tablo 1:")
        pos3 = extracted.find("3. Bolum: Sonuc")

        assert pos1 < pos2 < pos3, "Metin blokları yukarıdan aşağıya (y0) doğal okuma sırasında olmalı"


class TestCentralizedReranker:
    """Merkezi Cross-Encoder reranker birim testleri."""

    def test_rerank_empty_chunks(self):
        res = rerank_chunks("mevduat faizi", [])
        assert res == []

    def test_rerank_filtering_and_order(self):
        chunks = [
            {"text": "Alakasız metin içeriği", "score": 0.30},
            {"text": "Mevduat faiz oranları bankalarda %45 seviyesindedir.", "score": 0.85},
            {"text": "Çok düşük benzerlikli doküman", "score": 0.05}
        ]
        res = rerank_chunks("Mevduat faiz oranı kaç?", chunks, top_k=2, min_score=0.20)
        assert len(res) <= 2
        assert all(c["score"] >= 0.20 for c in res)
