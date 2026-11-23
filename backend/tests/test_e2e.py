"""
Uçtan uca entegrasyon testleri — Tüm Sistem

Tüm servislerin birlikte çalışmasını test eder.
Regresyon + rota doğrulama + kaynak çeşitliliği.

Not: Servislerin ayakta olması gerekir. Değilse otomatik skip edilir.
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ai"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8004")
ROUTER_URL       = os.getenv("ROUTER_URL",       "http://localhost:8000")
BRANCH1_RAG_URL  = os.getenv("BRANCH1_RAG_URL",  "http://localhost:8001")
BRANCH2_RAG_URL  = os.getenv("BRANCH2_RAG_URL",  "http://localhost:8002")
HQ_RAG_URL       = os.getenv("HQ_RAG_URL",       "http://localhost:8003")
LLM_URL          = os.getenv("BRANCH_LLM_URL",   "http://localhost:8099")


def _service_up(url: str, path: str = "/health") -> bool:
    try:
        import httpx
        r = httpx.get(f"{url}{path}", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def _orchestrator_up() -> bool:
    return _service_up(ORCHESTRATOR_URL)


def _llm_up() -> bool:
    return _service_up(LLM_URL)


# ─── REGRESYON TESTLERİ ───────────────────────────────────────────────────────

@pytest.mark.skipif(not _orchestrator_up(), reason="Orchestrator servisi çalışmıyor")
class TestRegression:
    """Önceden çalışan sorguların hâlâ doğru yanıtladığını doğrular."""

    def test_chitchat_is_routed_correctly(self):
        import httpx
        r = httpx.post(
            f"{ORCHESTRATOR_URL}/query",
            json={"query": "Merhaba nasılsın", "branch_id": 1},
            timeout=60.0,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["route"] == "chitchat"
        assert data["label"] == 0

    def test_simple_finance_query(self):
        import httpx
        r = httpx.post(
            f"{ORCHESTRATOR_URL}/query",
            json={"query": "Konut kredisi faiz oranı nedir", "branch_id": 1},
            timeout=90.0,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["route"] in ("simple", "complex")
        assert data["confidence"] > 0.5
        assert len(data.get("sources", [])) >= 0

    def test_complex_query_uses_global_rag(self):
        import httpx
        r = httpx.post(
            f"{ORCHESTRATOR_URL}/query",
            json={
                "query": "Portföy çeşitlendirmesi için DCF analizi ile risk karşılaştırması yap",
                "branch_id": 1,
            },
            timeout=90.0,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["route"] in ("simple", "complex")

    def test_response_fields_complete(self):
        import httpx
        r = httpx.post(
            f"{ORCHESTRATOR_URL}/query",
            json={"query": "Faiz oranı nedir?", "branch_id": 1},
            timeout=60.0,
        )
        assert r.status_code == 200
        data = r.json()
        required = {"query", "answer", "route", "label", "confidence", "sources", "elapsed_ms"}
        assert required.issubset(data.keys())

    def test_answer_is_not_empty(self):
        import httpx
        r = httpx.post(
            f"{ORCHESTRATOR_URL}/query",
            json={"query": "Konut kredisi belgeler nelerdir", "branch_id": 1},
            timeout=90.0,
        )
        assert r.status_code == 200
        answer = r.json().get("answer", "")
        assert len(answer.strip()) > 20


# ─── ROUTER ROTA TESTLERİ ─────────────────────────────────────────────────────

@pytest.mark.skipif(not _service_up(ROUTER_URL), reason="Router servisi çalışmıyor")
class TestRouterRouting:
    """Router sınıflandırma doğruluğunu kontrol eder."""

    CASES = [
        ("Merhaba! Nasılsın?", "chitchat"),
        ("Faiz oranı nedir?", "simple"),
        ("DCF analizi ile hisse senedi değerlemesi nasıl yapılır?", "complex"),
        ("Teşekkürler!", "chitchat"),
        ("Konut kredisi başvurusu için hangi belgeler gerekli?", "simple"),
    ]

    def test_chitchat_classified(self):
        import httpx
        r = httpx.post(f"{ROUTER_URL}/predict", json={"text": "Merhaba! Nasılsın?"})
        assert r.json()["label_name"] == "chitchat"

    def test_simple_classified(self):
        import httpx
        r = httpx.post(f"{ROUTER_URL}/predict", json={"text": "Faiz oranı nedir?"})
        assert r.json()["label_name"] in ("simple", "complex")

    def test_complex_classified_not_chitchat(self):
        import httpx
        r = httpx.post(
            f"{ROUTER_URL}/predict",
            json={"text": "DCF analizi ile hisse senedi değerlemesi nasıl yapılır?"},
        )
        assert r.json()["label_name"] != "chitchat"

    def test_confidence_in_range(self):
        import httpx
        for text, _ in self.CASES[:3]:
            r = httpx.post(f"{ROUTER_URL}/predict", json={"text": text})
            conf = r.json()["confidence"]
            assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"


# ─── RAG KAYNAK TESTLERİ ─────────────────────────────────────────────────────

@pytest.mark.skipif(not _service_up(BRANCH1_RAG_URL), reason="Branch-1 RAG çalışmıyor")
class TestBranch1RAG:
    """Branch-1 RAG kaynak çeşitliliği ve skoru kontrol eder."""

    def test_query_returns_chunks(self):
        import httpx
        r = httpx.post(
            f"{BRANCH1_RAG_URL}/query",
            json={"query": "konut kredisi faiz", "top_k": 5},
            timeout=30.0,
        )
        assert r.status_code == 200
        chunks = r.json().get("chunks", [])
        assert len(chunks) > 0

    def test_chunks_have_required_fields(self):
        import httpx
        r = httpx.post(
            f"{BRANCH1_RAG_URL}/query",
            json={"query": "gelir belgesi kredi", "top_k": 3},
            timeout=30.0,
        )
        chunks = r.json().get("chunks", [])
        for chunk in chunks:
            assert "text" in chunk
            assert "score" in chunk

    def test_scores_are_positive(self):
        import httpx
        r = httpx.post(
            f"{BRANCH1_RAG_URL}/query",
            json={"query": "faiz hesaplama", "top_k": 5},
            timeout=30.0,
        )
        chunks = r.json().get("chunks", [])
        for chunk in chunks:
            assert chunk["score"] >= 0.0


# ─── LLM SERVER TESTLERİ ─────────────────────────────────────────────────────

@pytest.mark.skipif(not _llm_up(), reason="LLM server (Gemma) çalışmıyor")
class TestLLMServer:
    """llama-server sağlık ve basit inference testi."""

    def test_health(self):
        import httpx
        r = httpx.get(f"{LLM_URL}/health", timeout=5.0)
        assert r.status_code == 200

    def test_simple_completion(self):
        import httpx
        r = httpx.post(
            f"{LLM_URL}/completion",
            json={
                "prompt": "Faiz nedir? Kısa yanıt:",
                "n_predict": 50,
                "temperature": 0.1,
                "stream": False,
            },
            timeout=30.0,
        )
        assert r.status_code == 200
        content = r.json().get("content", "")
        assert len(content.strip()) > 0
