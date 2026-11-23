"""
Entegrasyon testleri — FastAPI Servisleri

Router ve Orchestrator servislerini test eder.
Not: Bu testler servisler ayakta olduğunda çalışır.
Ayakta değilse otomatik olarak skip edilir.
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ai"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

ROUTER_URL       = os.getenv("ROUTER_URL",       "http://localhost:8000")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8004")
RAG_URL          = os.getenv("RAG_URL",          "http://localhost:8003")


def _service_up(url: str) -> bool:
    try:
        import httpx
        r = httpx.get(f"{url}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


# ─── ROUTER SERVİS TESTLERİ ───────────────────────────────────────────────────

@pytest.mark.skipif(not _service_up(ROUTER_URL), reason="Router servisi çalışmıyor")
class TestRouterService:
    def test_health(self):
        import httpx
        r = httpx.get(f"{ROUTER_URL}/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"

    def test_chitchat_prediction(self):
        import httpx
        r = httpx.post(f"{ROUTER_URL}/predict", json={"text": "Merhaba nasılsın?"})
        assert r.status_code == 200
        data = r.json()
        assert data["label_name"] in ["chitchat", "simple", "complex"]
        assert 0.0 <= data["confidence"] <= 1.0

    def test_complex_finance_prediction(self):
        import httpx
        r = httpx.post(
            f"{ROUTER_URL}/predict",
            json={"text": "DCF analizi ile hisse senedi değerlemesini açıklar mısın?"}
        )
        assert r.status_code == 200
        data = r.json()
        assert data["label_name"] in ["simple", "complex"]

    def test_batch_endpoint(self):
        import httpx
        texts = ["Merhaba", "Faiz oranı", "DCF analizi nedir?"]
        r = httpx.post(f"{ROUTER_URL}/predict/batch", json=texts)
        assert r.status_code == 200
        results = r.json()
        assert len(results) == 3

    def test_empty_text_rejected(self):
        import httpx
        r = httpx.post(f"{ROUTER_URL}/predict", json={"text": ""})
        assert r.status_code == 422  # Validation error

    def test_response_schema(self):
        import httpx
        r = httpx.post(f"{ROUTER_URL}/predict", json={"text": "Kredi faiz oranı"})
        assert r.status_code == 200
        data = r.json()
        required_fields = {"text", "label", "label_name", "confidence", "scores", "elapsed_ms"}
        assert required_fields.issubset(data.keys())


# ─── RAG SERVİS TESTLERİ ─────────────────────────────────────────────────────

@pytest.mark.skipif(not _service_up(RAG_URL), reason="RAG servisi çalışmıyor")
class TestRAGService:
    def test_health(self):
        import httpx
        r = httpx.get(f"{RAG_URL}/health")
        assert r.status_code == 200

    def test_query_returns_results(self):
        import httpx
        r = httpx.post(
            f"{RAG_URL}/query",
            json={"query": "mevduat faiz oranı", "top_k": 3, "branch_id": 1},
            timeout=30.0,
        )
        assert r.status_code == 200
        data = r.json()
        # RAG servisi "chunks" anahtar kelimesini kullanıyor
        assert "chunks" in data
        assert isinstance(data["chunks"], list)

    def test_stats_endpoint(self):
        import httpx
        r = httpx.get(f"{RAG_URL}/stats")
        assert r.status_code == 200


# ─── ORCHESTRATOR SERVİS TESTLERİ ────────────────────────────────────────────

@pytest.mark.skipif(
    not _service_up(ORCHESTRATOR_URL),
    reason="Orchestrator servisi çalışmıyor"
)
class TestOrchestratorService:
    def test_health(self):
        import httpx
        r = httpx.get(f"{ORCHESTRATOR_URL}/health")
        assert r.status_code == 200

    def test_query_response_schema(self):
        import httpx
        r = httpx.post(
            f"{ORCHESTRATOR_URL}/query",
            json={"query": "Faiz oranı nedir?", "branch_id": 1},
            timeout=60.0,
        )
        assert r.status_code == 200
        data = r.json()
        required_fields = {"query", "answer", "route", "label", "confidence", "sources", "elapsed_ms"}
        assert required_fields.issubset(data.keys())

    def test_route_label_valid(self):
        import httpx
        r = httpx.post(
            f"{ORCHESTRATOR_URL}/query",
            json={"query": "Hisse senedi nedir?", "branch_id": 1},
            timeout=60.0,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["route"] in ["chitchat", "simple", "complex"]
        assert data["label"] in [0, 1, 2]
