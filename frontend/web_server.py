import os
import sys
import json
import httpx
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ai"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

if os.path.exists(PROJECT_ROOT / ".env"):
    load_dotenv(PROJECT_ROOT / ".env")
elif os.path.exists(PROJECT_ROOT / ".env.client"):
    load_dotenv(PROJECT_ROOT / ".env.client")

from fastapi.staticfiles import StaticFiles

app = FastAPI(title="MOSAIC-v2 Corporate Banking Web UI Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent
DIST_DIR = FRONTEND_DIR / "dist"
ASSETS_DIR = DIST_DIR / "assets"
STATIC_DIR = FRONTEND_DIR / "static"

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── ENVIRONMENT CONFIG DETECTION ───────────────────────────────────────────

MOSAIC_ENV = os.getenv("MOSAIC_ENV", "").lower()
branch_id_env = os.getenv("BRANCH_ID", "")
if branch_id_env and not MOSAIC_ENV:
    MOSAIC_ENV = "branch"

if not MOSAIC_ENV:
    MOSAIC_ENV = "hq"

if MOSAIC_ENV == "hq":
    ENV_NAME = "hq"
    BRANCH_ID_STR = "hq"
    TITLE = "MOSAIC Bank — Global Finans & Analitik Platformu"
    ACCENT_COLOR = "#2563eb"
    ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8004")
    RAG_URL = os.getenv("HQ_RAG_URL", "http://localhost:8003")
else:
    ENV_NAME = "branch"
    BRANCH_ID_STR = branch_id_env or "branch_01"
    if "02" in BRANCH_ID_STR or "2" in BRANCH_ID_STR:
        TITLE = "MOSAIC Bank — Şube-2 Kurumsal & Yatırım Bankacılığı"
        ACCENT_COLOR = "#059669"
    else:
        TITLE = "MOSAIC Bank — Şube-1 Bireysel & Kredi Bankacılığı"
        ACCENT_COLOR = "#2563eb"

    server_host = os.getenv("SERVER_HOST", "localhost")
    server_llm_port = os.getenv("SERVER_LLM_PORT", "8004")
    ORCHESTRATOR_URL = f"http://{server_host}:{server_llm_port}"

    rag_port = os.getenv("PORT", os.getenv("BRANCH_RAG_PORT", "8001" if "01" in BRANCH_ID_STR else "8002"))
    # Şube RAG servisi tls_serve.py ile mTLS üzerinden çalışıyorsa (RAG_TLS_CERT tanımlı) yerel adres de https olmalı.
    RAG_URL = f"{'https' if os.getenv('RAG_TLS_CERT') else 'http'}://localhost:{rag_port}"


# ─── mTLS İSTEMCİSİ (Şube RAG servisleri yalnızca kurum CA'sının imzaladığı istemci sertifikalarını kabul eder) ───

def _rag_ssl_context():
    """https RAG adresleri için CA doğrulamalı ve (varsa) istemci sertifikalı SSL bağlamı; aksi halde True."""
    import ssl
    ca = os.getenv("RAG_CA_CERT_PATH", "")
    cert, key = os.getenv("RAG_CLIENT_CERT", ""), os.getenv("RAG_CLIENT_KEY", "")
    ca_path = (PROJECT_ROOT / ca) if ca and not os.path.isabs(ca) else Path(ca) if ca else None
    if os.getenv("RAG_TLS_VERIFY", "true").lower() == "false":
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    elif ca_path and ca_path.exists():
        ctx = ssl.create_default_context(cafile=str(ca_path))
    else:
        ctx = ssl.create_default_context()
    if cert and key:
        cp = (PROJECT_ROOT / cert) if not os.path.isabs(cert) else Path(cert)
        kp = (PROJECT_ROOT / key) if not os.path.isabs(key) else Path(key)
        if cp.exists() and kp.exists():
            ctx.load_cert_chain(str(cp), str(kp))
        else:
            logger.warning(f"[mTLS] İstemci sertifikası bulunamadı: {cp} / {kp}")
    return ctx


_RAG_SSL = None


def rag_client(timeout: float) -> httpx.AsyncClient:
    """Şube/HQ RAG servislerine (http veya mTLS https) bağlanan istemci."""
    global _RAG_SSL
    if _RAG_SSL is None:
        _RAG_SSL = _rag_ssl_context()
    return httpx.AsyncClient(timeout=timeout, verify=_RAG_SSL, trust_env=False)


def branch_rag_url(n: int) -> str:
    """Şube-n RAG adresi: bu UI o şubede çalışıyorsa yerel servis, değilse BRANCHn_RAG_URL."""
    if ENV_NAME == "branch" and str(n) in BRANCH_ID_STR:
        return RAG_URL
    scheme = "https" if (
        os.getenv("RAG_CLIENT_CERT")
        or os.getenv("RAG_TLS_CERT")
        or os.getenv("RAG_TLS_VERIFY", "").lower() == "true"
    ) else "http"
    default = f"{scheme}://10.0.0.11:8001" if n == 1 else f"{scheme}://10.0.0.12:8002"
    return os.getenv(f"BRANCH{n}_RAG_URL", default)

logger.info(f"Loaded UI Config | Mode: {ENV_NAME.upper()} | Branch: {BRANCH_ID_STR} | Orchestrator: {ORCHESTRATOR_URL} | RAG: {RAG_URL}")

# ─── API ENDPOINTS ──────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    return {
        "env": ENV_NAME,
        "branch_id": BRANCH_ID_STR,
        "title": TITLE,
        "accent_color": ACCENT_COLOR,
        "orchestrator_url": ORCHESTRATOR_URL,
        "rag_url": RAG_URL
    }

@app.get("/api/health")
async def get_health():
    rag_status = "down"
    orchestrator_status = "down"

    async with rag_client(1.5) as rc:
        try:
            r = await rc.get(f"{RAG_URL}/health")
            if r.status_code == 200:
                rag_status = "up"
        except Exception:
            pass

    async with httpx.AsyncClient(timeout=1.5) as client:
        try:
            r = await client.get(f"{ORCHESTRATOR_URL}/health")
            if r.status_code == 200:
                orchestrator_status = "up"
        except Exception:
            pass

    return {
        "status": "ok",
        "services": {
            "rag": rag_status,
            "orchestrator": orchestrator_status
        }
    }

@app.get("/api/network_status")
async def get_network_status():
    """
    HQ, VM1 ve VM2 düğümlerine gerçek zamanlı canlı HTTP ping ve gecikme ölçümü yapar.
    Simülasyon içermez, gerçek soket yanıt sürelerini milisaniye cinsinden döner.
    """
    import time
    server_host = os.getenv("SERVER_HOST", "10.0.0.10")

    if ENV_NAME == "branch":
        # Şube VM'sinde çalışırken HQ bağlantısı
        hq_url = os.getenv("ORCHESTRATOR_URL", f"http://{server_host}:8004")
        hq_rag_fallback = os.getenv("HQ_RAG_URL", f"http://{server_host}:8003")
    else:
        # HQ yerelinde çalışırken
        hq_url = os.getenv("ORCHESTRATOR_URL", "http://localhost:8004")
        hq_rag_fallback = os.getenv("HQ_RAG_URL", "http://localhost:8003")

    # Yerel şubede çalışıyorsa localhost'u önceliklendir, aksi halde LAN adresini (mTLS) kullan
    b1_url = branch_rag_url(1)
    b2_url = branch_rag_url(2)

    targets = [
        ("hq", "Düğüm 1: HQ (Genel Merkez)", f"{server_host} (RTX 5060 Ti 16GB)", "Orkestratör & Qwen3-8B Expert", hq_url, hq_rag_fallback),
        ("vm1", "Düğüm 2: Şube-1 VM (Bireysel / Kredi)", "10.0.0.11 (2 vCPU, 10 GB RAM)", "Branch-1 ChromaDB & Yerel UI", b1_url, None),
        ("vm2", "Düğüm 3: Şube-2 VM (Kurumsal / Yatırım)", "10.0.0.12 (2 vCPU, 10 GB RAM)", "Branch-2 ChromaDB & Yerel UI", b2_url, None)
    ]

    results = []
    async with rag_client(2.0) as client:
        for node_id, name, ip, role, url, fallback in targets:
            health_url = f"{url}/health"
            t0 = time.perf_counter()
            status = "down"
            latency_str = "Zaman Aşımı (Timeout)"
            status_code = None
            try:
                resp = await client.get(health_url)
                dt = (time.perf_counter() - t0) * 1000
                status_code = resp.status_code
                if resp.status_code == 200:
                    status = "up"
                    latency_str = f"{dt:.1f} ms"
                else:
                    latency_str = f"HTTP {resp.status_code} ({dt:.1f} ms)"
            except Exception as e:
                # HQ için yedek endpoint denemesi (Orkestratör vs RAG)
                if fallback:
                    try:
                        t_fb = time.perf_counter()
                        resp_fb = await client.get(f"{fallback}/health")
                        dt_fb = (time.perf_counter() - t_fb) * 1000
                        if resp_fb.status_code == 200:
                            status = "up"
                            status_code = 200
                            latency_str = f"{dt_fb:.1f} ms"
                    except Exception:
                        pass

                if status != "up":
                    err_name = type(e).__name__
                    if "Timeout" in err_name:
                        latency_str = "Çevrimdışı (2.0s Timeout)"
                    else:
                        latency_str = "Erişilemedi (Bağlantı Reddi)"

            results.append({
                "id": node_id,
                "name": name,
                "ip": ip,
                "role": role,
                "url": url,
                "status": status,
                "latency": latency_str,
                "status_code": status_code
            })

    return {"status": "ok", "nodes": results}

@app.get("/api/stats")
async def get_stats():
    try:
        async with rag_client(3.0) as client:
            resp = await client.get(f"{RAG_URL}/stats")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch stats from RAG: {e}")
    return {"collection_count": 0}

@app.post("/api/query")
async def api_query(request: Request):
    data = await request.json()
    query = data.get("query", "")
    history = data.get("history", [])

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Allow user-selected branch_id or fallback to env default
    if "branch_id" in data:
        branch_id = int(data["branch_id"])
    elif ENV_NAME == "hq":
        branch_id = 0
    else:
        branch_id = 2 if "02" in BRANCH_ID_STR or "2" in BRANCH_ID_STR else 1

    payload = {
        "query": query,
        "branch_id": branch_id,
        "history": history
    }

    logger.info(f"[UI Query] Routing to Orchestrator: {ORCHESTRATOR_URL}/query/stream (branch_id={branch_id})")

    async def stream_generator():
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                async with client.stream("POST", f"{ORCHESTRATOR_URL}/query/stream", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line:
                            yield line + "\n"
        except Exception as e:
            logger.error(f"[UI Query Stream] Orchestrator connection error: {e}")
            yield json.dumps({
                "step": "route",
                "route": "HATA",
                "model": "Bağlantı Hatası",
                "confidence": 0.0
            }) + "\n"
            yield json.dumps({
                "step": "rag",
                "status": "error",
                "sources": [],
                "doc_count": 0
            }) + "\n"
            yield json.dumps({
                "step": "llm_token",
                "token": f"### [WARN] Servis Bağlantı Hatası\n\nOrkestratör servisine ({ORCHESTRATOR_URL}) erişim sağlanamadı: `{e}`.\n\nLütfen ana makinede HQ servislerinin ve Orkestratörün ayakta olduğunu kontrol ediniz."
            }) + "\n"
            yield json.dumps({
                "step": "result",
                "model_name": "Sistem Hatası",
                "sources": []
            }) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    source_name: Optional[str] = Form(None),
    branch_id: Optional[str] = Form(None)
):
    target_rag_url = RAG_URL
    if branch_id:
        b_str = str(branch_id).strip().lower()
        if "1" in b_str:
            target_rag_url = branch_rag_url(1)
        elif "2" in b_str:
            target_rag_url = branch_rag_url(2)
        elif "hq" in b_str or "0" in b_str:
            server_host = os.getenv("SERVER_HOST", "10.0.0.10")
            default_hq = "http://localhost:8003" if ENV_NAME == "hq" else f"http://{server_host}:8003"
            target_rag_url = os.getenv("HQ_RAG_URL", default_hq)

    logger.info(f"[UI Upload] Uploading document to RAG: {target_rag_url}/upload (source: {source_name or file.filename})")
    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        data = {}
        if source_name:
            data["source_name"] = source_name

        async with rag_client(300.0) as client:
            resp = await client.post(f"{target_rag_url}/upload", files=files, data=data)
            if resp.status_code != 200:
                detail = resp.text
                try:
                    detail = resp.json().get("detail", detail)
                except Exception:
                    pass
                raise HTTPException(status_code=resp.status_code, detail=f"RAG ayrıştırma hatası: {detail}")
            return resp.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[UI Upload] RAG service connection error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"RAG servisine erişilemedi ({target_rag_url}): {e}. Lütfen ilgili RAG servisinin ayakta olduğundan emin olun."
        )

@app.post("/api/trigger_fl")
async def trigger_fl(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    fl_type = data.get("type", "llm")
    if fl_type not in ("llm", "embedding"):
        raise HTTPException(status_code=400, detail="Invalid training type")

    admin_host = os.getenv("FL_ADMIN_HOST", "127.0.0.1")
    admin_port = 9081
    url = f"http://{admin_host}:{admin_port}/admin/trigger/{fl_type}"

    logger.info(f"[UI FL Trigger] Triggering Flower {fl_type} training via: {url}")
    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            resp = await client.post(url)
            data = resp.json() if resp.status_code == 200 else {}
    except Exception as e:
        logger.warning(f"[UI FL Trigger] Flower admin sunucusuna ulaşılamadı: {e}")
        raise HTTPException(status_code=503, detail=f"Flower federasyon sunucusu (admin {admin_host}:{admin_port}) çevrimdışı; federe tur başlatılamadı.")
    if data.get("status") != "triggered":
        raise HTTPException(status_code=502, detail=f"Flower sunucusu isteği kabul etmedi: {data or resp.status_code}")
    return {
        "status": "triggered",
        "type": fl_type,
        "message": (f"Flower sunucusu {fl_type} federe tur isteğini kabul etti. Tur, sunucu tarafında yeterli sayıda şube "
                    f"istemcisi bağlıysa yürütülür; ilerleme Flower sunucu günlüğünde izlenir.")
    }

@app.get("/api/fl_status")
async def fl_status():
    admin_host = os.getenv("FL_ADMIN_HOST", "127.0.0.1")
    admin_port = 9081
    url = f"http://{admin_host}:{admin_port}/admin/status"
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "active": True,
                    "scheduler_running": data.get("scheduler_running", False),
                    "connected_branches": data.get("connected_branches", 2)
                }
    except Exception:
        pass
    return {"active": False, "connected_branches": 0}

@app.get("/api/fl_audit")
async def fl_audit():
    """
    Federe Öğrenme & DP-SGD Matematiksel Kanıt Motoru Verisi
    """
    try:
        from backend.federated.proof_engine import proof_engine
        return proof_engine.get_full_audit_payload()
    except Exception as e:
        logger.error(f"Error fetching FL audit payload: {e}")
        return {"status": "error", "message": str(e)}

# ─── FRONTEND STATIC SERVING (SPA) ──────────────────────────────────────────

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_spa(full_path: str):
    # If the requested path is a real file inside dist/, serve it
    requested_file = DIST_DIR / full_path
    if full_path and requested_file.is_file():
        return FileResponse(str(requested_file))

    # Otherwise return index.html for SPA client-side routing
    dist_index = DIST_DIR / "index.html"
    if dist_index.exists():
        with open(dist_index, "r", encoding="utf-8") as f:
            return f.read()

    # Fallback to legacy index if dist not built
    fallback_index = FRONTEND_DIR / "index_legacy.html"
    if fallback_index.exists():
        with open(fallback_index, "r", encoding="utf-8") as f:
            return f.read()

    raise HTTPException(status_code=404, detail="Index file not found")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("UI_PORT", "8005"))
    logger.info(f"Starting MOSAIC-v2 Corporate Banking UI on port {port}...")
    uvicorn.run("web_server:app", host="0.0.0.0", port=port, reload=False)
