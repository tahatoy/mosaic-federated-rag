"""
MOSAIC-v2 Consolidated Backend API
==================================
A unified FastAPI backend serving all three microservices:
  1. BERT Router API (Port 8000)
  2. RAG & ChromaDB API (Ports 8001 - 8003)
  3. Orchestrator API (Port 8004)

You can launch a specific service by setting the `SERVICE_TYPE` environment variable
or passing the `--service` command line argument (e.g. `router`, `rag`, `orchestrator`).
"""

import os
import re
import sys
import time
import json
import math
import asyncio
import argparse
from collections import Counter
from pathlib import Path
from typing import Optional, AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ai"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Load environment variables
if os.path.exists(PROJECT_ROOT / ".env"):
    load_dotenv(PROJECT_ROOT / ".env")
elif os.path.exists(PROJECT_ROOT / ".env.client"):
    load_dotenv(PROJECT_ROOT / ".env.client")
else:
    load_dotenv()

# Import the Unified AI Engine
from backend.ai_engine import (
    RouterModel,
    EmbeddingEngine,
    _apply_router_overrides,
    _filter_pii,
    demask_stream,
    build_system_prompt,
    build_user_prompt,
    _build_llm_prompt,
    _clean_llm_token,
    _clean_llm_output,
    detect_branch_from_query,
    strip_pii_from_chunks,
    extract_pdf_content,
    _chunk_text,
    rerank_chunks
)

# Parse command line argument or environment variable
parser = argparse.ArgumentParser(description="MOSAIC-v2 Unified Backend API")
parser.add_argument("--service", choices=["router", "rag", "orchestrator"], default=None,
                    help="Which backend service to start")
parser.add_argument("--port", type=int, default=None, help="Port to run the service on")

# Strip flags that are not ours to prevent Uvicorn/FastAPI parser conflicts
sys_args = [arg for arg in sys.argv[1:] if not arg.startswith("--port") and not arg.startswith("--service")]
args, unknown = parser.parse_known_args()

SERVICE_TYPE = os.getenv("SERVICE_TYPE", "").lower()
if args.service:
    SERVICE_TYPE = args.service
if not SERVICE_TYPE:
    # Auto-detect fallback
    SERVICE_TYPE = "orchestrator"

PORT = args.port
if PORT is None:
    if SERVICE_TYPE == "router":
        PORT = int(os.getenv("ROUTER_PORT", "8000"))
    elif SERVICE_TYPE == "rag":
        bid = os.getenv("BRANCH_ID", "").lower()
        default_rag_p = "8001" if "1" in bid else ("8002" if "2" in bid else "8003")
        PORT = int(os.getenv("PORT", os.getenv("BRANCH_RAG_PORT", default_rag_p)))
    else:
        PORT = int(os.getenv("ORCHESTRATOR_PORT", "8004"))

# Initialize FastAPI app
app = FastAPI(
    title=f"MOSAIC-v2 Backend API - {SERVICE_TYPE.upper()}",
    description=f"Consolidated endpoint hosting for the MOSAIC {SERVICE_TYPE} service.",
    version="2.0.0"
)

# ── KVKK md.12 / BDDK BS Yönetmeliği teknik tedbirleri: iz kaydı + kimlik doğrulama/yetkilendirme ──
from compliance.audit import get_audit
from compliance import access as _access

@app.middleware("http")
async def _compliance_middleware(request: Request, call_next):
    path = request.url.path; t0 = time.perf_counter()
    ip = request.client.host if request.client else None
    actor = "mtls-istemci" if request.url.scheme == "https" else "yerel"
    if SERVICE_TYPE == "orchestrator" and _access.auth_required() and path not in _access.PUBLIC:
        who = _access.authenticate(request.headers.get("X-API-Key", ""))
        if not who:
            get_audit().record("kimlik_dogrulama_red", "anonim", path=path, ip=ip)
            return JSONResponse(status_code=401, content={"detail": "Geçerli API anahtarı gerekli (X-API-Key)."})
        actor = f"api:{who[0]}"
        if not _access.is_allowed(who[1], path):
            get_audit().record("yetki_red", actor, path=path, rol=who[1], ip=ip)
            return JSONResponse(status_code=403, content={"detail": f"'{who[1]}' rolünün bu işleme yetkisi yok."})
    request.state.actor = actor
    response = await call_next(request)
    if path not in _access.PUBLIC:
        get_audit().record("http", actor, path=path, method=request.method, status=response.status_code,
                           ms=round((time.perf_counter() - t0) * 1000, 1), ip=ip)
    return response


# ──────────────────────────────────────────────────────────────────────────────
# HTTPX Client Managers
# ──────────────────────────────────────────────────────────────────────────────
_http_client: Optional[httpx.AsyncClient] = None
_rag_client: Optional[httpx.AsyncClient] = None

def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=60.0)
    return _http_client

# Kaynakta maskeleme: şube /query cevabındaki parçalar şubeden çıkmadan önce maskelenir (varsayılan açık)
BRANCH_PII_MASKING = os.getenv("BRANCH_PII_MASKING", "true").lower() != "false"

def _branch_mask(text: str) -> str:
    """Şube tarafı maskeleme: yer tutucular HQ sorgu eşlemesiyle çakışmasın ve geri açılamasın diye ayrı ad alanı ([GİZLİ_TÜR])."""
    return re.sub(r"\[(IBAN|TC_NO|EMAIL|PHONE|CREDIT_CARD|NAME)_\d+\]", r"[GİZLİ_\1]", _filter_pii(text, {}))

def _mtls(verify, cert):
    """httpx 0.28 uyumlu mTLS bağlamı: verify=<yol> iken cert= yok sayıldığı için sertifikayı SSLContext'e yükle."""
    import ssl as _ssl
    if verify is False and not cert:
        return False
    if verify is False:
        ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT); ctx.check_hostname = False; ctx.verify_mode = _ssl.CERT_NONE
    else:
        ctx = _ssl.create_default_context(cafile=verify) if isinstance(verify, str) else _ssl.create_default_context()
    if cert and cert[0]:
        ctx.load_cert_chain(*cert)
    return ctx

def get_rag_tls_client() -> httpx.AsyncClient:
    global _rag_client
    if _rag_client is not None and not _rag_client.is_closed:
        return _rag_client
    ca_cert = os.getenv("RAG_CA_CERT_PATH", "backend/certs/ca.crt")
    client_cert, client_key = os.getenv("RAG_CLIENT_CERT"), os.getenv("RAG_CLIENT_KEY")
    cert = (client_cert, client_key) if client_cert and client_key and os.path.exists(client_cert) else None
    if os.getenv("RAG_TLS_VERIFY", "true").lower() == "false":
        _rag_client = httpx.AsyncClient(verify=_mtls(False, cert), timeout=30.0)
    elif os.path.exists(ca_cert):
        logger.info(f"[TLS] RAG Client initialized with CA certificate: {ca_cert} (mTLS client cert: {'yes' if cert else 'no'})")
        _rag_client = httpx.AsyncClient(verify=_mtls(ca_cert, cert), timeout=30.0)
    else:
        logger.warning("[TLS] backend/certs/ca.crt not found — using verify=False (Insecure test mode)")
        _rag_client = httpx.AsyncClient(verify=False, timeout=30.0)
    return _rag_client

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Kullanıcı sorusu")

class PredictResponse(BaseModel):
    text: str
    label: int
    label_name: str
    confidence: float
    scores: dict[str, float]
    elapsed_ms: float

class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = Field(default=True)

class DocumentChunk(BaseModel):
    id: str
    text: str
    score: float
    source: str
    branch: str

class RAGQueryResponse(BaseModel):
    query: str
    chunks: list[DocumentChunk]
    elapsed_ms: float

class OrchestratorQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    branch_id: int = Field(default=0, description="0=All (HQ), 1=Branch 1, 2=Branch 2")
    history: list[dict] = Field(default_factory=list)
    local_chunks: list[dict] = Field(default_factory=list)

class OrchestratorQueryResponse(BaseModel):
    query: str
    answer: str
    route: str
    label: int
    confidence: float
    model_name: str
    elapsed_ms: float
    sources: list[DocumentChunk]

# ──────────────────────────────────────────────────────────────────────────────
# 1. ROUTER ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────
if SERVICE_TYPE == "router":
    logger.info("Configuring Router API Endpoints")

    @app.on_event("startup")
    async def startup_router():
        logger.info("[Router] Lazy loading BERT model...")
        RouterModel.get_instance()
        logger.info("[Router] Model loaded successfully.")

    @app.get("/health")
    async def router_health():
        rm = RouterModel.get_instance()
        return {
            "status": "ok",
            "model_path": rm.model_path,
            "device": str(rm.device),
            "backend": rm.backend
        }

    @app.post("/predict", response_model=PredictResponse)
    async def predict_query(request: PredictRequest):
        start = time.perf_counter()
        rm = RouterModel.get_instance()
        result = rm.predict(request.text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return PredictResponse(
            text=request.text,
            elapsed_ms=round(elapsed_ms, 2),
            **result
        )

    @app.post("/predict/batch")
    async def predict_batch(texts: list[str]):
        rm = RouterModel.get_instance()
        return [rm.predict(t) for t in texts[:100]]

# ──────────────────────────────────────────────────────────────────────────────
# 2. RAG & CHROMADB ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────
elif SERVICE_TYPE == "rag":
    BRANCH_ID = os.getenv("BRANCH_ID", "hq")
    logger.info(f"Configuring RAG API Endpoints for Branch: {BRANCH_ID}")

    @app.on_event("startup")
    async def startup_rag():
        logger.info(f"[RAG-{BRANCH_ID}] Eagerly initializing vector DB client...")
        try:
            engine = EmbeddingEngine.get_instance(branch_id=BRANCH_ID)
            # İlk encode çağrısında PyTorch/oneDNN CPU ısınma maliyetini açılışta tüket.
            engine.embed_model.encode(["passage: ısınma"], convert_to_numpy=True, show_progress_bar=False)
            logger.info(f"[RAG-{BRANCH_ID}] Engine initialization & warmup complete.")
        except Exception as e:
            logger.error(f"[RAG-{BRANCH_ID}] Startup initialization warning: {e}. Will retry on query.")

    @app.get("/health")
    async def rag_health():
        try:
            engine = EmbeddingEngine.get_instance(branch_id=BRANCH_ID)
            count = 0
            try:
                count = engine.collection.count()
            except Exception:
                pass
            return {
                "status": "ok",
                "branch_id": BRANCH_ID,
                "chroma_path": getattr(engine, "chroma_path", "unknown"),
                "chroma_doc_count": count,
                "embedding_model": getattr(engine, "embedding_model_key", "unknown"),
                "port": PORT
            }
        except Exception as e:
            logger.warning(f"[RAG-{BRANCH_ID}] Health check warning: {e}")
            return JSONResponse(status_code=200, content={"status": "degraded", "branch_id": BRANCH_ID, "detail": str(e), "port": PORT})

    @app.get("/stats")
    async def rag_stats():
        engine = EmbeddingEngine.get_instance(branch_id=BRANCH_ID)
        return {
            "branch_id": BRANCH_ID,
            "collection_count": engine.collection.count(),
            "embedding_model": engine.embedding_model_key,
            "reranker_available": engine.reranker is not None,
            "chromadb_path": engine.chroma_path
        }

    @app.post("/reset")
    async def reset_rag():
        try:
            engine = EmbeddingEngine.get_instance(branch_id=BRANCH_ID)
            try:
                engine.chroma_client.delete_collection(name=engine.collection_name)
            except Exception:
                pass
            engine.collection = engine.chroma_client.get_or_create_collection(
                name=engine.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.warning(f"ChromaDB collection reset: {engine.collection_name}")
            return {"status": "ok", "message": f"Collection '{engine.collection_name}' reset successfully"}
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            return {"status": "error", "message": str(e)}

    @app.post("/add_chunks")
    async def add_chunks(chunks: list[dict]):
        if not chunks:
            return {"success": False, "message": "Empty chunk list"}

        engine = EmbeddingEngine.get_instance(branch_id=BRANCH_ID)
        texts = [c["content"] for c in chunks]
        ids = [c.get("id") or f"distill_{hash(c['content'])}_{i}" for i, c in enumerate(chunks)]

        passage_chunks = [(f"passage: {t}" if not t.lower().startswith("passage: ") else t) for t in texts]
        embeddings = engine.embed_model.encode(passage_chunks, convert_to_numpy=True, show_progress_bar=False).tolist()

        metas = []
        for c in chunks:
            m = {"source": c.get("source", "distilled"), "branch": c.get("branch", "unknown"), "is_distilled": "true"}
            metas.append(m)

        engine.collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
        return {"success": True, "added": len(chunks)}

    @app.post("/query", response_model=RAGQueryResponse)
    async def query_rag(request: RAGQueryRequest, req: Request = None):
        start = time.perf_counter()
        source_branch = req.headers.get("X-Source-Branch") if req else "unknown"
        logger.info(f"[RAG-Query] Query received. Source: {source_branch}, Query (maskeli): '{_filter_pii(request.query, {})}'")

        engine = EmbeddingEngine.get_instance(branch_id=BRANCH_ID)
        normalized = engine.normalize_query(request.query)

        query_embedding = engine.embed_model.encode(normalized, convert_to_numpy=True).tolist()
        n_results = request.top_k * 4 if request.use_reranker and engine.reranker else request.top_k

        try:
            results = engine.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, engine.collection.count() or 1),
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"ChromaDB query error: {e}")
            raise HTTPException(status_code=500, detail=f"ChromaDB query error: {e}")

        docs = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        ids = results["ids"][0] if results["ids"] else []

        if request.use_reranker and engine.reranker and docs:
            pairs = [[request.query, doc] for doc in docs]
            raw_scores = engine.reranker.predict(pairs, show_progress_bar=False).tolist()
            sigmoid_scores = [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]
            ranked = sorted(zip(sigmoid_scores, docs, metadatas, ids), reverse=True)
            docs = [r[1] for r in ranked[:request.top_k]]
            metadatas = [r[2] for r in ranked[:request.top_k]]
            ids = [r[3] for r in ranked[:request.top_k]]
            scores_final = [r[0] for r in ranked[:request.top_k]]
        else:
            scores_final = [1.0 - d for d in distances[:request.top_k]]
            docs = docs[:request.top_k]
            metadatas = metadatas[:request.top_k]
            ids = ids[:request.top_k]

        # Minimum kosinüs benzerliği (alakasız yoğun arama sonuçlarını şube düzeyinde ön eleme).
        # TurkEmbed'de ilgili parçalar ~0.5-0.6 aralığındadır; asıl alaka elemesini HQ'daki çapraz kodlayıcı yapar.
        MIN_CHROMA_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.45"))
        chunks = []
        for doc_id, doc, meta, score in zip(ids, docs, metadatas, scores_final):
            if score >= MIN_CHROMA_SIMILARITY:
                chunks.append(DocumentChunk(
                    id=doc_id,
                    text=_branch_mask(doc) if BRANCH_PII_MASKING else doc,
                    score=round(score, 4),
                    source=meta.get("source", "unknown"),
                    branch=meta.get("branch", BRANCH_ID)
                ))

        elapsed_ms = (time.perf_counter() - start) * 1000
        # Denetim görünürlüğü: bu şubenin isteyen şubeye kaç parça döndürdüğü ve en alakalı parçanın
        # maskeli kısa önizlemesi (önizleme her durumda kaynakta maskelenir; ham PII loga yazılmaz).
        if chunks:
            top = chunks[0]
            preview = re.sub(r"\s+", " ", top.text if BRANCH_PII_MASKING else _branch_mask(top.text)).strip()[:160]
            per_source = dict(Counter(c.source for c in chunks).most_common())
            logger.info(f"[RAG-Response] {BRANCH_ID} -> {source_branch}: {len(chunks)} parça döndü | "
                        f"en iyi skor={top.score:.3f} | kaynak başına parça={per_source} | {elapsed_ms:.0f} ms")
            logger.info(f"[RAG-Response] En alakalı parça ({BRANCH_ID}, maskeli): '{preview}...'")
        else:
            logger.info(f"[RAG-Response] {BRANCH_ID} -> {source_branch}: benzerlik eşiği ({MIN_CHROMA_SIMILARITY}) "
                        f"üstünde parça yok, boş yanıt | {elapsed_ms:.0f} ms")
        get_audit().record("rag_sorgu", getattr(req.state, "actor", "?") if req else "?", kaynak_sube=source_branch,
                           sorgu_ozet=get_audit().digest(_filter_pii(request.query, {})), belgeler=[c.id for c in chunks])
        return RAGQueryResponse(query=request.query, chunks=chunks, elapsed_ms=round(elapsed_ms, 2))

    class EraseRequest(BaseModel):
        ids: list[str] = Field(default_factory=list)
        contains: list[str] = Field(default_factory=list, description="Belge metninde geçen tanımlayıcı (ör. TC no) ile imha")

    @app.post("/erase")
    async def erase_rag(request: EraseRequest, req: Request):
        """KVKK md.7 / imha: belirtilen kayıtları kalıcı olarak siler; silinen kimlikleri döndürür."""
        engine = EmbeddingEngine.get_instance(branch_id=BRANCH_ID)
        ids = set(request.ids)
        for term in request.contains:
            if term.strip():
                ids.update(engine.collection.get(where_document={"$contains": term.strip()}, include=[])["ids"])
        existing = engine.collection.get(ids=list(ids), include=[])["ids"] if ids else []
        if existing: engine.collection.delete(ids=existing)
        get_audit().record("imha", getattr(req.state, "actor", "?"), silinen=existing, terim_ozet=[get_audit().digest(t) for t in request.contains])
        return {"silinen": existing, "adet": len(existing)}

    @app.get("/audit/verify")
    async def audit_verify_rag():
        ok, n, bad = get_audit().verify(); return {"gecerli": ok, "kayit": n, "bozuk": bad}

    @app.post("/upload")
    async def upload_document(file: UploadFile = File(...), source_name: Optional[str] = Form(default=None)):
        t0 = time.perf_counter()
        logger.info(f"[Upload] Processing uploaded file: {file.filename}, source: {source_name}")
        try:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Boş dosya yüklendi")

            fname = file.filename or "document"
            src = (source_name or "").strip() or Path(fname).stem
            import re
            clean_src = re.sub(r"[^\w\-]", "_", src, flags=re.UNICODE)

            if fname.lower().endswith(".pdf"):
                text = extract_pdf_content(content)
                if not text or not text.strip():
                    raise HTTPException(status_code=400, detail="PDF belgesinden metin çıkarılamadı (boş veya taranamadı)")
            else:
                try:
                    text = content.decode("utf-8")
                except Exception:
                    text = content.decode("latin-1", errors="replace")

            chunks = _chunk_text(text)
            if not chunks:
                raise HTTPException(status_code=400, detail="Belge içeriği parçalama için yetersiz")

            engine = EmbeddingEngine.get_instance(branch_id=BRANCH_ID)
            passage_chunks = [(f"passage: {c}" if not c.lower().startswith("passage: ") else c) for c in chunks]
            embeddings = engine.embed_model.encode(passage_chunks, convert_to_numpy=True, show_progress_bar=False).tolist()

            timestamp_suffix = int(time.time())
            ids = [f"upload_{clean_src}_{i}_{timestamp_suffix}" for i in range(len(chunks))]
            metas = [{"source": src, "branch": BRANCH_ID, "is_upload": "true"} for _ in chunks]

            engine.collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metas)
            elapsed = round((time.perf_counter() - t0) * 1000, 1)

            logger.success(f"[Upload] Başarıyla indekslendi: {len(chunks)} chunk ({src}) -> {engine.collection_name}")
            return {
                "success": True,
                "source": src,
                "branch": BRANCH_ID,
                "chunks_added": len(chunks),
                "collection_total": engine.collection.count(),
                "elapsed_ms": elapsed
            }
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"[Upload Error] Belge işleme hatası ({file.filename}): {e}\n{tb}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "detail": f"{type(e).__name__}: {str(e)}", "traceback": tb}
            )

# ──────────────────────────────────────────────────────────────────────────────
# 3. ORCHESTRATOR ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────
else:
    logger.info("Configuring Orchestrator API Endpoints")

    # Read environment URLs
    ROUTER_URL = os.getenv("ROUTER_URL", "http://localhost:8000")
    BRANCH1_RAG_URL = os.getenv("BRANCH1_RAG_URL", "http://localhost:8001")
    BRANCH2_RAG_URL = os.getenv("BRANCH2_RAG_URL", "http://localhost:8002")
    HQ_RAG_URL = os.getenv("HQ_RAG_URL", "http://localhost:8003")
    BRANCH_LLM_URL = os.getenv("BRANCH_LLM_URL", "http://localhost:8099")  # Gemma
    HQ_LLM_URL = os.getenv("HQ_LLM_URL", "http://localhost:8098")          # Qwen

    BRANCH_MODEL_NAME = "Gemma-3-4B"
    HQ_MODEL_NAME = "Qwen3-8B"
    RAG_FINAL_K = int(os.getenv("RAG_FINAL_K", "6"))  # bağlama en fazla 6 parça girer (build_user_prompt)

    async def call_router(query: str) -> dict:
        client = get_http_client()
        try:
            r = await client.post(f"{ROUTER_URL}/predict", json={"text": query}, timeout=5.0)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.error(f"Router connectivity error: {e}")
        return {"label": 1, "label_name": "simple", "confidence": 1.0}

    async def call_rag(url: str, query: str, branch_id: int, source_branch: str) -> list[dict]:
        client = get_rag_tls_client()
        headers = {"X-Source-Branch": source_branch}
        try:
            r = await client.post(
                f"{url.rstrip('/')}/query",
                json={"query": query, "top_k": 8, "use_reranker": False},
                headers=headers,
                timeout=10.0
            )
            if r.status_code == 200:
                data = r.json()
                return data.get("chunks", [])
        except Exception as e:
            logger.error(f"RAG Service unreachable ({url}): {e}")
        return []

    async def call_llm(llm_url: str, model_name: str, system: str, user: str, route: str) -> str:
        prompt, stop, n_predict, temp = _build_llm_prompt(model_name, system, user, route)
        payload = {
            "prompt": prompt, "n_predict": n_predict, "temperature": temp,
            "top_p": 0.95, "repeat_penalty": 1.1, "stream": False, "stop": stop
        }
        client = get_http_client()
        try:
            r = await client.post(f"{llm_url}/completion", json=payload, timeout=180.0)
            r.raise_for_status()
            answer = r.json().get("content", "").strip()
            return _clean_llm_output(answer)
        except Exception as e:
            if llm_url != HQ_LLM_URL:
                logger.warning(f"Branch LLM unreachable ({e}), falling back to HQ LLM ({HQ_LLM_URL})...")
                return await call_llm(HQ_LLM_URL, HQ_MODEL_NAME, system, user, route)
            logger.error(f"LLM completed generation failed: {e}")
            return f"Üzgünüm, şu an yanıt üretemiyorum ({e})"

    async def stream_llm_tokens(llm_url: str, model_name: str, system: str, user: str, route: str) -> AsyncGenerator[str, None]:
        prompt, stop, n_predict, temp = _build_llm_prompt(model_name, system, user, route)
        payload = {
            "prompt": prompt, "n_predict": n_predict, "temperature": temp,
            "top_p": 0.95, "repeat_penalty": 1.1, "stream": True, "stop": stop
        }
        client = get_http_client()
        target_url = llm_url
        target_model = model_name

        # Quick health pre-check for fallback
        try:
            r = await client.get(f"{target_url}/health", timeout=1.0)
            if r.status_code != 200 and target_url != HQ_LLM_URL:
                logger.warning(f"Branch LLM ({target_url}) not healthy, routing to HQ LLM ({HQ_LLM_URL})")
                target_url = HQ_LLM_URL
                target_model = HQ_MODEL_NAME
                prompt, stop, n_predict, temp = _build_llm_prompt(target_model, system, user, route)
                payload["prompt"] = prompt
                payload["stop"] = stop
        except Exception:
            if target_url != HQ_LLM_URL:
                logger.warning(f"Branch LLM ({target_url}) unreachable, routing to HQ LLM ({HQ_LLM_URL})")
                target_url = HQ_LLM_URL
                target_model = HQ_MODEL_NAME
                prompt, stop, n_predict, temp = _build_llm_prompt(target_model, system, user, route)
                payload["prompt"] = prompt
                payload["stop"] = stop

        try:
            async with client.stream("POST", f"{target_url}/completion", json=payload, timeout=180.0) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "): continue
                    try:
                        chunk = json.loads(line[6:])
                    except Exception:
                        continue
                    if chunk.get("stop"):
                        break
                    token = chunk.get("content", "")
                    if token:
                        yield _clean_llm_token(token)
        except Exception as e:
            logger.error(f"LLM streaming tokens failed: {e}")
            yield f" [LLM Hatası: {e}]"

    # Şube parçalarını HQ koleksiyonuna kalıcı kopyalayan damıtma, tezdeki tasarımda (Bölüm 3.2.3: şube parçaları
    # yalnızca yanıt üretimi için döner) yer almadığından varsayılan olarak KAPALIDIR.
    HQ_DISTILL_ENABLED = os.getenv("HQ_DISTILL_ENABLED", "false").lower() == "true"
    # Şubeye özgü bilgi yalnızca şubelerde bulunduğundan (HQ koleksiyonu yalnızca ortak bilgi içerir), şube
    # arayüzünden gelen her bilgi getirme sorgusunda diğer şube de paralel sorgulanır (Tez Bölüm 3.2.1).
    # false: diğer şube yalnızca "complex" rotada sorgulanır (tez ölçümlerindeki davranış).
    CROSS_BRANCH_ALWAYS = os.getenv("CROSS_BRANCH_ALWAYS", "true").lower() == "true"

    def _log_orch_selection(req_branch_id: int, route: str, sources: list[dict]):
        b_counts = dict(Counter(s.get("branch", "?") for s in sources).most_common())
        top_b = (
            f"{sources[0].get('branch', '?')} ({sources[0].get('source', '?')}, skor={sources[0].get('score', 0.0):.3f})"
            if sources else "yok"
        )
        logger.info(f"[Orch-Select] İstek=branch_{req_branch_id:02d} | Rota={route} | Seçilen kaynaklar={b_counts} | 1. kaynak={top_b}")

    async def distill_knowledge(chunks: list[dict]):
        # Distills high score chunks into HQ ChromaDB
        distill_list = []
        for c in chunks:
            if float(c.get("score", 0.0)) >= 0.60:
                distill_list.append({
                    "content": c.get("text", c.get("content", "")),
                    "source": c.get("source", "distilled"),
                    "branch": c.get("branch", "unknown"),
                    "id": f"distill_{c.get('id', hash(c.get('text', '')))}"
                })
        if distill_list:
            try:
                client = get_rag_tls_client()
                await client.post(f"{HQ_RAG_URL}/add_chunks", json=distill_list, timeout=10.0)
                logger.info(f"[Distill] Knowledge transfer completed for {len(distill_list)} chunks.")
            except Exception as e:
                logger.error(f"[Distill] Failed to transfer distilled knowledge to HQ: {e}")

    @app.get("/health")
    async def orchestrator_health():
        client = get_http_client()
        tls_client = get_rag_tls_client()
        services_list = [
            ("router", ROUTER_URL),
            ("hq_rag", HQ_RAG_URL),
            ("branch1_rag", BRANCH1_RAG_URL),
            ("branch2_rag", BRANCH2_RAG_URL),
            ("gemma_llm", BRANCH_LLM_URL),
            ("qwen_llm", HQ_LLM_URL)
        ]

        async def check_service(svc_name, url):
            try:
                c = tls_client if url.lower().startswith("https://") else client
                r = await c.get(f"{url.rstrip('/')}/health" if "llm" not in svc_name else url, timeout=2.0)
                return svc_name, ("up" if r.status_code == 200 else "down")
            except Exception:
                return svc_name, "down"

        tasks = [check_service(svc, url) for svc, url in services_list]
        results = await asyncio.gather(*tasks)
        services = dict(results)
        return {"status": "ok", "services": services}

    @app.post("/reset_hq")
    async def reset_hq():
        try:
            client = get_rag_tls_client()
            r = await client.post(f"{HQ_RAG_URL}/reset", timeout=15.0)
            return r.json()
        except Exception as e:
            return {"status": "error", "message": f"Could not connect to HQ RAG: {e}"}

    class OrchEraseRequest(BaseModel):
        contains: list[str] = Field(..., min_length=1, description="İmha edilecek ilgili kişiye ait tanımlayıcılar (TC, IBAN...)")

    @app.post("/erase")
    async def orchestrator_erase(request: OrchEraseRequest, req: Request):
        """İlgili kişi silme talebi: şubelerde kaynak kayıtları, HQ'da damıtılmış kopyaları imha eder."""
        client = get_rag_tls_client(); report = {}
        deleted_src = []
        for name, url in (("branch_01", BRANCH1_RAG_URL), ("branch_02", BRANCH2_RAG_URL), ("hq", HQ_RAG_URL)):
            try:
                r = await client.post(f"{url.rstrip('/')}/erase", json={"contains": request.contains}, timeout=30.0)
                report[name] = r.json(); deleted_src += report[name].get("silinen", [])
            except Exception as e:
                report[name] = {"hata": str(e)}
        if deleted_src:  # damıtılmış kopyalar: distill_<kaynak_id>
            r = await client.post(f"{HQ_RAG_URL.rstrip('/')}/erase", json={"ids": [f"distill_{i}" for i in deleted_src]}, timeout=30.0)
            report["hq_damitilmis"] = r.json()
        get_audit().record("imha_talebi", getattr(req.state, "actor", "?"), terim_ozet=[get_audit().digest(t) for t in request.contains],
                           sonuc={k: v.get("adet", v.get("hata")) for k, v in report.items()})
        return report

    @app.get("/audit/verify")
    async def audit_verify_orch():
        ok, n, bad = get_audit().verify(); return {"gecerli": ok, "kayit": n, "bozuk": bad}

    @app.post("/query", response_model=OrchestratorQueryResponse)
    async def orchestrator_query(request: OrchestratorQueryRequest):
        start = time.perf_counter()
        original_query = request.query

        mappings = {}
        auto_branch_id, branch_detected = detect_branch_from_query(original_query, default_branch_id=request.branch_id)
        detected_branch = auto_branch_id if branch_detected else 0
        if branch_detected and request.branch_id == 0:
            request.branch_id = auto_branch_id
            logger.info(f"[KVKK Routing] Auto-routed branch to: {request.branch_id}")

        router_res = await call_router(original_query)
        router_res = _apply_router_overrides(original_query, router_res)

        label = router_res.get("label", 1)
        route = router_res.get("label_name", "simple")
        confidence = router_res.get("confidence", 0.0)

        sources = []
        local_chunks = request.local_chunks

        if label > 0:
            tasks = []
            source_branch = "hq" if request.branch_id == 0 else f"branch_{request.branch_id:02d}"
            tasks.append(call_rag(HQ_RAG_URL, original_query, branch_id=0, source_branch=source_branch))

            if request.branch_id == 0:
                tasks.append(call_rag(BRANCH1_RAG_URL, original_query, branch_id=1, source_branch=source_branch))
                tasks.append(call_rag(BRANCH2_RAG_URL, original_query, branch_id=2, source_branch=source_branch))
            else:
                if not local_chunks:
                    branch_url = BRANCH1_RAG_URL if request.branch_id == 1 else BRANCH2_RAG_URL
                    tasks.append(call_rag(branch_url, original_query, branch_id=request.branch_id, source_branch=source_branch))
                if label == 2 or CROSS_BRANCH_ALWAYS:
                    other_url = BRANCH2_RAG_URL if request.branch_id == 1 else BRANCH1_RAG_URL
                    other_id = 2 if request.branch_id == 1 else 1
                    tasks.append(call_rag(other_url, original_query, branch_id=other_id, source_branch=source_branch))

            results = await asyncio.gather(*tasks)
            ext_chunks = results[0]
            if len(results) > 1:
                for r_chunks in results[1:]:
                    ext_chunks.extend(r_chunks)

            merged = []
            if local_chunks:
                for lc in local_chunks:
                    merged.append({"text": lc.get("text", ""), "score": float(lc.get("score", 0.5)),
                                   "source": lc.get("source", "local"), "branch": f"branch_{request.branch_id:02d}"})
            for ec in ext_chunks:
                merged.append({"text": ec.text if hasattr(ec, "text") else ec.get("text", ""),
                               "score": ec.score if hasattr(ec, "score") else ec.get("score", 0.0),
                               "source": ec.source if hasattr(ec, "source") else ec.get("source", "unknown"),
                               "branch": ec.branch if hasattr(ec, "branch") else ec.get("branch", "unknown")})

            # Sort and deduplicate
            # Aynı metin hem HQ koleksiyonundan hem bir şube servisinden döndüyse (HQ koleksiyonu şube
            # korpuslarının kopyasını içerir) en yüksek skor korunur, kaynak etiketi şube olarak işaretlenir.
            seen = {}
            unique_chunks = []
            for item in sorted(merged, key=lambda x: x["score"], reverse=True):
                txt = item["text"]
                if txt not in seen:
                    seen[txt] = item
                    unique_chunks.append(item)
                elif str(seen[txt].get("branch")) in ("hq", "None", "unknown") and str(item.get("branch")) not in ("hq", "None", "unknown"):
                    seen[txt]["branch"] = item["branch"]
                    seen[txt]["source"] = item.get("source", seen[txt].get("source"))

            unique_chunks = strip_pii_from_chunks(unique_chunks)
            sources = rerank_chunks(original_query, unique_chunks, top_k=RAG_FINAL_K, min_score=0.20)
            _log_orch_selection(request.branch_id, route, sources)

        # Prompt construction
        llm_url = HQ_LLM_URL if label == 2 else BRANCH_LLM_URL
        model_name = HQ_MODEL_NAME if label == 2 else BRANCH_MODEL_NAME

        system = build_system_prompt(route)
        masked_query = _filter_pii(original_query, mappings)
        user_prompt = build_user_prompt(masked_query, sources, route, history=request.history, mappings=mappings)

        answer = await call_llm(llm_url, model_name, system, user_prompt, route)

        # De-mask mappings in answers and source contexts
        for placeholder, original in mappings.items():
            answer = answer.replace(placeholder, original)

        formatted_sources = []
        for s in sources:
            content = s.get("text", s.get("content", ""))
            for placeholder, original in mappings.items():
                content = content.replace(placeholder, original)
            formatted_sources.append(DocumentChunk(
                id=s.get("id", ""),
                text=content,
                score=s.get("score", 0.0),
                source=s.get("source", "unknown"),
                branch=s.get("branch", "unknown")
            ))

        if HQ_DISTILL_ENABLED and request.branch_id == 0 and any(s.get("branch") != "hq" for s in sources):
            asyncio.create_task(distill_knowledge(sources))

        get_audit().record("orkestrator_sorgu", "api", sube=request.branch_id, rota=route,
                           sorgu_ozet=get_audit().digest(masked_query), kaynaklar=[f"{x.branch}:{x.id}" for x in formatted_sources])
        elapsed_ms = (time.perf_counter() - start) * 1000
        return OrchestratorQueryResponse(
            query=original_query,
            answer=answer,
            route=route,
            label=label,
            confidence=confidence,
            model_name=model_name,
            elapsed_ms=round(elapsed_ms, 2),
            sources=formatted_sources
        )

    @app.post("/query/stream")
    async def orchestrator_query_stream(request: OrchestratorQueryRequest):
        original_query = request.query
        mappings = {}

        auto_branch_id, branch_detected = detect_branch_from_query(original_query, default_branch_id=request.branch_id)
        detected_branch = auto_branch_id if branch_detected else 0
        if branch_detected and request.branch_id == 0:
            request.branch_id = auto_branch_id

        async def generate():
            router_res = await call_router(original_query)
            router_res = _apply_router_overrides(original_query, router_res)
            label = router_res.get("label", 1)
            route = router_res.get("label_name", "simple")
            confidence = router_res.get("confidence", 0.0)

            yield json.dumps({"step": "router", "status": "done", "route": route, "label": label, "confidence": confidence}) + "\n"

            sources = []
            local_chunks = request.local_chunks

            if label > 0:
                tasks = []
                source_branch = "hq" if request.branch_id == 0 else f"branch_{request.branch_id:02d}"
                tasks.append(call_rag(HQ_RAG_URL, original_query, branch_id=0, source_branch=source_branch))

                if request.branch_id == 0:
                    tasks.append(call_rag(BRANCH1_RAG_URL, original_query, branch_id=1, source_branch=source_branch))
                    tasks.append(call_rag(BRANCH2_RAG_URL, original_query, branch_id=2, source_branch=source_branch))
                else:
                    if not local_chunks:
                        branch_url = BRANCH1_RAG_URL if request.branch_id == 1 else BRANCH2_RAG_URL
                        tasks.append(call_rag(branch_url, original_query, branch_id=request.branch_id, source_branch=source_branch))
                    if label == 2 or CROSS_BRANCH_ALWAYS:
                        other_url = BRANCH2_RAG_URL if request.branch_id == 1 else BRANCH1_RAG_URL
                        other_id = 2 if request.branch_id == 1 else 1
                        tasks.append(call_rag(other_url, original_query, branch_id=other_id, source_branch=source_branch))

                results = await asyncio.gather(*tasks)
                ext_chunks = results[0]
                if len(results) > 1:
                    for r_chunks in results[1:]:
                        ext_chunks.extend(r_chunks)

                merged = []
                if local_chunks:
                    for lc in local_chunks:
                        merged.append({"text": lc.get("text", ""), "score": float(lc.get("score", 0.5)),
                                       "source": lc.get("source", "local"), "branch": f"branch_{request.branch_id:02d}"})
                for ec in ext_chunks:
                    merged.append({"text": ec.text if hasattr(ec, "text") else ec.get("text", ""),
                                   "score": ec.score if hasattr(ec, "score") else ec.get("score", 0.0),
                                   "source": ec.source if hasattr(ec, "source") else ec.get("source", "unknown"),
                                   "branch": ec.branch if hasattr(ec, "branch") else ec.get("branch", "unknown")})

                # Aynı metin HQ'dan ve bir şubeden döndüyse en yüksek skor korunur, etiket şube olur
                seen = {}
                unique_chunks = []
                for item in sorted(merged, key=lambda x: x["score"], reverse=True):
                    txt = item["text"]
                    if txt not in seen:
                        seen[txt] = item
                        unique_chunks.append(item)
                    elif str(seen[txt].get("branch")) in ("hq", "None", "unknown") and str(item.get("branch")) not in ("hq", "None", "unknown"):
                        seen[txt]["branch"] = item["branch"]
                        seen[txt]["source"] = item.get("source", seen[txt].get("source"))

                unique_chunks = strip_pii_from_chunks(unique_chunks)
                sources = rerank_chunks(original_query, unique_chunks, top_k=RAG_FINAL_K, min_score=0.20)
                _log_orch_selection(request.branch_id, route, sources)

            # De-mask sources content for user-facing UI
            ui_sources = []
            for s in sources:
                content = s.get("text", s.get("content", ""))
                for placeholder, original in mappings.items():
                    content = content.replace(placeholder, original)
                ui_sources.append({
                    "content": content,
                    "source": s.get("source", "unknown"),
                    "score": s.get("score", 0.0),
                    "branch": s.get("branch", "unknown")
                })

            yield json.dumps({"step": "rag", "status": "done", "sources": ui_sources, "doc_count": len(ui_sources)}) + "\n"

            llm_url = HQ_LLM_URL if label == 2 else BRANCH_LLM_URL
            model_name = HQ_MODEL_NAME if label == 2 else BRANCH_MODEL_NAME
            system = build_system_prompt(route)

            masked_query = _filter_pii(original_query, mappings)
            user_prompt = build_user_prompt(masked_query, sources, route, history=request.history, mappings=mappings)

            token_stream = stream_llm_tokens(llm_url, model_name, system, user_prompt, route)
            async for token in demask_stream(token_stream, mappings):
                yield json.dumps({"step": "llm_token", "token": token}) + "\n"

            yield json.dumps({"step": "result", "model_name": model_name, "sources": ui_sources}) + "\n"

            if HQ_DISTILL_ENABLED and request.branch_id == 0 and any(s.get("branch") != "hq" for s in sources):
                asyncio.create_task(distill_knowledge(sources))

        return StreamingResponse(generate(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting MOSAIC-v2 {SERVICE_TYPE.upper()} Backend Service on Port: {PORT}...")
    uvicorn.run(app, host=os.getenv("BIND_HOST", "0.0.0.0"), port=PORT)
