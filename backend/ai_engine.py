"""
MOSAIC-v2 Unified AI Engine
===========================
Consolidates all AI, NLP, and document processing logic:
  - BERT Query Router (RouterModel)
  - TurkEmbed & e5-small Embedding & Vector DB Retrieval (EmbeddingEngine)
  - PDF Document Parsing & Extraction (PyMuPDF, pdfplumber, pdfminer.six)
  - Text Chunking & Semantic Indexing (chunk_text, _chunk_text, index_corpus)
  - KVKK/GDPR Compliance (PII masking/de-masking, CJK stripping)
  - Orchestration Prompts, LLM Prompt Building, & Routing Overrides
"""

import os
import re
import sys
import json
import random
import io
import math
from pathlib import Path
from typing import Optional, Tuple, AsyncGenerator

import numpy as np
import torch
from loguru import logger

# Set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ai"))

# ─── CJK & THINK REGEX ────────────────────────────────────────────────────────
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u2e80-\u2eff\u2f00-\u2fdf\u3000-\u303f\u31c0-\u31ef\uac00-\ud7af\u3040-\u309f]")
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

# ─── 1. ROUTER AI (BERT CLASSIFIER) ──────────────────────────────────────────

LABEL_NAMES = ["chitchat", "simple", "complex"]

class RouterModel:
    _instance: Optional["RouterModel"] = None

    def __init__(self):
        from transformers import AutoTokenizer

        _default_path = str(PROJECT_ROOT / "ai" / "models" / "router" / "bert_router_best")
        self.model_path = os.getenv("MODEL_PATH", _default_path)
        self.backend = os.getenv("ROUTER_BACKEND", "pytorch").lower()  # "pytorch" | "onnx"

        if self.model_path.startswith("/app") and not os.path.exists(self.model_path):
            self.model_path = _default_path

        logger.info(f"[RouterModel] Tokenizer loading from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        _default_onnx = str(Path(self.model_path).parent / "onnx" / "router_int8.onnx")
        self.onnx_model_path = os.getenv("ONNX_MODEL_PATH", _default_onnx)

        if self.backend == "onnx":
            self._load_onnx()
        else:
            self._load_pytorch()

    def _load_pytorch(self):
        from transformers import AutoModelForSequenceClassification
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[RouterModel] PyTorch backend — device: {self.device}, path: {self.model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device).eval()
        self.ort_session = None

    def _load_onnx(self):
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("[RouterModel] onnxruntime not found — falling back to PyTorch.")
            self.backend = "pytorch"
            self._load_pytorch()
            return

        onnx_path = Path(self.onnx_model_path)
        if not onnx_path.exists():
            logger.warning(f"[RouterModel] ONNX model not found: {onnx_path} — falling back to PyTorch.")
            self.backend = "pytorch"
            self._load_pytorch()
            return

        logger.info(f"[RouterModel] ONNX INT8 backend — path: {onnx_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "4"))
        self.ort_session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.device = torch.device("cpu")
        self.model = None

    @classmethod
    def get_instance(cls) -> "RouterModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict(self, text: str) -> dict:
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="np" if self.backend == "onnx" else "pt",
        )

        if self.backend == "onnx":
            feed = {k: v for k, v in encoding.items()
                    if k in [i.name for i in self.ort_session.get_inputs()]}
            logits = self.ort_session.run(None, feed)[0][0]
            logits_exp = np.exp(logits - logits.max())
            probs = logits_exp / logits_exp.sum()
        else:
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_label = int(np.argmax(probs))
        return {
            "label": pred_label,
            "label_name": LABEL_NAMES[pred_label],
            "confidence": float(probs[pred_label]),
            "scores": {name: float(probs[i]) for i, name in enumerate(LABEL_NAMES)},
        }

# ─── 2. EMBEDDING ENGINE & RETRIEVAL ──────────────────────────────────────────

_TR_FIX: dict[str, str] = {
    "Ý": "İ", "ý": "ı", "Þ": "Ş", "þ": "ş",
    "Ð": "Ğ", "ð": "ğ",
    "î": "i", "â": "a", "û": "u",
    "\x00": "", "\ufeff": "",
}

def _normalize_turkish_chars(text: str) -> str:
    for broken, fixed in _TR_FIX.items():
        text = text.replace(broken, fixed)
    return text

MOSAIC_EMBED_HF_ID = "Toxotes/mosaic-embed"


def _embedding_model_path(key: str) -> Optional[Path]:
    """Local model directory for EMBEDDING_MODEL_KEY; None means the base intfloat/multilingual-e5-small.

    e5small_base  base multilingual-e5-small (deployed testbed configuration)
    mosaic_embed  MOSAIC-Embed, ai/models/embeddings/turkembed_v3 (on the Hub as Toxotes/mosaic-embed)
    The earlier instruction-tuned model (turkembed_finetuned) is withdrawn: it retrieved worse than its base model.
    """
    base = PROJECT_ROOT / "ai" / "models" / "embeddings"
    if key == "mosaic_embed":
        return base / "turkembed_v3"
    if key != "e5small_base":
        logger.warning(f"EMBEDDING_MODEL_KEY={key} is not supported; using the base multilingual-e5-small")
    return None


class EmbeddingEngine:
    _instance: Optional["EmbeddingEngine"] = None

    def __init__(self, branch_id: str = "hq"):
        from sentence_transformers import SentenceTransformer, CrossEncoder
        import chromadb

        self.branch_id = branch_id
        self.embedding_model_key = os.getenv("EMBEDDING_MODEL_KEY", "e5small_base")
        self.reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

        # Normalize branch id
        b_str = str(self.branch_id).lower().strip()
        if "1" in b_str:
            clean_id = "branch_01"
        elif "2" in b_str:
            clean_id = "branch_02"
        else:
            clean_id = "hq"
        self.clean_branch_id = clean_id

        # Self-healing ChromaDB path resolution (Works seamlessly on WSL, Host, or VMware Ubuntu)
        env_chroma = os.getenv("CHROMADB_PATH", "").strip()
        resolved_path = None
        if env_chroma:
            try:
                candidate = Path(os.path.expanduser(env_chroma))
                candidate.mkdir(parents=True, exist_ok=True)
                resolved_path = str(candidate)
            except Exception as pe:
                logger.warning(f"[EmbeddingEngine] CHROMADB_PATH '{env_chroma}' erişilemedi ({pe}), yerel dizine geçiliyor.")

        if not resolved_path:
            # Try user home directory
            try:
                candidate = Path.home() / "mosaic_data" / "chroma_db" / clean_id
                candidate.mkdir(parents=True, exist_ok=True)
                resolved_path = str(candidate)
            except Exception:
                # Fallback to repository data directory (guaranteed writable)
                candidate = PROJECT_ROOT / "ai" / "data" / "chromadb" / clean_id
                candidate.mkdir(parents=True, exist_ok=True)
                resolved_path = str(candidate)

        self.chroma_path = resolved_path

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[EmbeddingEngine] Initializing — Device: {_device} — Branch: {self.branch_id}")

        # Loading Embedding Model
        embed_path = _embedding_model_path(self.embedding_model_key)
        logger.info(f"[EmbeddingEngine] Loading embedding: {embed_path}")

        if embed_path is None:
            # Q1 revizyonu: taban model, geri çekilen ilk ince ayarlı modelden daha iyi erişim sağladı
            logger.info("[EmbeddingEngine] Taban multilingual-e5-small kullanılıyor (EMBEDDING_MODEL_KEY=e5small_base)")
            self.embed_model = SentenceTransformer("intfloat/multilingual-e5-small", device=_device, trust_remote_code=True)
        elif embed_path.exists():
            try:
                self.embed_model = SentenceTransformer(str(embed_path), device=_device, trust_remote_code=True)
                logger.success("[EmbeddingEngine] Fine-tuned local embedding loaded.")
            except Exception as e:
                logger.error(f"[EmbeddingEngine] Load error: {e}, falling back to HF model.")
                self.embed_model = SentenceTransformer("intfloat/multilingual-e5-small", device=_device, trust_remote_code=True)
        else:
            fallback = MOSAIC_EMBED_HF_ID if self.embedding_model_key == "mosaic_embed" else "intfloat/multilingual-e5-small"
            logger.warning(f"[EmbeddingEngine] Local path not found, using standard HF model: {fallback}")
            try:
                self.embed_model = SentenceTransformer(fallback, device=_device, trust_remote_code=True)
            except Exception as e:
                logger.error(f"[EmbeddingEngine] HuggingFace embedding loading failed ({e}).")
                raise e

        # Loading Hybrid Tokenizer
        self.hybrid_tokenizer = None
        try:
            from tokenizer.hybrid_tokenizer import HybridTokenizer
            self.hybrid_tokenizer = HybridTokenizer(bpe_model_path=None, use_zemberek=True)
            logger.info("[EmbeddingEngine] Hybrid tokenizer (Zemberek+BPE) loaded.")
        except Exception as e:
            logger.warning(f"[EmbeddingEngine] Hybrid tokenizer could not be loaded: {e}")

        # Loading Reranker (HQ Only)
        if self.branch_id == "hq":
            try:
                self.reranker = CrossEncoder(self.reranker_model, device=_device)
                logger.info(f"[EmbeddingEngine] Reranker loaded: {self.reranker_model} ({_device})")
            except Exception as e:
                logger.warning(f"[EmbeddingEngine] Reranker load failed: {e}")
                self.reranker = None
        else:
            self.reranker = None

        # ChromaDB setup
        logger.info(f"[EmbeddingEngine] ChromaDB path: {self.chroma_path}")
        collection_map = {
            "hq": "finance_global",
            "branch_01": "finance_branch1",
            "branch_02": "finance_branch2",
        }
        self.collection_name = collection_map.get(self.clean_branch_id, f"corpus_{self.branch_id}")
        # İsteğe bağlı HNSW ayarı (yalnızca koleksiyon ilk oluşturulurken etkili): RAG_HNSW_EF
        _coll_meta = {"hnsw:space": "cosine"}
        if os.getenv("RAG_HNSW_EF"):
            _ef = int(os.getenv("RAG_HNSW_EF"))
            _coll_meta.update({"hnsw:search_ef": _ef, "hnsw:construction_ef": _ef, "hnsw:M": 32})

        try:
            self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
            except Exception:
                self.collection = self.chroma_client.create_collection(name=self.collection_name, metadata=_coll_meta)
        except Exception as ce:
            logger.warning(f"[EmbeddingEngine] ChromaDB path '{self.chroma_path}' failed ({ce}). Attempting safe project-local path...")
            try:
                safe_path = PROJECT_ROOT / "data" / "chromadb" / self.clean_branch_id
                safe_path.mkdir(parents=True, exist_ok=True)
                self.chroma_path = str(safe_path)
                self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(f"[EmbeddingEngine] Successfully recovered with safe path: {self.chroma_path}")
            except Exception as ce2:
                logger.error(f"[EmbeddingEngine] Safe path failed ({ce2}). Using in-memory EphemeralClient fallback.")
                self.chroma_client = chromadb.EphemeralClient()
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )

        try:
            doc_cnt = self.collection.count()
        except Exception:
            doc_cnt = 0
        logger.info(f"[EmbeddingEngine] ChromaDB collection ready: {self.collection_name} ({doc_cnt} chunks)")

    def normalize_query(self, query: str) -> str:
        normalized = _normalize_turkish_chars(query)
        if self.hybrid_tokenizer is not None:
            try:
                normalized = self.hybrid_tokenizer._normalize(normalized)
            except Exception:
                pass
        prefix = "query: "
        if normalized.lower().startswith(prefix):
            return normalized
        return prefix + normalized

    @classmethod
    def get_instance(cls, branch_id: str = "hq") -> "EmbeddingEngine":
        if cls._instance is None:
            cls._instance = cls(branch_id=branch_id)
        return cls._instance

# ─── 3. CHUNKING & INDEXING LOGIC ─────────────────────────────────────────────

def _sentence_split(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZĞÜŞİÖÇ0-9])', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text: str, tokenizer, max_tokens: int = 400, overlap_tokens: int = 50) -> list[str]:
    sentences = _sentence_split(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens: int = 0

    for sent in sentences:
        if hasattr(tokenizer, "tokenize"):
            sent_tokens = len(tokenizer.tokenize(sent))
        else:
            sent_tokens = len(tokenizer.encode(sent))

        if current_tokens + sent_tokens > max_tokens and current_sentences:
            chunks.append(" ".join(current_sentences))
            # Calculate overlap
            overlap_accum = 0
            overlap_sents: list[str] = []
            for s in reversed(current_sentences):
                s_tok = len(tokenizer.tokenize(s)) if hasattr(tokenizer, "tokenize") else len(tokenizer.encode(s))
                if overlap_accum + s_tok <= overlap_tokens:
                    overlap_sents.insert(0, s)
                    overlap_accum += s_tok
                else:
                    break
            current_sentences = overlap_sents
            current_tokens = overlap_accum

        current_sentences.append(sent)
        current_tokens += sent_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks

def _recursive_split(text: str, max_len: int, separators: list) -> list:
    if not text.strip():
        return []
    if len(text) <= max_len:
        return [text]
    if not separators:
        return [text[i:i + max_len] for i in range(0, len(text), max_len)]
    sep = separators[0]
    rest = separators[1:]
    parts = text.split(sep)
    if len(parts) == 1:
        return _recursive_split(text, max_len, rest)
    chunks = []
    current = ""
    for part in parts:
        candidate = (current + sep + part).strip() if current else part.strip()
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part.strip()) > max_len:
                sub = _recursive_split(part.strip(), max_len, rest)
                if sub:
                    chunks.extend(sub[:-1])
                    current = sub[-1]
                else:
                    current = ""
            else:
                current = part.strip()
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]

def _chunk_text(text: str, max_len: int = 600, min_len: int = 60, overlap: int = 120) -> list[str]:
    if not text or not text.strip(): return []
    text = _normalize_turkish_chars(text)
    caption_re = re.compile(r"^(Tablo|Grafik|Şekil|Figür|Diyagram|Çizelge|Figure|Table|Fig\.?|Chart|Graf|\[Grafik)\s*\.?\s*\d*[:\s.]", re.IGNORECASE | re.MULTILINE)
    lines = text.splitlines()
    segments, buf_lines, i = [], [], 0
    while i < len(lines):
        line = lines[i]
        if caption_re.match(line.strip()):
            if buf_lines: segments.append("\n".join(buf_lines)); buf_lines = []
            caption_block = [line]
            i += 1
            is_graphic = line.strip().startswith("[Grafik") or line.strip().startswith("Grafik") or line.strip().startswith("Şekil")
            if is_graphic:
                while i < len(lines):
                    l_str = lines[i].strip()
                    if (l_str.startswith("[Grafik") and l_str != line.strip()) or l_str.startswith("Tablo"):
                        break
                    caption_block.append(lines[i])
                    if not l_str and i + 1 < len(lines) and not lines[i + 1].strip():
                        i += 1
                        break
                    i += 1
            else:
                while i < len(lines) and ("|" in lines[i] or not lines[i].strip()):
                    caption_block.append(lines[i]); i += 1
            segments.append("\n".join(caption_block))
            continue
        buf_lines.append(line); i += 1
    if buf_lines: segments.append("\n".join(buf_lines))
    separators = ["\n\n", ".\n", ". ", "!\n", "! ", "?\n", "? ", ";\n", "; ", "\n", " "]
    raw = []
    for seg in segments:
        if not seg.strip(): continue
        # Detect table or graphic blocks to keep them intact
        is_table_or_graphic = (
            (sum(1 for l in seg.splitlines() if "|" in l and ":" in l) >= 2) or
            (sum(1 for l in seg.splitlines() if l.strip().startswith("|") and l.strip().endswith("|")) >= 2) or
            ("[Grafik / Görsel:" in seg)
        )
        if is_table_or_graphic and len(seg) <= max_len * 2.5:
            raw.append(seg.strip())
        else:
            raw.extend(_recursive_split(seg.strip(), max_len, separators))
    merged, buf = [], ""
    for chunk in raw:
        chunk = chunk.strip()
        if not chunk: continue
        is_atomic = ("|" in chunk and "\n|" in chunk) or ("[Grafik / Görsel:" in chunk)
        if is_atomic:
            if buf:
                if len(buf) >= min_len:
                    merged.append(buf)
                buf = ""
            merged.append(chunk)
            continue

        if not buf: buf = chunk
        elif len(buf) + 2 + len(chunk) <= max_len: buf = buf + "\n\n" + chunk
        else:
            if len(buf) >= min_len: merged.append(buf)
            buf = chunk
    if buf and len(buf) >= min_len: merged.append(buf)
    if len(merged) > 1 and len(merged[-1]) < min_len:
        last = merged.pop()
        merged[-1] = merged[-1] + "\n\n" + last
    if not merged: return []
    if overlap <= 0 or len(merged) <= 1: return merged
    result = [merged[0]]
    for i in range(1, len(merged)):
        # Avoid slicing partial overlap if previous chunk is an atomic table/graphic
        prev_is_atomic = ("|" in merged[i - 1] and "\n|" in merged[i - 1]) or ("[Grafik / Görsel:" in merged[i - 1])
        if prev_is_atomic:
            result.append(merged[i])
        else:
            tail = merged[i - 1][-overlap:]
            result.append(tail + "\n\n" + merged[i])
    return result

def index_corpus(
    branch: str,
    embedding_model_key: str = "e5small_base",
    max_tokens: int = 400,
    overlap_tokens: int = 50,
    batch_size: int = 64,
    chroma_path: str | None = None,
) -> int:
    splits_dir = PROJECT_ROOT / "ai" / "data" / "splits" / branch
    corpus_files = list(splits_dir.glob("*.jsonl")) if splits_dir.exists() else []
    if not corpus_files:
        logger.error(f"No corpus files found in splits_dir: {splits_dir}")
        return 0

    from sentence_transformers import SentenceTransformer
    embed_path = _embedding_model_path(embedding_model_key)
    if embed_path is not None and embed_path.exists():
        embed_model = SentenceTransformer(str(embed_path), trust_remote_code=True)
    else:
        fallback = MOSAIC_EMBED_HF_ID if embedding_model_key == "mosaic_embed" else "intfloat/multilingual-e5-small"
        embed_model = SentenceTransformer(fallback, trust_remote_code=True)

    tokenizer = None
    try:
        from tokenizer.hybrid_tokenizer import HybridTokenizer
        tokenizer = HybridTokenizer(bpe_model_path=None, use_zemberek=True)
    except Exception:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

    import chromadb
    clean_b = "branch_01" if "1" in str(branch) else ("branch_02" if "2" in str(branch) else "hq")
    c_path = chroma_path
    if not c_path:
        try:
            cand = Path.home() / "mosaic_data" / "chroma_db" / clean_b
            cand.mkdir(parents=True, exist_ok=True)
            c_path = str(cand)
        except Exception:
            c_path = str(PROJECT_ROOT / "ai" / "data" / "chromadb" / clean_b)
            Path(c_path).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=c_path)
    col_name = "finance_global" if branch == "hq" else f"finance_{branch.replace('branch-', 'branch')}"
    collection = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(f"Indexing branch corpus: {branch} (files={len(corpus_files)})")
    total_chunks = 0
    existing_ids = set(collection.get()["ids"])

    for corpus_file in corpus_files:
        with open(corpus_file, encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]

        for rec_idx, rec in enumerate(records):
            text = rec.get("text") or rec.get("response") or ""
            if not text:
                continue

            chunks = chunk_text(text, tokenizer, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

            batch_chunks = []
            batch_meta = []
            batch_ids = []

            for chunk_idx, chunk in enumerate(chunks):
                doc_id = f"{corpus_file.stem}_{rec_idx}_{chunk_idx}"
                if doc_id in existing_ids:
                    continue
                batch_chunks.append(chunk)
                batch_ids.append(doc_id)
                batch_meta.append({
                    "source": str(corpus_file.stem),
                    "branch": branch,
                    "doc_idx": rec_idx,
                    "title": rec.get("title", "")[:100],
                })

            if not batch_chunks:
                continue

            for start in range(0, len(batch_chunks), batch_size):
                end = start + batch_size
                sub_texts = batch_chunks[start:end]
                sub_ids = batch_ids[start:end]
                sub_meta = batch_meta[start:end]

                prepended_texts = []
                for t in sub_texts:
                    prefix = "passage: "
                    prepended_texts.append(t if t.lower().startswith(prefix) else prefix + t)

                embeddings = embed_model.encode(
                    prepended_texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).tolist()

                collection.add(
                    documents=sub_texts,
                    embeddings=embeddings,
                    ids=sub_ids,
                    metadatas=sub_meta,
                )
                total_chunks += len(sub_texts)

    logger.info(f"Indexing completed for {branch}: {total_chunks} chunks indexed.")
    return total_chunks

# ─── 4. PDF PARSING & TABLE EXTRACTORS ────────────────────────────────────────

def _format_table_markdown(table_data: list) -> str:
    """Formats 2D table data into a clean Github Flavored Markdown (GFM) table."""
    if not table_data:
        return ""
    cleaned = []
    for row in table_data:
        cells = [str(c).replace("\n", " ").strip() if c is not None else "" for c in row]
        if any(cells):
            cleaned.append(cells)
    if not cleaned:
        return ""
    max_cols = max(len(r) for r in cleaned)
    if max_cols == 0:
        return ""
    padded = [r + [""] * (max_cols - len(r)) for r in cleaned]
    header = padded[0]
    sep = [" :--- "] * max_cols
    md_lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(sep) + "|"
    ]
    for row in padded[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(md_lines)

def _format_table_rows(table_data: list) -> str:
    """
    Format table data for Turkish financial RAG:
    Combines key-value representation (for high vector recall & test compatibility)
    and Markdown GFM table format (for rich structured LLM context).
    """
    if not table_data:
        return ""
    cleaned = []
    for row in table_data:
        cells = [str(c).strip() if c is not None else "" for c in row]
        if any(cells):
            cleaned.append(cells)
    if not cleaned:
        return ""

    header = cleaned[0]
    has_header = any(h for h in header)
    lines = []
    data_rows = cleaned[1:] if has_header else cleaned
    if has_header:
        for row in data_rows:
            if not any(row): continue
            pairs = []
            for i, (h, v) in enumerate(zip(header, row)):
                if v:
                    label = h if h else f"Column{i + 1}"
                    pairs.append(f"{label}: {v}")
            if pairs:
                lines.append(" | ".join(pairs))
    else:
        for row in cleaned:
            vals = [v for v in row if v]
            if vals:
                lines.append(" | ".join(vals))

    md = _format_table_markdown(table_data)
    if md:
        return "\n".join(lines) + "\n\n" + md
    return "\n".join(lines)

_GRAPHIC_CAPTION_RE = re.compile(
    r"^(Grafik|Şekil|Diyagram|Çizelge|Figür|Figure|Chart|Fig\.?|Diagram)\s*\.?\s*\d*[:\s.-]\s*([^\n]+)",
    re.IGNORECASE | re.MULTILINE
)

def _extract_pdf_text_pymupdf(content: bytes) -> str:
    try:
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        page_texts: list[str] = []
        for page_num, page in enumerate(doc, 1):
            page_elements: list[tuple[float, float, str, str]] = []  # (y0, x0, type, text)
            table_rects: list[fitz.Rect] = []

            # 1. Extract tables with bounding boxes
            try:
                tabs = page.find_tables()
                for tbl in tabs.tables:
                    bbox = tbl.bbox  # (x0, y0, x1, y1)
                    table_rects.append(fitz.Rect(bbox))
                    data = tbl.extract()
                    fmt = _format_table_rows(data)
                    if fmt.strip():
                        page_elements.append((bbox[1], bbox[0], "table", fmt.strip()))
            except Exception as e:
                logger.debug(f"PyMuPDF table extraction failed (Page {page_num}): {e}")

            # 2. Extract graphic and image regions
            graphic_rects: list[fitz.Rect] = []
            try:
                # Vector drawing areas (charts / plots)
                drawings = page.get_drawings()
                if len(drawings) >= 4:
                    d_rects = [fitz.Rect(d["rect"]) for d in drawings if fitz.Rect(d["rect"]).get_area() > 300]
                    if d_rects:
                        u_rect = d_rects[0]
                        for r in d_rects[1:]:
                            u_rect |= r
                        if 3000 < u_rect.get_area() < (page.rect.get_area() * 0.85):
                            if not any((u_rect & tr).get_area() > 1500 for tr in table_rects):
                                graphic_rects.append(u_rect)
            except Exception as e:
                logger.debug(f"PyMuPDF drawings inspection failed (Page {page_num}): {e}")

            # 3. Extract text blocks, image blocks, and graphic captions
            try:
                text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES)
                for block in text_dict.get("blocks", []):
                    b_type = block.get("type", 0)
                    b_bbox = block.get("bbox", (0, 0, 0, 0))
                    b_rect = fitz.Rect(b_bbox)
                    b_area = b_rect.get_area()

                    # Handle image blocks
                    if b_type == 1:
                        if b_area > 1500 and not any((b_rect & tr).get_area() > 1000 for tr in table_rects):
                            graphic_rects.append(b_rect)
                        continue

                    # Text blocks (type 0): skip if inside table to avoid duplication
                    if table_rects and b_area > 0 and any((b_rect & tr).get_area() / b_area > 0.50 for tr in table_rects):
                        continue

                    block_text = " ".join(span["text"] for line in block.get("lines", []) for span in line.get("spans", [])).strip()
                    if not block_text:
                        continue

                    # Check if block is a chart/graphic caption or metadata
                    is_graphic_caption = False
                    for gr in graphic_rects:
                        expanded_gr = fitz.Rect(gr.x0 - 15, gr.y0 - 45, gr.x1 + 15, gr.y1 + 45)
                        if (b_rect & expanded_gr).get_area() > 0 or expanded_gr.contains(b_rect):
                            if _GRAPHIC_CAPTION_RE.search(block_text):
                                is_graphic_caption = True
                                cap_match = _GRAPHIC_CAPTION_RE.search(block_text)
                                cap_name = cap_match.group(0).strip()
                                page_elements.append((b_bbox[1], b_bbox[0], "graphic", f"[Grafik / Görsel: {cap_name}]\n{block_text}"))
                                break
                    if is_graphic_caption:
                        continue

                    page_elements.append((b_bbox[1], b_bbox[0], "text", block_text))
            except Exception as e:
                logger.debug(f"PyMuPDF text extraction failed (Page {page_num}): {e}")
                t = page.get_text("text")
                if t.strip():
                    page_elements.append((0.0, 0.0, "text", t.strip()))

            # 4. Sort strictly by vertical reading order (y0, then x0)
            page_elements.sort(key=lambda elem: (elem[0], elem[1]))
            ordered_texts = [elem[3] for elem in page_elements if elem[3].strip()]
            combined = "\n\n".join(ordered_texts)
            if combined:
                page_texts.append(_normalize_turkish_chars(combined))
        doc.close()
        result = "\n\n".join(page_texts)
        if result.strip():
            return result
    except Exception as e:
        logger.warning(f"PyMuPDF parser failed: {e}")
    return ""

def _extract_pdf_text_pdfplumber(content: bytes) -> str:
    try:
        import pdfplumber
        page_texts = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                pw, ph = float(page.width), float(page.height)
                sections: list[str] = []
                tables = page.find_tables()
                tables_sorted = sorted(tables, key=lambda t: t.bbox[1])
                prev_bottom = 0.0
                for tbl in tables_sorted:
                    x0, top, x1, bottom = (float(v) for v in tbl.bbox)
                    if top > prev_bottom + 2:
                        try:
                            above = page.crop((0, prev_bottom, pw, top)).extract_text(x_tolerance=3, y_tolerance=3) or ""
                            if above.strip(): sections.append(above.strip())
                        except: pass
                    fmt = _format_table_rows(tbl.extract())
                    if fmt.strip(): sections.append(fmt.strip())
                    prev_bottom = bottom
                if tables_sorted:
                    if prev_bottom < ph - 2:
                        try:
                            below = page.crop((0, prev_bottom, pw, ph)).extract_text(x_tolerance=3, y_tolerance=3) or ""
                            if below.strip(): sections.append(below.strip())
                        except: pass
                else:
                    full = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                    if full.strip(): sections.append(full.strip())
                if sections: page_texts.append("\n\n".join(sections))
        result = "\n\n".join(page_texts)
        if result.strip(): return result
    except Exception as e:
        logger.warning(f"pdfplumber parser failed: {e}")
    return ""

def _extract_pdf_text_pdfminer(content: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(io.BytesIO(content))
        if text and text.strip():
            return text
    except Exception as e:
        logger.warning(f"pdfminer parser failed: {e}")
    return ""

def extract_pdf_content(content: bytes) -> str:
    logger.info("[PDF-Extract] Trying PyMuPDF...")
    result = _extract_pdf_text_pymupdf(content)
    if result.strip():
        return result

    logger.info("[PDF-Extract] Falling back to pdfplumber...")
    result = _extract_pdf_text_pdfplumber(content)
    if result.strip():
        return result

    logger.info("[PDF-Extract] Falling back to pdfminer...")
    return _extract_pdf_text_pdfminer(content)

# ─── 5. PII MASKING & GDPR COMPLIANCE ─────────────────────────────────────────

# Kişisel veri maskeleme v3 (2026-10-14): ayraç toleranslı desenler, sağlama doğrulaması (TC, IBAN mod-97, Luhn),
# ad sözlüğü + unvan/alan etiketi/bağlam ipuçları. Önceki sürümler: backups/ai_engine_v1_*.py, backups/ai_engine_v2_*.py
# Yaygın Türkçe ad sözlüğü (TÜİK en sık verilen adlar ve genel kullanım; test listelerinden bağımsız derlenmiştir)
_PII_FIRST_NAMES = set("""
ahmet mehmet mustafa ali hüseyin hasan ibrahim ismail osman yusuf murat ömer ramazan halil süleyman abdullah mahmut
recep salih fatih kadir emre hakan adem kemal yaşar bayram mustafa orhan cem can burak serkan erkan volkan onur kaan
kerem barış uğur tolga yasin enes furkan berk eren efe arda emir mert umut caner deniz ozan sinan tuncay tayfun
cengiz levent selim serdar sedat erdem gökhan alper koray tarık metin necati nihat ercan engin oğuz oğuzhan batuhan
yiğit kağan alperen emirhan berkay doruk görkem ilker kubilay erhan halit şükrü şahin taner turgut zafer
fatma ayşe emine hatice zeynep elif meryem şerife zehra sultan hanife merve özlem esra büşra kübra tuğba gizem
derya selin pınar ebru sevgi gül nur irem ece ceren melis buse damla dilek aslı aysel ayten nermin songül sibel
seda serap gülay gülşen hülya filiz yasemin yeliz canan esin şeyma rabia sena beyza nisa eylül zeynep defne ela
azra ecrin asya ilayda melike sude aleyna tuba hande gamze burcu didem özge çağla başak berna nazlı
""".split())
_PII_NOT_SURNAME = {"Bayramı", "Bayram", "Pasifik", "Sultan", "Han", "Paşa", "Camii", "Caddesi", "Sokak", "Bulvarı",
                    "Mahallesi", "Üniversitesi", "Hastanesi", "Köprüsü", "Holding", "Bank", "Bankası", "Grubu"}

_PII_NOT_SURNAME_UP = {"TL", "USD", "EUR", "TCMB", "BDDK", "SPK", "KAP", "BIST", "KKDF", "BSMV", "IBAN", "SWIFT", "EFT", "FAST", "ATM", "POS",
                       "A.Ş", "AŞ", "LTD", "ŞTİ", "TÜFE", "ÜFE", "GSYH", "KDV", "ÖTV", "SGK", "MASAK", "KVKK", "TBB", "BES", "TEFAS", "NAV", "CEO", "CFO"}

def _pii_lower(w):
    return w.replace("İ", "i").replace("I", "ı").lower()

# Türkçe banka belgelerinde kişi adını önceleyen alan etiketleri ve bağlam ipuçları (v3)
_PII_NAME_LABELS = (r"(?:Ad[ıi]?\s*[-/]?\s*Soyad[ıi]?|Ad\s+Soyad|M[üu]şteri(?:\s+Ad[ıi])?|Hesap\s+Sahibi|Kart\s+Sahibi|Al[ıi]c[ıi]|G[öo]nderen|"
                    r"Kefil|Borçlu|Vekil|Yetkili|[İI]lgili\s+Ki[şs]i|Talep\s+Sahibi|Ba[şs]vuru\s+Sahibi|Ad[ıi])")
_PII_HONORIFIC = r"(?:Sayın|Sn\.|Bay|Bayan|Sayin)"
_PII_SEP = r"[\s.\-]?"

def _pii_digits(s):
    return re.sub(r"\D", "", s)

def _pii_tc_ok(d):
    if len(d) != 11 or d[0] == "0": return False
    n = list(map(int, d))
    return n[9] == ((sum(n[0:9:2]) * 7) - sum(n[1:8:2])) % 10 and n[10] == sum(n[:10]) % 10

def _pii_luhn_ok(d):
    if not 13 <= len(d) <= 19: return False
    s = 0
    for i, ch in enumerate(reversed(d)):
        x = int(ch)
        if i % 2 == 1:
            x *= 2
            if x > 9: x -= 9
        s += x
    return s % 10 == 0

def _pii_iban_ok(raw):
    s = re.sub(r"[\s\-]", "", raw).upper()
    if not re.fullmatch(r"TR\d{24}", s): return False
    r = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in r)) % 97 == 1

def _filter_pii(text: str, mappings_dict: dict = None) -> str:
    """Kişisel veri maskeleme v3.1: ayraç toleranslı desenler + sağlama doğrulaması + ad sözlüğü + ad yayılımı."""
    original_text = text
    global_mappings = {} if mappings_dict is None else mappings_dict
    spans = []  # (start, end, category)

    def add(m, cat, group=0):
        s, e = m.span(group)
        if not any(s < e2 and s2 < e for s2, e2, _ in spans):
            spans.append((s, e, cat))

    # 1) IBAN (TR + 24 hane, boşluk/tire toleranslı, büyük-küçük harf duyarsız, mod-97 doğrulamalı)
    for m in re.finditer(r"(?i)\bTR[\s\-]?\d{2}(?:[\s\-]?\d){22}\b", text):
        if _pii_iban_ok(m.group()): add(m, "IBAN")
    # 2) E-posta
    for m in re.finditer(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", text):
        add(m, "EMAIL")
    # 3) Kart numarası: gerçek kart düzeni (4-4-4-4 veya 4-6-5, tutarlı ayraç), Luhn doğrulamalı
    for m in re.finditer(r"(?<!\d)(?<!\d[.,])(?:\d{4}([\s.\-]?)\d{4}\1\d{4}\1\d{4}|\d{4}([\s\-]?)\d{6}\2\d{5})(?!\d|[.,]\d)", text):
        d = _pii_digits(m.group())
        if _pii_luhn_ok(d) and not _pii_tc_ok(d): add(m, "CREDIT_CARD")
    # 4) TC kimlik no (11 hane, ayraç toleranslı, sağlama doğrulamalı)
    for m in re.finditer(r"(?<!\d)(?<!\d[.,])[1-9](?:[\s.\-]?\d){10}(?!\d|[.,]\d)", text):
        if _pii_tc_ok(_pii_digits(m.group())): add(m, "TC_NO")
    # 5) Cep telefonu (+90 / 0 önekli veya öneksiz 5xx; boşluk, nokta, tire, parantez toleranslı)
    for m in re.finditer(r"(?<![\d])(?:\+?90[\s.\-]?)?(?:0[\s.\-]?)?\(?5\d{2}\)?[\s.\-]?\d{3}[\s.\-]?\d{2}[\s.\-]?\d{2}(?![\d])", text):
        d = _pii_digits(m.group())
        if len(d) in (10, 11, 12) and d[-10] == "5": add(m, "PHONE")
    # 6) Ad-soyad: (a) unvan + 1-3 büyük harfli sözcük, (b) sözlükteki bir ad + büyük harfli/tamamı büyük soyad,
    #    (c) TAMAMI BÜYÜK soyad + sözlükteki ad
    W = r"[A-ZÇĞİÖŞÜ][a-zçğıöşü]+"; U = r"[A-ZÇĞİÖŞÜ]{2,}"
    for m in re.finditer(rf"\b{_PII_HONORIFIC}\s+((?:{W}|{U})(?:\s+(?:{W}|{U})){{0,2}})", text):
        add(m, "NAME", 1)
    # (d) alan etiketi + ":" + 1-3 büyük harfli sözcük (aynı satır); Kefil/Borçlu/Vekil için ":" isteğe bağlı  [v3]
    SP = r"[ \t]+"
    for m in re.finditer(rf"(?i:\b{_PII_NAME_LABELS})[ \t]*[:：][ \t]*((?:{W}|{U})(?:{SP}(?:{W}|{U})){{0,2}})", text):
        add(m, "NAME", 1)
    for m in re.finditer(rf"\b(?:Kefil|Borçlu|Vekil|Sayın|Sn\.)[ \t]+((?:{W}|{U})(?:{SP}(?:{W}|{U})){{0,2}})", text):
        add(m, "NAME", 1)
    # (e) 2-3 büyük harfli sözcük (aynı satır) + isimli/adlı/adındaki  [v3]
    for m in re.finditer(rf"\b((?:{W}|{U})(?:{SP}(?:{W}|{U})){{1,2}})(?:'\w+)?[ \t]+(?:isimli|adlı|adındaki)\b", text):
        add(m, "NAME", 1)
    # (f) sözlükteki ad + TAMAMI BÜYÜK soyad (aynı satır; Türkçe belge geleneği)  [v3]
    for m in re.finditer(rf"\b({W})[ \t]+({U})\b", text):
        if _pii_lower(m.group(1)) in _PII_FIRST_NAMES and m.group(2) not in _PII_NOT_SURNAME_UP: add(m, "NAME")
    # (b) sözlükteki ad + 0-1 ara ad + soyad: her büyük harfli sözcük konumundan ayrı ayrı denenir (örtüşen eşleşme)
    for m in re.finditer(rf"(?=\b({W})((?:\s+{W})?)\s+({W}|{U})\b)", text):
        first, mid, last = m.group(1), m.group(2), m.group(3)
        if _pii_lower(first) in _PII_FIRST_NAMES and last not in _PII_NOT_SURNAME and mid.strip() not in _PII_NOT_SURNAME:
            s0 = m.start(1); e0 = m.end(3); spans_add = (s0, e0)
            if not any(s0 < e2 and s2 < e0 for s2, e2, _ in spans): spans.append((s0, e0, "NAME"))
    # (c) TAMAMI BÜYÜK soyad + sözlükteki ad
    for m in re.finditer(rf"\b({U})\s+({W})\b", text):
        if _pii_lower(m.group(2)) in _PII_FIRST_NAMES: add(m, "NAME")
    # (g) yayılım [v3.1]: yukarıdaki kurallardan biriyle tespit edilen çok sözcüklü ad-soyad, metnin başka
    #     yerlerinde (etiketsiz, ekli biçimler dahil: "Leyla Karabulut'un") geçtiğinde de maskelenir.
    _found = {text[s:e] for s, e, c in spans if c == "NAME" and len(text[s:e].split()) >= 2}
    for _nm in sorted(_found, key=len, reverse=True):
        for m in re.finditer(rf"(?<!\w){re.escape(_nm)}(?!\w)", text):
            add(m, "NAME")

    if not spans:
        return text

    def placeholder(val, cat):
        for p, orig in global_mappings.items():
            if orig == val: return p
        n = sum(1 for p in global_mappings if p.startswith(f"[{cat}_")) + 1
        p = f"[{cat}_{n}]"; global_mappings[p] = val
        return p

    out = []; last = 0
    for s, e, cat in sorted(spans):
        out.append(text[last:s]); out.append(placeholder(text[s:e], cat)); last = e
    out.append(text[last:])
    text = "".join(out)
    if os.getenv("ENABLE_PII_LOGGING", "false").lower() == "true":
        logger.info(f"[PII Masking] Original: '{original_text}'")
    logger.info(f"[PII Masking] Masking complete. Items changed: {len(spans)}")
    return text

async def demask_stream(token_generator: AsyncGenerator[str, None], mappings: dict) -> AsyncGenerator[str, None]:
    if not mappings:
        async for token in token_generator:
            yield token
        return

    buffer = ""
    async for token in token_generator:
        buffer += token

        while True:
            idx = buffer.find("[")
            if idx == -1:
                yield buffer
                buffer = ""
                break
            else:
                if idx > 0:
                    yield buffer[:idx]
                    buffer = buffer[idx:]

                end_idx = buffer.find("]")
                if end_idx != -1:
                    placeholder = buffer[:end_idx + 1]
                    if placeholder in mappings:
                        yield mappings[placeholder]
                    else:
                        yield placeholder
                    buffer = buffer[end_idx + 1:]
                else:
                    if len(buffer) > 100:
                        yield buffer[0]
                        buffer = buffer[1:]
                    else:
                        break
    if buffer:
        yield buffer

def _strip_cjk(token: str) -> str:
    cjk_char_re = re.compile(
        r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf]'
    )
    if not token:
        return ""
    cjk_chars = cjk_char_re.findall(token)
    if len(cjk_chars) / len(token) > 0.40:
        return ""
    return cjk_char_re.sub("", token)

# ─── 6. ROUTER OVERRIDES & ORCHESTRATION ──────────────────────────────────────

ROUTER_CONFIDENCE_THRESHOLD = 0.55
_COMPLEX_KEYWORDS = [
    "dcf", "indirgenmiş nakit akımı", "npv", "net bugünkü değer",
    "vadeye kadar getiri", "durasyon", "convexity", "türev araç",
    "swap", "opsiyon", "hedging", "risk yönetimi", "stres testi",
    "baz puan", "libor", "euribor", "tüfe", "enflasyon düzeltmesi"
]

def _apply_router_overrides(query: str, result: dict) -> dict:
    label = result.get("label", 1)
    confidence = result.get("confidence", 1.0)
    query_low = query.lower()

    # Rule 1: Complex financial terms force complex route
    has_complex_kw = any(kw in query_low for kw in _COMPLEX_KEYWORDS)
    if has_complex_kw and label < 2:
        logger.info(f"Router override: label {label}->{2} (complex keyword match)")
        result = {**result, "label": 2, "label_name": "complex", "override_reason": "complex_keyword"}

    # Rule 2: Low confidence upgrade
    elif confidence < ROUTER_CONFIDENCE_THRESHOLD:
        if label == 0:
            logger.info(f"Router override: chitchat->simple (conf={confidence:.3f} < {ROUTER_CONFIDENCE_THRESHOLD})")
            result = {**result, "label": 1, "label_name": "simple", "override_reason": f"low_confidence({confidence:.3f})"}
        elif label == 1:
            logger.info(f"Router override: simple->complex (conf={confidence:.3f} < {ROUTER_CONFIDENCE_THRESHOLD})")
            result = {**result, "label": 2, "label_name": "complex", "override_reason": f"low_confidence({confidence:.3f})"}

    return result

_BRANCH_KEYWORDS = {
    1: ["şube 1", "şube1", "branch 1", "branch1", "branch_1", "şube_1", "birinci şube", "ilk şube"],
    2: ["şube 2", "şube2", "branch 2", "branch2", "branch_2", "şube_2", "ikinci şube"],
}

def detect_branch_from_query(query: str, default_branch_id: int = 0) -> Tuple[int, bool]:
    q_low = query.lower()
    for branch_id, kws in _BRANCH_KEYWORDS.items():
        if any(kw in q_low for kw in kws):
            return branch_id, True
    return default_branch_id, False

PII_FIELDS = {"customer_id", "tc_no", "account_no", "musteri_adi", "musteri_id"}

def strip_pii_from_chunks(chunks: list[dict]) -> list[dict]:
    for chunk in chunks:
        if "metadata" in chunk:
            removed = {k: v for k, v in chunk["metadata"].items() if k in PII_FIELDS}
            if removed:
                logger.info(f"[KVKK] Metadata stripped — branch: {chunk.get('branch', '?')}, fields: {list(removed.keys())}")
            chunk["metadata"] = {k: v for k, v in chunk["metadata"].items() if k not in PII_FIELDS}
    return chunks

# ─── 7. CENTRALIZED CROSS-ENCODER RERANKER ────────────────────────────────────

_RERANKER_INSTANCE = None

def get_reranker(model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        from sentence_transformers import CrossEncoder
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            logger.info(f"[Reranker] Loading centralized CrossEncoder: {model_name} on {_device}...")
            _RERANKER_INSTANCE = CrossEncoder(model_name, device=_device)
            logger.success("[Reranker] CrossEncoder loaded successfully.")
        except Exception as e:
            logger.warning(f"[Reranker] Failed to load CrossEncoder ({e}), will fallback to score sorting.")
            _RERANKER_INSTANCE = None
    return _RERANKER_INSTANCE

def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
    min_score: float = 0.20
) -> list[dict]:
    """
    HQ Centralized Cross-Encoder Reranker:
    Reranks merged candidate chunks from all branches using a single unified cross-attention model.
    Transforms raw logits to calibrated sigmoid relevance scores (0.0 to 1.0).
    Filters out chunks below min_score to prevent irrelevant topic bleed.
    """
    if not chunks:
        return []

    texts = []
    valid_chunks = []
    for c in chunks:
        t = (c.get("text") or c.get("content") or "").strip()
        if t:
            texts.append(t)
            valid_chunks.append(c)

    if not valid_chunks:
        return []

    reranker = get_reranker()
    if reranker is None:
        sorted_chunks = sorted(valid_chunks, key=lambda c: float(c.get("score", 0.0)), reverse=True)
        return [c for c in sorted_chunks if float(c.get("score", 0.0)) >= min_score][:top_k]

    pairs = [[query, txt] for txt in texts]

    try:
        raw_scores = reranker.predict(pairs, show_progress_bar=False).tolist()
        calibrated = []
        for s in raw_scores:
            val = float(s)
            sig = 1.0 / (1.0 + math.exp(-max(min(val, 20.0), -20.0)))
            calibrated.append(round(sig, 4))
    except Exception as e:
        logger.error(f"[Reranker] Scoring error ({e}), falling back to existing scores.")
        sorted_chunks = sorted(valid_chunks, key=lambda c: float(c.get("score", 0.0)), reverse=True)
        return [c for c in sorted_chunks if float(c.get("score", 0.0)) >= min_score][:top_k]

    reranked = []
    for chunk, new_score in zip(valid_chunks, calibrated):
        c_copy = dict(chunk)
        c_copy["score"] = new_score
        reranked.append(c_copy)

    reranked.sort(key=lambda c: c["score"], reverse=True)
    filtered = [c for c in reranked if c["score"] >= min_score]
    return filtered[:top_k]

# ─── 8. PROMPT BUILDING & LLM HELPERS ─────────────────────────────────────────

def build_system_prompt(route: str) -> str:
    if route == "chitchat":
        return (
            "You are an expert Turkish banking assistant. ALWAYS respond in natural, polite and professional Turkish.\n"
            "Kullanıcıyla kurumsal, kibar ve doğal bir dille sohbet edin. "
            "Genel sohbet, teşekkür ve selamlaşmalarda KESİNLİKLE tablo veya madde KULLANMAYIN; doğrudan, samimi ve akıcı yanıt verin. "
            "Kullanıcı bankacılık işlemleri veya finansal analiz konuları sorduğunda yardımcı olabileceğinizi belirtin."
        )

    return (
        "You are an expert Turkish banking and financial AI advisor. ALWAYS respond in clear, professional Turkish.\n\n"
        "KESİN DOĞRULUK VE BAĞLAM KURALLARI (SIFIR HALÜSİNASYON):\n"
        "1. YALNIZCA sağlanan Bağlam Belgelerindeki doğrulanmış gerçekleri, oranları ve verileri kullanın.\n"
        "2. Belgelerde açıkça yer almayan hiçbir faiz oranını, LTV oranını, vadeyi, teminat şartını veya şube kuralını ASLA tahmin etmeyin, varsaymayın veya uydurmayın.\n"
        "3. Belgede açıkça bulunmayan bir bilgi sorulduğunda veya belge yetersiz kaldığında, spekülasyon yapmayın; 'Belgelerde bu hususa dair bir kayıt bulunmamaktadır' şeklinde dürüstçe belirtin.\n\n"
        "BİÇİMLENDİRME VE YANIT STANDARTLARI:\n"
        "1. DOĞAL VE AMACA UYGUN DİL:\n"
        "   - Kısa sorular ve tekil tanımlar için (örn. '... kimdir?', '... nedir?') doğrudan, sade ve akıcı bir açıklama paragrafı ile yanıt verin; gereksiz yere tablo veya madde KULLANMAYIN.\n"
        "   - YALNIZCA birden fazla şart, başvuru adımı veya kriter sıralanması gerektiğinde maddeler (- ) kullanın.\n"
        "2. KESİN TABLO KURALI (GFM Markdown Formatı):\n"
        "   - Karşılaştırmalı veriler, oranlar (faiz, LTV, BES devlet katkısı), vadeler, limitler veya tutar kademeleri sunarken ya da kullanıcı açıkça 'tablo' istediğinde MUTLAKA geçerli bir Markdown Tablosu oluşturun.\n"
        "   - Tabloları ASLA boşluklarla veya düz metin hizalamasıyla yazmayın! Her sütunu dikey çizgi (|) ile ayırın ve başlık satırının hemen altına mutlaka (| :--- | :--- |) ayırıcı satırını ekleyin.\n"
        "   - Tablo Formatı Örneği:\n\n"
        "| Parametre / Kriter | Değer / Oran | Açıklama |\n"
        "| :--- | :--- | :--- |\n"
        "| Örnek Parametre | %25 | Açıklayıcı bilgi |\n\n"
        "3. OKUNABİLİRLİK VE TAMAMLILIK:\n"
        "   - Uzun analizlerde ara başlıklar (###) ve kritik finansal terimlerde kalın (**bold**) yazı kullanın.\n"
        "   - Cümlelerinizi ve tablolarınızı asla yarım bırakmayın; mantıksal bir sonuçla eksiksiz tamamlayın."
    )

def build_user_prompt(
    query: str,
    context_chunks: list[dict],
    route: str,
    history: list[dict] = None,
    max_turns: int = 3,
    mappings: dict = None
) -> str:
    def format_history(history_list, limit):
        if not history_list:
            return ""
        lines = []
        for msg in history_list[-limit:]:
            role = "Kullanıcı" if msg.get("role") == "user" else "Asistan"
            lines.append(f"{role}: {msg.get('content')}")
        return "\n".join(lines)

    history_ctx = format_history(history, max_turns)

    if route == "chitchat":
        return f"Geçmiş Sohbet:\n{history_ctx}\n\nKullanıcı: {query}\nAsistan:"

    # 1. Yalnızca geçerli eşik üzerindeki chunk'ları al (alakasız veya eşik altı olanları zorla tutma!)
    selected_chunks = [c for c in context_chunks if float(c.get("score", 0.0)) >= 0.20][:6]

    # Bağlam bütçesi: llama-server 4096 tokenlik bağlamla çalışır (Tez Bölüm 4.7.3); uzun parçalar istemi taşırmasın.
    per_chunk = int(os.getenv("RAG_CHUNK_CHAR_LIMIT", "1400"))
    budget = int(os.getenv("RAG_CONTEXT_CHAR_BUDGET", "5600"))
    context_parts = []
    for chunk in selected_chunks:
        snippet = (chunk.get("text") or chunk.get("content") or "").strip()
        if snippet:
            snippet = _filter_pii(snippet, mappings)
            if len(snippet) > per_chunk:
                snippet = snippet[:per_chunk].rsplit(" ", 1)[0] + " …"
            if context_parts and budget - len(snippet) < 0:
                break
            budget -= len(snippet)
            label = f"[Belge: {chunk.get('source', 'bilinmiyor')} | Şube: {chunk.get('branch', 'bilinmiyor')} | Skor: {chunk.get('score', 0.0):.3f}]"
            context_parts.append(f"{label}\n{snippet}")

    has_docs = bool(context_parts)
    context_str = "\n\n".join(context_parts) if has_docs else "İlgili doğrudan belge bulunamadı."

    history_section = (
        f"Önceki Sohbet Geçmişi (Yalnızca bağlam devamı içindir; mevcut soru farklı bir konuysa bu geçmişi mevcut soruyla KARIŞTIRMAYIN):\n{history_ctx}\n\n"
        if history_ctx else ""
    )

    if has_docs:
        if route == "simple":
            doc_inst = (
                "YALNIZCA yukarıdaki Bağlam Belgelerini esas alarak soruyu doğrudan, net ve doğru bir Türkçe ile yanıtlayın. "
                "Belgelerde açıkça yazmayan veya belirtilmeyen hiçbir kural, oran, şart veya bilgiyi ASLA uydurmayın, varsaymayın veya tahmin etmeyin. "
                "Geçmiş sohbetteki farklı konuları mevcut soruya karıştırmayın. "
                "Kısa veya tekil sorular için doğrudan açıklama yapın; her yanıta zoraki tablo veya madde eklemeyin. "
                "Yalnızca birden fazla şart/kriter varsa maddeleme yapın. "
                "Belgede açıkça geçen oranlar, vadeler veya karşılaştırmalı veriler varsa ya da kullanıcı tablo istediyse, MUTLAKA standart Markdown Tablosu oluşturun. "
                "Eğer kullanıcının sorusunun cevabı belgede yoksa veya eksikse, bunu dürüstçe 'Belgelerde bu hususa dair bilgi yer almamaktadır' şeklinde net olarak belirtin."
            )
        else:  # complex
            doc_inst = (
                "Yukarıdaki doğrulanmış belgeleri ve finansal verileri derinlemesine analiz ederek profesyonel ve yapılandırılmış bir finansal değerlendirme sunun. "
                "YALNIZCA belgelerde ve resmi finansal mevzuatta yer alan kesin verilere dayanın; hayali oran veya varsayımsal şart türetmeyin. "
                "Geçmiş sohbetteki farklı konuları mevcut soruya karıştırmayın. "
                "Analizin gerektirdiği ölçüde ara başlıklar (###) ve önemli noktalar için maddeler (- ) kullanabilirsiniz. "
                "Belgedeki sayısal göstergeler, oranlar, vadeler veya limitler için MUTLAKA standart Markdown Tablosu oluşturun. "
                "Analizinizi tam bir sonuç ve değerlendirme ile eksiksiz olarak bitirin."
            )
        return (
            f"Bağlam Belgeleri:\n{context_str}\n\n"
            f"{history_section}"
            f"Kullanıcı Sorusu / Analiz İstemi: {query}\n\n"
            f"Yönerge: {doc_inst}\nAsistan:"
        )
    else:
        # Doğrudan doğrulanmış belge bulunamadığında
        query_lower = query.lower()
        is_branch_specific = any(kw in query_lower for kw in [
            "şube 1", "şube 2", "şube1", "şube2", "bu şube", "şubemiz",
            "kılavuz", "prosedür", "yönerge", "şartname", "rehin", "kasko",
            "azami kredi", "kredi oranı", "ltv", "faiz baremi", "faiz oranı",
            "erken ödeme", "tahsisat yetki", "onay yetki", "çks", "elüs", "gelir beyanı",
            "şube müdürü", "kredi şefi", "baş denetçi", "denetçi"
        ])

        if is_branch_specific:
            doc_inst = (
                "BİLGİ: Sistem veritabanında bu konuya ilişkin doğrulanmış bir şube prosedür/kılavuz belgesi henüz bulunmamaktadır.\n"
                "1. KESİNLİKLE herhangi bir şubeye aitmiş gibi hayali/varsayımsal faiz oranı, LTV yüzdesi, vade, yetkili adı veya iç kural UYDURMAYIN.\n"
                "2. Şubeye ait resmi prosedür kılavuzunun sistemde henüz kayıtlı olmadığını nazikçe ve profesyonelce belirtin.\n"
                "3. Soruya ilişkin genel BDDK mevzuatını, sektörel kuralları veya bankacılık prensiplerini açıklayın.\n"
                "4. Şubeye özel kesin oran ve koşullar için kullanıcının ilgili prosedür belgesini sisteme yükleyebileceğini hatırlatın."
            )
        else:
            doc_inst = (
                "Sistemde bu konuya ilişkin özel bir kurum içi doküman bulunmamaktadır. "
                "Kullanıcının sorusunu genel finans ve bankacılık alan bilginizi, sektörel mevzuatı ve formülleri kullanarak "
                "açıklayıcı, profesyonel, yardımsever ve akıcı bir Türkçe ile yanıtlayın. "
                "Herhangi bir şubeye aitmiş gibi özel iç politika uydurmayın."
            )

        return (
            f"Bağlam Belgeleri ve Finansal Veriler:\n{context_str}\n\n"
            f"{history_section}"
            f"Kullanıcı Sorusu: {query}\n\n"
            f"Yönerge: {doc_inst}\nAsistan:"
        )

def _build_llm_prompt(model_name: str, system: str, user: str, route: str = "simple") -> tuple:
    is_qwen = "qwen" in model_name.lower()
    is_gemma = "gemma" in model_name.lower()

    if route == "chitchat": temperature = 0.4
    elif route == "complex": temperature = 0.1
    else: temperature = 0.1

    if is_qwen:
        full_prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user} /no_think<|im_end|>\n<|im_start|>assistant\n"
        stop_tokens = ["<|im_end|>", "<|im_start|>", "<end_of_turn>"]
        n_predict = 2048
    elif is_gemma:
        full_prompt = f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n"
        stop_tokens = ["<end_of_turn>", "<start_of_turn>"]
        n_predict = 2048
    else:
        full_prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        stop_tokens = ["<|im_end|>"]
        n_predict = 2048
    return full_prompt, stop_tokens, n_predict, temperature

def _clean_llm_output(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = text.replace("<think>", "").replace("</think>", "")
    return _strip_cjk(text).strip()

def _clean_llm_token(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = text.replace("<think>", "").replace("</think>", "")
    return _strip_cjk(text)
