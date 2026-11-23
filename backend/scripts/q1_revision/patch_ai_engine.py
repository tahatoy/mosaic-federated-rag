"""Şube kopyasındaki backend/ai_engine.py'ye taban e5-small seçeneği ve HNSW ayarı ekler (idempotent)."""
import sys
p = sys.argv[1]; s = open(p, encoding="utf-8").read()
if "e5small_base" in s:
    print("zaten yamalı"); sys.exit(0)
old = '''        if embed_path.exists():
            try:
                self.embed_model = SentenceTransformer(str(embed_path), device=_device, trust_remote_code=True)'''
new = '''        if self.embedding_model_key == "e5small_base":
            logger.info("[EmbeddingEngine] Taban multilingual-e5-small kullanılıyor (EMBEDDING_MODEL_KEY=e5small_base)")
            self.embed_model = SentenceTransformer("intfloat/multilingual-e5-small", device=_device, trust_remote_code=True)
        elif embed_path.exists():
            try:
                self.embed_model = SentenceTransformer(str(embed_path), device=_device, trust_remote_code=True)'''
assert s.count(old) == 1, "model bloğu bulunamadı"
s = s.replace(old, new)
old2 = '''            self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )'''
new2 = '''            self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
            except Exception:
                _m = {"hnsw:space": "cosine"}
                if os.getenv("RAG_HNSW_EF"):
                    _ef = int(os.getenv("RAG_HNSW_EF")); _m.update({"hnsw:search_ef": _ef, "hnsw:construction_ef": _ef, "hnsw:M": 32})
                self.collection = self.chroma_client.create_collection(name=self.collection_name, metadata=_m)'''
assert s.count(old2) == 1, "koleksiyon bloğu bulunamadı"
s = s.replace(old2, new2)
open(p, "w", encoding="utf-8").write(s); print("yamalandı")
