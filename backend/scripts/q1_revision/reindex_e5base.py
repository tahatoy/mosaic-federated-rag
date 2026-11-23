"""
Şube (veya HQ) ChromaDB koleksiyonunu taban multilingual-e5-small ile yeniden indeksler.

Kaynak koleksiyondaki parçalar (id, metin, üst veri) olduğu gibi okunur; servisteki ekleme kuralıyla ("passage: " öneki)
taban e5-small vektörleri hesaplanır ve HNSW ayarları yükseltilmiş yeni bir koleksiyona yazılır. Kaynak koleksiyon
değiştirilmez (geri dönüş için saklanır). Metin şubeden çıkmaz; işlem şubenin kendi makinesinde yapılır.
Kullanım: python reindex_e5base.py --src data/chromadb/branch_01 --dst data/chromadb/branch_01_e5 --name finance_branch1
"""
import argparse
import json
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--ef", type=int, default=200); ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    a = ap.parse_args()
    import chromadb
    import torch
    from sentence_transformers import SentenceTransformer
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    src = chromadb.PersistentClient(path=a.src).get_collection(a.name)
    n = src.count()
    ids, docs, metas = [], [], []
    for off in range(0, n, 2000):
        r = src.get(include=["documents", "metadatas"], limit=2000, offset=off)
        ids += r["ids"]; docs += r["documents"]; metas += [m or {} for m in r["metadatas"]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(a.model, device=dev)
    dst_client = chromadb.PersistentClient(path=a.dst)
    try:
        dst_client.delete_collection(a.name)
    except Exception:
        pass
    dst = dst_client.create_collection(a.name, metadata={"hnsw:space": "cosine", "hnsw:search_ef": a.ef, "hnsw:construction_ef": a.ef, "hnsw:M": 32})
    B = 256
    for i in range(0, len(docs), B):
        sub = docs[i:i + B]
        emb = m.encode([t if t.lower().startswith("passage: ") else "passage: " + t for t in sub], batch_size=32,
                       normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).tolist()
        dst.add(ids=ids[i:i + B], documents=sub, embeddings=emb, metadatas=[x if x else {"source": "unknown"} for x in metas[i:i + B]])
        if (i // B) % 10 == 0:
            print(f"{i + len(sub)}/{len(docs)}", flush=True)
    print(json.dumps({"src": a.src, "dst": a.dst, "name": a.name, "n_src": n, "n_dst": dst.count(), "sec": round(time.time() - t0, 1),
                      "device": dev, "model": a.model, "hnsw_ef": a.ef}), flush=True)


if __name__ == "__main__":
    main()
