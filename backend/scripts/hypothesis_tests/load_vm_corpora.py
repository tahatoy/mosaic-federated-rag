"""Şube korpuslarını (ai/data/splits/branchN/corpus.jsonl) ilgili şube VM'inin RAG servisine
/add_chunks ile yükler. Embedding VM üzerinde hesaplanır (veri şubede indekslenir).
Kullanım: python load_vm_corpora.py --url http://10.0.0.11:8001 --corpus ai/data/splits/branch1/corpus.jsonl --prefix b1
"""
import argparse, json, sys, time
import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--corpus", required=True)
ap.add_argument("--prefix", required=True); ap.add_argument("--batch", type=int, default=64)
a = ap.parse_args()
base = a.url.rstrip("/")
docs = [json.loads(l) for l in open(a.corpus, encoding="utf-8") if l.strip()]
with httpx.Client(timeout=600) as cl:
    have = cl.get(f"{base}/stats").json().get("collection_count", 0)
    print(f"[{a.prefix}] VM'deki mevcut parça: {have}, yüklenecek: {len(docs)}", flush=True)
    if have >= len(docs):
        print(f"[{a.prefix}] zaten yüklü, atlanıyor"); sys.exit(0)
    # kaldığı yerden devam: id'ler deterministik (upsert), güvenlik payıyla geri sar
    start = max(0, (have - 256) // a.batch * a.batch)
    if start: print(f"[{a.prefix}] {start}. parçadan devam ediliyor", flush=True)
    t0 = time.time()
    for i in range(start, len(docs), a.batch):
        b = docs[i:i + a.batch]
        payload = [{"id": f"{a.prefix}_{i + j}", "content": d["text"], "source": d.get("source", "corpus"),
                    "branch": a.prefix} for j, d in enumerate(b)]
        for attempt in range(3):
            try:
                r = cl.post(f"{base}/add_chunks", json=payload); r.raise_for_status(); break
            except Exception as e:
                print(f"[{a.prefix}] hata ({e}), yeniden deneniyor", flush=True); time.sleep(5)
        else:
            sys.exit(f"[{a.prefix}] yükleme başarısız: parti {i}")
        if (i // a.batch) % 20 == 0:
            print(f"[{a.prefix}] {i + len(b)}/{len(docs)}  {time.time() - t0:.0f}s", flush=True)
    print(f"[{a.prefix}] tamam: {cl.get(f'{base}/stats').json()}  süre {time.time() - t0:.0f}s", flush=True)
