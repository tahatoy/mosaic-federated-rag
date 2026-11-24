"""
E16 — Son yapılandırmada (MOSAIC-Embed) bilgi getirme gecikmesi: şube servisleri (gömme + HNSW arama + kaynakta maskeleme,
mTLS) ve HQ ortak koleksiyonu eşzamanlı sorgulanır, adaylar HQ'da çapraz kodlayıcıyla yeniden sıralanır. LLM hariç.
Çıktı: ai/experiments/results/q1/e16_retrieval_latency_me.json
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import torch

from q1_common import questions, save
from e2_crossbranch_generation import mtls_client, remote


def main():
    from sentence_transformers import CrossEncoder
    qs, _, _ = questions(500)
    qs = qs[:200]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=dev)
    b = [os.environ["BRANCH1_RAG_URL"], os.environ["BRANCH2_RAG_URL"]]
    hq = os.getenv("HQ_RAG_URL", "http://127.0.0.1:8003")
    plain = httpx.Client(timeout=120)
    rec = {"search_parallel_ms": [], "rerank_ms": [], "total_ms": [], "per_service_ms": {"b1": [], "b2": [], "hq": []}}
    with mtls_client() as mt, ThreadPoolExecutor(3) as ex:
        for q in qs[:5]:  # ısınma
            remote(mt, b[0], q["search_query"]); remote(mt, b[1], q["search_query"]); remote(plain, hq, q["search_query"])
        for q in qs:
            sq = q["search_query"]

            def timed(name, cl, url):
                t = time.perf_counter(); r = remote(cl, url, sq); rec["per_service_ms"][name].append((time.perf_counter() - t) * 1000); return r
            t0 = time.perf_counter()
            fs = [ex.submit(timed, "b1", mt, b[0]), ex.submit(timed, "b2", mt, b[1]), ex.submit(timed, "hq", plain, hq)]
            cand = [c for f in fs for c in f.result()]
            t1 = time.perf_counter()
            ce.predict([[sq, c["text"]] for c in cand], show_progress_bar=False)
            t2 = time.perf_counter()
            rec["search_parallel_ms"].append((t1 - t0) * 1000); rec["rerank_ms"].append((t2 - t1) * 1000); rec["total_ms"].append((t2 - t0) * 1000)
    summ = lambda v: {"mean": round(float(np.mean(v)), 1), "median": round(float(np.median(v)), 1), "p95": round(float(np.percentile(v, 95)), 1)}
    out = {"n": len(qs), "candidates_per_query": "10 + 10 + 10", "search_parallel_ms": summ(rec["search_parallel_ms"]),
           "rerank_ms": summ(rec["rerank_ms"]), "total_ms": summ(rec["total_ms"]),
           "per_service_ms": {k: summ(v) for k, v in rec["per_service_ms"].items()}}
    save("e16_retrieval_latency_me.json", out)
    print(out)


if __name__ == "__main__":
    main()
