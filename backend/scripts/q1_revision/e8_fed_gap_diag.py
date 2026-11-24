"""
E8 — Federe (C) ile merkezi (D) arasındaki 3 puanlık farkın kaynağı.

E1'de tam aramayla federe birleştirmenin kayıpsız olduğu görüldü; E2'de ise canlı şube servisleriyle (C) merkezi dizin (D)
arasında ~3 puanlık EM farkı ölçüldü. Bu betik LLM kullanmadan, aynı 500 kalıp soruda şu kaynakları ayırır:
  * aday kümesi: şube servisleri (ChromaDB HNSW + RAG_MIN_SIMILARITY süzgeci, top_k) altın parçayı döndürüyor mu?
  * k etkisi: şube başına 10 ve 20 aday
  * kaynakta maskeleme: dönen parça metni özgün metinden farklı mı, altın sayı maskelemeden sonra korunuyor mu?
  * aynı protokolün tam aramalı (harness) sürümü: şube başına ilk 10 + HQ ortak, aynı yeniden sıralayıcı
Çıktı: ai/experiments/results/q1/e8_fed_gap.json
"""
import os
import re

import numpy as np
import torch

from q1_common import ROOT, mean_ci, norm_num, questions, save
from e2_crossbranch_generation import mtls_client, remote


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="turkembed", choices=["turkembed", "e5", "me"]); ap.add_argument("--out", default="e8_fed_gap.json")
    a = ap.parse_args()
    from sentence_transformers import CrossEncoder, SentenceTransformer
    qs, docs, owner = questions(500)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=dev)
    m = SentenceTransformer({"turkembed": str(ROOT / "ai/models/embeddings/turkembed_finetuned"), "e5": "intfloat/multilingual-e5-small", "me": str(ROOT / "ai/models/embeddings/turkembed_v3")}[a.model], device=dev); m.max_seq_length = 512
    E = m.encode(["passage: " + d for d in docs], batch_size=128, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    shards = {"b1": [i for i, o in enumerate(owner) if o in (0, 1)], "b2": [i for i, o in enumerate(owner) if o in (0, 2)], "hq": [i for i, o in enumerate(owner) if o == 0]}
    b_url = {1: os.environ["BRANCH1_RAG_URL"], 2: os.environ["BRANCH2_RAG_URL"]}
    import httpx
    plain = httpx.Client(timeout=120)
    hq_url = os.getenv("HQ_RAG_URL", "http://127.0.0.1:8003")

    def rr(q, texts, k=3):
        if not texts:
            return []
        s = ce.predict([[q, t] for t in texts], show_progress_bar=False)
        return [texts[j] for j in np.argsort(-np.asarray(s))[:k]]

    rec = []
    with mtls_client() as mt:
        for q in qs:
            sq = q["search_query"]; gold_doc = docs[q["gold_idx"][0]]
            r = {}
            for k in (10, 20):
                own = remote(mt, b_url[q["owner_branch"]], sq, k)
                cand = own + remote(mt, b_url[3 - q["owner_branch"]], sq, k) + remote(plain, hq_url, sq, k)
                texts = [c["text"] for c in cand]
                r[f"n_own_k{k}"] = len(own)
                r[f"gold_in_cand_k{k}"] = int(any(q["reference"] in t for t in texts))
                r[f"gold_num_in_cand_k{k}"] = int(any(q["gold"] in t for t in texts))
                r[f"ctx_hit_k{k}"] = int(any(q["reference"] in t for t in rr(sq, texts)))
                if k == 10:
                    masked = [t for t in texts if "GİZLİ_" in t]
                    r["cand_masked_frac"] = len(masked) / max(1, len(texts))
                    # altın parça servis üzerinden döndüyse özgün metinle karşılaştır
                    g = [t for t in texts if t[:80] == gold_doc[:80]]
                    r["gold_chunk_returned_by_prefix"] = int(bool(g))
                    r["gold_chunk_text_changed"] = int(bool(g) and g[0] != gold_doc)
                    r["gold_reference_lost_by_masking"] = int(bool(g) and q["reference"] not in g[0])
            qv = m.encode(["query: " + sq], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
            S = (qv @ E.T)[0].float()
            fed = []
            for ids in shards.values():
                sub = S[torch.as_tensor(ids, device=S.device)]
                fed += [int(ids[j]) for j in torch.topk(sub, 10).indices.tolist()]
            fed = list(dict.fromkeys(fed))
            r["harness_fed_gold_in_cand"] = int(any(i in q["gold_idx"] for i in fed))
            r["harness_fed_ctx_hit"] = int(any(q["reference"] in t for t in rr(sq, [docs[i] for i in fed])))
            cen = torch.topk(S, 20).indices.tolist()
            r["harness_central_ctx_hit"] = int(any(q["reference"] in t for t in rr(sq, [docs[i] for i in cen])))
            rec.append(r)
    out = {"n": len(rec), "summary": {k: mean_ci([x[k] for x in rec]) for k in rec[0]}}
    out["harness_model"] = a.model
    save(a.out, out)
    for k, v in out["summary"].items():
        print(k, v, flush=True)


if __name__ == "__main__":
    main()
