"""
E13 — Cross-branch answer accuracy (H1 protocol) with MOSAIC-Embed versus the base multilingual-e5-small.

Same questions, prompt, generator (Gemma-3-4B Q4_K_M, temperature 0), cross-encoder and k = 3 as E2. The deployed
testbed indexes the branches with the base model, so both retrievers are compared offline on identical inputs:

  D_e5      central index, base multilingual-e5-small, cross-encoder re-ranking
  D_mosaic  central index, MOSAIC-Embed, cross-encoder re-ranking
  F_e5      federated simulation: Branch-1, Branch-2 and the HQ collection each return their top 10 (exact search),
            candidates are merged at HQ and re-ranked; base model
  F_mosaic  the same federated simulation with MOSAIC-Embed

Metrics as in E2 (EM, ctx_hit, grounded, unsupported); paired McNemar tests and bootstrap CIs.
Output: results/q1/e13_embed_generation.json
"""
import argparse
import json

import httpx
import numpy as np
import torch

from e2_crossbranch_generation import ask, nums
from q1_common import OUT_DIR, ROOT, mcnemar_exact, mean_ci, norm_num, paired_diff_ci, questions, save

MODELS = {"e5": "intfloat/multilingual-e5-small", "mosaic": str(ROOT / "ai/models/embeddings/turkembed_v3")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--llm", default="http://127.0.0.1:8099/v1/chat/completions")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    from sentence_transformers import CrossEncoder, SentenceTransformer

    qs, docs, owner = questions(a.n)
    para = {r["reference"]: r["natural"] for r in
            json.loads((OUT_DIR / "e0_paraphrases.json").read_text(encoding="utf-8"))["kayitlar"] if r["natural"]}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=dev)
    shards = {"b1": [i for i, o in enumerate(owner) if o in (0, 1)],
              "b2": [i for i, o in enumerate(owner) if o in (0, 2)],
              "hq": [i for i, o in enumerate(owner) if o == 0]}
    shard_t = {n: torch.as_tensor(ix) for n, ix in shards.items()}
    embs = {}
    for nm, path in MODELS.items():
        m = SentenceTransformer(path, device=dev)
        m.max_seq_length = 512
        E = m.encode(["passage: " + d for d in docs], batch_size=128, normalize_embeddings=True,
                     convert_to_tensor=True, show_progress_bar=False)
        embs[nm] = (m, E)

    def scores(nm, q):
        m, E = embs[nm]
        qv = m.encode(["query: " + q], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        return (qv @ E.T)[0].float().cpu()

    def central(nm, q, k=20):
        s, ix = torch.topk(scores(nm, q), k)
        return [(int(i), float(v)) for v, i in zip(s.tolist(), ix.tolist())]

    def federated(nm, q, k_shard=10):
        s = scores(nm, q)
        cand = {}
        for ix in shard_t.values():
            v, j = torch.topk(s[ix], min(k_shard, len(ix)))
            for sc, jj in zip(v.tolist(), j.tolist()):
                d = int(ix[jj])
                cand[d] = max(cand.get(d, -9.0), sc)
        return sorted(cand.items(), key=lambda x: -x[1])

    def rerank(q, cand, k):
        sc = ce.predict([[q, docs[d]] for d, _ in cand], show_progress_bar=False)
        return [docs[cand[j][0]] for j in np.argsort(-np.asarray(sc))[:k]]

    sets = {"cloze": [(q, q["question"], q["search_query"]) for q in qs],
            "natural": [(q, para[q["reference"]] + " Sadece değeri yaz.", para[q["reference"]])
                        for q in qs if q["reference"] in para]}
    arms = ["D_e5", "D_mosaic", "F_e5", "F_mosaic"]
    out = {"n": {k: len(v) for k, v in sets.items()}, "llm": "Gemma-3-4B Q4_K_M (8099)", "k": a.k,
           "retrievers": MODELS, "sonuclar": {}, "testler": {}, "kayitlar": {}}
    with httpx.Client(timeout=180) as cl:
        for sname, items in sets.items():
            rec = {arm: [] for arm in arms}
            for i, (q, question, sq) in enumerate(items, 1):
                ctx = {"D_e5": rerank(sq, central("e5", sq), a.k), "D_mosaic": rerank(sq, central("mosaic", sq), a.k),
                       "F_e5": rerank(sq, federated("e5", sq), a.k), "F_mosaic": rerank(sq, federated("mosaic", sq), a.k)}
                for arm in arms:
                    ans = ask(cl, a.llm, question, ctx[arm])
                    cnum, anum = nums(" ".join(ctx[arm])), nums(ans)
                    rec[arm].append({"em": int(q["gold_norm"] in norm_num(ans)),
                                     "ctx_hit": int(any(q["reference"] in t for t in ctx[arm])),
                                     "grounded": int(bool(anum) and anum <= cnum),
                                     "unsupported": int(bool(anum - cnum)), "answer": ans[:200]})
                if i % 50 == 0:
                    print(sname, i, {k: round(np.mean([r["em"] for r in v]), 3) for k, v in rec.items()}, flush=True)
            out["sonuclar"][sname] = {arm: {m: mean_ci([r[m] for r in v]) for m in ("em", "ctx_hit", "grounded", "unsupported")}
                                      for arm, v in rec.items()}
            t = {}
            for x, y in [("D_mosaic", "D_e5"), ("F_mosaic", "F_e5"), ("F_e5", "D_e5"), ("F_mosaic", "D_mosaic")]:
                for metric in ("em", "ctx_hit"):
                    ex, ey = [r[metric] for r in rec[x]], [r[metric] for r in rec[y]]
                    b = sum(p and not r for p, r in zip(ex, ey))
                    c = sum(r and not p for p, r in zip(ex, ey))
                    t[f"{x}_vs_{y}_{metric}"] = {"only_first": b, "only_second": c, "mcnemar_p": mcnemar_exact(b, c),
                                                 "diff_ci": paired_diff_ci(ex, ey)}
            out["testler"][sname] = t
            out["kayitlar"][sname] = {arm: [r["answer"] for r in v] for arm, v in rec.items()}
            print(sname, json.dumps(out["sonuclar"][sname], ensure_ascii=False), flush=True)
            print(sname, json.dumps(t, ensure_ascii=False), flush=True)
    save("e13_embed_generation.json", out)


if __name__ == "__main__":
    main()
