"""
E12c — Cross-branch retrieval with TurkEmbed v2, under the same protocol as E1.

Question sets: the 600 cloze questions and the accepted natural paraphrases (E0). Models: base multilingual-e5-small,
the first TurkEmbed (turkembed-ft) and TurkEmbed v2; each alone and with cross-encoder re-ranking of the top 20, in
the central and the federated setting. Paired bootstrap confidence intervals compare TurkEmbed v2 with the base model.

Output: results/q1/e12_turkembed_v2.json
"""
import json
import sys

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from e1_retrieval_baselines import ce_rerank, dense_rank, shard_ids, summarize
from q1_common import OUT_DIR, ROOT, paired_diff_ci, questions, save

MODELS = {
    "e5-small": "intfloat/multilingual-e5-small",
    "turkembed-ft": str(ROOT / "ai/models/embeddings/turkembed_finetuned"),
    "turkembed-v2": str(ROOT / "ai/models/embeddings/turkembed_v2"),
}
EXTRA = sys.argv[1] if len(sys.argv) > 1 else None  # e.g. turkembed_v3: evaluated as an additional model
if EXTRA:
    MODELS[EXTRA.replace("_", "-")] = str(ROOT / "ai/models/embeddings" / EXTRA)


def main():
    qs, docs, owner = questions(600)
    shards = shard_ids(owner)
    para = {r["reference"]: r["natural"] for r in
            json.loads((OUT_DIR / "e0_paraphrases.json").read_text(encoding="utf-8"))["kayitlar"] if r["natural"]}
    nq = [q for q in qs if q["reference"] in para]
    sets = {"cloze": (qs, [q["search_query"] for q in qs]), "natural": (nq, [para[q["reference"]] for q in nq])}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ranks = {s: {} for s in sets}
    for name, path in MODELS.items():
        m = SentenceTransformer(path, device=dev)
        m.max_seq_length = 512
        if dev == "cuda":
            m.half()
        E = m.encode(["passage: " + d for d in docs], batch_size=32, normalize_embeddings=True, convert_to_tensor=True,
                     show_progress_bar=False)
        for s, (_, qt) in sets.items():
            Q = m.encode(["query: " + q for q in qt], batch_size=128, normalize_embeddings=True,
                         convert_to_tensor=True, show_progress_bar=False)
            ranks[s][name] = dense_rank((Q @ E.T).float(), shards)
        del m, E
        torch.cuda.empty_cache()
    ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=dev)
    for s, (_, qt) in sets.items():
        for name in list(MODELS):
            c, f = ranks[s][name]
            ranks[s][name + "+ce"] = (ce_rerank(ce, docs, qt, c), ce_rerank(ce, docs, qt, f))

    res = {"n_docs": len(docs), "n_questions": {k: len(v[0]) for k, v in sets.items()}, "methods": {}, "v2_minus_base": {}}
    for s, (sq, _) in sets.items():
        per = {}
        for name, (c, f) in ranks[s].items():
            mc, xc = summarize(c, sq)
            mf, xf = summarize(f, sq)
            per[name] = (xc, xf)
            res["methods"].setdefault(name, {})[s] = {"central": mc, "federated": mf}
            print(s, name, "MRR@10 central", mc["mrr@10"], "federated", mf["mrr@10"], flush=True)
        for cand in [m for m in MODELS if m not in ("e5-small", "turkembed-ft")]:
            for suffix in ("", "+ce"):
                v2, base = per[cand + suffix], per["e5-small" + suffix]
                res["v2_minus_base"].setdefault(cand, {}).setdefault(s, {})["dense" if not suffix else "with_reranker"] = {
                    "central": paired_diff_ci(v2[0], base[0]), "federated": paired_diff_ci(v2[1], base[1])}
    print(json.dumps(res["v2_minus_base"], indent=1))
    save(f"e12_eval{'_' + EXTRA if EXTRA else ''}.json", res)


if __name__ == "__main__":
    main()
