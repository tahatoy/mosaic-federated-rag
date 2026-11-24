"""
E6 — H4'ün geniş örneklemle yeniden sınanması: Zemberek kök/morfem işlemenin erişime etkisi.

Tezde/makalede H4 erişim etkisi 15 sorgu üzerinden ölçülmüştü. Burada E1'deki 600 kalıp ve 419 yeniden ifade
edilmiş çapraz şube sorusu ile 22.192 şube parçasının tamamı kullanılır. Üç ön işleme karşılaştırılır:
  ham     : metin olduğu gibi
  kok     : Zemberek (analyze_and_disambiguate) ile her sözcük köküne indirgenir (h4_token_cost.py ile aynı)
  morfem  : Zemberek morfemleri boşlukla ayrılır (morfem bölütlemeli hibrit girdi)
Yöntemler: BM25 ve yoğun erişim (multilingual-e5-small, TurkEmbed), sorgu ve parçalara aynı ön işleme uygulanır.
Çıktı: ai/experiments/results/q1/e6_h4_zemberek.json (parça önbelleği: e6_zemberek_cache.json)
"""
import json
import multiprocessing as mp
import time

from q1_common import OUT_DIR, ROOT, mean_ci, paired_diff_ci, questions, rank_metrics, save, tr_tokens

_M = None


def _init():
    global _M
    from zemberek import TurkishMorphology
    _M = TurkishMorphology.create_with_defaults()


def _proc(text):
    try:
        an = _M.analyze_and_disambiguate(text).best_analysis()
    except Exception:
        return text, text
    kok, mor = [], []
    for a in an:
        try:
            s = a.get_stem()
            kok.append(s if s and s != "UNK" and len(s) > 1 else a.surface_form())
        except Exception:
            pass
        try:
            parts = [m.surface for m in a.morpheme_data_list if getattr(m, "surface", "")]
            mor.extend(parts if parts else [a.surface_form()])
        except Exception:
            pass
    return " ".join(x for x in kok if x), " ".join(x for x in mor if x)


def main():
    qs, docs, owner = questions(600)
    para = {r["reference"]: r["natural"] for r in json.loads((OUT_DIR / "e0_paraphrases.json").read_text(encoding="utf-8"))["kayitlar"] if r["natural"]}
    nq = [q for q in qs if q["reference"] in para]
    sets = {"cloze": (qs, [q["search_query"] for q in qs]), "natural": (nq, [para[q["reference"]] for q in nq])}
    cache_p = OUT_DIR / "e6_zemberek_cache.json"
    t0 = time.time()
    if cache_p.exists():
        cache = json.loads(cache_p.read_text(encoding="utf-8"))
    else:
        with mp.Pool(6, initializer=_init) as pool:
            res = pool.map(_proc, docs, chunksize=64)
        cache = {"kok": [r[0] for r in res], "morfem": [r[1] for r in res]}
        cache_p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    t_docs = time.time() - t0
    print("parça işleme", round(t_docs, 1), "s", flush=True)
    with mp.Pool(6, initializer=_init) as pool:
        qproc = {s: pool.map(_proc, qt) for s, (_, qt) in sets.items()}
    variants = {"ham": (docs, {s: qt for s, (_, qt) in sets.items()}),
                "kok": (cache["kok"], {s: [r[0] for r in qproc[s]] for s in sets}),
                "morfem": (cache["morfem"], {s: [r[1] for r in qproc[s]] for s in sets})}
    out = {"n_docs": len(docs), "n": {s: len(v[0]) for s, v in sets.items()}, "doc_processing_s": round(t_docs, 1),
           "avg_tokens_per_doc": {k: round(sum(len(tr_tokens(d)) for d in v[0]) / len(docs), 2) for k, v in variants.items()},
           "ornek": {"ham": sets["natural"][1][0], "kok": variants["kok"][1]["natural"][0], "morfem": variants["morfem"][1]["natural"][0]},
           "results": {}}
    print(json.dumps(out["ornek"], ensure_ascii=False), flush=True)
    import numpy as np
    import torch
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    per = {}
    for vname, (vd, vq) in variants.items():
        bm = BM25Okapi([tr_tokens(d) for d in vd])
        for s, (sq, _) in sets.items():
            ranks = [list(np.argsort(-bm.get_scores(tr_tokens(q)))[:100]) for q in vq[s]]
            per[(s, "bm25", vname)] = [rank_metrics(r, q["gold_idx"]) for r, q in zip(ranks, sq)]
        print("bm25", vname, "tamam", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for mname, path in (("e5-small", "intfloat/multilingual-e5-small"), ("mosaic-embed", str(ROOT / "ai/models/embeddings/turkembed_v3"))):
        m = SentenceTransformer(path, device=dev); m.max_seq_length = 512
        for vname in ("ham", "kok"):
            vd, vq = variants[vname]
            E = m.encode(["passage: " + d for d in vd], batch_size=128, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
            for s, (sq, _) in sets.items():
                Q = m.encode(["query: " + q for q in vq[s]], batch_size=128, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
                top = torch.topk((Q @ E.T).float(), 100, dim=1).indices.cpu().tolist()
                per[(s, mname, vname)] = [rank_metrics(t, q["gold_idx"]) for t, q in zip(top, sq)]
            print(mname, vname, "tamam", flush=True)
        del m; torch.cuda.empty_cache()
    for (s, meth, vname), rows in per.items():
        r = out["results"].setdefault(s, {}).setdefault(meth, {})
        r[vname] = {k: mean_ci([x[k] for x in rows]) for k in ("mrr@10", "hit@1", "hit@10")}
        if vname != "ham":
            r[vname]["minus_ham_mrr10"] = paired_diff_ci([x["mrr@10"] for x in rows], [x["mrr@10"] for x in per[(s, meth, "ham")]])
    for s in out["results"]:
        for meth, v in out["results"][s].items():
            print(s, meth, {k: v[k]["mrr@10"][0] for k in v}, flush=True)
    save("e6_h4_zemberek_me.json", out)


if __name__ == "__main__":
    main()
