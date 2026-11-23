"""
E1 — Güncel erişim yöntemleriyle karşılaştırma ve merkezi / federe erişim eşdeğerliği.

Soru kümeleri: H1 üretecinden 600 çapraz şube sorusu (cloze) ve E0'da kabul edilen doğal dil sürümleri.
Altın parça: kaynak cümleyi içeren, sahip şubeye özgü parça(lar). Arama alanı: iki şube korpusunun tamamı.

Yöntemler
  bm25                 : BM25Okapi (Türkçe küçük harf, sözcük sınırı tokenizasyonu)
  e5-small (taban)     : intfloat/multilingual-e5-small
  turkembed-ft (MOSAIC): e5-small'ın finans alanına ince ayarlı sürümü
  e5-base / e5-large   : intfloat/multilingual-e5-{base,large}
  bge-m3               : BAAI/bge-m3 (yoğun vektör)
  turkembed4retrieval  : newmindai/TurkEmbed4Retrieval (Türkçe erişim modeli)
  hybrid-rrf           : bm25 + turkembed-ft, karşılıklı sıra füzyonu (k=60)
  *+ce                 : ilk 20 aday cross-encoder/mmarco-mMiniLMv2 ile yeniden sıralanır

Senaryolar
  central   : tüm parçalar tek dizinde (üst sınır; MOSAIC'in kaçındığı yapı)
  federated : her şube (ve HQ ortak koleksiyonu) kendi dizininde ilk 10'u döndürür; adaylar HQ'da birleştirilir
              (yoğun yöntemlerde puanla, BM25'te şube içi IDF ile; +ce'de birleşik adaylar yeniden sıralanır)
Çıktı: ai/experiments/results/q1/e1_retrieval.json
"""
import argparse
import json
import time

import numpy as np
import torch

from q1_common import OUT_DIR, ROOT, mean_ci, paired_diff_ci, questions, rank_metrics, save, tr_tokens

DENSE = {
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "turkembed-ft": (str(ROOT / "ai/models/embeddings/turkembed_finetuned"), "query: ", "passage: "),
    "e5-base": ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    "e5-large": ("intfloat/multilingual-e5-large", "query: ", "passage: "),
    "bge-m3": ("BAAI/bge-m3", "", ""),
    "turkembed4retrieval": ("newmindai/TurkEmbed4Retrieval", "", ""),
}


def shard_ids(owner):
    s1 = [i for i, o in enumerate(owner) if o in (0, 1)]
    s2 = [i for i, o in enumerate(owner) if o in (0, 2)]
    hq = [i for i, o in enumerate(owner) if o == 0]
    return {"b1": np.array(s1), "b2": np.array(s2), "hq": np.array(hq)}


def dense_rank(S, shards, k_shard=10, depth=100):
    """S: [Q, N] benzerlik. Merkezi ve federe sıralamalar."""
    central = torch.topk(S, depth, dim=1).indices.cpu().numpy()
    fed = []
    for qi in range(S.shape[0]):
        cand = {}
        for ids in shards.values():
            sub = S[qi, torch.as_tensor(ids, device=S.device)]
            v, ix = torch.topk(sub, min(k_shard, len(ids)))
            for s, j in zip(v.tolist(), ix.tolist()):
                d = int(ids[j]); cand[d] = max(cand.get(d, -9), s)
        fed.append([d for d, _ in sorted(cand.items(), key=lambda x: -x[1])])
    return central, fed


def bm25_rank(docs, shards, qtexts, k_shard=10, depth=100):
    from rank_bm25 import BM25Okapi
    toks = [tr_tokens(d) for d in docs]
    full = BM25Okapi(toks)
    sh = {n: (ids, BM25Okapi([toks[i] for i in ids])) for n, ids in shards.items()}
    central, fed = [], []
    for q in qtexts:
        qt = tr_tokens(q)
        sc = full.get_scores(qt)
        central.append(list(np.argsort(-sc)[:depth]))
        cand = {}
        for ids, bm in sh.values():
            s = bm.get_scores(qt)
            for j in np.argsort(-s)[:k_shard]:
                cand[int(ids[j])] = float(s[j])
        fed.append([d for d, _ in sorted(cand.items(), key=lambda x: -x[1])])
    return central, fed


def rrf(a, b, k=60, depth=100):
    sc = {}
    for lst in (a, b):
        for r, d in enumerate(lst[:depth], 1):
            sc[d] = sc.get(d, 0) + 1.0 / (k + r)
    return [d for d, _ in sorted(sc.items(), key=lambda x: -x[1])]


def ce_rerank(ce, docs, qtexts, ranks, top=20):
    out = []
    for q, r in zip(qtexts, ranks):
        cand = list(r[:top])
        s = ce.predict([[q, docs[d]] for d in cand], batch_size=64, show_progress_bar=False)
        out.append([cand[j] for j in np.argsort(-np.asarray(s))])
    return out


def summarize(ranks, qs):
    per = [rank_metrics(list(r), q["gold_idx"]) for r, q in zip(ranks, qs)]
    keys = per[0].keys()
    return {k: mean_ci([p[k] for p in per]) for k in keys}, [p["mrr@10"] for p in per]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--models", default=",".join(DENSE))
    a = ap.parse_args()
    from sentence_transformers import CrossEncoder, SentenceTransformer
    qs, docs, owner = questions(a.n)
    shards = shard_ids(owner)
    para = {}
    pp = OUT_DIR / "e0_paraphrases.json"
    if pp.exists():
        para = {r["reference"]: r["natural"] for r in json.loads(pp.read_text(encoding="utf-8"))["kayitlar"] if r["natural"]}
    sets = {"cloze": (qs, [q["search_query"] for q in qs])}
    nq = [q for q in qs if q["reference"] in para]
    if nq:
        sets["natural"] = (nq, [para[q["reference"]] for q in nq])
    print("parça:", len(docs), "soru:", {k: len(v[0]) for k, v in sets.items()}, "şube parça:", {k: len(v) for k, v in shards.items()}, flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"n_docs": len(docs), "n_questions": {k: len(v[0]) for k, v in sets.items()}, "methods": {}, "latency_ms": {}}
    ranks = {s: {} for s in sets}
    for name in a.models.split(","):
        path, qp, dp = DENSE[name]
        try:
            t0 = time.time()
            m = SentenceTransformer(path, device=dev, trust_remote_code=True)
            m.max_seq_length = 512
            if dev == "cuda":
                m.half()
            E = m.encode([dp + d for d in docs], batch_size=32, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
            print(name, "dizin", round(time.time() - t0, 1), "s", flush=True)
            for s, (sq, qt) in sets.items():
                t1 = time.time()
                Q = m.encode([qp + q for q in qt], batch_size=128, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
                S = (Q @ E.T).float()
                ranks[s][name] = dense_rank(S, shards)
                res["latency_ms"].setdefault(name, round((time.time() - t1) * 1000 / len(qt), 2))
            del m, E
            torch.cuda.empty_cache()
        except Exception as e:  # model indirilemezse kaydet, diğerlerine devam et
            print("ATLANDI", name, type(e).__name__, str(e)[:200], flush=True)
            res["methods"][name] = {"hata": f"{type(e).__name__}: {str(e)[:200]}"}
    for s, (sq, qt) in sets.items():
        t1 = time.time()
        ranks[s]["bm25"] = bm25_rank(docs, shards, qt)
        res["latency_ms"].setdefault("bm25", round((time.time() - t1) * 1000 / len(qt) / 2, 2))
        if "turkembed-ft" in ranks[s]:
            c, f = ranks[s]["turkembed-ft"]; bc, bf = ranks[s]["bm25"]
            ranks[s]["hybrid-rrf"] = ([rrf(x, y) for x, y in zip(bc, c)], [rrf(x, y) for x, y in zip(bf, f)])
    ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=dev)
    for s, (sq, qt) in sets.items():
        for base in ("turkembed-ft", "e5-small", "hybrid-rrf", "bm25"):
            if base in ranks[s]:
                c, f = ranks[s][base]
                ranks[s][base + "+ce"] = (ce_rerank(ce, docs, qt, c), ce_rerank(ce, docs, qt, f))
    for s, (sq, qt) in sets.items():
        for name, (c, f) in ranks[s].items():
            mc, xc = summarize(c, sq)
            mf, xf = summarize(f, sq)
            res["methods"].setdefault(name, {})[s] = {
                "central": mc, "federated": mf, "fed_minus_central_mrr10": paired_diff_ci(xf, xc),
                "identical_top10_rate": round(float(np.mean([list(x[:10]) == list(y[:10]) for x, y in zip(c, f)])), 4)}
            print(s, name, "MRR@10 merkezi", mc["mrr@10"], "federe", mf["mrr@10"], flush=True)
    save("e1_retrieval.json", res)


if __name__ == "__main__":
    main()
