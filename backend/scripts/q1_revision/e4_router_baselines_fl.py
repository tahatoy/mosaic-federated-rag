"""
E4 — Yönlendirici: klasik karşılaştırma tabanları ve tekrarlı federe eğitim.

1) Tabanlar (router_train ile eğitilir, router_test n=3000 üzerinde ölçülür):
   çoğunluk sınıfı, TF-IDF (sözcük 1-2 + karakter 2-5 gram) + lojistik regresyon,
   multilingual-e5-small gömmesi + lojistik regresyon, konuşlandırılan BERT yönlendirici (8000/predict/batch).
2) Federe yönlendirici: model_tool.train_federated ile aynı non-IID istemci verisi, 5 tur, 2 yerel dönem,
   FedAvg ve FedProx (μ=0.01). Fark: başlangıç noktası, merkezi olarak eğitilmiş yönlendirici yerine önceden
   eğitilmiş dbmdz/bert-base-turkish-cased'dir (merkezi bilgiden arındırılmış başlangıç) ve deney 3 tohumla
   tekrarlanır. Aynı istemci verilerinin birleşimiyle eğitilen merkezi model karşılaştırma tabanıdır.
Çıktı: ai/experiments/results/q1/e4_router.json
"""
import argparse
import json
import random
import statistics
import time

import httpx
import numpy as np
import torch

from q1_common import ROOT, save

DATA = ROOT / "ai" / "data" / "processed"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = "dbmdz/bert-base-turkish-cased"


def load(name):
    t, y = [], []
    for line in (DATA / name).open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("text", "").strip() and r.get("label") in (0, 1, 2):
            t.append(r["text"].strip()); y.append(r["label"])
    return t, y


def scores(y, p):
    from sklearn.metrics import accuracy_score, f1_score
    return {"acc": round(accuracy_score(y, p), 4), "macro_f1": round(f1_score(y, p, average="macro"), 4)}


def branch_data(train_t, train_y, bid):
    by = {0: [], 1: [], 2: []}
    for t, y in zip(train_t, train_y):
        by[y].append(t)
    ratios = {1: {0: 0.70, 1: 0.50, 2: 0.30}, 2: {0: 0.30, 1: 0.50, 2: 0.70}}
    rnd = random.Random(42 + bid)
    T, Y = [], []
    for c, r in ratios[bid].items():
        pool = by[c][:]; rnd.shuffle(pool)
        n = max(300, min(600, int(len(pool) * r * 0.15)))
        T += pool[:n]; Y += [c] * n
    return T, Y


def train_bert(tok, init_sd, T, Y, epochs, seed, prox=None, mu=0.0):
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    m = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=3).to(DEV)
    if init_sd is not None:
        m.load_state_dict(init_sd)
    enc = tok(T, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    idx = list(range(len(Y))); rnd = random.Random(seed)
    steps = (len(idx) + 31) // 32 * epochs
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5)
    sch = get_linear_schedule_with_warmup(opt, max(1, int(steps * 0.1)), steps)
    yt = torch.tensor(Y)
    m.train()
    for _ in range(epochs):
        rnd.shuffle(idx)
        for i in range(0, len(idx), 32):
            b = idx[i:i + 32]
            out = m(input_ids=enc["input_ids"][b].to(DEV), attention_mask=enc["attention_mask"][b].to(DEV), labels=yt[b].to(DEV))
            loss = out.loss
            if prox is not None and mu > 0:
                loss = loss + mu / 2 * sum(((p - g) ** 2).sum() for p, g in zip(m.parameters(), prox))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    return m


def predict(m, tok, T):
    m.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(T), 128):
            e = tok(T[i:i + 128], max_length=128, padding=True, truncation=True, return_tensors="pt").to(DEV)
            P += m(**e).logits.argmax(-1).cpu().tolist()
    return P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--rounds", type=int, default=5)
    a = ap.parse_args()
    trT, trY = load("router_train.jsonl"); teT, teY = load("router_test.jsonl")
    out = {"n_train": len(trY), "n_test": len(teY), "baselines": {}, "federated": {}}
    # 1) tabanlar
    from collections import Counter
    maj = Counter(trY).most_common(1)[0][0]
    out["baselines"]["majority"] = scores(teY, [maj] * len(teY))
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    t0 = time.time()
    tf = make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                  TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
                       LogisticRegression(max_iter=2000, C=4.0))
    tf.fit(trT, trY); t1 = time.time(); p = tf.predict(teT)
    out["baselines"]["tfidf_logreg"] = {**scores(teY, p), "infer_ms_per_query": round((time.time() - t1) * 1000 / len(teT), 3), "train_s": round(t1 - t0, 1)}
    print("tfidf", out["baselines"]["tfidf_logreg"], flush=True)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("intfloat/multilingual-e5-small", device=str(DEV))
    Xtr = st.encode(["query: " + t for t in trT], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    Xte = st.encode(["query: " + t for t in teT], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    lr = LogisticRegression(max_iter=3000, C=10.0).fit(Xtr, trY)
    out["baselines"]["e5small_logreg"] = scores(teY, lr.predict(Xte)); print("e5", out["baselines"]["e5small_logreg"], flush=True)
    del st; torch.cuda.empty_cache()
    try:
        P = []
        with httpx.Client(timeout=120) as cl:
            t1 = time.time()
            for i in range(0, len(teT), 100):
                P += [int(r["label"]) for r in cl.post("http://127.0.0.1:8000/predict/batch", json=teT[i:i + 100]).json()]
        out["baselines"]["bert_router_deployed"] = {**scores(teY, P), "infer_ms_per_query_http_batch": round((time.time() - t1) * 1000 / len(teT), 3)}
    except Exception as e:
        out["baselines"]["bert_router_deployed"] = {"hata": str(e)[:200]}
    print("bert", out["baselines"]["bert_router_deployed"], flush=True)
    save("e4_router.json", out)
    # 2) federe, 3 tohum, önceden eğitilmiş BERTurk'ten başlangıç
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    cl = {b: branch_data(trT, trY, b) for b in (1, 2)}
    out["federated"]["client_sizes"] = {b: len(v[1]) for b, v in cl.items()}
    rows = {"central_pooled": [], "FedAvg": [], "FedProx": [], "local_b1": [], "local_b2": []}
    for seed in [int(s) for s in a.seeds.split(",")]:
        T = cl[1][0] + cl[2][0]; Y = cl[1][1] + cl[2][1]
        m = train_bert(tok, None, T, Y, a.rounds * 2, seed); rows["central_pooled"].append(scores(teY, predict(m, tok, teT))); del m
        for b in (1, 2):
            m = train_bert(tok, None, cl[b][0], cl[b][1], a.rounds * 2, seed); rows[f"local_b{b}"].append(scores(teY, predict(m, tok, teT))); del m
        for alg, mu in (("FedAvg", 0.0), ("FedProx", 0.01)):
            from transformers import AutoModelForSequenceClassification
            torch.manual_seed(seed)
            g = {k: v.detach().clone() for k, v in AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=3).state_dict().items()}
            t0 = time.time()
            for r in range(a.rounds):
                agg, tot = None, 0
                for b in (1, 2):
                    prox = None
                    if mu > 0:
                        tmp = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=3); tmp.load_state_dict(g)
                        prox = [p.detach().to(DEV) for p in tmp.parameters()]; del tmp
                    m = train_bert(tok, g, cl[b][0], cl[b][1], 2, seed * 100 + r * 10 + b, prox=prox, mu=mu)
                    n = len(cl[b][1]); sd = m.state_dict()
                    if agg is None:
                        agg = {k: v.detach().float().cpu() * n for k, v in sd.items()}
                    else:
                        for k, v in sd.items():
                            agg[k] += v.detach().float().cpu() * n
                    tot += n; del m; torch.cuda.empty_cache()
                g = {k: (v / tot).to(g[k].dtype) for k, v in agg.items()}
            m = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=3).to(DEV); m.load_state_dict(g)
            rows[alg].append({**scores(teY, predict(m, tok, teT)), "train_s": round(time.time() - t0, 1)}); del m
        print("tohum", seed, {k: v[-1] for k, v in rows.items()}, flush=True)
        out["federated"]["runs"] = rows; save("e4_router.json", out)
    out["federated"]["summary"] = {k: {m: [round(statistics.mean(x[m] for x in v), 4), round(statistics.pstdev(x[m] for x in v), 4)]
                                       for m in ("acc", "macro_f1")} for k, v in rows.items() if v}
    save("e4_router.json", out)


if __name__ == "__main__":
    main()
