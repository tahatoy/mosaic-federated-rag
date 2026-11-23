"""
H1 deneyi: federe bilgi getirme, tekil (RAG'sız / tek şube) kurulumdan daha doğru mu?

Kolların karşılaştırması (aynı LLM, aynı sorular, aynı istem):
  A  no_rag      : LLM tek başına (parametrik bilgi)
  B  local_only  : yalnızca sorunun SORULDUĞU şubenin yerel koleksiyonu
  C  federated   : iki şube koleksiyonu taranır, adaylar HQ'da çapraz-kodlayıcıyla yeniden sıralanır

Soru üretimi (LLM'siz, tekrarlanabilir): şube korpusundaki bir cümledeki sayısal değer boşluk bırakılır
(cloze). Altın yanıt = o sayı. Başarı = altın sayının yanıtta geçmesi (doğruluk). Ek olarak ROUGE-L raporlanır.
Soru, altın parçanın BULUNMADIĞI şubede sorulur (veri adacığı senaryosu) -> yerel arama yetersiz kalmalıdır.

İstatistik: eşleştirilmiş sonuçlarda McNemar (kesin iki yönlü), %95 bootstrap GA.
Önkoşul: ports 8001 (Şube-1 RAG), 8002 (Şube-2 RAG), llama-server (varsayılan 8099).
Çalıştırma:  python backend/scripts/hypothesis_tests/h1_federated_rag_eval.py --n 200 --llm-url http://127.0.0.1:8099/v1/chat/completions
Çıktı: ai/experiments/results/h1_federated_rag.json
"""
import argparse, json, math, os, random, re, sys
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
NUM = re.compile(r"(?<![\w.,])(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?)(?=\s?(?:%|TL\b|trilyon|milyar|milyon|baz puan|puan))")

def norm_num(s): return re.sub(r"[.,\s]", "", s)

def load_branch(name):
    rows = []
    with open(ROOT / "ai" / "data" / "splits" / name / "corpus.jsonl", encoding="utf-8") as f:
        for l in f:
            if l.strip(): rows.append(json.loads(l))
    return rows

def build_questions(n, seed=42):
    rnd = random.Random(seed)
    b1, b2 = load_branch("branch1"), load_branch("branch2")
    t1 = {r["text"] for r in b1}; t2 = {r["text"] for r in b2}
    qs = []
    pool = [(1, r) for r in b1 if r["text"] not in t2] + [(2, r) for r in b2 if r["text"] not in t1]  # yalnızca tek şubede bulunan parçalar
    rnd.shuffle(pool)
    for owner, r in pool:
        text = r.get("text", "")
        body = text.split("\n\n", 1)[-1]
        for sent in re.split(r"(?<=[.!?])\s+", body):
            ms = list(NUM.finditer(sent))
            if 70 < len(sent) < 300 and len(ms) == 1 and 'S&P' not in sent and len(norm_num(ms[0].group(1))) >= 2:
                m = ms[0]
                blank = sent[:m.start()] + "___" + sent[m.end():]
                if blank.lstrip().startswith(('___', ':', '%')): continue
                qs.append({"owner_branch": owner, "asked_branch": 3 - owner, "gold": m.group(1), "gold_norm": norm_num(m.group(1)),
                           "question": f"Aşağıdaki cümledeki boşluğa gelen sayısal değer nedir? Sadece değeri yaz.\n\"{blank}\"",
                           "search_query": blank, "reference": sent})
                break
        if len(qs) >= n: break
    return qs

def retrieve(client, port, query, k):
    base = os.environ.get(f"BRANCH{port-8000}_RAG_URL", f"http://127.0.0.1:{port}").rstrip("/")
    r = client.post(f"{base}/query", json={"query": query, "top_k": k, "use_reranker": False}, timeout=60)
    r.raise_for_status()
    return r.json()["chunks"]

def rerank(query, chunks, top_k):
    try:
        from ai_engine import rerank_chunks
        return rerank_chunks(query, chunks, top_k=top_k, min_score=0.0)
    except Exception:
        return sorted(chunks, key=lambda c: -c.get("score", 0))[:top_k]

def ask(client, url, question, context=None, max_tokens=40):
    sys_p = "Sen Türkçe bir bankacılık asistanısın. Kısa ve doğrudan yanıt ver."
    user = question if not context else "Bağlam:\n" + "\n---\n".join(context) + "\n\nYalnızca bağlama dayanarak yanıtla.\n" + question
    r = client.post(url, json={"messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                               "max_tokens": max_tokens, "temperature": 0.0}, timeout=120)
    out = r.json()["choices"][0]["message"]["content"].strip()
    return out.split("</think>")[-1].strip()

def lcs(a, b):
    a, b = a.split(), b.split()
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i+1][j+1] = dp[i][j]+1 if a[i] == b[j] else max(dp[i][j+1], dp[i+1][j])
    return dp[-1][-1]

def rouge_l(ref, hyp):
    l = lcs(ref.lower(), hyp.lower())
    if not l: return 0.0
    p, r = l/max(1, len(hyp.split())), l/max(1, len(ref.split()))
    return 2*p*r/(p+r)

def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k+1)) / 2**n
    return min(1.0, 2*p)

def boot_ci(x, y, B=5000, seed=1):
    rnd = random.Random(seed); n = len(x); d = []
    for _ in range(B):
        idx = [rnd.randrange(n) for _ in range(n)]
        d.append(sum(x[i]-y[i] for i in idx)/n)
    d.sort(); return d[int(.025*B)], d[int(.975*B)]


def _preflight(cl):
    """Ölçüm öncesi: her şubeye gerçek bir /query gönderilebildiğini doğrula (yoksa geçersiz veri üretme)."""
    import os as _o, sys as _s
    for b in (1, 2):
        u = _o.environ.get(f"BRANCH{b}_RAG_URL")
        if not u: continue
        try:
            r = cl.post(u.rstrip("/") + "/query", json={"query": "mevduat faizi", "top_k": 1, "use_reranker": False}, timeout=60); r.raise_for_status()
        except Exception as e:
            print(f"ÖN KONTROL BAŞARISIZ: şube {b} ({u}): {type(e).__name__}: {e}", flush=True); _s.exit(3)
    print("ön kontrol: şubeler /query yanıtlıyor", flush=True)

def _mtls(verify, cert):
    """httpx 0.28 uyumlu mTLS bağlamı: verify=<yol> iken cert= yok sayıldığı için sertifikayı SSLContext'e yükle."""
    import ssl as _ssl
    if verify is False and not cert:
        return False
    if verify is False:
        ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT); ctx.check_hostname = False; ctx.verify_mode = _ssl.CERT_NONE
    else:
        ctx = _ssl.create_default_context(cafile=verify) if isinstance(verify, str) else _ssl.create_default_context()
    if cert and cert[0]:
        ctx.load_cert_chain(*cert)
    return ctx

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--llm-url", default="http://127.0.0.1:8099/v1/chat/completions"); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="h1_federated_rag.json", help="ai/experiments/results altındaki çıktı dosyası adı")
    a = ap.parse_args()
    qs = build_questions(a.n); print("soru sayısı:", len(qs))
    arms = {"A_no_rag": [], "B_local_only": [], "C_federated": []}; rl = {k: [] for k in arms}; recs = []
    _ca = os.getenv("RAG_CA_CERT_PATH", "backend/certs/ca.crt")
    _verify = _ca if os.getenv("RAG_TLS_VERIFY", "false").lower() == "true" and os.path.exists(_ca) else False
    _cert = (os.getenv("RAG_CLIENT_CERT"), os.getenv("RAG_CLIENT_KEY")) if os.getenv("RAG_CLIENT_CERT") else None
    with httpx.Client(verify=_mtls(_verify, _cert)) as cl:  # mTLS (şube) ve düz http (yerel LLM) birlikte
        _preflight(cl)
        for i, q in enumerate(qs, 1):
            ctxs = {"A_no_rag": None}
            ctxs["B_local_only"] = [c["text"] for c in rerank(q["search_query"], retrieve(cl, 8000 + q["asked_branch"], q["search_query"], 10), a.k)]
            cand = retrieve(cl, 8001, q["search_query"], 10) + retrieve(cl, 8002, q["search_query"], 10)
            ctxs["C_federated"] = [c["text"] for c in rerank(q["search_query"], cand, a.k)]
            row = {"q": q["question"], "gold": q["gold"], "owner": q["owner_branch"], "asked": q["asked_branch"]}
            for arm, ctx in ctxs.items():
                ans = ask(cl, a.llm_url, q["question"], ctx)
                ok = int(q["gold_norm"] in norm_num(ans)); arms[arm].append(ok); rl[arm].append(rouge_l(q["reference"], ans))
                row[arm] = {"answer": ans, "correct": ok}
            recs.append(row)
            if i % 20 == 0: print(i, {k: round(sum(v)/len(v), 3) for k, v in arms.items()})
    res = {"n": len(qs), "accuracy": {k: round(sum(v)/len(v), 4) for k, v in arms.items()},
           "rouge_l": {k: round(sum(v)/len(v), 4) for k, v in rl.items()}, "tests": {}}
    for x, y in (("C_federated", "A_no_rag"), ("C_federated", "B_local_only")):
        b = sum(1 for p, q in zip(arms[x], arms[y]) if p and not q); c = sum(1 for p, q in zip(arms[x], arms[y]) if q and not p)
        lo, hi = boot_ci(arms[x], arms[y])
        res["tests"][f"{x}_vs_{y}"] = {"only_first_correct": b, "only_second_correct": c, "mcnemar_p": round(mcnemar_exact(b, c), 6),
                                        "acc_diff": round(sum(arms[x])/len(qs) - sum(arms[y])/len(qs), 4), "boot95": [round(lo, 4), round(hi, 4)]}
    res["h1_karar"] = "DESTEKLENDİ" if res["tests"]["C_federated_vs_A_no_rag"]["mcnemar_p"] < 0.05 and res["accuracy"]["C_federated"] > res["accuracy"]["A_no_rag"] else "DESTEKLENMEDİ"
    res["kayıtlar"] = recs
    out = ROOT / "ai" / "experiments" / "results" / a.out; out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "kayıtlar"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
