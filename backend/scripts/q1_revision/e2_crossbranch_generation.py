"""
E2 — Genişletilmiş çapraz şube deneyi (H1) ve ayırt edici bileşen ablasyonu.

H1'in özgün düzeni korunur (aynı soru üreteci, aynı istem, Gemma-3-4B Q4_K_M, sıcaklık 0) fakat:
  * örneklem 200 -> --n (varsayılan 500) soruya çıkarılır, E0'daki doğal dil sorularıyla ikinci bir küme eklenir;
  * ROUGE-L yerine kısa sayısal yanıta uygun ölçütler kullanılır:
      EM        : altın sayının yanıtta yer alması
      ctx_hit   : altın sayının modele verilen bağlamda bulunması (erişim başarısı)
      grounded  : yanıttaki tüm sayıların bağlamda geçmesi (kaynağa dayalılık)
      unsupported: bağlamda olmayan bir sayı üretilmesi (halüsinasyon göstergesi)
  * merkezi RAG (D) üst sınır olarak eklenir; federe yapının (C) merkezi yapıya göre kaybı eşleştirilmiş
    bootstrap ve McNemar ile sınanır;
  * ablasyon kolları aynı soru üzerinde ölçülür (tavan etkisi yok, çünkü yanıt sorunun sorulduğu şubede değildir).

Kollar
  A        bilgi getirme yok
  B        yalnız sorunun sorulduğu şube (VM, mTLS) + HQ çapraz kodlayıcı
  C        MOSAIC federe: Şube-1 + Şube-2 (VM, mTLS, kaynakta maskeleme) + HQ ortak koleksiyonu, HQ'da yeniden sıralama
  D        merkezi: iki şube korpusu tek dizinde (TurkEmbed, tam arama) + aynı yeniden sıralama
  D_noce   D, yeniden sıralama yok (ilk 3 yoğun aday)
  D_e5     D, ince ayarsız multilingual-e5-small gömmesi
  C_base_q4km / C_base_q2k : C bağlamı, ince ayarsız Gemma-3-4B'nin Q4_K_M ve Q2_K sürümleri (kuantizasyon etkisi)
Çıktı: ai/experiments/results/q1/e2_generation.json
"""
import argparse
import json
import os
import re

import httpx
import numpy as np
import torch

from q1_common import OUT_DIR, ROOT, mcnemar_exact, mean_ci, norm_num, paired_diff_ci, questions, save

NUMRE = re.compile(r"\d+(?:[.,]\d+)*")
SYS = "Sen Türkçe bir bankacılık asistanısın. Kısa ve doğrudan yanıt ver."


def nums(s):
    return {norm_num(x) for x in NUMRE.findall(s or "") if len(norm_num(x)) >= 1}


def mtls_client():
    import ssl
    ca = os.getenv("RAG_CA_CERT_PATH", "backend/certs/ca.crt")
    ctx = ssl.create_default_context(cafile=str(ROOT / ca))
    cert, key = os.getenv("RAG_CLIENT_CERT"), os.getenv("RAG_CLIENT_KEY")
    if cert:
        ctx.load_cert_chain(str(ROOT / cert), str(ROOT / key) if key else None)
    return httpx.Client(verify=ctx, timeout=120)


def remote(cl, url, q, k=10):
    r = cl.post(url.rstrip("/") + "/query", json={"query": q, "top_k": k, "use_reranker": False})
    r.raise_for_status()
    return [{"text": c.get("text") or c.get("content") or "", "score": float(c.get("score", 0))} for c in r.json().get("chunks", [])]


def ask(cl, url, question, ctx):
    # Orkestratördeki bağlam bütçesiyle aynı: parça başına en fazla 1.400 karakter (RAG_CHUNK_CHAR_LIMIT)
    ctx = None if ctx is None else [c if len(c) <= 1400 else c[:1400].rsplit(" ", 1)[0] + " …" for c in ctx]
    user = question if ctx is None else "Bağlam:\n" + "\n---\n".join(ctx) + "\n\nYalnızca bağlama dayanarak yanıtla.\n" + question
    r = cl.post(url, json={"messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                           "max_tokens": 40, "temperature": 0.0}, timeout=180)
    return r.json()["choices"][0]["message"]["content"].split("</think>")[-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--llm", default="http://127.0.0.1:8099/v1/chat/completions")
    ap.add_argument("--extra", default="C_base_q4km=http://127.0.0.1:8097,C_base_q2k=http://127.0.0.1:8096",
                    help="C bağlamını farklı üreticilerle yanıtlayan ek kollar: ad=url,...")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--arms", default="A,B,C,D,D_noce,D_e5", help="cloze kümesinde koşulacak ana kollar")
    ap.add_argument("--natural", type=int, default=1, help="doğal dil kümesini de koş")
    ap.add_argument("--natural-arms", default="A,C,D")
    ap.add_argument("--out", default="e2_generation.json")
    a = ap.parse_args()
    from sentence_transformers import CrossEncoder, SentenceTransformer
    qs, docs, owner = questions(a.n)
    para = {}
    pp = OUT_DIR / "e0_paraphrases.json"
    if pp.exists():
        para = {r["reference"]: r["natural"] for r in json.loads(pp.read_text(encoding="utf-8"))["kayitlar"] if r["natural"]}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=dev)
    embs = {}
    for nm, path in (("turkembed", str(ROOT / "ai/models/embeddings/turkembed_finetuned")), ("e5", "intfloat/multilingual-e5-small")):
        m = SentenceTransformer(path, device=dev); m.max_seq_length = 512
        embs[nm] = (m, m.encode(["passage: " + d for d in docs], batch_size=128, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False))

    def central(nm, q, k=20):
        m, E = embs[nm]
        qv = m.encode(["query: " + q], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        s, ix = torch.topk((qv @ E.T)[0].float(), k)
        return [{"text": docs[i], "score": float(v)} for v, i in zip(s.tolist(), ix.tolist())]

    def rr(q, cand, k):
        if not cand:
            return []
        sc = ce.predict([[q, c["text"]] for c in cand], show_progress_bar=False)
        return [cand[j]["text"] for j in np.argsort(-np.asarray(sc))[:k]]

    b_url = {1: os.environ["BRANCH1_RAG_URL"], 2: os.environ["BRANCH2_RAG_URL"]}
    hq_url = os.getenv("HQ_RAG_URL", "http://127.0.0.1:8003")
    plain = httpx.Client(timeout=180)
    extra = {}
    for spec in filter(None, a.extra.split(",")):
        nm, url = spec.split("=", 1)
        try:
            if plain.get(url + "/health", timeout=5).status_code == 200:
                extra[nm] = url + "/v1/chat/completions"
        except Exception:
            pass
    print("soru:", len(qs), "doğal:", sum(q["reference"] in para for q in qs), "ek kollar:", list(extra), flush=True)

    sets = {"cloze": [(q, q["question"], q["search_query"]) for q in qs],
            "natural": [(q, para[q["reference"]] + " Sadece değeri yaz.", para[q["reference"]]) for q in qs if q["reference"] in para]}
    out = {"n": {k: len(v) for k, v in sets.items()}, "llm": "Gemma-3-4B Q4_K_M (8099)", "k": a.k, "sonuclar": {}, "testler": {}, "kayitlar": {}}
    with mtls_client() as mt:
        for sname, items in sets.items():
            arms = [x for x in a.arms.split(",") if x] + list(extra)
            if sname == "natural":
                if not a.natural:
                    continue
                arms = [x for x in a.natural_arms.split(",") if x]
            rec = {arm: [] for arm in arms}
            for i, (q, question, sq) in enumerate(items, 1):
                ctx = {"A": None}
                if "B" in arms:
                    ctx["B"] = rr(sq, remote(mt, b_url[q["asked_branch"]], sq), a.k)
                fed = remote(mt, b_url[1], sq) + remote(mt, b_url[2], sq) + remote(plain, hq_url, sq)
                ctx["C"] = rr(sq, fed, a.k)
                cen = central("turkembed", sq) if any(x.startswith("D") for x in arms) else []
                ctx["D"] = rr(sq, cen, a.k)
                if "D_noce" in arms:
                    ctx["D_noce"] = [c["text"] for c in cen[:a.k]]
                if "D_e5" in arms:
                    ctx["D_e5"] = rr(sq, central("e5", sq), a.k)
                for nm in extra:
                    if nm in arms:
                        ctx[nm] = ctx["C"]
                for arm in arms:
                    if arm == "C_ref":
                        continue
                    ans = ask(plain, extra.get(arm, a.llm), question, ctx[arm])
                    cnum = nums(" ".join(ctx[arm] or []))
                    anum = nums(ans)
                    rec[arm].append({
                        "em": int(q["gold_norm"] in norm_num(ans)),
                        "ctx_hit": int(ctx[arm] is not None and any(q["reference"] in t for t in ctx[arm])),
                        "grounded": int(ctx[arm] is not None and bool(anum) and anum <= cnum),
                        "unsupported": int(ctx[arm] is not None and bool(anum - cnum)),
                        "answer": ans[:200]})
                if i % 50 == 0:
                    print(sname, i, {k: round(np.mean([r["em"] for r in v]), 3) for k, v in rec.items()}, flush=True)
            out["sonuclar"][sname] = {arm: {m: mean_ci([r[m] for r in v]) for m in ("em", "ctx_hit", "grounded", "unsupported")} for arm, v in rec.items()}
            t = {}
            for x, y in [("C", "A"), ("C", "B"), ("C", "D"), ("C", "D_e5"), ("D", "D_noce"), ("D", "D_e5"), ("C", "C_base_q4km"), ("C_base_q4km", "C_base_q2k")]:
                if x in rec and y in rec:
                    ex, ey = [r["em"] for r in rec[x]], [r["em"] for r in rec[y]]
                    b = sum(p and not r for p, r in zip(ex, ey)); c = sum(r and not p for p, r in zip(ex, ey))
                    t[f"{x}_vs_{y}"] = {"only_first": b, "only_second": c, "mcnemar_p": mcnemar_exact(b, c), "diff_ci": paired_diff_ci(ex, ey)}
            out["testler"][sname] = t
            out["kayitlar"][sname] = {arm: [r["answer"] for r in v] for arm, v in rec.items()}
            print(sname, json.dumps(out["sonuclar"][sname], ensure_ascii=False), flush=True)
            print(sname, json.dumps(t, ensure_ascii=False), flush=True)
    save(a.out, out)


if __name__ == "__main__":
    main()
