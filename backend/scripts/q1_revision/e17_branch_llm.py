"""
E17 — Gemma-3-4B'nin şube sanal makinelerinde (yalnızca CPU) çalıştırılması: canlı uç çıkarım testi.

500 çapraz şube sorusundan her şubeye eşit sayıda, eşit aralıklı seçilen sorular (varsayılan 50) için E2'nin C koşuluyla aynı bağlam kurulur
(iki şube RAG servisi + genel merkez ortak koleksiyonu, çapraz kodlayıcıyla ilk 3 parça). Aynı bağlam ve aynı
istemle soru iki kez yanıtlanır:
  hq     : genel merkezdeki Gemma-3-4B Q4_K_M (GPU, 8099)
  branch : sorunun sorulduğu şubenin kendi sanal makinesindeki Gemma-3-4B Q4_K_M (2 vCPU, yalnızca CPU)
Şube modeline SSH tüneliyle erişilir (--branch-llm). Ölçülenler: tam eşleşme (eşleştirilmiş McNemar),
yanıt süresi, llama-server'ın bildirdiği istem işleme ve üretim hızları. Şube belleği betik tarafından eklenir.
Çıktı: ai/experiments/results/q1/e17_branch_llm.json
"""
import argparse
import os
import statistics as st
import threading
import time

import httpx
import numpy as np
import torch

from q1_common import mcnemar_exact, mean_ci, norm_num, paired_diff_ci, questions, save

SYS = "Sen Türkçe bir bankacılık asistanısın. Kısa ve doğrudan yanıt ver."


def mtls_client():
    import ssl
    from q1_common import ROOT
    ctx = ssl.create_default_context(cafile=str(ROOT / os.getenv("RAG_CA_CERT_PATH", "backend/certs/ca.crt")))
    cert, key = os.getenv("RAG_CLIENT_CERT"), os.getenv("RAG_CLIENT_KEY")
    if cert:
        ctx.load_cert_chain(str(ROOT / cert), str(ROOT / key) if key else None)
    return httpx.Client(verify=ctx, timeout=120)


def remote(cl, url, q, k=10):
    r = cl.post(url.rstrip("/") + "/query", json={"query": q, "top_k": k, "use_reranker": False})
    r.raise_for_status()
    return [{"text": c.get("text") or c.get("content") or ""} for c in r.json().get("chunks", [])]


def ask(cl, url, question, ctx):
    # E2 ile aynı istem ve bağlam bütçesi (parça başına en fazla 1.400 karakter)
    ctx = [c if len(c) <= 1400 else c[:1400].rsplit(" ", 1)[0] + " …" for c in ctx]
    user = "Bağlam:\n" + "\n---\n".join(ctx) + "\n\nYalnızca bağlama dayanarak yanıtla.\n" + question
    t0 = time.perf_counter()
    r = cl.post(url, json={"messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                           "max_tokens": 40, "temperature": 0.0}, timeout=600)
    wall = time.perf_counter() - t0
    j = r.json()
    tm = j.get("timings", {})
    return {"answer": j["choices"][0]["message"]["content"].split("</think>")[-1].strip(), "wall_s": wall,
            "prompt_n": tm.get("prompt_n"), "prompt_tok_s": tm.get("prompt_per_second"),
            "gen_n": tm.get("predicted_n"), "gen_tok_s": tm.get("predicted_per_second")}


def summ(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return {"mean": round(st.mean(v), 3), "median": round(st.median(v), 3), "min": round(min(v), 3),
            "max": round(max(v), 3), "p95": round(float(np.percentile(v, 95)), 3), "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--hq-llm", default="http://127.0.0.1:8099/v1/chat/completions")
    ap.add_argument("--branch-llm", default="1=http://127.0.0.1:18101,2=http://127.0.0.1:18102")
    ap.add_argument("--out", default="e17_branch_llm.json")
    a = ap.parse_args()
    from sentence_transformers import CrossEncoder

    bllm = {int(k): v.rstrip("/") + "/v1/chat/completions" for k, v in (s.split("=", 1) for s in a.branch_llm.split(","))}
    qs_all, _, _ = questions(500)
    qs = []  # her şubeden eşit sayıda, kendi içinde eşit aralıklı soru
    for b in (1, 2):
        pool = [q for q in qs_all if q["asked_branch"] == b]
        step = max(1, len(pool) // (a.n // 2))
        qs += pool[::step][:a.n // 2]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=dev)
    b_url = {1: os.environ["BRANCH1_RAG_URL"], 2: os.environ["BRANCH2_RAG_URL"]}
    hq_url = os.getenv("HQ_RAG_URL", "http://127.0.0.1:8003")
    plain = httpx.Client(timeout=600)

    # 1) bağlamlar (E2 C koşulu) ve genel merkez yanıtı
    recs = []
    with mtls_client() as mt:
        for i, q in enumerate(qs, 1):
            sq = q["search_query"]
            fed = remote(mt, b_url[1], sq) + remote(mt, b_url[2], sq) + remote(plain, hq_url, sq)
            sc = ce.predict([[sq, c["text"]] for c in fed], show_progress_bar=False) if fed else []
            ctx = [fed[j]["text"] for j in np.argsort(-np.asarray(sc))[:a.k]] if fed else []
            hq = ask(plain, a.hq_llm, q["question"], ctx)
            recs.append({"branch": q["asked_branch"], "q": q, "ctx": ctx, "hq": hq})
            if i % 10 == 0:
                print("bağlam+hq", i, flush=True)

    # 2) her soru, sorulduğu şubenin kendi modelinde (iki şube paralel, her şube kendi sırasıyla)
    def run_branch(b):
        cl = httpx.Client(timeout=600)
        for r in recs:
            if r["branch"] == b:
                for attempt in range(3):
                    try:
                        r["br"] = ask(cl, bllm[b], r["q"]["question"], r["ctx"])
                        break
                    except Exception as e:  # geçici bağlantı hatasında yeniden dene
                        print(f"şube-{b} hata ({attempt + 1}/3):", e, flush=True)
                        time.sleep(5)
                else:
                    raise RuntimeError(f"şube-{b} modeli yanıt vermedi")
                print(f"şube-{b}", round(r["br"]["wall_s"], 1), "s", flush=True)
    t0 = time.perf_counter()
    th = [threading.Thread(target=run_branch, args=(b,)) for b in (1, 2)]
    [t.start() for t in th]
    [t.join() for t in th]
    total = time.perf_counter() - t0

    em = {k: [int(r["q"]["gold_norm"] in norm_num(r[k]["answer"])) for r in recs] for k in ("hq", "br")}
    only_hq = sum(x and not y for x, y in zip(em["hq"], em["br"]))
    only_br = sum(y and not x for x, y in zip(em["hq"], em["br"]))
    out = {
        "setup": {"model": "GEMMA3_FINANCE_V1.Q4_K_M.gguf", "questions": "evenly spaced subset of the 500 cross-branch questions, half asked from each branch",
                  "n": len(recs), "per_branch": {b: sum(r["branch"] == b for r in recs) for b in (1, 2)}, "k": a.k,
                  "context": "E2 condition C (both branch RAG services + HQ shared collection, cross-encoder top 3)",
                  "hq": "Gemma-3-4B Q4_K_M on the HQ GPU (8099)",
                  "branch": "Gemma-3-4B Q4_K_M on the asking branch's VMware VM, CPU only, 2 threads, alongside the branch RAG service"},
        "em": {"hq": mean_ci(em["hq"]), "branch": mean_ci(em["br"]),
               "same_answer_correctness": sum(x == y for x, y in zip(em["hq"], em["br"])),
               "only_hq": only_hq, "only_branch": only_br, "mcnemar_p": mcnemar_exact(only_hq, only_br),
               "diff_ci": paired_diff_ci(em["br"], em["hq"])},
        "identical_answers": sum(r["hq"]["answer"] == r["br"]["answer"] for r in recs),
        "latency_s": {"hq": summ([r["hq"]["wall_s"] for r in recs]), "branch": summ([r["br"]["wall_s"] for r in recs]),
                      "branch_by_branch": {b: summ([r["br"]["wall_s"] for r in recs if r["branch"] == b]) for b in (1, 2)}},
        "branch_llm_speed": {"prompt_tokens": summ([r["br"]["prompt_n"] for r in recs]),
                             "prompt_tok_s": summ([r["br"]["prompt_tok_s"] for r in recs]),
                             "gen_tokens": summ([r["br"]["gen_n"] for r in recs]),
                             "gen_tok_s": summ([r["br"]["gen_tok_s"] for r in recs])},
        "branch_phase_total_s": round(total, 1),
        "records": [{"branch": r["branch"], "gold": r["q"]["gold_norm"], "hq": r["hq"]["answer"][:200],
                     "branch_answer": r["br"]["answer"][:200], "branch_wall_s": round(r["br"]["wall_s"], 2)} for r in recs],
    }
    save(a.out, out)
    print("EM hq / şube:", out["em"]["hq"], out["em"]["branch"], "McNemar p:", out["em"]["mcnemar_p"], flush=True)
    print("şube yanıt süresi (s):", out["latency_s"]["branch"], flush=True)


if __name__ == "__main__":
    main()
