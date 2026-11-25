"""
E5 — Canlı uçtan uca hat ve sözcük örtüşmesinden bağımsız değerlendirme (LLM hakem).

Doğal dil çapraz şube soruları (E0) canlı orkestratöre (yönlendirici + HQ ortak koleksiyonu + mTLS şube servisleri +
yeniden sıralama + dil modeli) yanıtın BULUNMADIĞI şubeden gönderilir. Karşılaştırma için aynı sorular bilgi
getirmeden doğrudan merkez modeline (Qwen3-8B, 8098) sorulur.

Ölçütler
  EM          : altın sayının yanıtta geçmesi (deterministik)
  judge_correct: hakem, altın cümleye göre yanıtın sorulan değeri doğru verip vermediğine karar verir (0/1)
  judge_faithful: hakem, yanıttaki olgusal iddiaların tümünün getirilen kaynak parçalarla desteklenip desteklenmediğine karar verir (0/1)
Hakem: ince ayar görmemiş Qwen3-8B Q4_K_M (port 8096; sistemdeki modellerden ayrı süreç), sıcaklık 0.
Hakem geçerliliği: judge_correct ile EM arasındaki uyum (Cohen κ) raporlanır.
Çıktı: ai/experiments/results/q1/e5_e2e_judge.json
"""
import argparse
import json
import re

import httpx

from q1_common import OUT_DIR, mean_ci, mcnemar_exact, norm_num, paired_diff_ci, questions, save

JUDGE_PROMPT = """Bir soru-cevap sisteminin yanıtını değerlendiriyorsun.
Soru: {q}
Doğru bilgiyi içeren kaynak cümle: {ref}
Sistemin getirdiği kaynak parçalar:
{ctx}
Sistemin yanıtı: {ans}

İki karar ver:
1) correct: Yanıt, soruda istenen değeri kaynak cümleyle uyumlu biçimde veriyorsa 1, vermiyor veya yanlış veriyorsa 0.
2) faithful: Yanıttaki olgusal iddiaların (sayılar, adlar, tarihler) tamamı yukarıdaki getirilen kaynak parçalarda destekleniyorsa 1, desteklenmeyen en az bir iddia varsa 0. Kaynak parça yoksa ve yanıt olgusal iddia içeriyorsa 0.
Yalnızca şu biçimde JSON döndür: {{"correct": 0 veya 1, "faithful": 0 veya 1}} /no_think"""


def kappa(x, y):
    n = len(x); po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


def judge(cl, url, q, ref, ctx, ans):
    msg = JUDGE_PROMPT.format(q=q, ref=ref, ctx="\n---\n".join(c[:700] for c in ctx) if ctx else "(yok)", ans=ans[:1500])
    r = cl.post(url, json={"messages": [{"role": "user", "content": msg}], "max_tokens": 60, "temperature": 0.0}).json()
    t = r["choices"][0]["message"]["content"].split("</think>")[-1]
    m = re.search(r"\{.*?\}", t, re.S)
    try:
        d = json.loads(m.group(0)); return int(d.get("correct", 0)), int(d.get("faithful", 0))
    except Exception:
        return 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["collect", "judge"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--orch", default="http://127.0.0.1:8004/query")
    ap.add_argument("--qwen", default="http://127.0.0.1:8098/v1/chat/completions")
    ap.add_argument("--judge", default="http://127.0.0.1:8096/v1/chat/completions")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-norag", action="store_true", help="bilgi getirmeyen karşılaştırmayı atla (geliştirme taramaları)")
    ap.add_argument("--offset", type=int, default=0, help="geliştirme kümesi için ilk --offset soruyu atla (ör. 200: test dışındaki 219 soru)")
    ap.add_argument("--reuse-norag", default="", help="bilgi getirmeyen Qwen yanıtlarını bu etiketli ham dosyadan al (ör. _v3); aynı sorular, aynı model")
    a = ap.parse_args()
    raw = OUT_DIR / f"e5_e2e_raw{a.tag}.json"
    if a.phase == "collect":
        qs, _, _ = questions(600)
        para = {r["reference"]: r["natural"] for r in json.loads((OUT_DIR / "e0_paraphrases.json").read_text(encoding="utf-8"))["kayitlar"] if r["natural"]}
        items = [q for q in qs if q["reference"] in para][a.offset:a.offset + a.n]
        rec = []
        prev = {}
        if a.reuse_norag:
            prev = {r["q"]: r for r in json.loads((OUT_DIR / f"e5_e2e_raw{a.reuse_norag}.json").read_text(encoding="utf-8"))}
        with httpx.Client(timeout=600) as cl:
            for i, q in enumerate(items, 1):
                nq = para[q["reference"]]
                d = cl.post(a.orch, json={"query": nq, "branch_id": q["asked_branch"]}).json()
                ans = d.get("answer", "")
                if a.no_norag:
                    ans0 = ""
                elif nq in prev:
                    ans0 = prev[nq]["norag_answer"]
                else:
                    r0 = cl.post(a.qwen, json={"messages": [{"role": "system", "content": "Sen Türkçe bir bankacılık asistanısın."},
                                                             {"role": "user", "content": nq + " /no_think"}], "max_tokens": 400, "temperature": 0.0}).json()
                    ans0 = r0["choices"][0]["message"]["content"].split("</think>")[-1].strip()
                rec.append({"q": nq, "reference": q["reference"], "gold": q["gold"], "asked": q["asked_branch"], "owner": q["owner_branch"],
                            "route": d.get("route"), "elapsed_ms": d.get("elapsed_ms"), "escalated": d.get("escalated", False), "model_name": d.get("model_name"),
                            "src_branches": [s.get("branch") for s in d.get("sources", [])], "sources": [s.get("text", "") for s in d.get("sources", [])],
                            "em": int(q["gold_norm"] in norm_num(ans)), "norag_em": int(q["gold_norm"] in norm_num(ans0)),
                            "answer": ans, "norag_answer": ans0})
                if i % 20 == 0:
                    print(i, "EM", round(sum(r["em"] for r in rec) / i, 3), "RAG'sız EM", round(sum(r["norag_em"] for r in rec) / i, 3), flush=True)
        raw.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print("yazıldı:", raw)
        return
    rec = json.loads(raw.read_text(encoding="utf-8"))
    with httpx.Client(timeout=600) as cl:
        for i, r in enumerate(rec, 1):
            r["judge_correct"], r["judge_faithful"] = judge(cl, a.judge, r["q"], r["reference"], r["sources"], r["answer"])
            r["norag_judge_correct"], r["norag_judge_faithful"] = judge(cl, a.judge, r["q"], r["reference"], [], r["norag_answer"])
            if i % 50 == 0:
                print(i, flush=True)
    keys = ("em", "judge_correct", "judge_faithful", "norag_em", "norag_judge_correct", "norag_judge_faithful")
    res = {"n": len(rec), "summary": {k: mean_ci([r[k] for r in rec]) for k in keys},
           "judge_validity": {"kappa_judge_vs_em_mosaic": kappa([r["judge_correct"] for r in rec], [r["em"] for r in rec]),
                              "kappa_judge_vs_em_norag": kappa([r["norag_judge_correct"] for r in rec], [r["norag_em"] for r in rec]),
                              "agreement_mosaic": round(sum(r["judge_correct"] == r["em"] for r in rec) / len(rec), 4)},
           "routes": {k: sum(r["route"] == k for r in rec) for k in {r["route"] for r in rec}},
           "src_owner_in_context": round(sum(any(str(b) in (f"b{r['owner']}", f"branch_0{r['owner']}", str(r['owner'])) for b in r["src_branches"]) for r in rec) / len(rec), 4),
           "tests": {}, "mean_elapsed_ms": round(sum(r["elapsed_ms"] or 0 for r in rec) / len(rec), 1)}
    for x, y in (("judge_correct", "norag_judge_correct"), ("em", "norag_em")):
        ex, ey = [r[x] for r in rec], [r[y] for r in rec]
        b = sum(p and not q for p, q in zip(ex, ey)); c = sum(q and not p for p, q in zip(ex, ey))
        res["tests"][f"{x}_vs_{y}"] = {"mcnemar_p": mcnemar_exact(b, c), "diff_ci": paired_diff_ci(ex, ey)}
    res["records"] = [{k: v for k, v in r.items() if k != "sources"} for r in rec]
    save(f"e5_e2e_judge{a.tag}.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "records"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
