"""E10 TR-MMLU özeti: doğruluk, %95 GA, finans bölümleri alt kümesi, modeller arası McNemar."""
import json, math, random
from pathlib import Path

R = Path("ai/experiments/results/q1")
FIN = ["Bankacılık ve Sigortacılık", "Maliye", "Menkul Kıymetler ve Sermaye Piyasası",
       "Muhasebe ve Vergi Uygulamaları", "İktisat"]


def preds(name, mode):
    return [json.loads(l) for l in open(R / f"e10_tr_mmlu_{name}_{mode}_preds.jsonl", encoding="utf-8")]


def ci(h, B=2000, seed=7):
    rnd = random.Random(seed); n = len(h)
    ms = sorted(sum(h[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(ms[int(.025 * B)], 4), round(ms[int(.975 * B) - 1], 4)]


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10; k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return n01, n10, p


out = {}
H = {}
for name in ("qwen", "gemma"):
    for mode in ("chatml", "native"):
        f = R / f"e10_tr_mmlu_{name}_{mode}.json"
        if not f.exists():
            continue
        s = json.load(open(f, encoding="utf-8"))
        P = preds(name, mode)
        h = [int(p["pred"] == p["gold"]) for p in P]
        H[(name, mode)] = h
        fin = [int(p["pred"] == p["gold"]) for p in P if p["sec"] in FIN]
        secs = sorted(s["per_section"].items(), key=lambda kv: kv[1]["acc"])
        out[f"{name}_{mode}"] = dict(n=s["n"], acc=s["accuracy"], ci95=s["ci95"], unparseable=s["unparseable"],
                                     latency=s["avg_latency_s"], fin_n=len(fin), fin_acc=round(sum(fin) / len(fin), 4),
                                     fin_ci95=ci(fin), worst3=secs[:3], best3=secs[-3:],
                                     pred_dist={L: sum(1 for p in P if p["pred"] == L) for L in "ABCDE"} | {"None": sum(1 for p in P if p["pred"] is None)})
for mode in ("chatml", "native"):
    if ("qwen", mode) in H and ("gemma", mode) in H:
        out[f"mcnemar_qwen_vs_gemma_{mode}"] = mcnemar(H[("qwen", mode)], H[("gemma", mode)])
for name in ("qwen", "gemma"):
    if (name, "chatml") in H and (name, "native") in H:
        out[f"mcnemar_{name}_chatml_vs_native"] = mcnemar(H[(name, "chatml")], H[(name, "native")])
print(json.dumps(out, ensure_ascii=False, indent=1))
(R / "e10_tr_mmlu_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
