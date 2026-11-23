"""E11: human evaluation of the live end-to-end answers.

Draws the 100-answer sample from the second live run (e5_e2e_judge_v2.json) and compares the labels of two human
raters with each other, with the LLM judge and with exact match (EM).

Usage:
    python e11_human_eval.py sample   # prints the sampled record ids in presentation order
    python e11_human_eval.py analyse  # writes human_eval/human_eval_summary.json

Sampling: seed 2026, proportional to the judge's (correct, faithful) decision combinations.
Raters labelled each answer with the definitions used by the judge (see e5_e2e_judge.py) in annotation_form.html,
without seeing the judge or EM decisions or each other's labels.
"""
import json
import random
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "ai" / "experiments" / "results" / "q1"
OUT = RES / "human_eval"
SEED, N_SAMPLE = 2026, 100


def load_judge():
    return json.loads((RES / "e5_e2e_judge_v2.json").read_text(encoding="utf-8"))["records"]


def sample(records):
    rng = random.Random(SEED)
    strata = {}
    for i, r in enumerate(records):
        strata.setdefault((r["judge_correct"], r["judge_faithful"]), []).append(i)
    picked = []
    for _, ids in sorted(strata.items()):
        picked += rng.sample(ids, round(len(ids) * N_SAMPLE / len(records)))
    picked = picked[:N_SAMPLE]
    order = picked[:]
    rng.shuffle(order)
    return order


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def kappa_ci(x, y, b=5000, seed=2):
    rng = random.Random(seed)
    n, ks = len(x), []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            ks.append(kappa([x[i] for i in idx], [y[i] for i in idx]))
        except ZeroDivisionError:
            pass
    ks.sort()
    return [round(ks[int(0.025 * len(ks))], 3), round(ks[int(0.975 * len(ks)) - 1], 3)]


def rate_ci(v, b=10000, seed=1):
    rng = random.Random(seed)
    n = len(v)
    bs = sorted(sum(rng.choice(v) for _ in range(n)) / n for _ in range(b))
    return [round(sum(v) / n, 3), [round(bs[int(0.025 * b)], 3), round(bs[int(0.975 * b) - 1], 3)]]


def mcnemar(x, y):
    b = sum(1 for p, q in zip(x, y) if p == 1 and q == 0)
    c = sum(1 for p, q in zip(x, y) if p == 0 and q == 1)
    n = b + c
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return {"only_first": b, "only_second": c, "p": round(p, 5)}


def compare(x, y):
    return {"agree": sum(a == b for a, b in zip(x, y)), "kappa": round(kappa(x, y), 3),
            "kappa_ci95": kappa_ci(x, y), "mcnemar": mcnemar(x, y)}


def analyse():
    rec = load_judge()
    raters = {}
    for name in ("rater1", "rater2"):
        d = json.loads((OUT / f"{name}_labels.json").read_text(encoding="utf-8"))
        raters[name] = {e["id"]: e for e in d["etiketler"]}
    ids = sample(rec)
    assert all(set(ids) == set(r) for r in raters.values()), "labels do not match the sample"
    jc = [int(rec[i]["judge_correct"]) for i in ids]
    jf = [int(rec[i]["judge_faithful"]) for i in ids]
    em = [int(rec[i]["em"]) for i in ids]
    lab = {n: ([r[i]["correct"] for i in ids], [r[i]["faithful"] for i in ids]) for n, r in raters.items()}
    out = {"n": len(ids), "seed": SEED,
           "rates": {"judge_correct": rate_ci(jc), "judge_faithful": rate_ci(jf), "em": rate_ci(em)},
           "inter_rater": {"correct": compare(lab["rater1"][0], lab["rater2"][0]),
                           "faithful": compare(lab["rater1"][1], lab["rater2"][1])}}
    for n, (c, f) in lab.items():
        out["rates"][f"{n}_correct"] = rate_ci(c)
        out["rates"][f"{n}_faithful"] = rate_ci(f)
        out[f"{n}_vs_judge"] = {"correct": compare(c, jc), "faithful": compare(f, jf)}
        out[f"{n}_vs_em"] = compare(c, em)
    out["judge_vs_em"] = compare(jc, em)
    agreed = [k for k, i in enumerate(ids) if lab["rater1"][1][k] == lab["rater2"][1][k]]
    out["faithful_consensus"] = {
        "n": len(agreed),
        "accepted_by_both_rejected_by_judge": sum(1 for k in agreed if lab["rater1"][1][k] == 1 and jf[k] == 0),
        "rejected_by_both_accepted_by_judge": sum(1 for k in agreed if lab["rater1"][1][k] == 0 and jf[k] == 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "human_eval_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["inter_rater"], indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    if cmd == "sample":
        print(sample(load_judge()))
    else:
        analyse()
