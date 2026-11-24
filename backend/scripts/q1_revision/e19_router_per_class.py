"""E19: router per-class recall on the template test set (router_test.jsonl), raw and with the live rules.
Usage: python e19_router_per_class.py bert_router_best bert_router_v2  (CPU is enough)."""
import json, re, sys, torch
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ROOT = Path(__file__).resolve().parents[3]
src = (ROOT / "backend" / "ai_engine.py").read_text(encoding="utf-8")
blk = src[src.index("_COMPLEX_KEYWORDS = ["):]; blk = blk[:blk.index("]") + 1]
KW = eval(blk.split("=", 1)[1]); TH = 0.55
def rules(t, l, c):
    if any(k in t.lower() for k in KW) and l < 2: return 2
    if c < TH and l == 0: return 1
    if c < TH and l == 1: return 2
    return l
T, Y = [], []
for line in (ROOT / "ai/data/processed/router_test.jsonl").open(encoding="utf-8"):
    r = json.loads(line)
    if r.get("text", "").strip() and r.get("label") in (0, 1, 2): T.append(r["text"].strip()); Y.append(r["label"])
torch.set_num_threads(max(1, torch.get_num_threads()))
out = {"note": "E19: per-class recall on router_test.jsonl (n=3000, 1000 per class); CPU re-run of E18 checkpoints. bert_router_best = live router (first), bert_router_v2 = retrained router (last seed; all three seeds gave identical template-test results in E18).", "keywords": len(KW), "n": len(Y)}
for name in sys.argv[1:]:
    p = ROOT / "ai/models/router" / name
    tok = AutoTokenizer.from_pretrained(p); m = AutoModelForSequenceClassification.from_pretrained(p).eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(T), 64):
            e = tok(T[i:i + 64], max_length=64, padding=True, truncation=True, return_tensors="pt")
            P += torch.softmax(m(**e).logits.float(), -1).tolist()
    raw = [max(range(3), key=lambda k: p_[k]) for p_ in P]
    ru = [rules(t, l, p_[l]) for t, l, p_ in zip(T, raw, P)]
    res = {}
    for kind, pred in (("raw", raw), ("with_rules", ru)):
        cm = [[0] * 3 for _ in range(3)]
        for a, b in zip(Y, pred): cm[a][b] += 1
        res[kind] = {"acc": round(sum(cm[i][i] for i in range(3)) / len(Y), 4), "confusion_true_rows": cm,
                     "recall": {c: round(cm[i][i] / sum(cm[i]), 4) for i, c in enumerate(["chitchat", "simple", "complex"])}}
    out[name] = res; print(name, json.dumps(res), flush=True)
(ROOT / "ai/experiments/results/q1" / "e19_router_per_class.json").write_text(json.dumps(out, indent=1))
