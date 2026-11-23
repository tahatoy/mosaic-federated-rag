"""
Q1 revizyon deneyleri için ortak yardımcılar.

- Şube korpusları ve H1 ile aynı üreteçten (h1_federated_rag_eval.build_questions) türetilen çapraz şube soruları
- Altın parça eşlemesi: sorunun kaynak cümlesini içeren, sahip şubeye özgü parçalar
- Erişim ölçütleri (Hit@k, MRR@10, nDCG@10) ve eşleştirilmiş bootstrap güven aralıkları
Tüm rastgelelik sabit tohumla yönetilir; sonuçlar ai/experiments/results/q1/ altına yazılır.
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend" / "scripts" / "hypothesis_tests"))
sys.path.insert(0, str(ROOT / "backend"))
OUT_DIR = ROOT / "ai" / "experiments" / "results" / "q1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from h1_federated_rag_eval import build_questions, norm_num, load_branch, mcnemar_exact  # noqa: E402,F401


def load_corpora():
    """Şube korpuslarını döndürür: (docs, branch_of) — docs: benzersiz metin listesi, branch_of: {0,1,2} (0 = iki şubede ortak)."""
    b1 = [r["text"] for r in load_branch("branch1")]
    b2 = [r["text"] for r in load_branch("branch2")]
    s1, s2 = set(b1), set(b2)
    docs, owner, seen = [], [], set()
    for t in b1 + b2:
        if t in seen:
            continue
        seen.add(t)
        docs.append(t)
        owner.append(0 if (t in s1 and t in s2) else (1 if t in s1 else 2))
    return docs, owner


def attach_gold(qs, docs, owner):
    """Her soru için altın parça indeksleri: kaynak cümleyi içeren, sahip şubeye ait parçalar."""
    out = []
    for q in qs:
        g = [i for i, t in enumerate(docs) if q["reference"] in t and owner[i] == q["owner_branch"]]
        if g:
            q = dict(q, gold_idx=g)
            out.append(q)
    return out


def questions(n, seed=42):
    docs, owner = load_corpora()
    qs = attach_gold(build_questions(n, seed=seed), docs, owner)
    return qs, docs, owner


_TR = str.maketrans({"I": "ı", "İ": "i"})


def tr_tokens(s: str):
    return [w for w in re.split(r"[^\wçğıöşüÇĞİÖŞÜ]+", s.translate(_TR).lower()) if w]


def rank_metrics(ranked, gold, ks=(1, 3, 5, 10)):
    """ranked: sıralı belge indeksleri; gold: küme. Hit@k, MRR@10, nDCG@10 (ikili alaka)."""
    gold = set(gold)
    first = next((r for r, d in enumerate(ranked[:100], 1) if d in gold), None)
    m = {f"hit@{k}": float(first is not None and first <= k) for k in ks}
    m["mrr@10"] = 1.0 / first if first and first <= 10 else 0.0
    dcg = sum(1.0 / math.log2(r + 1) for r, d in enumerate(ranked[:10], 1) if d in gold)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold), 10) + 1))
    m["ndcg@10"] = dcg / idcg if idcg else 0.0
    return m


def mean_ci(x, B=2000, seed=7):
    rnd = random.Random(seed)
    n = len(x)
    mu = sum(x) / n
    bs = sorted(sum(x[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return round(mu, 4), [round(bs[int(0.025 * B)], 4), round(bs[int(0.975 * B) - 1], 4)]


def paired_diff_ci(x, y, B=2000, seed=11):
    rnd = random.Random(seed)
    n = len(x)
    d = [a - b for a, b in zip(x, y)]
    bs = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return round(sum(d) / n, 4), [round(bs[int(0.025 * B)], 4), round(bs[int(0.975 * B) - 1], 4)]


def save(name, obj):
    p = OUT_DIR / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print("yazıldı:", p)
    return p
