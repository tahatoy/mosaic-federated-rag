"""
Genel merkez (HQ) koleksiyonu için yalnızca ORTAK bilgiyi içeren korpus üretir (savunma sonrası iyileştirme).

Tezdeki deney düzeneğinde HQ koleksiyonu iki şube korpusunun birleşimiydi (22.192 parça); şubeye özgü
21.763 parçanın kopyası merkezde de bulunuyordu (Tez Bölüm 4.7.2, 5.4). Bu betik HQ'ya yalnızca iki şubede
birebir ortak olan genel bankacılık parçalarını (%6 ortak havuz, 429 parça) bırakır; şubeye özgü bilgiye
yalnızca çapraz şube RAG servisi (mTLS, kaynakta maskeleme) üzerinden erişilebilir.

Çıktı: ai/data/splits/hq_ortak/corpus.jsonl
Kullanım: python backend/scripts/build_hq_ortak_corpus.py
"""
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "ai" / "data" / "splits"


def load(name):
    with open(S / name / "corpus.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


b1, b2 = load("branch1"), load("branch2")
t2 = {r["text"] for r in b2}
seen, ortak = set(), []
for r in b1:
    if r["text"] in t2 and r["text"] not in seen:
        seen.add(r["text"]); ortak.append(r)
out = S / "hq_ortak" / "corpus.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    for r in ortak:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"şube-1 {len(b1)}, şube-2 {len(b2)}, ortak {len(ortak)} -> {out}")
