"""
İki uçtan uca koşuyu (h1_e2e_<tag>.json) aynı sorular üzerinden eşleştirerek karşılaştırır (McNemar kesin testi).
Kullanım: python h1_e2e_compare.py once_hq_kopya_gpu_bos sonra_hq_ortak_gpu_bos
Çıktı: ai/experiments/results/h1_e2e_karsilastirma.json
"""
import json, math, sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[3] / "ai" / "experiments" / "results"
a_tag, b_tag = sys.argv[1], sys.argv[2]
A = json.loads((R / f"h1_e2e_{a_tag}.json").read_text(encoding="utf-8"))
B = json.loads((R / f"h1_e2e_{b_tag}.json").read_text(encoding="utf-8"))
ka, kb = A["kayitlar"], B["kayitlar"]
assert [r["soru"] for r in ka] == [r["soru"] for r in kb], "soru kümeleri farklı"
only_a = sum(1 for x, y in zip(ka, kb) if x["dogru"] and not y["dogru"])
only_b = sum(1 for x, y in zip(ka, kb) if y["dogru"] and not x["dogru"])
n = only_a + only_b
p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n)
res = {"once": a_tag, "sonra": b_tag, "n": len(ka),
       "dogruluk_once": A["dogruluk"], "dogruluk_sonra": B["dogruluk"],
       "ortalama_sure_ms_once": A["ortalama_sure_ms"], "ortalama_sure_ms_sonra": B["ortalama_sure_ms"],
       "yalniz_once_dogru": only_a, "yalniz_sonra_dogru": only_b, "mcnemar_p": round(p, 4),
       "yorum": "Fark istatistiksel olarak anlamlı değil (p>=0.05): HQ'daki şube kopyaları kaldırıldığında doğruluk korunmuştur."
       if p >= 0.05 else "Fark istatistiksel olarak anlamlı (p<0.05)."}
(R / "h1_e2e_karsilastirma.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
