"""Tüm deney sonuçlarını tek özet dosyada toplar: ai/experiments/results/OZET_SONUCLAR.json (tezdeki sayıların tek kaynağı)."""
import json, math
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "ai/experiments/results"
def J(n):
    p = R / n; return json.load(open(p, encoding="utf-8")) if p.exists() else None
def wilson(k, n, z=1.96):
    if not n: return None
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)); return [round((c-h)/d, 4), round((c+h)/d, 4)]
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n)
O = {}
for m in ("qwen", "gemma"):
    b = J(f"bench_{m}.json"); nb = J(f"bench_native_{m}.json")
    O[f"kiyas_{m}"] = {"chatml_ham": None if not b else {k: {x: v[x] for x in v if x != "results"} for k, v in b.items()}, "model_sablonu": nb}
for tag in ("http", "mtls", "mtls_iki_sube"):
    h = J(f"h1_federated_rag_{tag}.json")
    if h: O[f"H1_{tag}"] = {k: h[k] for k in ("n", "accuracy", "rouge_l", "tests", "h1_karar") if k in h}
for tag in ("before", "after", "kvkk", "iki_sube"):
    w = J(f"h2_canary_wire_{tag}.json"); e = J(f"h2_canary_e2e_{tag}.json"); s = J(f"h2_security_asr_{tag}.json")
    if w: O[f"H2_kanarya_hat_{tag}"] = {k: v for k, v in w.items() if k != "örnekler"}
    if e:
        rec = e["kayıtlar"]; by = {}
        for r in rec:
            b = int(r["id"].split("_")[1][1]); key = f"kaynak_branch_0{b}"
            d = by.setdefault(key, {"n": 0, "sızan": 0}); d["n"] += 1; d["sızan"] += int(bool(r["sızan"]))
        for d in by.values(): d["oran"] = round(d["sızan"] / d["n"], 4); d["wilson95"] = wilson(d["sızan"], d["n"])
        O[f"H2_kanarya_uctan_uca_{tag}"] = {"n": e["n"], "sızan": e["cevapta_sır_olan"], "oran": e["sızıntı_oranı"], "wilson95": wilson(e["cevapta_sır_olan"], e["n"]),
                                           "tür": e["sır_türleri"], "hata_atlanan": e.get("hata_atlanan", 0), "kaynak_subeye_gore": by}
    if s: O[f"H2_saldiri_{tag}"] = {k: v for k, v in s.items() if k != "kayıtlar"}
for t in ("v1", "v2", "v3", "v3_engine"):
    for st in ("heldout", "confirm", "confirm2"):
        p = J(f"h2_pii_{st}_{t}.json")
        if p: O.setdefault("H2_pii", {})[f"{st}_{t}"] = {"genel": p["genel_recall"], "kategori": {k: v["_toplam_recall"] for k, v in p["kategoriler"].items()},
                                                         "negatif_FP": p["negatif_cümleler"]["oran"], "korpus_FP": p["gerçek_korpus_yanlış_alarm"]["oran"]}
for t in ("v1", "v2", "v3_engine"):
    p = J(f"h2_pii_leakage_{t}.json")
    if p: O.setdefault("H2_pii", {})[f"ilk_test_{t}"] = {k: (v.get("overall_recall") if isinstance(v, dict) and "overall_recall" in v else v.get("oran") if isinstance(v, dict) else v) for k, v in p.items()}
O["H2_dp"] = J("h2_dp_analysis.json")
O["H3_cpu"] = J("h3_cpu_inference.json")
g = J("h3_gpu_bench.json")
if g: O["H3_gpu"] = [{"test": ("pp" if r["n_gen"] == 0 else "tg"), "n": r["n_prompt"] or r["n_gen"], "tok_s": round(r["avg_ts"], 2), "std": round(r["stddev_ts"], 2)} for r in g]
O["kvkk_bddk"] = J("kvkk_bddk_dogrulama.json")
O["kvkk_bddk_sube2"] = J("kvkk_bddk_dogrulama_sube2.json")
O["kvkk_bddk_iki_sube"] = J("kvkk_bddk_dogrulama_iki_sube.json")
(R / "OZET_SONUCLAR.json").write_text(json.dumps(O, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: (list(v.keys())[:6] if isinstance(v, dict) else type(v).__name__) for k, v in O.items()}, ensure_ascii=False, indent=0))
