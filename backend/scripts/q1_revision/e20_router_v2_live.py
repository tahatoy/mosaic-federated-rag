"""
E20 — Düzeltilmiş yönlendirici (bert_router_v2, E18) ile canlı 200 soru: ilk yönlendiriciye (E15/_v3) göre eşleştirilmiş karşılaştırma.

Aynı 200 yeniden ifade edilmiş çapraz şube sorusu, aynı canlı hat (HQ ortak koleksiyonu + mTLS şube servisleri, MOSAIC-Embed,
yeniden sıralama, CROSS_BRANCH_ALWAYS=true) üzerinden gönderilir; yalnızca yönlendirici değişir. İlk yönlendirici bu soruların
tamamını karmaşık sınıfa (Qwen3-8B) göndermişti; v2'nin çoğunu basit sınıfa (Gemma-3-4B) göndermesi beklenir.
Ölçülenler: EM, hakem doğruluğu ve kaynağa sadakat (McNemar + eşleştirilmiş bootstrap GA), yanıt süresi (ortalama, ortanca,
p95; eşleştirilmiş fark GA), rota dağılımı, rota başına EM ve süre, yönlendiricinin doğru sınıfı (basit) seçme oranı.
Girdi: e5_e2e_judge_v3.json (ilk yönlendirici), e5_e2e_judge{--tag}.json (v2). Çıktı: e20_router_v2_live.json
"""
import argparse
import json
import random
import statistics as st

from q1_common import OUT_DIR, mcnemar_exact, mean_ci, paired_diff_ci, save


def pct(v, p):
    v = sorted(v)
    k = (len(v) - 1) * p / 100
    f = int(k)
    return v[f] + (v[min(f + 1, len(v) - 1)] - v[f]) * (k - f)


def med_diff_ci(x, y, B=2000, seed=13):
    rnd = random.Random(seed); n = len(x)
    bs = []
    for _ in range(B):
        ix = [rnd.randrange(n) for _ in range(n)]
        bs.append(st.median([x[i] for i in ix]) - st.median([y[i] for i in ix]))
    bs.sort()
    return round(st.median(x) - st.median(y), 1), [round(bs[int(0.025 * B)], 1), round(bs[int(0.975 * B) - 1], 1)]


def lat(v):
    return {"mean": round(st.mean(v), 1), "median": round(st.median(v), 1), "p95": round(pct(v, 95), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="_v3")
    ap.add_argument("--tag", default="_v4_router2")
    a = ap.parse_args()
    old = {r["q"]: r for r in json.loads((OUT_DIR / f"e5_e2e_judge{a.base}.json").read_text(encoding="utf-8"))["records"]}
    new_all = json.loads((OUT_DIR / f"e5_e2e_judge{a.tag}.json").read_text(encoding="utf-8"))
    new = {r["q"]: r for r in new_all["records"]}
    qs = [q for q in old if q in new]
    O = [old[q] for q in qs]; N = [new[q] for q in qs]
    out = {"n_paired": len(qs), "base": a.base, "tag": a.tag, "metrics": {}, "tests": {}, "latency_ms": {}, "routes": {}, "per_route": {}}
    for k in ("em", "judge_correct", "judge_faithful"):
        x, y = [r[k] for r in N], [r[k] for r in O]
        b = sum(p and not q for p, q in zip(x, y)); c = sum(q and not p for p, q in zip(x, y))
        out["metrics"][k] = {"router_v1": mean_ci(y), "router_v2": mean_ci(x)}
        out["tests"][k] = {"only_v2": b, "only_v1": c, "mcnemar_p": mcnemar_exact(b, c), "diff_ci": paired_diff_ci(x, y)}
    lo, ln = [float(r["elapsed_ms"] or 0) for r in O], [float(r["elapsed_ms"] or 0) for r in N]
    out["latency_ms"] = {"router_v1": lat(lo), "router_v2": lat(ln), "mean_diff_ci": paired_diff_ci(ln, lo),
                         "median_diff_ci": med_diff_ci(ln, lo), "speedup_median": round(st.median(lo) / st.median(ln), 2)}
    for tag, R in (("router_v1", O), ("router_v2", N)):
        out["routes"][tag] = {k: sum(r["route"] == k for r in R) for k in sorted({r["route"] for r in R})}
        out["per_route"][tag] = {k: {"n": sum(r["route"] == k for r in R),
                                     "em": mean_ci([r["em"] for r in R if r["route"] == k]),
                                     "judge_correct": mean_ci([r["judge_correct"] for r in R if r["route"] == k]),
                                     "latency_ms": lat([float(r["elapsed_ms"] or 0) for r in R if r["route"] == k])}
                                 for k in sorted({r["route"] for r in R})}
    # Bu soruların tamamı tek bir değer istediği için doğru sınıf "simple"dır.
    out["router_accuracy_simple"] = {t: round(sum(r["route"] == "simple" for r in R) / len(R), 4) for t, R in (("router_v1", O), ("router_v2", N))}
    # Yalnızca v2'nin basit sınıfa yönlendirdiği sorularda: aynı soru v1'de Qwen3-8B ile yanıtlanmıştı
    sel = [i for i, r in enumerate(N) if r["route"] == "simple"]
    if sel:
        x, y = [N[i]["em"] for i in sel], [O[i]["em"] for i in sel]
        b = sum(p and not q for p, q in zip(x, y)); c = sum(q and not p for p, q in zip(x, y))
        out["rerouted_subset"] = {"n": len(sel), "em_gemma_v2": mean_ci(x), "em_qwen_v1": mean_ci(y), "mcnemar_p": mcnemar_exact(b, c),
                                  "diff_ci": paired_diff_ci(x, y), "latency_v2": lat([ln[i] for i in sel]), "latency_v1": lat([lo[i] for i in sel])}
    out["judge_validity_v2"] = new_all.get("judge_validity")
    save("e20_router_v2_live.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
