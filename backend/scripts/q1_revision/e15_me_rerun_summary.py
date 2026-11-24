"""
E15 — İlk alana özgü gömme modelinin çıkarılmasından sonraki yeniden koşuların özeti (MOSAIC-Embed son yapılandırma).

Girdi: e2_generation_me_full.json (A,B,C,D_me,D_me_noce,D_e5; doğal A,C,D_me), e2_generation_me_quant.json (C bağlamı,
taban Gemma Q4_K_M / Q2_K), e2_generation_me_nomask.json (C, kaynakta maskeleme kapalı), e2_generation_me.json (önceki C koşusu).
Dosyalar arası eşleştirilmiş testler, kayıtlı yanıtlardan soru başına tam eşleşme yeniden hesaplanarak yapılır
(aynı 500 soru, aynı sıra). Çıktı: ai/experiments/results/q1/e15_me_rerun_summary.json
"""
import json

from q1_common import OUT_DIR, mcnemar_exact, mean_ci, norm_num, paired_diff_ci, questions, save


def load(name):
    p = OUT_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def em_vec(ans, qs):
    return [int(q["gold_norm"] in norm_num(a)) for a, q in zip(ans, qs)]


def test(x, y):
    b = sum(p and not r for p, r in zip(x, y)); c = sum(r and not p for p, r in zip(x, y))
    return {"only_first": b, "only_second": c, "mcnemar_p": mcnemar_exact(b, c), "diff_ci": paired_diff_ci(x, y)}


def main():
    qs, _, _ = questions(500)
    full, quant, nomask, prev = (load(n) for n in ("e2_generation_me_full.json", "e2_generation_me_quant.json",
                                                    "e2_generation_me_nomask.json", "e2_generation_me.json"))
    out = {"tables": {}, "tests": {}, "checks": {}}
    em = {}
    if full:
        out["tables"]["ablation_cloze"] = full["sonuclar"]["cloze"]
        out["tables"]["natural"] = full["sonuclar"].get("natural")
        out["tests"]["within_full_cloze"] = full["testler"]["cloze"]
        out["tests"]["within_full_natural"] = full["testler"].get("natural")
        for arm, ans in full["kayitlar"]["cloze"].items():
            em[arm] = em_vec(ans, qs)
        out["checks"]["em_recomputed_matches"] = {arm: round(sum(v) / len(v), 4) for arm, v in em.items()}
    if quant:
        out["tables"]["quant_cloze"] = quant["sonuclar"]["cloze"]
        for arm, ans in quant["kayitlar"]["cloze"].items():
            em[arm] = em_vec(ans, qs)
    if nomask:
        out["tables"]["nomask_cloze"] = nomask["sonuclar"]["cloze"]
        em["C_nomask"] = em_vec(nomask["kayitlar"]["cloze"]["C"], qs)
    if prev:
        em["C_prev"] = em_vec(prev["kayitlar"]["cloze"]["C"], qs)
    for x, y in [("C", "C_base_q4km"), ("C_base_q4km", "C_base_q2k"), ("C", "C_base_q2k"), ("C", "C_nomask"), ("C", "C_prev")]:
        if x in em and y in em:
            out["tests"][f"{x}_vs_{y}"] = test(em[x], em[y])
    out["em"] = {k: mean_ci(v) for k, v in em.items()}
    save("e15_me_rerun_summary.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "tables"}, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
