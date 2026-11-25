"""E21 sonu: federe (Flower, VM) MOSAIC-Embed ile canlı hattın 500 + 352 soruluk çapraz şube sonuçlarının, merkezi MOSAIC-Embed'li
son yapılandırmayla (e2_generation_me.json) aynı sorularda eşleştirilmiş karşılaştırması (McNemar, eşleştirilmiş bootstrap)."""
import json
from q1_common import OUT_DIR, mcnemar_exact, mean_ci, norm_num, paired_diff_ci, questions, save

def ems(rec, qs, para=None):
    return [int(q["gold_norm"] in norm_num(a)) for q, a in zip(qs, rec)]

qs, _, _ = questions(500)
pp = json.loads((OUT_DIR / "e0_paraphrases.json").read_text(encoding="utf-8"))
para = {r["reference"]: r["natural"] for r in pp["kayitlar"] if r["natural"]}
nq = [q for q in qs if q["reference"] in para]
fl = json.loads((OUT_DIR / "e2_generation_fl.json").read_text(encoding="utf-8"))
me = json.loads((OUT_DIR / "e2_generation_me.json").read_text(encoding="utf-8"))
out = {}
for s, Q in (("cloze", qs), ("natural", nq)):
    row = {}
    for a_fl, a_me in (("C", "C"), ("D_fl", "D_me")):
        if a_fl in fl["kayitlar"].get(s, {}) and a_me in me["kayitlar"].get(s, {}):
            x, y = ems(fl["kayitlar"][s][a_fl], Q), ems(me["kayitlar"][s][a_me], Q)
            b = sum(p and not q for p, q in zip(x, y)); c = sum(q and not p for p, q in zip(x, y))
            row[f"{a_fl}_flower_vs_{a_me}_central"] = {"n": len(x), "em_flower": mean_ci(x), "em_central": mean_ci(y), "only_flower": b, "only_central": c,
                                                      "mcnemar_p": mcnemar_exact(b, c), "diff_ci": paired_diff_ci(x, y)}
    row["fl_run_summary"] = fl["sonuclar"].get(s)
    out[s] = row
save("e24_fl_vs_me.json", out)
print(json.dumps(out, ensure_ascii=False, indent=1))
