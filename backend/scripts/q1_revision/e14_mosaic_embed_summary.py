"""E14: paired comparisons for the final configuration with MOSAIC-Embed.

Compares the live MOSAIC-Embed run (e2_generation_me.json, e5_e2e_raw_v3.json) with the base e5-small run
(e2_generation_v2.json, e5_e2e_raw_v2.json) and the first configuration (e5_e2e_raw.json) on the same questions.
Output: ai/experiments/results/q1/e14_mosaic_embed_live_summary.json
"""
import json

from q1_common import OUT_DIR, mcnemar_exact, mean_ci, norm_num, paired_diff_ci, questions


def em(items, answers):
    return [int(q["gold_norm"] in norm_num(a)) for q, a in zip(items, answers)]


def paired(x, y):
    b = sum(p and not r for p, r in zip(x, y)); c = sum(r and not p for p, r in zip(x, y))
    return {"only_first": b, "only_second": c, "mcnemar_p": mcnemar_exact(b, c), "diff_ci": paired_diff_ci(x, y)}


def main():
    qs, _, _ = questions(500)
    para = {r["reference"]: r["natural"] for r in json.loads((OUT_DIR / "e0_paraphrases.json").read_text(encoding="utf-8"))["kayitlar"] if r["natural"]}
    sets = {"cloze": qs, "natural": [q for q in qs if q["reference"] in para]}
    me = json.loads((OUT_DIR / "e2_generation_me.json").read_text(encoding="utf-8"))
    v2 = json.loads((OUT_DIR / "e2_generation_v2.json").read_text(encoding="utf-8"))
    out = {"source": "e2_generation_me.json (live MOSAIC-Embed services) vs e2_generation_v2.json (live base e5 services, arm A no retrieval)",
           "tests": {}, "arms": {}}
    for s, items in sets.items():
        a = em(items, v2["kayitlar"][s]["A"]); ce5 = em(items, v2["kayitlar"][s]["C"])
        cme = em(items, me["kayitlar"][s]["C"]); dme = em(items, me["kayitlar"][s]["D_me"])
        out["arms"][s] = {"A_v2": mean_ci(a), "C_e5": mean_ci(ce5), "C_me": mean_ci(cme), "D_me": mean_ci(dme)}
        out["tests"][f"{s}_C_me_vs_A"] = paired(cme, a)
        out["tests"][f"{s}_C_me_vs_C_e5"] = paired(cme, ce5)
        out["tests"][f"{s}_C_me_vs_D_me"] = paired(cme, dme)
    l3 = json.loads((OUT_DIR / "e5_e2e_raw_v3.json").read_text(encoding="utf-8"))
    l2 = {r["reference"]: r for r in json.loads((OUT_DIR / "e5_e2e_raw_v2.json").read_text(encoding="utf-8"))}
    l1 = {r["reference"]: r for r in json.loads((OUT_DIR / "e5_e2e_raw.json").read_text(encoding="utf-8"))}
    a = [r["em"] for r in l3]; z = [r["norag_em"] for r in l3]
    out["live"] = {"n": len(a), "em": mean_ci(a), "norag_em": mean_ci(z),
                   "mean_latency_ms": sum(r["elapsed_ms"] for r in l3) / len(l3),
                   "routes": {k: sum(r["route"] == k for r in l3) for k in {r["route"] for r in l3}}}
    out["tests"]["live_vs_norag"] = paired(a, z)
    out["tests"]["live_me_vs_live_e5_run2"] = paired(a, [l2[r["reference"]]["em"] for r in l3])
    out["tests"]["live_me_vs_live_first"] = paired(a, [l1[r["reference"]]["em"] for r in l3])
    root = OUT_DIR.parent
    out["canary"] = {"wire": json.loads((root / "h2_canary_wire_sonra_me.json").read_text(encoding="utf-8")).get("hatta_açık_sır"),
                     "e2e": json.loads((root / "h2_canary_e2e_sonra_me.json").read_text(encoding="utf-8")).get("cevapta_sır_olan")}
    (OUT_DIR / "e14_mosaic_embed_live_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["arms"], ensure_ascii=False))


if __name__ == "__main__":
    main()
