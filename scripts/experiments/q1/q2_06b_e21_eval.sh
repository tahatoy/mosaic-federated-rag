#!/usr/bin/env bash
# Q2-06 (E21): tüm gömme modellerinin çapraz şube erişim değerlendirmesi ve canlıya alınacak federe modelin seçimi.
. "$(dirname "${BASH_SOURCE[0]}")/../../../backend/scripts/q1_revision/q2_lib.sh"
durum "Q2-06 başladı"
llm_stop
python3 -u $Q/e21_flower_mosaic_embed.py eval --models mosaic_embed_central_frozen,mosaic_embed_flower,mosaic_embed_flower_vm; echo "eval rc=$?"
python3 - <<'PY'
import json
d = json.load(open("ai/experiments/results/q1/e21_flower_mosaic_embed.json", encoding="utf-8"))
ch = None
for nm in ("mosaic_embed_flower_vm", "mosaic_embed_flower"):
    k = f"{nm}_vs_base_e5_small_cloze"
    if k in d["diffs"]:
        diff, ci = d["diffs"][k]
        print(nm, "taban farkı (kalıp):", diff, ci, "| yeniden ifade:", d["diffs"].get(f"{nm}_vs_base_e5_small_natural"))
        if ci[0] > 0 and ch is None:
            ch = nm
json.dump({"deploy": ch}, open("ai/experiments/results/q1/e21_deploy_choice.json", "w"))
print("CANLIYA ALINACAK:", ch)
PY
durum "Q2-06 bitti: $(cat $RES/e21_deploy_choice.json)"
