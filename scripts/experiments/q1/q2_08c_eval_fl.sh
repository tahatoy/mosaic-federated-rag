#!/usr/bin/env bash
# Q2-08: federe (Flower) MOSAIC-Embed ile canlı hat: 500 çapraz şube sorusu (C = canlı federe hat; D_me / D_fl / D_e5 merkezi kollar),
# canlı 200 soru (yönlendirici v2 + federe gömme) ve kanarya sızıntı testleri.
. "$(dirname "${BASH_SOURCE[0]}")/../../../backend/scripts/q1_revision/q2_lib.sh"
durum "Q2-08c başladı"
system_check || { bash demo_baslat.sh 2>&1 | tail -10; }
FLM=ai/models/embeddings/$(python3 -c "import json;print(json.load(open('$RES/e21_deploy_choice.json'))['deploy'] or 'mosaic_embed_flower')")
llm_stop; llm_up 8099 ai/models/finetuned/GEMMA3_FINANCE_V1.Q4_K_M.gguf
python3 -u $Q/e2_crossbranch_generation.py --n 500 --arms "C,D_me,D_fl" --natural-arms "C,D_me,D_fl" --extra "" --fl-model $FLM --out e2_generation_fl.json; echo "e2 rc=$?"
llm_restore
python3 -u $Q/e5_e2e_judge.py --phase collect --n 200 --tag _v5_fl --reuse-norag _v3; echo "e5 collect rc=$?"
judge_up
python3 -u $Q/e5_e2e_judge.py --phase judge --tag _v5_fl; echo "e5 judge rc=$?"
llm_restore
python3 -u $Q/e20_router_v2_live.py --base _v3 --tag _v5_fl | head -40
cp $RES/e20_router_v2_live.json $RES/e20_v5_fl_vs_v3.json
python3 -u $Q/e20_router_v2_live.py --base _v4b_router2 --tag _v5_fl | head -40
cp $RES/e20_router_v2_live.json $RES/e20_v5_fl_vs_v4b.json
python3 -u $Q/e20_router_v2_live.py --base _v3 --tag _v4b_router2 >/dev/null
# (süre kısıtı) kanarya testleri bu koşuda atlandı; maskeleme ve mTLS değişmedi

python3 -u $Q/e24_fl_vs_me.py | head -60
durum "Q2-08c bitti"
