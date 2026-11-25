#!/usr/bin/env bash
# Q2-02 (E20): düzeltilmiş yönlendirici (bert_router_v2) ile canlı 200 soru, hakem, ilk yönlendiriciyle eşleştirilmiş karşılaştırma.
. "$(dirname "${BASH_SOURCE[0]}")/../../../backend/scripts/q1_revision/q2_lib.sh"
durum "Q2-02 E20 başladı"
system_check >/dev/null || { bash demo_baslat.sh 2>&1 | tail -15; }
router_restart ai/models/router/bert_router_v2
python3 -u $Q/e5_e2e_judge.py --phase collect --n 200 --tag _v4_router2 --reuse-norag _v3; echo "collect rc=$?"
judge_up
python3 -u $Q/e5_e2e_judge.py --phase judge --tag _v4_router2; echo "judge rc=$?"
llm_restore
python3 -u $Q/e20_router_v2_live.py --base _v3 --tag _v4_router2 | head -60
if [ -f $RES/e5_e2e_judge_v4_router2.json ]; then
  cp -n .env.hq backups/env_hq_router_v1_bak_$(date +%Y%m%d)
  python3 $Q/set_env.py .env.hq MODEL_PATH=ai/models/router/bert_router_v2
  cp .env.hq .env
  durum "E20 tamam; yönlendirici v2 kalıcı olarak etkin (.env.hq MODEL_PATH)"
else
  durum "HATA: E20 sonucu yok"
fi
durum "Q2-02 bitti"
