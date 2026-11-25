#!/usr/bin/env bash
# Q2-02c (E20 devamı): ikinci yükseltme tetiği — hafif modelin yanıtındaki sayı bağlamda yoksa (desteklenmeyen sayı) da yükselt.
# Yalnızca 219 geliştirme sorusunda Q2-02b'nin seçtiği ayarla karşılaştırılır; geliştirmede daha iyiyse test kümesinde BİR KEZ koşulur.
. "$(dirname "${BASH_SOURCE[0]}")/../../../backend/scripts/q1_revision/q2_lib.sh"
durum "Q2-02c başladı"
orch_restart(){ # K ESC UNSUP
  pkill -f "backend_api[.]py --service orchestrator" 2>/dev/null; fuser -k 8004/tcp >/dev/null 2>&1; sleep 2
  env AUTH_REQUIRED=false PORT=8004 SIMPLE_FINAL_K=$1 ESCALATE_ON_ABSTAIN=$2 ESCALATE_UNSUPPORTED=$3 setsid nohup python3 backend/backend_api.py --service orchestrator --port 8004 > logs/orch.log 2>&1 < /dev/null &
  for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8004/health >/dev/null && break; sleep 3; done
  curl -s -m 300 -o /dev/null -X POST http://127.0.0.1:8004/query -H 'Content-Type: application/json' -d '{"query":"Kredi kartı asgari ödeme tutarı nasıl hesaplanır?","branch_id":0}'
  echo "orkestratör k=$1 kaçınma=$2 desteksiz=$3: $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8004/health)"
}
system_check >/dev/null || llm_restore
set -- $(tail -1 $RES/e20_dev_selection.txt); K=$1; E=$2
orch_restart $K true true
python3 -u $Q/e5_e2e_judge.py --phase collect --offset 200 --n 219 --no-norag --tag _dev_k${K}_esctrue_unsup 2>&1 | tail -2
python3 - <<PY > /tmp/q2c.txt
import json, statistics as st
R = "ai/experiments/results/q1/"
def sc(t):
    r = json.load(open(R + f"e5_e2e_raw_{t}.json", encoding="utf-8"))
    return sum(x["em"] for x in r) / len(r), st.median(x["elapsed_ms"] for x in r), sum(bool(x.get("escalated")) for x in r) / len(r)
a = sc("dev_k${K}_esc${E}"); b = sc("dev_k${K}_esctrue_unsup")
print(f"# önceki seçim k=${K} esc=${E}: EM={a[0]:.4f} ortanca={a[1]:.0f} yükseltme={a[2]:.3f}")
print(f"# + desteklenmeyen sayı tetiği: EM={b[0]:.4f} ortanca={b[1]:.0f} yükseltme={b[2]:.3f}")
print("yeni" if b[0] > a[0] + 1e-9 else "eski")
PY
cat /tmp/q2c.txt; cat /tmp/q2c.txt >> $RES/e20_dev_selection.txt
if [ "$(tail -1 /tmp/q2c.txt)" = yeni ]; then
  orch_restart $K true true
  python3 -u $Q/e5_e2e_judge.py --phase collect --n 200 --tag _v4c_router2 --reuse-norag _v3; echo "collect rc=$?"
  judge_up
  python3 -u $Q/e5_e2e_judge.py --phase judge --tag _v4c_router2; echo "judge rc=$?"
  llm_restore
  python3 -u $Q/e20_router_v2_live.py --base _v3 --tag _v4c_router2 | head -40
  cp $RES/e20_router_v2_live.json $RES/e20_v4c_vs_v3.json
  python3 $Q/set_env.py .env.hq SIMPLE_FINAL_K=$K ESCALATE_ON_ABSTAIN=true ESCALATE_UNSUPPORTED=true
  cp .env.hq .env
  durum "Q2-02c: desteklenmeyen sayı tetiği geliştirmede daha iyi; test koşuldu ve kalıcı yapıldı"
else
  python3 $Q/set_env.py .env.hq ESCALATE_UNSUPPORTED=false; cp .env.hq .env
  durum "Q2-02c: ikinci tetik geliştirmede iyileştirmedi; Q2-02b ayarı kalıyor"
fi
set -a; . ./.env.hq; set +a
orch_restart ${SIMPLE_FINAL_K:-$K} ${ESCALATE_ON_ABSTAIN:-$E} ${ESCALATE_UNSUPPORTED:-false}
durum "Q2-02c bitti"
