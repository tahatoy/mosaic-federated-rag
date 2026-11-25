#!/usr/bin/env bash
# Q2-02b (E20 devamı): yönlendirici v2 ile Gemma-3-4B'nin yeniden ifade edilmiş sorularda bağlam varken yanıttan kaçınması
# (E20: basit sınıfta 153 sorunun 51'inde "belgelerde bilgi yok") için kademeli yönlendirme ve hafif modele verilen parça sayısı.
# Seçim TEST DIŞI 219 geliştirme sorusunda yapılır (aynı 419'luk kümenin 200 test sorusu dışındaki kısmı); seçilen ayar 200 test
# sorusunda bir kez koşulur, hakemlenir ve ilk yönlendiriciyle karşılaştırılır.
. "$(dirname "${BASH_SOURCE[0]}")/../../../backend/scripts/q1_revision/q2_lib.sh"
durum "Q2-02b başladı"
orch_restart(){ # SIMPLE_FINAL_K ESCALATE
  pkill -f "backend_api[.]py --service orchestrator" 2>/dev/null; fuser -k 8004/tcp >/dev/null 2>&1; sleep 2
  env AUTH_REQUIRED=false PORT=8004 SIMPLE_FINAL_K=$1 ESCALATE_ON_ABSTAIN=$2 setsid nohup python3 backend/backend_api.py --service orchestrator --port 8004 > logs/orch.log 2>&1 < /dev/null &
  for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8004/health >/dev/null && break; sleep 3; done
  curl -s -m 300 -o /dev/null -X POST http://127.0.0.1:8004/query -H 'Content-Type: application/json' -d '{"query":"Kredi kartı asgari ödeme tutarı nasıl hesaplanır?","branch_id":0}'
  echo "orkestratör k=$1 kademeli=$2: $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8004/health)"
}
llm_restore
router_restart ai/models/router/bert_router_v2
for v in "6 false" "3 false" "6 true" "3 true"; do
  set -- $v
  orch_restart $1 $2
  python3 -u $Q/e5_e2e_judge.py --phase collect --offset 200 --n 219 --no-norag --tag _dev_k$1_esc$2 2>&1 | tail -2
done
python3 - <<'PY' > /tmp/q2_choice.txt
import json, statistics as st
R = "ai/experiments/results/q1/"
best = None
for k in (6, 3):
    for e in ("false", "true"):
        r = json.load(open(f"{R}e5_e2e_raw_dev_k{k}_esc{e}.json", encoding="utf-8"))
        em = sum(x["em"] for x in r) / len(r); lat = st.median(x["elapsed_ms"] for x in r)
        esc = sum(bool(x.get("escalated")) for x in r) / len(r)
        print(f"# k={k} esc={e} n={len(r)} EM={em:.4f} median_ms={lat:.0f} escalated={esc:.3f}")
        if best is None or em > best[0] + 1e-9 or (abs(em - best[0]) < 1e-9 and lat < best[1]):
            best = (em, lat, k, e)
print(best[2], best[3])
PY
cat /tmp/q2_choice.txt; cp /tmp/q2_choice.txt $RES/e20_dev_selection.txt
set -- $(tail -1 /tmp/q2_choice.txt)
K=$1; E=$2
durum "E20 geliştirme seçimi: SIMPLE_FINAL_K=$K ESCALATE_ON_ABSTAIN=$E"
orch_restart $K $E
python3 -u $Q/e5_e2e_judge.py --phase collect --n 200 --tag _v4b_router2 --reuse-norag _v3; echo "collect rc=$?"
judge_up
python3 -u $Q/e5_e2e_judge.py --phase judge --tag _v4b_router2; echo "judge rc=$?"
llm_restore
python3 -u $Q/e20_router_v2_live.py --base _v3 --tag _v4b_router2 | head -70
cp $RES/e20_router_v2_live.json $RES/e20_v4b_vs_v3.json
python3 -u $Q/e20_router_v2_live.py --base _v3 --tag _v4_router2 > /dev/null
python3 $Q/set_env.py .env.hq SIMPLE_FINAL_K=$K ESCALATE_ON_ABSTAIN=$E MODEL_PATH=ai/models/router/bert_router_v2
cp .env.hq .env
orch_restart $K $E
durum "Q2-02b bitti (kalıcı: router v2, SIMPLE_FINAL_K=$K, ESCALATE_ON_ABSTAIN=$E)"
