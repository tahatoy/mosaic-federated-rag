#!/usr/bin/env bash
# Q1 revizyon — son yapılandırma (MOSAIC-Embed) değerlendirmesi: kanarya, 500 + 352 soru (B, C, D_me), canlı 200 soru + hakem, eşleştirilmiş karşılaştırmalar (E14).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
[ -f .venv/bin/activate ] && . .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/ai:$PWD/backend" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="$PWD/ai/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"
Q=backend/scripts/q1_revision
LS=ai/llama.cpp/build/bin/llama-server
echo "başladı: $(date -Is)"
llm_stop(){ pkill -9 -f "llama-server" ; for i in $(seq 1 20); do pgrep -f llama-server >/dev/null || break; sleep 1; done; sleep 2
  echo "llm_stop: kalan=$(pgrep -fc llama-server) GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; }
llm_up(){ local p=$1 m=$2; shift 2
  setsid nohup $LS -m "$m" --host 127.0.0.1 --port "$p" -ngl 99 "$@" > "logs/llm_$p.log" 2>&1 < /dev/null &
  for i in $(seq 1 80); do curl -sf -m 3 "http://127.0.0.1:$p/health" >/dev/null && return 0; sleep 3; done; echo "ZAMAN AŞIMI $p"; return 1; }
llm_restore(){ llm_stop; llm_up 8099 ai/models/finetuned/GEMMA3_FINANCE_V1.Q4_K_M.gguf -c 4096 -np 1; llm_up 8098 ai/models/finetuned/QWEN3_EXPERT_V2.Q4_K_M.gguf -c 4096 -np 1
  echo "LLM geri yüklendi: 8099=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8099/health) 8098=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8098/health)"; }
set -a; . ./.env.hq; set +a
KEY=$HOME/.ssh/mosaic_vm
SSHO=(-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.ssh/mosaic_known" -o ConnectTimeout=8 -o BatchMode=yes -i "$KEY")
b=1
for ip in 10.0.0.11 10.0.0.12; do   # şube servisleri MOSAIC-Embed'i yüklemiş olmalı (yalnızca anahtar adına güvenme)
  ssh "${SSHO[@]}" mosaic2@$ip "grep -h 'Loading embedding' ~/Mosaic/logs/rag_tls_800$b.log | tail -n 1" | grep -q turkembed_v3 || { echo "HATA: şube $b MOSAIC-Embed yüklemedi"; exit 1; }
  b=$((b+1))
done
python3 -u backend/scripts/hypothesis_tests/h2_canary.py wire --tag sonra_me | tail -n 3
python3 -u $Q/e2_crossbranch_generation.py --n 500 --arms "B,C,D_me" --natural-arms "C,D_me" --extra "" --out e2_generation_me.json; echo "e2 rc=$?"
python3 -u $Q/e5_e2e_judge.py --phase collect --n 200 --tag _v3; echo "e5 collect rc=$?"
llm_stop; llm_up 8096 ai/models/quantized/qwen3-8b-Q4_K_M.gguf -c 8192 -np 1
python3 -u $Q/e5_e2e_judge.py --phase judge --tag _v3; echo "e5 judge rc=$?"
llm_restore
python3 -u backend/scripts/hypothesis_tests/h2_canary.py e2e --tag sonra_me | tail -n 3
(cd $Q && python3 -u e14_mosaic_embed_summary.py); echo "e14 rc=$?"
echo "bitti: $(date -Is)"
