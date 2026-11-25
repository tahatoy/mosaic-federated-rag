#!/usr/bin/env bash
# Q1 ikinci tur (q2_*) iş betiklerinin ortak yardımcıları. İş kuyruğu (job_runner.sh) proje kökünde çalıştırır.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
ROOTDIR=$PWD
[ -f .venv/bin/activate ] && . .venv/bin/activate
set -a; . ./.env.hq; set +a
export PYTHONPATH="$PWD:$PWD/ai:$PWD/backend:$PWD/backend/scripts/q1_revision" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="$PWD/ai/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"
Q=backend/scripts/q1_revision
RES=ai/experiments/results/q1
LS=ai/llama.cpp/build/bin/llama-server
KEY=$HOME/.ssh/mosaic_vm
SSHO=(-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.ssh/mosaic_known" -o ConnectTimeout=8 -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=20 -i "$KEY")
VMS=(10.0.0.11 10.0.0.12)
HQIP=${HQ_LAN_IP:-10.0.0.10}
MT=(--cacert backend/certs/ca.crt --cert backend/certs/hq_client.crt --key backend/certs/hq_client.key)
mkdir -p logs backups
DURUM=logs/q2_durum.txt
durum(){ echo "[$(date '+%F %T')] $*" | tee -a "$DURUM"; }

llm_stop(){ pkill -9 -f "llama-server" ; for i in $(seq 1 20); do pgrep -f llama-server >/dev/null || break; sleep 1; done; sleep 2
  echo "llm_stop: kalan llama-server=$(pgrep -fc llama-server) GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; }
llm_up(){ # port model [ctx]
  setsid nohup $LS -m "$2" --host 127.0.0.1 --port "$1" -ngl 99 -c "${3:-4096}" -np 1 > "logs/llm_$1.log" 2>&1 < /dev/null &
  for i in $(seq 1 80); do curl -sf -m 3 "http://127.0.0.1:$1/health" >/dev/null && return 0; sleep 3; done; echo "ZAMAN AŞIMI $1"; return 1; }
llm_restore(){ llm_stop; llm_up 8099 ai/models/finetuned/GEMMA3_FINANCE_V1.Q4_K_M.gguf; llm_up 8098 ai/models/finetuned/QWEN3_EXPERT_V2.Q4_K_M.gguf
  echo "LLM geri yüklendi: 8099=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8099/health) 8098=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8098/health)"; }
judge_up(){ llm_stop; llm_up 8096 ai/models/quantized/qwen3-8b-Q4_K_M.gguf 8192; }

router_restart(){ # yönlendirici servisini .env.hq'daki (ya da verilen) MODEL_PATH ile yeniden başlatır
  local mp=${1:-$MODEL_PATH}
  pkill -f "backend_api[.]py --service router" 2>/dev/null; fuser -k 8000/tcp >/dev/null 2>&1; sleep 2
  env MODEL_PATH="$mp" BIND_HOST=127.0.0.1 PORT=8000 setsid nohup python3 backend/backend_api.py --service router --port 8000 > logs/router.log 2>&1 < /dev/null &
  for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null && break; sleep 3; done
  echo "yönlendirici ($mp): $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)"
  curl -s -m 10 -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
    -d '{"text":"Apache Corporation 2023 mali yılında piyasa değeri yaklaşık kaç milyar dolardı?"}'; echo
}

system_check(){ # 0: canlı hat hazır
  local ok=0
  for b in 1 2; do v=BRANCH${b}_RAG_URL; c=$(curl -s -m 8 "${MT[@]}" "${!v}/stats"); echo "şube $b: $c"; echo "$c" | grep -q collection_count || ok=1; done
  echo "HQ: $(curl -s -m 5 http://127.0.0.1:8003/stats)"
  for u in 8000/health 8003/health 8004/health 8098/health 8099/health; do
    c=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$u"); echo "  $u -> $c"; [ "$c" = 200 ] || ok=1; done
  return $ok
}

vm(){ ssh "${SSHO[@]}" "mosaic2@$1" "${@:2}"; }
