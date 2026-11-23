#!/usr/bin/env bash
# Q1 revizyon deneyi. Herhangi bir dizinden çalıştırılabilir; betik proje köküne geçer.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
[ -f .venv/bin/activate ] && . .venv/bin/activate
set -a; . ./.env.hq; set +a
export PYTHONPATH="$PWD:$PWD/ai:$PWD/backend" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="$PWD/ai/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"
Q=backend/scripts/q1_revision
echo "başladı: $(date -Is)"
LS=ai/llama.cpp/build/bin/llama-server
llm_stop(){ pkill -9 -f "llama-server" ; for i in $(seq 1 20); do pgrep -f llama-server >/dev/null || break; sleep 1; done; sleep 2
  echo "llm_stop: kalan llama-server=$(pgrep -fc llama-server) GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; }
llm_up(){ # port model
  setsid nohup $LS -m "$2" --host 127.0.0.1 --port "$1" -ngl 99 -c "${3:-4096}" -np 1 > "logs/llm_$1.log" 2>&1 < /dev/null &
  for i in $(seq 1 80); do curl -sf -m 3 "http://127.0.0.1:$1/health" >/dev/null && return 0; sleep 3; done; echo "ZAMAN AŞIMI $1"; }
llm_restore(){ llm_stop; llm_up 8099 ai/models/finetuned/GEMMA3_FINANCE_V1.Q4_K_M.gguf; llm_up 8098 ai/models/finetuned/QWEN3_EXPERT_V2.Q4_K_M.gguf
  echo "LLM geri yüklendi: 8099=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8099/health) 8098=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8098/health)"; }
llm_stop
python3 -u $Q/e4_router_baselines_fl.py --seeds 0,1,2 --rounds 5; rc=$?
llm_restore
echo "bitti: $(date -Is) rc=$rc"; exit $rc
