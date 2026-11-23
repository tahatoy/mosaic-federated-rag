#!/usr/bin/env bash
# Q1 revizyon — E10: TR-MMLU (Bayram vd.) sıfır-örnek, iki model × iki istem koşulu.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
[ -f .venv/bin/activate ] && . .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/ai:$PWD/backend" PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$PWD/ai/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"
Q=backend/scripts/q1_revision
D=ai/data/benchmarks/tr_mmlu_bayram/tr_mmlu_6200.parquet
LS=ai/llama.cpp/build/bin/llama-server
echo "başladı: $(date -Is)"
[ -f "$D" ] || { echo "VERİ YOK: $D"; exit 3; }
python3 -u $Q/e10_tr_mmlu.py --data $D --url x --name x --mode chatml --inspect | head -40
llm_stop(){ pkill -9 -f "llama-server" ; for i in $(seq 1 20); do pgrep -f llama-server >/dev/null || break; sleep 1; done; sleep 2
  echo "llm_stop: kalan=$(pgrep -fc llama-server) GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; }
llm_up(){ # port model [ek argümanlar]
  local p=$1 m=$2; shift 2
  setsid nohup $LS -m "$m" --host 127.0.0.1 --port "$p" -ngl 99 -c 4096 -np 1 "$@" > "logs/llm_$p.log" 2>&1 < /dev/null &
  for i in $(seq 1 80); do curl -sf -m 3 "http://127.0.0.1:$p/health" >/dev/null && return 0; sleep 3; done; echo "ZAMAN AŞIMI $p"; return 1; }
llm_restore(){ llm_stop; llm_up 8099 ai/models/finetuned/GEMMA3_FINANCE_V1.Q4_K_M.gguf; llm_up 8098 ai/models/finetuned/QWEN3_EXPERT_V2.Q4_K_M.gguf
  echo "LLM geri yüklendi: 8099=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8099/health) 8098=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8098/health)"; }
for spec in "qwen:QWEN3_EXPERT_V2.Q4_K_M.gguf:8198" "gemma:GEMMA3_FINANCE_V1.Q4_K_M.gguf:8199"; do
  IFS=: read name gguf port <<< "$spec"
  llm_stop
  if llm_up $port ai/models/finetuned/$gguf --jinja; then
    for mode in chatml native; do
      python3 -u $Q/e10_tr_mmlu.py --data $D --url http://127.0.0.1:$port --name $name --mode $mode | tail -25
      echo "$name/$mode rc=$?"
    done
  else tail -20 logs/llm_$port.log; fi
done
llm_restore
echo "bitti: $(date -Is)"
