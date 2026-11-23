#!/usr/bin/env bash
# Q1 revizyon — E12: MOSAIC-Embed eğitimi. Qwen3-8B ile sorgu üretimi (2 × 6.000 parça), 6 dönem eğitim, geliştirme kümesiyle dönem seçimi, 1.019 soruluk erişim değerlendirmesi.
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
llm_stop
if llm_up 8098 ai/models/finetuned/QWEN3_EXPERT_V2.Q4_K_M.gguf -c 12288 -np 6 --jinja; then
  python3 -u $Q/e12a_gen_train_queries.py --n 6000 --workers 6 --max-minutes 110; echo "e12a (1) rc=$?"
  python3 -u $Q/e12a_gen_train_queries.py --n 6000 --offset 6000 --append --workers 6 --max-minutes 110; echo "e12a (2) rc=$?"
else tail -20 logs/llm_8098.log; fi
llm_stop
[ -s ai/data/e12/train_queries.jsonl ] || { echo "eğitim sorgusu yok, durduruluyor"; llm_restore; exit 2; }
python3 -u $Q/e12b_train_turkembed_v2.py --epochs 6 --lr 2e-5 --batch 64 --out turkembed_v3; echo "e12b rc=$?"
python3 -u $Q/e12c_eval_turkembed_v2.py turkembed_v3; echo "e12c rc=$?"
llm_restore
echo "bitti: $(date -Is)"
