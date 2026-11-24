#!/usr/bin/env bash
# Q1 revizyon (E18, E19) — yönlendirici v2: rapor dilinde eşleştirilmiş sorular, yeniden eğitim ve değerlendirme (genel merkez GPU'su).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
mkdir -p logs
LOG=logs/router_v2.log; exec > >(tee -a "$LOG") 2>&1
echo "=== Yönlendirici v2: $(date -Is) ==="
[ -f .venv/bin/activate ] && . .venv/bin/activate
set -a; . ./.env.hq; set +a
export PYTHONPATH="$PWD:$PWD/ai:$PWD/backend" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="$PWD/ai/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"
LS=ai/llama.cpp/build/bin/llama-server
STARTED=0
if ! curl -sf -m 5 http://127.0.0.1:8098/health >/dev/null; then
  echo "Qwen3-8B (8098) kapalı, başlatılıyor..."
  setsid nohup $LS -m ai/models/finetuned/QWEN3_EXPERT_V2.Q4_K_M.gguf --host 127.0.0.1 --port 8098 -ngl 99 -c 4096 -np 4 > logs/llm_8098_router_v2.log 2>&1 < /dev/null &
  for i in $(seq 1 80); do curl -sf -m 3 http://127.0.0.1:8098/health >/dev/null && break; sleep 3; done
  curl -sf -m 3 http://127.0.0.1:8098/health >/dev/null || { echo "HATA: Qwen3-8B başlamadı (logs/llm_8098_router_v2.log)"; exit 1; }
  STARTED=1
fi
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
cd backend/scripts/q1_revision && python3 -u e18_router_v2.py "$@" && python3 -u e19_router_per_class.py bert_router_best bert_router_v2; RC=$?
cd - >/dev/null
pkill -f "llama-serve[r] .*--port 8098 .*-np 4" 2>/dev/null  # yalnızca bu betiğin açtığı (-np 4) Qwen sunucusu
echo "=== bitti rc=$RC: $(date -Is) — ai/experiments/results/q1/e18_router_v2.json, e19_router_per_class.json ==="
