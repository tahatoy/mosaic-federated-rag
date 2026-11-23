#!/usr/bin/env bash
# Ek kıyaslama: modelin kendi sohbet şablonuyla TR-MMLU + ikili duygu. run_all_experiments.sh bittikten sonra çalıştırın.
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
trap '' HUP
R=ai/experiments/results; LOG=$R/run_extra.log; mkdir -p logs; exec > >(tee -a "$LOG") 2>&1
echo "=== Ek kıyaslama başladı: $(date -Is) ==="
[ -f .venv/bin/activate ] && source .venv/bin/activate
export LD_LIBRARY_PATH="$(pwd)/ai/llama.cpp/build/bin:${LD_LIBRARY_PATH}"
LS=ai/llama.cpp/build/bin/llama-server
wait_http(){ for i in $(seq 1 $2); do curl -sf -m 3 "$1" >/dev/null && return 0; sleep 3; done; echo "ZAMAN AŞIMI: $1"; return 1; }
for spec in "qwen:QWEN3_EXPERT_V2.Q4_K_M.gguf:8198" "gemma:GEMMA3_FINANCE_V1.Q4_K_M.gguf:8199"; do
  IFS=: read name gguf port <<< "$spec"
  fuser -k ${port}/tcp 2>/dev/null
  $LS -m ai/models/finetuned/$gguf --port $port -ngl 99 -c 4096 -np 1 --jinja > logs/extra_$name.log 2>&1 & pid=$!
  if wait_http http://127.0.0.1:$port/health 60; then
    python3 backend/scripts/hypothesis_tests/bench_chat_native.py --url http://127.0.0.1:$port --name $name
  else tail -20 logs/extra_$name.log; fi
  kill $pid 2>/dev/null; sleep 3
done
echo "=== Ek kıyaslama BİTTİ: $(date -Is) ==="
