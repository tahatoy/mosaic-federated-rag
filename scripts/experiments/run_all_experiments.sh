#!/usr/bin/env bash
# MOSAIC-v2 hipotez deneyleri: tek komutla (WSL). Sonuçlar: ai/experiments/results/ ve run_all.log
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
trap '' HUP   # pencere kapanırsa da süreçler sürsün
ROOT="$(pwd)"; R=ai/experiments/results; mkdir -p "$R" logs
LOG="$R/run_all.log"; exec > >(tee -a "$LOG") 2>&1
echo "=== MOSAIC deneyleri başladı: $(date -Is) ==="
[ -f .venv/bin/activate ] && source .venv/bin/activate
export LD_LIBRARY_PATH="$(pwd)/ai/llama.cpp/build/bin:${LD_LIBRARY_PATH}"
PY=python3; PIDS=()
cleanup(){ echo "[temizlik] servisler durduruluyor"; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "UYARI: nvidia-smi yok"
# tanı: WSL bellek/GPU kaydı (kapanma nedenini görmek için)
( while true; do echo "$(date +%T) $(free -m | awk '/Mem:/{print "used="$3"MB avail="$7"MB"}') $(free -m | awk '/Swap:/{print "swap="$3"MB"}') $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"; sleep 30; done >> logs/mem.log 2>&1 ) & PIDS+=($!)

wait_http(){ for i in $(seq 1 $2); do curl -sf -m 3 "$1" >/dev/null && return 0; sleep 3; done; echo "ZAMAN AŞIMI: $1"; return 1; }
step(){ echo; echo "################ $1 ################"; }

step "0) Eski süreçler temizleniyor"
for p in $(pgrep -f run_all_experiments.sh); do [ "$p" != "$$" ] && [ "$p" != "$PPID" ] && kill "$p" 2>/dev/null; done; sleep 2
for port in 8000 8001 8002 8003 8004 8098 8099; do fuser -k ${port}/tcp 2>/dev/null; done; sleep 2

step "1) ChromaDB indeksleri (yoksa kurulur; TurkEmbed ince ayarlı model, GPU)"
$PY - <<'E' > /tmp/idx_state.txt
import chromadb
need=[("ai/data/chromadb/global","finance_global"),("ai/data/chromadb/branch1","finance_branch1"),("ai/data/chromadb/branch2","finance_branch2")]
ok=True
for p,n in need:
    try: ok &= chromadb.PersistentClient(path=p).get_collection(n).count()>1000
    except Exception: ok=False
print("OK" if ok else "BUILD")
E
if grep -q BUILD /tmp/idx_state.txt; then
  $PY backend/scripts/data_tool.py index --use-gpu --embed-model ai/models/embeddings/turkembed_finetuned || echo "İNDEKS HATASI"
else echo "indeksler hazır"; fi

step "2) LLM sunucuları (Gemma 8099, Qwen 8098)"
LS=ai/llama.cpp/build/bin/llama-server
$LS -m ai/models/finetuned/GEMMA3_FINANCE_V1.Q4_K_M.gguf --port 8099 -ngl 99 -c 4096 -np 1 > logs/llm_gemma.log 2>&1 & PIDS+=($!)
$LS -m ai/models/finetuned/QWEN3_EXPERT_V2.Q4_K_M.gguf --port 8098 -ngl 99 -c 4096 -np 1 > logs/llm_qwen.log 2>&1 & PIDS+=($!)
wait_http http://127.0.0.1:8099/health 60 || { echo "HATA: Gemma başlamadı"; tail -20 logs/llm_gemma.log; exit 1; }
wait_http http://127.0.0.1:8098/health 60 || { echo "HATA: Qwen başlamadı"; tail -20 logs/llm_qwen.log; exit 1; }

step "3) Servisler (router 8000, şube RAG 8001/8002, HQ RAG 8003, orkestratör 8004)"
export RAG_TLS_VERIFY=false BRANCH1_RAG_URL=http://127.0.0.1:8001 BRANCH2_RAG_URL=http://127.0.0.1:8002 HQ_RAG_URL=http://127.0.0.1:8003 \
  ROUTER_URL=http://127.0.0.1:8000 HQ_LLM_URL=http://127.0.0.1:8098 BRANCH_LLM_URL=http://127.0.0.1:8099 \
  TURKEMBED_MODEL_PATH=ai/models/embeddings/turkembed_finetuned EMBEDDING_MODEL=ai/models/embeddings/turkembed_finetuned
start_svc(){ name=$1; shift; env "$@" $PY backend/backend_api.py --service "$SVC" --port "$PORT" > logs/$name.log 2>&1 & PIDS+=($!); }
SVC=router PORT=8000 start_svc router PORT=8000 MODEL_PATH=ai/models/router/bert_router_best
if [ "${MODE:-vm}" = "vm" ]; then
  # Gerçek dağıtım: şube RAG servisleri VMware VM'lerinde (köprü ağ)
  export BRANCH1_RAG_URL=${VM1_URL:-http://10.0.0.11:8001} BRANCH2_RAG_URL=${VM2_URL:-http://10.0.0.12:8002}
  echo "MOD=vm  şube-1=$BRANCH1_RAG_URL  şube-2=$BRANCH2_RAG_URL"
  for u in "$BRANCH1_RAG_URL" "$BRANCH2_RAG_URL"; do
    wait_http "$u/health" 40 || { echo "HATA: VM servisi yanıt vermiyor: $u"; exit 1; }
    echo "$u/health -> $(curl -s -m 5 "$u/health" | head -c 300)"
    for i in 1 2 3 4 5; do curl -s -o /dev/null -m 10 -w "  gecikme(health) %{time_total}s\n" "$u/health"; done
  done
  # VM korpus yüklemesi arka planda (GPU kıyaslamalarıyla paralel); H1 öncesi beklenir
  ( $PY backend/scripts/hypothesis_tests/load_vm_corpora.py --url "$BRANCH1_RAG_URL" --corpus ai/data/splits/branch1/corpus.jsonl --prefix b1 > logs/vm_load_b1.log 2>&1 ) & LOAD1=$!
  ( $PY backend/scripts/hypothesis_tests/load_vm_corpora.py --url "$BRANCH2_RAG_URL" --corpus ai/data/splits/branch2/corpus.jsonl --prefix b2 > logs/vm_load_b2.log 2>&1 ) & LOAD2=$!
else
SVC=rag PORT=8001 start_svc b1 PORT=8001 BRANCH_ID=branch_01 CHROMADB_PATH=ai/data/chromadb/branch1
SVC=rag PORT=8002 start_svc b2 PORT=8002 BRANCH_ID=branch_02 CHROMADB_PATH=ai/data/chromadb/branch2
fi
SVC=rag PORT=8003 start_svc hq PORT=8003 BRANCH_ID=hq CHROMADB_PATH=ai/data/chromadb/global
PORTS="8000 8003"; [ "${MODE:-vm}" = "vm" ] || PORTS="8000 8001 8002 8003"
for p in $PORTS; do wait_http http://127.0.0.1:$p/health 100 || { echo "HATA: servis $p başlamadı"; tail -15 logs/*.log | tail -40; exit 1; }; done
SVC=orchestrator PORT=8004 start_svc orch PORT=8004
wait_http http://127.0.0.1:8004/health 40

step "4) Akademik kıyaslamalar (sıfır-örnek): Qwen 8098 ve Gemma 8099"
$PY -c "import json,sys;d=json.load(open('$R/bench_qwen.json'));sys.exit(0 if d['tr_mmlu']['unparseable']<d['tr_mmlu']['total'] and d['sentiment']['unparseable']<d['sentiment']['total'] else 1)" 2>/dev/null && echo "bench_qwen.json geçerli, atlanıyor" || $PY backend/scripts/eval_tool.py bench --url http://127.0.0.1:8098 --max-mmlu 500 --max-sentiment 500 && cp $R/bench_results.json $R/bench_qwen.json
$PY -c "import json,sys;d=json.load(open('$R/bench_gemma.json'));sys.exit(0 if d['tr_mmlu']['unparseable']<d['tr_mmlu']['total'] and d['sentiment']['unparseable']<d['sentiment']['total'] else 1)" 2>/dev/null && echo "bench_gemma.json geçerli, atlanıyor" || $PY backend/scripts/eval_tool.py bench --url http://127.0.0.1:8099 --max-mmlu 500 --max-sentiment 500 && cp $R/bench_results.json $R/bench_gemma.json

step "5) H1 federe RAG deneyi"
if [ -n "$LOAD1" ]; then echo "VM korpus yüklemesi bekleniyor..."; wait $LOAD1; r1=$?; wait $LOAD2; r2=$?; tail -2 logs/vm_load_b1.log logs/vm_load_b2.log; [ $r1 -eq 0 ] && [ $r2 -eq 0 ] || { echo "HATA: VM yüklemesi başarısız"; exit 1; }; fi
[ -f $R/h1_federated_rag.json ] && grep -q '"n": 200' $R/h1_federated_rag.json && echo "H1 sonucu mevcut, atlanıyor" || $PY backend/scripts/hypothesis_tests/h1_federated_rag_eval.py --n 200 --llm-url http://127.0.0.1:8099/v1/chat/completions

step "6) H2 saldırı deneyi (300 istem)"
$PY backend/scripts/hypothesis_tests/h2_security_eval_runner.py

step "7) H2 PII + DP (yeniden)"
$PY backend/scripts/hypothesis_tests/h2_pii_leakage_test.py | tail -8
$PY backend/scripts/hypothesis_tests/h2_dp_analysis.py > /dev/null && echo "DP analizi tamam"

step "8) H3 GPU çıkarım hızı (llama-bench, GPU) – karşılaştırma için"
ai/llama.cpp/build/bin/llama-bench -m ai/models/finetuned/GEMMA3_FINANCE_V1.Q4_K_M.gguf -ngl 99 -p 128 -n 64 -r 3 -o json > $R/h3_gpu_bench.json 2>>"$LOG" && echo "H3 GPU tamam"

echo "=== BİTTİ: $(date -Is) ==="
