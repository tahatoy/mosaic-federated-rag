#!/usr/bin/env bash
# ==============================================================================
# MOSAIC — Genel Merkez (HQ) başlatıcı (Windows 11 + WSL2 Ubuntu, RTX 5060 Ti)
#   8099 llama-server  GEMMA3_FINANCE_V1 Q4_K_M  (basit / günlük konuşma)
#   8098 llama-server  QWEN3_EXPERT_V2  Q4_K_M  (karmaşık finansal analiz)
#   8000 BERT yönlendirici | 8003 HQ RAG (yalnızca ortak bilgi, 429 parça) | 8004 orkestratör (şubelere mTLS)
#   8005 HQ arayüzü        | 9080/9081 Flower sunucusu (FedProx; admin yalnızca 127.0.0.1)
#   Kullanım: bash scripts/hq_start.sh [--no-flower]
#   Günlükler: logs/llm_gemma.log logs/llm_qwen.log logs/router.log logs/hq.log logs/orch.log logs/webui.log logs/flower.log
# ==============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
FLOWER=1; [ "${1:-}" = "--no-flower" ] && FLOWER=0

[ -f .venv/bin/activate ] && . .venv/bin/activate
cp .env.hq .env
set -a; . ./.env; set +a
mkdir -p logs
export PYTHONPATH="$PWD:$PWD/ai:$PWD/backend" PYTHONUNBUFFERED=1 MOSAIC_ENV=hq
export LD_LIBRARY_PATH="$PWD/ai/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"
LS=ai/llama.cpp/build/bin/llama-server

wait_http(){ for _ in $(seq 1 "$2"); do curl -sf -m 3 "$1" >/dev/null && return 0; sleep 3; done; echo "  ZAMAN AŞIMI: $1"; return 1; }

echo "[1/6] Eski HQ süreçleri durduruluyor..."
pkill -f "backend_api[.]py --service" 2>/dev/null; pkill -f "frontend[.]web_server:app" 2>/dev/null
pkill -f "federated/server/server[.]py" 2>/dev/null; pkill -f "llama-server .*--port 809[89]" 2>/dev/null
for p in 8000 8003 8004 8005 8098 8099 9080 9081; do fuser -k "$p/tcp" >/dev/null 2>&1; done
sleep 2

echo "[2/6] Dil modeli sunucuları (GPU, 4096 bağlam)..."
setsid nohup $LS -m ai/models/finetuned/GEMMA3_FINANCE_V1.Q4_K_M.gguf --host 127.0.0.1 --port 8099 -ngl 99 -c 4096 -np 1 > logs/llm_gemma.log 2>&1 < /dev/null &
setsid nohup $LS -m ai/models/finetuned/QWEN3_EXPERT_V2.Q4_K_M.gguf  --host 127.0.0.1 --port 8098 -ngl 99 -c 4096 -np 1 > logs/llm_qwen.log  2>&1 < /dev/null &
wait_http http://127.0.0.1:8099/health 60; wait_http http://127.0.0.1:8098/health 60

echo "[3/6] BERT yönlendirici ve HQ RAG (yalnızca ortak bilgi)..."
env BIND_HOST=127.0.0.1 PORT=8000 setsid nohup python3 backend/backend_api.py --service router --port 8000 > logs/router.log 2>&1 < /dev/null &
env BIND_HOST=127.0.0.1 PORT=8003 BRANCH_ID=hq setsid nohup python3 backend/backend_api.py --service rag --port 8003 > logs/hq.log 2>&1 < /dev/null &
wait_http http://127.0.0.1:8000/health 100; wait_http http://127.0.0.1:8003/health 100

echo "[4/6] Orkestratör (şubelere mTLS istemci sertifikasıyla bağlanır)..."
env AUTH_REQUIRED=false PORT=8004 setsid nohup python3 backend/backend_api.py --service orchestrator --port 8004 > logs/orch.log 2>&1 < /dev/null &
wait_http http://127.0.0.1:8004/health 60
echo "      Isınma sorgusu: yeniden sıralayıcı ve kütüphaneler belleğe alınıyor (ilk demo sorusu beklemesin)..."
curl -s -m 300 -o /dev/null -X POST http://127.0.0.1:8004/query -H 'Content-Type: application/json' \
  -d '{"query":"Kredi kartı asgari ödeme tutarı nasıl hesaplanır?","branch_id":0}'

echo "[5/6] HQ arayüzü (8005)..."
env UI_PORT=8005 setsid nohup python3 -m uvicorn frontend.web_server:app --host 0.0.0.0 --port 8005 > logs/webui.log 2>&1 < /dev/null &
wait_http http://127.0.0.1:8005/api/health 40

if [ $FLOWER = 1 ]; then
  echo "[6/6] Flower federasyon sunucusu (daemon, admin 127.0.0.1:9081)..."
  env FL_DAEMON=true FL_ADMIN_HOST=127.0.0.1 setsid nohup python3 backend/federated/server/server.py > logs/flower.log 2>&1 < /dev/null &
  wait_http http://127.0.0.1:9081/admin/status 20
else
  echo "[6/6] Flower atlandı (--no-flower)"
fi

echo "------------------------------------------------------------------"
for u in 8099/health 8098/health 8000/health 8003/health 8004/health 8005/api/health 9081/admin/status; do
  printf "  %-20s -> %s\n" "$u" "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$u")"
done
echo "  HQ arayüzü: http://localhost:8005  (LAN: http://${HQ_LAN_IP:-10.0.0.10}:8005)"
echo "------------------------------------------------------------------"
