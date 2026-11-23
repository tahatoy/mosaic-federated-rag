#!/usr/bin/env bash
# ==============================================================================
# MOSAIC — Şube düğümü başlatıcı (VMware Ubuntu VM üzerinde çalışır)
#   Kullanım:  bash scripts/branch_start.sh 1   |   bash scripts/branch_start.sh 2
#   - Şube RAG servisi (8001/8002): mTLS (TLS 1.3, yalnızca kurum CA'lı istemci sertifikası),
#     kaynakta kişisel veri maskeleme (BRANCH_PII_MASKING) ve denetim izi (logs/audit) etkin.
#   - Şube arayüzü (8005): soruyu genel merkez orkestratörüne iletir, yerel RAG'a mTLS ile bağlanır.
#   Günlükler: logs/rag_tls_800N.log, logs/webui.log
# ==============================================================================
set -u
B="${1:-}"
case "$B" in 1|2) ;; *) echo "Kullanım: bash scripts/branch_start.sh <1|2>"; exit 2;; esac
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PORT=$((8000 + B)); BID="branch_0$B"; C=backend/certs

[ -f .venv/bin/activate ] && . .venv/bin/activate
[ -f ".env.branch$B" ] && cp ".env.branch$B" .env
set -a; . ./.env 2>/dev/null; set +a
mkdir -p logs
chmod 600 $C/*.key 2>/dev/null

for f in ca.crt "branch$B.crt" "branch$B.key" "branch${B}_client.crt" "branch${B}_client.key"; do
  [ -f "$C/$f" ] || { echo "HATA: $C/$f bulunamadı (sertifikalar git ile gelir; branchN_client.key genel merkezden scp ile kopyalanır)"; exit 1; }
done

# Şube parçaları yerel TurkEmbed (ince ayarlı e5-small) ile vektörleştirilir (Tez Bölüm 3.2.3);
# model depoda tutulmaz, genel merkezden kopyalanır: ai/models/embeddings/turkembed_finetuned
[ -f ai/models/embeddings/turkembed_finetuned/model.safetensors ] || {
  echo "HATA: ai/models/embeddings/turkembed_finetuned bulunamadı. Genel merkezden kopyalayın:"
  echo "  scp -r ai/models/embeddings/turkembed_finetuned mosaic2@<VM_IP>:~/Mosaic/ai/models/embeddings/"; exit 1; }

echo "[1/3] Eski şube süreçleri durduruluyor..."
pkill -f "backend_api[.]py --service rag --port $PORT" 2>/dev/null
pkill -f "frontend[.]web_server:app" 2>/dev/null
fuser -k "$PORT/tcp" 8005/tcp >/dev/null 2>&1
sleep 2

export BRANCH_ID="$BID" PORT="$PORT" BRANCH_RAG_PORT="$PORT" MOSAIC_ENV=branch \
       BRANCH_PII_MASKING=true ENABLE_PII_LOGGING=false AUDIT_DIR=logs/audit AUDIT_RETENTION_DAYS=730 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       RAG_TLS_CERT="$C/branch$B.crt" RAG_TLS_KEY="$C/branch$B.key" RAG_TLS_CLIENT_CA="$C/ca.crt" \
       RAG_TLS_VERIFY=true RAG_CA_CERT_PATH="$C/ca.crt" \
       RAG_CLIENT_CERT="$C/branch${B}_client.crt" RAG_CLIENT_KEY="$C/branch${B}_client.key" \
       PYTHONPATH="$PWD:$PWD/ai:$PWD/backend" PYTHONUNBUFFERED=1

echo "[2/3] Şube-$B RAG servisi mTLS ile başlatılıyor (port $PORT)..."
setsid nohup python3 backend/scripts/tls_serve.py backend/backend_api.py --service rag --port "$PORT" \
  > "logs/rag_tls_$PORT.log" 2>&1 < /dev/null &
MT=(--cacert "$C/ca.crt" --cert "$C/branch${B}_client.crt" --key "$C/branch${B}_client.key")
ok=0
for _ in $(seq 1 80); do
  curl -sf -m 3 "${MT[@]}" "https://127.0.0.1:$PORT/health" >/dev/null && { ok=1; break; }
  sleep 3
done
[ $ok = 1 ] || { echo "HATA: RAG servisi ayağa kalkmadı, son günlük:"; tail -20 "logs/rag_tls_$PORT.log"; exit 1; }

echo "[3/3] Şube-$B arayüzü başlatılıyor (port 8005)..."
setsid nohup python3 -m uvicorn frontend.web_server:app --host 0.0.0.0 --port 8005 \
  > logs/webui.log 2>&1 < /dev/null &
for _ in $(seq 1 30); do curl -sf -m 3 http://127.0.0.1:8005/api/health >/dev/null && break; sleep 2; done

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "------------------------------------------------------------------"
echo " Şube-$B ($BID) — $(hostname) $IP"
echo " RAG (mTLS) : https://$IP:$PORT  -> $(curl -s -m 5 "${MT[@]}" "https://127.0.0.1:$PORT/stats")"
echo " Sertifikasız erişim denemesi: $(curl -s -m 4 -o /dev/null -w '%{http_code}' --cacert "$C/ca.crt" "https://127.0.0.1:$PORT/health"; echo " (000 = reddedildi)")"
echo " Arayüz     : http://$IP:8005  -> $(curl -s -m 5 http://127.0.0.1:8005/api/health)"
echo " Günlük     : tail -f logs/rag_tls_$PORT.log"
echo "------------------------------------------------------------------"
