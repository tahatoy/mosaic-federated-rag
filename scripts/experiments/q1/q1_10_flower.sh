#!/usr/bin/env bash
# E7 — İki şube VM'i arasında gerçek Flower koşusu (TLS), FedAvg ve FedProx; ardından sertifikasız istemci denemesi
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
[ -f .venv/bin/activate ] && . .venv/bin/activate
set -a; . ./.env.hq; set +a
export PYTHONPATH="$PWD:$PWD/ai:$PWD/backend:$PWD/backend/scripts/q1_revision" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
Q=backend/scripts/q1_revision
KEY=$HOME/.ssh/mosaic_vm
SSHO=(-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.ssh/mosaic_known" -o ConnectTimeout=8 -o BatchMode=yes -o ServerAliveInterval=30 -i "$KEY")
VMS=(10.0.0.11 10.0.0.12)
echo "başladı: $(date -Is)"
T=/tmp/q1fl; rm -rf $T; mkdir -p $T; cp backend/certs/ca.crt $T/
openssl req -new -newkey rsa:2048 -nodes -keyout $T/fl_server.key -subj "/CN=mosaic-hq-flower" -out $T/fl.csr 2>/dev/null
printf "subjectAltName=IP:10.0.0.10,IP:127.0.0.1,DNS:localhost\n" > $T/ext.cnf
openssl x509 -req -in $T/fl.csr -CA backend/certs/ca.crt -CAkey backend/certs/ca.key -CAserial $T/ca.srl -CAcreateserial -out $T/fl_server.crt -days 30 -extfile $T/ext.cnf 2>&1 | tail -1
python3 - <<'PY'
import hashlib, sys
sys.path.insert(0, "backend/scripts/q1_revision")
from q1_common import questions
qs, docs, _ = questions(600)
with open("/tmp/q1fl/test_sha1.txt", "w") as f:
    for i in sorted({i for q in qs for i in q["gold_idx"]}):
        f.write(hashlib.sha1(docs[i].encode("utf-8")).hexdigest() + "\n")
print("test parça özeti yazıldı")
PY
MODEL=$(python3 -c "from huggingface_hub import snapshot_download;print(snapshot_download('intfloat/multilingual-e5-small', local_files_only=True))")
echo "HQ model: $MODEL"
for ip in "${VMS[@]}"; do
  ssh "${SSHO[@]}" mosaic2@$ip "mkdir -p ~/Mosaic/q1fl" && scp -q "${SSHO[@]}" $Q/e7_flower_client.py $T/ca.crt $T/test_sha1.txt mosaic2@$ip:~/Mosaic/q1fl/ && echo "$ip: dosyalar kopyalandı"
done
run_round(){ # etiket strateji tur ca_kullan
  local label=$1 strat=$2 rounds=$3 useca=$4
  pkill -f e7_flower_server; sleep 2
  setsid nohup python3 -u $Q/e7_flower_server.py --strategy $strat --tag $label --rounds $rounds --model "$MODEL" --tls-dir $T > logs/q1_flower_$label.log 2>&1 < /dev/null &
  SP=$!
  for i in $(seq 1 120); do ss -ltn | grep -q ":9090 " && break; sleep 2; done
  echo "[$label] sunucu dinliyor: $(ss -ltn | grep -c ':9090 ')"
  local direct; direct=$(ssh "${SSHO[@]}" mosaic2@${VMS[0]} 'timeout 4 bash -c "echo > /dev/tcp/10.0.0.10/9090" && echo ok' 2>/dev/null)
  local ADDR TUN; if [ "$direct" = ok ]; then ADDR=10.0.0.10:9090; TUN=(); else ADDR=localhost:9090; TUN=(-R 9090:127.0.0.1:9090); fi
  echo "[$label] istemci adresi: $ADDR (doğrudan erişim: ${direct:-yok})"
  local CA="--ca q1fl/ca.crt"; [ "$useca" = 0 ] && CA=""
  local pids=() b=1
  for ip in "${VMS[@]}"; do
    ssh "${SSHO[@]}" "${TUN[@]}" mosaic2@$ip "cd ~/Mosaic && M=\$(.venv/bin/python -c \"from huggingface_hub import snapshot_download;print(snapshot_download('intfloat/multilingual-e5-small', local_files_only=True))\") && timeout ${CLIENT_TIMEOUT:-2400} .venv/bin/python q1fl/e7_flower_client.py --server $ADDR --branch $b --chroma data/chromadb/branch_0$b --exclude q1fl/test_sha1.txt --n 600 --model \$M $CA" > logs/q1_flower_${label}_b$b.log 2>&1 &
    pids+=($!); b=$((b+1))
  done
  for p in "${pids[@]}"; do wait $p; echo "[$label] istemci çıkış kodu: $?"; done
  for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 5; done
  kill $SP 2>/dev/null; wait $SP 2>/dev/null
  echo "[$label] sunucu günlüğü (son satırlar):"; grep -E "DEĞERLENDİRME|ERROR|Error|rounds|yazıldı" logs/q1_flower_$label.log | tail -12
  for b in 1 2; do echo "--- istemci b$b"; grep -v Warning logs/q1_flower_${label}_b$b.log | tail -6; done
}
run_round fedavg fedavg 3 1
run_round fedprox fedprox 3 1
# Olumsuz deneme: TLS sunucusuna sertifika doğrulaması olmadan (insecure) bağlanan istemci reddedilmeli
CLIENT_TIMEOUT=90 run_round insecure_test fedavg 1 0 2>/dev/null || true
echo "bitti: $(date -Is)"
