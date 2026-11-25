#!/usr/bin/env bash
# Q2-04 (E21): federe MOSAIC-Embed — veri hazırlığı, HQ GPU'sunda iki ayrı Flower istemci süreciyle (TLS) 6 tur, merkezi denetim.
. "$(dirname "${BASH_SOURCE[0]}")/../../../backend/scripts/q1_revision/q2_lib.sh"
durum "Q2-04 E21-HQ başladı"
llm_stop
T=/tmp/q2fl; rm -rf $T; mkdir -p $T; cp backend/certs/ca.crt $T/
openssl req -new -newkey rsa:2048 -nodes -keyout $T/fl_server.key -subj "/CN=mosaic-hq-flower" -out $T/fl.csr 2>/dev/null
printf "subjectAltName=IP:$HQIP,IP:127.0.0.1,DNS:localhost\n" > $T/ext.cnf
openssl x509 -req -in $T/fl.csr -CA backend/certs/ca.crt -CAkey backend/certs/ca.key -CAserial $T/ca.srl -CAcreateserial -out $T/fl_server.crt -days 30 -extfile $T/ext.cnf 2>&1 | tail -1
python3 -u $Q/e21_flower_mosaic_embed.py prep; echo "prep rc=$?"
pkill -f "e21_flower_mosaic_embed.py server"; sleep 1
python3 -u $Q/e21_flower_mosaic_embed.py server --rounds 6 --port 9092 --tls-dir $T --where hq --out-model mosaic_embed_flower > logs/q2_e21_hq_server.log 2>&1 &
SP=$!
for i in $(seq 1 120); do ss -ltn | grep -q ":9092 " && break; sleep 2; done
for b in 1 2; do
  python3 -u $Q/e21_flower_mosaic_embed.py client --server localhost:9092 --branch $b --data ai/data/e21/branch$b.json --ca $T/ca.crt > logs/q2_e21_hq_b$b.log 2>&1 &
done
wait $SP; echo "sunucu rc=$?"; sleep 3
grep -E "TEST|FEDERE|en iyi" logs/q2_e21_hq_server.log | tail -16
for b in 1 2; do grep "^{" logs/q2_e21_hq_b$b.log | tail -3; done
python3 -u $Q/e21_flower_mosaic_embed.py central --epochs 6 --freeze 1 --out-model mosaic_embed_central_frozen; echo "central rc=$?"
durum "Q2-04 bitti: $(ls -d ai/models/embeddings/mosaic_embed_flower 2>/dev/null)"
