#!/usr/bin/env bash
# Q2-05b (E21): iki şube VM'i arasında gerçek Flower eğitimi — Q2-05'teki ilk deneme VM'lerdeki eski sentence-transformers
# sürümü yüzünden (tokenize çıktısında metin alanı) başlamadan düştü; düzeltilmiş istemciyle, önce VM'lerde öz-sınamayla yeniden.
. "$(dirname "${BASH_SOURCE[0]}")/../../../backend/scripts/q1_revision/q2_lib.sh"
durum "Q2-05b başladı (VM Flower)"
llm_stop
[ -f $Q/e3_fl_embedding_dp_real.py ] && mv -f $Q/e3_fl_embedding_dp_real.py $Q/e3_fl_embedding_dp.py
head -3 $Q/e3_fl_embedding_dp.py
T=/tmp/q2fl; [ -f $T/fl_server.crt ] || { durum "HATA: TLS sertifikası yok"; exit 1; }
ROUNDS=${ROUNDS:-3}
b=1
for ip in "${VMS[@]}"; do
  vm $ip "pkill -f 'backend_api[.]py --service rag'; pkill -f 'frontend[.]web_server:app'; pkill -f e21_flower_mosaic_embed; mkdir -p ~/Mosaic/q1fl21"
  scp -q "${SSHO[@]}" $Q/e21_flower_mosaic_embed.py $T/ca.crt mosaic2@$ip:~/Mosaic/q1fl21/
  vm $ip "test -f ~/Mosaic/q1fl21/branch$b.json" || scp -q "${SSHO[@]}" ai/data/e21/branch$b.json mosaic2@$ip:~/Mosaic/q1fl21/
  echo "--- öz-sınama şube $b"
  vm $ip "cd ~/Mosaic && M=\$(.venv/bin/python -c \"from huggingface_hub import snapshot_download;print(snapshot_download('intfloat/multilingual-e5-small', local_files_only=True))\") && timeout 900 .venv/bin/python -u q1fl21/e21_flower_mosaic_embed.py selftest --data q1fl21/branch$b.json --model \$M" 2>&1 | grep -v -i warn | tail -6
  b=$((b+1))
done
pkill -f "e21_flower_mosaic_embed.py server"; sleep 1
python3 -u $Q/e21_flower_mosaic_embed.py server --rounds $ROUNDS --port 9092 --tls-dir $T --where vm --tag vm_fedavg --out-model mosaic_embed_flower_vm --round-timeout 7200 > logs/q2_e21_vm_server.log 2>&1 &
SP=$!
for i in $(seq 1 120); do ss -ltn | grep -q ":9092 " && break; sleep 2; done
pids=(); b=1
for ip in "${VMS[@]}"; do
  ssh "${SSHO[@]}" mosaic2@$ip "cd ~/Mosaic && M=\$(.venv/bin/python -c \"from huggingface_hub import snapshot_download;print(snapshot_download('intfloat/multilingual-e5-small', local_files_only=True))\") && timeout 30000 .venv/bin/python -u q1fl21/e21_flower_mosaic_embed.py client --server $HQIP:9092 --branch $b --data q1fl21/branch$b.json --model \$M --ca q1fl21/ca.crt --device cpu --threads 2 --grad-ckpt 1 --dev-pool 3000" > logs/q2_e21_vm_b$b.log 2>&1 &
  pids+=($!); b=$((b+1))
done
# E22 federe eğitim + DP fazı (Q2-05'ten ertelendi) VM eğitimiyle eşzamanlı, HQ GPU'sunda
python3 -u $Q/e22_scalability.py --phases fl > logs/q2_e22_fl.log 2>&1 &
EP=$!
for p in "${pids[@]}"; do wait $p; echo "VM istemci çıkış: $?"; done
for i in $(seq 1 120); do kill -0 $SP 2>/dev/null || break; sleep 5; done
kill $SP 2>/dev/null; wait $SP 2>/dev/null
grep -E "TEST|FEDERE|en iyi|rror" logs/q2_e21_vm_server.log | tail -14
for b in 1 2; do echo "--- b$b"; grep -E "^\{|rror|Killed" logs/q2_e21_vm_b$b.log | tail -12; done
wait $EP; echo "E22 fl rc=$?"; tail -3 logs/q2_e22_fl.log
durum "Q2-05b bitti: vm modeli=$(ls -d ai/models/embeddings/mosaic_embed_flower_vm 2>/dev/null)"
