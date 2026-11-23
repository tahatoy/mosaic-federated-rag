#!/usr/bin/env bash
# Q1 revizyon — son yapılandırma: şube ve HQ indekslerini MOSAIC-Embed (ai/models/embeddings/turkembed_v3) ile yeniden kurar,
# benzerlik eşiğini kaldırır (alaka elemesi HQ çapraz kodlayıcısında), HNSW ef=200; servisleri bu modelle yeniden başlatır.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
[ -f .venv/bin/activate ] && . .venv/bin/activate
export PYTHONUNBUFFERED=1
Q=backend/scripts/q1_revision
M=ai/models/embeddings/turkembed_v3
KEY=$HOME/.ssh/mosaic_vm
SSHO=(-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.ssh/mosaic_known" -o ConnectTimeout=8 -o BatchMode=yes -o ServerAliveInterval=30 -i "$KEY")
VMS=(10.0.0.11 10.0.0.12)
echo "başladı: $(date -Is)"
[ -f $M/model.safetensors ] || { echo "HATA: $M yok (huggingface-cli download Toxotes/mosaic-embed --local-dir $M)"; exit 1; }
pids=(); b=1
for ip in "${VMS[@]}"; do
  ( ssh "${SSHO[@]}" mosaic2@$ip "mkdir -p ~/Mosaic/q1fl ~/Mosaic/ai/models/embeddings" &&
    scp -q "${SSHO[@]}" $Q/reindex_e5base.py $Q/set_env.py mosaic2@$ip:~/Mosaic/q1fl/ &&
    scp -q -r "${SSHO[@]}" $M mosaic2@$ip:~/Mosaic/ai/models/embeddings/ &&
    ssh "${SSHO[@]}" mosaic2@$ip "cd ~/Mosaic && .venv/bin/python q1fl/reindex_e5base.py --src data/chromadb/branch_0$b --dst data/chromadb/branch_0${b}_me --name finance_branch$b --threads 2 --model $M"
  ) > logs/q1_reindex_me_b$b.log 2>&1 &
  pids+=($!); b=$((b+1))
done
python3 $Q/reindex_e5base.py --src ai/data/chromadb/hq_ortak --dst ai/data/chromadb/hq_ortak_me --name finance_global --model $M
for p in "${pids[@]}"; do wait $p; echo "VM yeniden indeksleme çıkış: $?"; done
for b in 1 2; do echo "--- b$b"; tail -n 2 logs/q1_reindex_me_b$b.log; done
b=1
for ip in "${VMS[@]}"; do
  ssh "${SSHO[@]}" mosaic2@$ip "cd ~/Mosaic && .venv/bin/python q1fl/set_env.py .env.branch$b CHROMADB_PATH=data/chromadb/branch_0${b}_me EMBEDDING_MODEL_KEY=mosaic_embed EMBEDDING_MODEL=$M RAG_MIN_SIMILARITY=0.0 RAG_HNSW_EF=200 && bash scripts/branch_start.sh $b && grep -h 'Loading embedding' logs/rag_tls_800$b.log | tail -n 1" 2>&1 | tail -n 6 | sed "s/^/[Şube-$b] /"
  b=$((b+1))
done
python3 $Q/set_env.py .env.hq CHROMADB_PATH=ai/data/chromadb/hq_ortak_me EMBEDDING_MODEL_KEY=mosaic_embed RAG_MIN_SIMILARITY=0.0 RAG_HNSW_EF=200
pkill -9 -f llama-server; sleep 3
bash scripts/hq_start.sh 2>&1 | tail -n 12
set -a; . ./.env.hq; set +a
for b in 1 2; do v=BRANCH${b}_RAG_URL; echo "şube $b stats: $(curl -s -m 8 --cacert $RAG_CA_CERT_PATH --cert $RAG_CLIENT_CERT --key $RAG_CLIENT_KEY ${!v}/stats)"; done
echo "HQ stats: $(curl -s -m 5 http://127.0.0.1:8003/stats)"
echo "bitti: $(date -Is)"
