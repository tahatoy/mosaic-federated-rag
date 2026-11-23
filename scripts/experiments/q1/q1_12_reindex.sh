#!/usr/bin/env bash
# Şube ve HQ indekslerini taban multilingual-e5-small ile yeniden kurar; benzerlik eşiğini kaldırır, HNSW ef=200; servisleri yeniden başlatır
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
[ -f .venv/bin/activate ] && . .venv/bin/activate
export PYTHONUNBUFFERED=1
Q=backend/scripts/q1_revision
KEY=$HOME/.ssh/mosaic_vm
SSHO=(-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.ssh/mosaic_known" -o ConnectTimeout=8 -o BatchMode=yes -o ServerAliveInterval=30 -i "$KEY")
VMS=(10.0.0.11 10.0.0.12)
echo "başladı: $(date -Is)"
pids=(); b=1
for ip in "${VMS[@]}"; do
  ssh "${SSHO[@]}" mosaic2@$ip "mkdir -p ~/Mosaic/q1fl" && scp -q "${SSHO[@]}" $Q/reindex_e5base.py $Q/set_env.py $Q/patch_ai_engine.py mosaic2@$ip:~/Mosaic/q1fl/
  ssh "${SSHO[@]}" mosaic2@$ip "cd ~/Mosaic && cp -n backend/ai_engine.py backend/ai_engine.py.pre_e5base && .venv/bin/python q1fl/patch_ai_engine.py backend/ai_engine.py && cp -n .env.branch$b .env.branch$b.turkembed_bak && .venv/bin/python q1fl/reindex_e5base.py --src data/chromadb/branch_0$b --dst data/chromadb/branch_0${b}_e5 --name finance_branch$b --threads 2" > logs/q1_reindex_b$b.log 2>&1 &
  pids+=($!); b=$((b+1))
done
python3 $Q/reindex_e5base.py --src ai/data/chromadb/hq_ortak --dst ai/data/chromadb/hq_ortak_e5 --name finance_global
for p in "${pids[@]}"; do wait $p; echo "VM yeniden indeksleme çıkış: $?"; done
for b in 1 2; do echo "--- b$b"; tail -2 logs/q1_reindex_b$b.log; done
b=1
for ip in "${VMS[@]}"; do
  ssh "${SSHO[@]}" mosaic2@$ip "cd ~/Mosaic && .venv/bin/python q1fl/set_env.py .env.branch$b CHROMADB_PATH=data/chromadb/branch_0${b}_e5 EMBEDDING_MODEL_KEY=e5small_base RAG_MIN_SIMILARITY=0.0 RAG_HNSW_EF=200 && bash scripts/branch_start.sh $b" 2>&1 | tail -4 | sed "s/^/[Şube-$b] /"
  b=$((b+1))
done
cp -n .env.hq backups/env_hq_turkembed_bak_20261122
python3 $Q/set_env.py .env.hq CHROMADB_PATH=ai/data/chromadb/hq_ortak_e5 EMBEDDING_MODEL_KEY=e5small_base RAG_MIN_SIMILARITY=0.0 RAG_HNSW_EF=200
pkill -9 -f llama-server; sleep 3
bash scripts/hq_start.sh 2>&1 | tail -12
set -a; . ./.env.hq; set +a
for b in 1 2; do v=BRANCH${b}_RAG_URL; echo "şube $b stats: $(curl -s -m 8 --cacert $RAG_CA_CERT_PATH --cert $RAG_CLIENT_CERT --key $RAG_CLIENT_KEY ${!v}/stats)"; done
echo "HQ stats: $(curl -s -m 5 http://127.0.0.1:8003/stats)"
echo "deneme: $(curl -s -m 120 -X POST http://127.0.0.1:8004/query -H 'Content-Type: application/json' -d '{"query":"Apache Corporation piyasa değeri yaklaşık kaç milyar dolar?","branch_id":1}' | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("route"),[s.get("branch") for s in d.get("sources",[])],d.get("answer","")[:200])')"
echo "bitti: $(date -Is)"
