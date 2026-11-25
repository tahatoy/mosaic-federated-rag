#!/usr/bin/env bash
# Q2-07: seçilen federe (Flower) MOSAIC-Embed ile iki şube ve HQ ortak koleksiyonunu yeniden indeksler, canlı hattı bu modelle başlatır.
# Eski MOSAIC-Embed koleksiyonları (_me) silinmez; geri dönüş: CHROMADB_PATH=..._me ve EMBEDDING_MODEL_PATH= (boş).
. "$(dirname "${BASH_SOURCE[0]}")/../../../backend/scripts/q1_revision/q2_lib.sh"
durum "Q2-07 başladı"
CH=$(python3 -c "import json;print(json.load(open('$RES/e21_deploy_choice.json'))['deploy'] or '')" 2>/dev/null)
if [ -z "$CH" ]; then durum "Q2-07: taban modeli geçen federe model yok — yeniden indeksleme ATLANDI; şube servisleri MOSAIC-Embed ile yeniden başlatılıyor"
  for b in 1 2; do vm ${VMS[$((b-1))]} "bash ~/Mosaic/scripts/branch_start.sh $b" 2>&1 | tail -3; done
  bash scripts/hq_start.sh 2>&1 | tail -10; exit 0; fi
M=ai/models/embeddings/$CH
pids=(); b=1
for ip in "${VMS[@]}"; do
  ( vm $ip "mkdir -p ~/Mosaic/q1fl ~/Mosaic/ai/models/embeddings" &&
    scp -q "${SSHO[@]}" $Q/reindex_e5base.py $Q/set_env.py $Q/patch_embed_path.py mosaic2@$ip:~/Mosaic/q1fl/ &&
    scp -q -r "${SSHO[@]}" $M mosaic2@$ip:~/Mosaic/ai/models/embeddings/ &&
    vm $ip "cd ~/Mosaic && cp -n .env.branch$b .env.branch$b.me_bak && .venv/bin/python q1fl/reindex_e5base.py --src data/chromadb/branch_0${b}_me --dst data/chromadb/branch_0${b}_fl --name finance_branch$b --threads 2 --model $M"
  ) > logs/q2_reindex_fl_b$b.log 2>&1 &
  pids+=($!); b=$((b+1))
done
python3 $Q/reindex_e5base.py --src ai/data/chromadb/hq_ortak_me --dst ai/data/chromadb/hq_ortak_fl --name finance_global --model $M
for p in "${pids[@]}"; do wait $p; echo "VM yeniden indeksleme çıkış: $?"; done
for b in 1 2; do echo "--- b$b"; tail -2 logs/q2_reindex_fl_b$b.log; done
b=1
for ip in "${VMS[@]}"; do
  if grep -q '"n_dst"' logs/q2_reindex_fl_b$b.log; then
    vm $ip "cd ~/Mosaic && .venv/bin/python q1fl/patch_embed_path.py backend/ai_engine.py && .venv/bin/python q1fl/set_env.py .env.branch$b CHROMADB_PATH=data/chromadb/branch_0${b}_fl EMBEDDING_MODEL_KEY=mosaic_embed EMBEDDING_MODEL_PATH=$M RAG_MIN_SIMILARITY=0.0 RAG_HNSW_EF=200"
  else durum "UYARI: şube $b yeniden indekslenemedi, MOSAIC-Embed ile kalıyor"; fi
  vm $ip "bash ~/Mosaic/scripts/branch_start.sh $b" 2>&1 | tail -4 | sed "s/^/[Şube-$b] /"
  b=$((b+1))
done
cp -n .env.hq backups/env_hq_me_bak_q2
python3 $Q/patch_embed_path.py backend/ai_engine.py
python3 $Q/set_env.py .env.hq CHROMADB_PATH=ai/data/chromadb/hq_ortak_fl EMBEDDING_MODEL_KEY=mosaic_embed EMBEDDING_MODEL_PATH=$M RAG_MIN_SIMILARITY=0.0 RAG_HNSW_EF=200
pkill -9 -f llama-server; sleep 3
bash scripts/hq_start.sh 2>&1 | tail -12
set -a; . ./.env.hq; set +a
system_check
echo "deneme: $(curl -s -m 120 -X POST http://127.0.0.1:8004/query -H 'Content-Type: application/json' -d '{"query":"Apache Corporation piyasa değeri yaklaşık kaç milyar dolar?","branch_id":1}' | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("route"),[s.get("branch") for s in d.get("sources",[])],d.get("answer","")[:160])')"
echo "{\"deployed\": \"$CH\"}" > $RES/q2_deployed_embedding.json
durum "Q2-07 bitti: canlı hat $CH ile"
