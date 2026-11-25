#!/usr/bin/env bash
# Q2-08a: şube VM'lerindeki eski ai_engine sürümü EMBEDDING_MODEL_PATH'i tanımadığı için (Q2-07'de yama uygulanamadı) federe modeli
# VM'lerde MOSAIC-Embed anahtarının yüklediği dizine kurar: turkembed_v3 -> turkembed_v3_merkezi_yedek, federe model -> turkembed_v3.
# Koleksiyon: data/chromadb/branch_0N_fl (Q2-07'de federe modelle yeniden indekslendi). Doğrulama: model dosyası md5 eşleşmesi.
. "$(dirname "${BASH_SOURCE[0]}")/../../../backend/scripts/q1_revision/q2_lib.sh"
durum "Q2-08a başladı (VM'lerde federe gömme modeli)"
M=ai/models/embeddings/mosaic_embed_flower_vm
H=$(md5sum $M/model.safetensors | cut -d' ' -f1); echo "HQ federe model md5: $H"
b=1
for ip in "${VMS[@]}"; do
  vm $ip "cd ~/Mosaic && test -d data/chromadb/branch_0${b}_fl && test -f $M/model.safetensors && \
    { [ -d ai/models/embeddings/turkembed_v3_merkezi_yedek ] || mv ai/models/embeddings/turkembed_v3 ai/models/embeddings/turkembed_v3_merkezi_yedek; } && \
    rm -rf ai/models/embeddings/turkembed_v3 && cp -r $M ai/models/embeddings/turkembed_v3 && \
    .venv/bin/python q1fl/set_env.py .env.branch$b CHROMADB_PATH=data/chromadb/branch_0${b}_fl EMBEDDING_MODEL_KEY=mosaic_embed RAG_MIN_SIMILARITY=0.0 RAG_HNSW_EF=200 && \
    echo \"şube $b md5: \$(md5sum ai/models/embeddings/turkembed_v3/model.safetensors | cut -d' ' -f1)\" && bash scripts/branch_start.sh $b" 2>&1 | grep -E "md5|RAG \(mTLS\)|HATA|rror" | cut -c1-260
  b=$((b+1))
done
set -a; . ./.env.hq; set +a
for b in 1 2; do v=BRANCH${b}_RAG_URL; echo "şube $b stats: $(curl -s -m 8 "${MT[@]}" "${!v}/stats")"; done
echo "HQ stats: $(curl -s -m 5 http://127.0.0.1:8003/stats)"
durum "Q2-08a bitti"
