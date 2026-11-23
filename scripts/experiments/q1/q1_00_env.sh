#!/usr/bin/env bash
# Q1 revizyon deneyi. Herhangi bir dizinden çalıştırılabilir; betik proje köküne geçer.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
[ -f .venv/bin/activate ] && . .venv/bin/activate
set -a; . ./.env.hq; set +a
export PYTHONPATH="$PWD:$PWD/ai:$PWD/backend" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="$PWD/ai/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"
Q=backend/scripts/q1_revision
echo "başladı: $(date -Is)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python3 -c "import torch,sentence_transformers,transformers,peft,rank_bm25,sklearn;print('torch',torch.__version__,torch.cuda.is_available(),'st',sentence_transformers.__version__,'tf',transformers.__version__,'peft',peft.__version__)"
for u in 8099/health 8098/health 8000/health 8003/health 8004/health; do echo "$u -> $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$u)"; done
for b in 1 2; do v=BRANCH${b}_RAG_URL; echo "şube $b: $(curl -s -m 8 -o /dev/null -w '%{http_code}' --cacert $RAG_CA_CERT_PATH --cert $RAG_CLIENT_CERT --key $RAG_CLIENT_KEY ${!v}/health)"; done
echo "HF: $(curl -s -m 10 -o /dev/null -w '%{http_code}' https://huggingface.co/api/models/BAAI/bge-m3)"
ls ~/.cache/huggingface/hub 2>/dev/null | head -40
python3 -c "
import sys; sys.path.insert(0,'$Q')
from q1_common import questions
qs,docs,owner=questions(600)
print('soru',len(qs),'parça',len(docs),'altın ort.',sum(len(q['gold_idx']) for q in qs)/len(qs))
from collections import Counter; print('sahip',Counter(q['owner_branch'] for q in qs)); print(qs[0]['search_query'])
"
echo "bitti: $(date -Is)"
