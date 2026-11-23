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
python3 -u $Q/e6_h4_zemberek_retrieval.py; rc=$?
echo "rc=$rc"
echo "bitti: $(date -Is)"
