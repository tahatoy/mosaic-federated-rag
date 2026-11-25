#!/usr/bin/env bash
# Q2-05e (E22): dağıtım gecikmesi fazının yeniden koşusu (ilk koşu N = 100'de taklit servisin bağlantı kesmesiyle düşmüştü).
. "$(dirname "${BASH_SOURCE[0]}")/../../../backend/scripts/q1_revision/q2_lib.sh"
durum "Q2-05e başladı"
python3 -u $Q/e22_scalability.py --phases latency 2>&1 | grep -v tokenizer | tail -10
durum "Q2-05e bitti"
