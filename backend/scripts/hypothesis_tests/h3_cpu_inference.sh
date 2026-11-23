#!/usr/bin/env bash
# H3 deneyi: 2 vCPU / sınırlı RAM'li CPU-only ortamda Gemma-3-4B Q4_K_M çıkarım hızı ve bellek.
# Kullanım (şube VM'inde):  bash h3_cpu_inference.sh /yol/GEMMA3_FINANCE_V1.Q4_K_M.gguf /yol/llama-bench [thread=2]
# Gerekli: CPU-only derlenmiş llama.cpp (cmake -B build -DGGML_CUDA=OFF && cmake --build build --target llama-bench)
set -euo pipefail
MODEL="${1:?model .gguf yolu}"; BENCH="${2:?llama-bench yolu}"; T="${3:-2}"
OUT="$(dirname "$0")/../../../ai/experiments/results/h3_cpu_inference.json"
TMP="$(mktemp)"; ERR="$(mktemp)"
/usr/bin/time -v "$BENCH" -m "$MODEL" -t "$T" -ngl 0 -p 128 -n 64 -r 3 -o json > "$TMP" 2> "$ERR"
RSS_KB=$(grep "Maximum resident" "$ERR" | awk '{print $NF}')
python3 - "$TMP" "$RSS_KB" "$T" "$MODEL" "$OUT" <<'PY'
import json, sys, platform
tmp, rss, t, model, out = sys.argv[1:]
rows = json.load(open(tmp))
res = {"model": model, "threads": int(t), "cpu": rows[0]["cpu_info"], "backend": rows[0]["backends"],
       "peak_rss_gib": round(int(rss)/1024/1024, 2), "kernel": platform.release()}
for r in rows:
    if r["n_gen"] == 0: res["prompt_processing_tok_s"] = round(r["avg_ts"], 2); res["pp_std"] = round(r["stddev_ts"], 2)
    if r["n_prompt"] == 0: res["generation_tok_s"] = round(r["avg_ts"], 2); res["tg_std"] = round(r["stddev_ts"], 2)
res["h3_esik_50_tok_s_karsilandi"] = res.get("generation_tok_s", 0) >= 50
json.dump(res, open(out, "w"), indent=2, ensure_ascii=False); print(json.dumps(res, indent=2, ensure_ascii=False))
PY
