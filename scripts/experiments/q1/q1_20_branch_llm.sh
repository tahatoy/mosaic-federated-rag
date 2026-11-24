#!/usr/bin/env bash
# Q1 revizyon (E17) — Gemma-3-4B'yi şube VM'lerinde (yalnızca CPU) çalıştırıp canlı uç çıkarım testi (E17).
#   1) Şube VM'lerinin işlemci özelliklerine göre CPU'ya özel llama-server derlenir (genel merkezde, bir kez).
#   2) llama-server ve Gemma-3-4B Q4_K_M şubelere kopyalanır, şubede yalnızca 127.0.0.1'e açılarak başlatılır.
#   3) Genel merkezden SSH tüneliyle erişilir; 50 soru hem genel merkez GPU'sunda hem sorunun sorulduğu şubede yanıtlanır.
#   4) Şube belleği saniyede bir kaydedilir; test bitince şube modelleri durdurulur.
# SSH: vm_deploy.sh'nin kurduğu anahtar (~/.ssh/mosaic_vm) kullanılır, şifre istenmez.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
mkdir -p logs
LOG=logs/branch_llm_test.log; exec > >(tee -a "$LOG") 2>&1
echo "=== Şube LLM testi: $(date -Is) ==="
[ -f .venv/bin/activate ] && . .venv/bin/activate
set -a; . ./.env.hq; set +a
export PYTHONPATH="$PWD:$PWD/ai:$PWD/backend" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="$PWD/ai/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"

N="${1:-50}"
U=mosaic2
declare -A IP=([1]=10.0.0.11 [2]=10.0.0.12) TP=([1]=18101 [2]=18102)
KEY=$HOME/.ssh/mosaic_vm
SSHO=(-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.ssh/mosaic_known" -o ConnectTimeout=8 -o BatchMode=yes -i "$KEY")
GG=ai/models/finetuned/GEMMA3_FINANCE_V1.Q4_K_M.gguf
BD=ai/llama.cpp/build-branch-cpu
R(){ local b=$1; shift; ssh "${SSHO[@]}" "$U@${IP[$b]}" "$@"; }
cleanup(){
  for b in 1 2; do R "$b" "bash ~/mosaic_llm/vm.sh stop" 2>/dev/null; done
  pkill -f "ssh .*-L 1810[12]:127.0.0.1:8099" 2>/dev/null
}
fail(){ echo; echo "HATA: $*"; cleanup; echo "Günlük: $LOG"; exit 1; }

echo "[0/6] Ön kontroller"
[ -f "$KEY" ] || fail "SSH anahtarı yok ($KEY). Şube VM'lerine anahtar tabanlı SSH erişimi kurulmalı."
for b in 1 2; do R "$b" true || fail "Şube-$b'ye anahtarla bağlanılamadı (${IP[$b]}). VM açık mı?"; done
curl -sf -m 5 http://127.0.0.1:8099/health >/dev/null || fail "Genel merkez Gemma (8099) kapalı. Önce genel merkez servislerini başlatın."
curl -sf -m 5 http://127.0.0.1:8003/health >/dev/null || fail "Genel merkez ortak RAG (8003) kapalı. Önce genel merkez servislerini başlatın."
[ -f "$GG" ] || fail "$GG bulunamadı"
for b in 1 2; do R "$b" "mkdir -p ~/mosaic_llm"; done
for b in 1 2; do scp "${SSHO[@]}" -q scripts/branch_llm_vm.sh "$U@${IP[$b]}:mosaic_llm/vm.sh" || fail "yardımcı betik kopyalanamadı"; done
for b in 1 2; do echo "  Şube-$b: $(R "$b" "bash ~/mosaic_llm/vm.sh info")"; done

echo "[1/6] Şubeye uygun CPU llama-server (genel merkezde derlenir)"
if [ ! -x "$BD/bin/llama-server" ]; then
  F1=$(R 1 "grep -m1 '^flags' /proc/cpuinfo"); F2=$(R 2 "grep -m1 '^flags' /proc/cpuinfo")
  has(){ [[ " $F1 " == *" $1 "* && " $F2 " == *" $1 "* ]]; }
  has avx2 || fail "Şube işlemcisi AVX2 desteklemiyor"
  OPTS=(-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON)
  has bmi2 && OPTS+=(-DGGML_BMI2=ON)
  if has avx512f && has avx512bw && has avx512dq && has avx512vl; then
    OPTS+=(-DGGML_AVX512=ON)
    has avx512_vnni && OPTS+=(-DGGML_AVX512_VNNI=ON)
    has avx512_bf16 && OPTS+=(-DGGML_AVX512_BF16=ON)
    has avx512vbmi && OPTS+=(-DGGML_AVX512_VBMI=ON)
  fi
  echo "  seçenekler: ${OPTS[*]}"
  cmake -S ai/llama.cpp -B "$BD" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF -DGGML_OPENMP=OFF \
        -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF "${OPTS[@]}" > logs/branch_llm_cmake.log 2>&1 \
    || fail "cmake başarısız (logs/branch_llm_cmake.log)"
  cmake --build "$BD" --target llama-server -j "$(nproc)" > logs/branch_llm_build.log 2>&1 \
    || fail "derleme başarısız (logs/branch_llm_build.log)"
fi
echo "  $($BD/bin/llama-server --version 2>&1 | grep -m1 -i version)"

echo "[2/6] llama-server ve model şubelere kopyalanıyor"
SZ=$(stat -c %s "$GG")
for b in 1 2; do
  scp "${SSHO[@]}" -q "$BD/bin/llama-server" "$U@${IP[$b]}:mosaic_llm/llama-server" || fail "llama-server kopyalanamadı (Şube-$b)"
  R "$b" "chmod +x ~/mosaic_llm/llama-server && ~/mosaic_llm/llama-server --version" >/dev/null 2>&1 \
    || fail "llama-server Şube-$b'de çalışmadı: $(R "$b" '~/mosaic_llm/llama-server --version 2>&1 | tail -2')"
  if [ "$(R "$b" "stat -c %s ~/mosaic_llm/gemma.gguf 2>/dev/null")" != "$SZ" ]; then
    FREE=$(R "$b" "df -Pk ~ | awk 'NR==2{print \$4}'")
    [ "${FREE:-0}" -gt 3000000 ] || fail "Şube-$b'de yeterli disk yok (${FREE} KB)"
    echo "  Şube-$b: model kopyalanıyor (2.3 GB)..."
    scp "${SSHO[@]}" -q "$GG" "$U@${IP[$b]}:mosaic_llm/gemma.gguf" || fail "model kopyalanamadı (Şube-$b)"
  else
    echo "  Şube-$b: model zaten var"
  fi
done

echo "[3/6] Şube modelleri başlatılıyor (yalnızca CPU, 2 iş parçacığı)"
for b in 1 2; do R "$b" "bash ~/mosaic_llm/vm.sh start" || fail "Şube-$b modeli başlamadı"; done

echo "[4/6] SSH tünelleri"
for b in 1 2; do
  pkill -f "ssh .*-L ${TP[$b]}:127.0.0.1:8099" 2>/dev/null
  ssh "${SSHO[@]}" -o ExitOnForwardFailure=yes -f -N -L "${TP[$b]}:127.0.0.1:8099" "$U@${IP[$b]}" || fail "tünel kurulamadı (Şube-$b)"
  curl -sf -m 5 "http://127.0.0.1:${TP[$b]}/health" >/dev/null || fail "tünelden şube modeline erişilemedi (Şube-$b)"
done

echo "[5/6] Test: $N soru (genel merkez GPU ve sorunun sorulduğu şubenin CPU'su)"
python3 -u backend/scripts/q1_revision/e17_branch_llm.py --n "$N" \
  --branch-llm "1=http://127.0.0.1:${TP[1]},2=http://127.0.0.1:${TP[2]}" || fail "test betiği hata verdi"

echo "[6/6] Şube bellek kayıtları alınıyor ve modeller durduruluyor"
for b in 1 2; do scp "${SSHO[@]}" -q "$U@${IP[$b]}:mosaic_llm/mem.log" "logs/branch_llm_mem_b$b.log"; done
cleanup
python3 - <<'EOF'
import json, pathlib
p = pathlib.Path("ai/experiments/results/q1/e17_branch_llm.json"); d = json.loads(p.read_text(encoding="utf-8"))
mem = {}
for b in (1, 2):
    rows = [l.split() for l in pathlib.Path(f"logs/branch_llm_mem_b{b}.log").read_text().splitlines() if len(l.split()) == 3]
    rows = [(int(t), int(u), int(r)) for t, u, r in rows]
    before = [u for _, u, r in rows if r == 0]
    mem[b] = {"used_before_llm_mb": max(before) if before else None,
              "peak_used_mb": max(u for _, u, _ in rows), "peak_llama_rss_mb": round(max(r for *_, r in rows) / 1024), "samples": len(rows)}
d["branch_memory"] = mem
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("şube belleği:", json.dumps(mem, ensure_ascii=False))
EOF
echo
echo "=== TAMAM: ai/experiments/results/q1/e17_branch_llm.json — $(date -Is) ==="
