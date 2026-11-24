#!/usr/bin/env bash
# MOSAIC — şube VM'inde Gemma-3-4B (yalnızca CPU) yardımcı betiği. branch_llm_test.sh tarafından ~/mosaic_llm/vm.sh olarak kopyalanır.
#   info | start | stop | memlog
cd ~/mosaic_llm || exit 1
case "${1:-}" in
info)
  echo "cpu=$(nproc) mem_total_mb=$(free -m | awk '/Mem:/{print $2}') mem_used_mb=$(free -m | awk '/Mem:/{print $3}') disk_free_kb=$(df -Pk ~ | awk 'NR==2{print $4}') os=$(. /etc/os-release; echo "$VERSION_ID")"
  ;;
memlog)
  while :; do
    echo "$(date +%s) $(free -m | awk '/Mem:/{print $3}') $(ps -o rss= -C llama-server | awk '{s+=$1} END{print s+0}')"
    sleep 1
  done
  ;;
start)
  pkill -f "llama-serve[r] .*--port 8099" 2>/dev/null
  [ -f memlog.pid ] && kill "$(cat memlog.pid)" 2>/dev/null
  pkill -f "vm.sh memlo[g]" 2>/dev/null
  setsid nohup bash ~/mosaic_llm/vm.sh memlog > mem.log 2>&1 < /dev/null &
  echo $! > memlog.pid
  sleep 3   # llama-server öncesi bellek (yalnızca RAG servisi ve arayüz)
  setsid nohup ./llama-server -m gemma.gguf --host 127.0.0.1 --port 8099 -t 2 -c 4096 -np 1 -ngl 0 --cache-ram 0 \
    > llama.log 2>&1 < /dev/null &
  for _ in $(seq 1 150); do
    curl -sf -m 2 http://127.0.0.1:8099/health >/dev/null && { echo "hazır"; exit 0; }
    sleep 2
  done
  echo "llama-server başlamadı:"; tail -20 llama.log; exit 1
  ;;
stop)
  pkill -f "llama-serve[r] .*--port 8099" 2>/dev/null
  [ -f memlog.pid ] && kill "$(cat memlog.pid)" 2>/dev/null
  pkill -f "vm.sh memlo[g]" 2>/dev/null
  rm -f memlog.pid; echo "durduruldu"
  ;;
*) echo "kullanım: vm.sh info|start|stop|memlog"; exit 2 ;;
esac
