#!/bin/bash
# Start a private qv_server, run tools/loop_bench.py against it, stop the server (by PID). Meant to run under
# `flock /tmp/qv_bench.lock`.  usage: loop_bench_server.sh PORT "SERVER ARGS" LOOP_BENCH_ARGS...
set -u
PORT=$1; SARGS=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=${QV_BUILD:-/tmp/loops-build}
MODEL=${QV_MODEL:-/workspace/q-volta-models/qv/inkling2-mini-q3a5.qv}
TOK=${QV_TOK:-/workspace/q-volta-models/Inkling-2-Mini/tokenizer.json}
$BUILD/qv_server --model $MODEL --tokenizer $TOK --backend cuda --port $PORT $SARGS > /tmp/loops-out/server-$PORT.log 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT
for i in $(seq 1 180); do
  curl -sf localhost:$PORT/health >/dev/null 2>&1 && break
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -5 /tmp/loops-out/server-$PORT.log; exit 1; }
  sleep 1
done
python3 -I $HERE/loop_bench.py run --url http://127.0.0.1:$PORT "$@"
