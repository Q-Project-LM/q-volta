#!/bin/bash
# Start qv_server, run longctx_replay.py against it, stop the server (only the PID started here).
# Usage: longctx_server_replay.sh SERVER_BIN OUT.jsonl "SERVER FLAGS" REPLAY_ARGS...   (hold flock /tmp/qv_bench.lock around the call)
set -u
BIN=$1; OUT=$2; FLAGS=$3; shift 3
PORT=${PORT:-8230}
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${QV_MODEL:-/workspace/q-volta-models/qv/inkling2-mini-q3a5.qv}
TOK=${QV_TOK:-/workspace/q-volta-models/Inkling-2-Mini/tokenizer.json}
LOG=${OUT%.jsonl}.server.log
$BIN --model "$MODEL" --tokenizer "$TOK" --port $PORT $FLAGS > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "server died"; tail -5 "$LOG"; exit 1; }
  sleep 2
done
python3 -I "$HERE/longctx_replay.py" "$@" --url "http://127.0.0.1:$PORT" --out "$OUT"
