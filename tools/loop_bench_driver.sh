#!/bin/bash
# Run a set of cells in lock-friendly chunks (each chunk = one flock acquisition, one server start).
# usage: loop_bench_driver.sh LABEL "SERVER ARGS" OUT.jsonl CHUNK N cell1 cell2 ... [-- extra loop_bench args]
LABEL=$1; SARGS=$2; OUT=$3; CHUNK=$4; N=$5; shift 5
CELLS=(); EXTRA=()
while [ $# -gt 0 ]; do [ "$1" = "--" ] && { shift; EXTRA=("$@"); break; }; CELLS+=("$1"); shift; done
HERE=$(cd "$(dirname "$0")" && pwd)
for c in "${CELLS[@]}"; do
  for ((s=${START:-0}; s<N; s+=CHUNK)); do
    e=$((s+CHUNK)); [ $e -gt $N ] && e=$N
    # resume support: skip generations already in OUT
    flock /tmp/qv_bench.lock $HERE/loop_bench_server.sh 8100 "$SARGS" --cells $c --start $s --n $e --label $LABEL --out $OUT --keep-text "${EXTRA[@]}"
  done
done
