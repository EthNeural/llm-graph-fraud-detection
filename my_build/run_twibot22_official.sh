#!/bin/bash
# E11 leaderboard reconciliation: GNN cells and tree cells in parallel (exit codes logged, G2).
# GNNs never start a cell while a LoRA trains (DGP's LoRA or the seed runs): they wait for the GPU queue to end.
cd "$(dirname "$0")/.." || exit 1
L=my_build/results/twibot22/official_protocol.log; D=my_build/results/twibot22/step6_dgp.log
run() { echo "=== $(date +%H:%M) $*" >> $L; env REPLICATION=1 "$@" python -u my_build/twibot22_official_protocol.py >> $L 2>&1; echo "=== exit code $? ($*)" >> $L; }
( for c in "official 1" "official 0" "ours 1"; do set -- $c
    while grep -aq "trainable params" $D 2>/dev/null && pgrep -f queue_5nov.sh >/dev/null; do sleep 60; done
    run KIND=gnn SPLIT=$1 VERIFIED=$2 THREADS=4; done ) &
( for c in "official 1" "official 0" "ours 1"; do set -- $c; run KIND=trees SPLIT=$1 VERIFIED=$2 OMP_NUM_THREADS=1; done ) &
wait
echo "=== $(date +%H:%M) all done" >> $L
