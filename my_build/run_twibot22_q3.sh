#!/bin/bash
# Q3 (E10): XGB-Graph + cross-relation inconsistency features. Exit codes logged (G2).
cd "$(dirname "$0")/.." || exit 1
L=my_build/results/twibot22/q3_inconsistency.log
run() { echo "=== $(date +%H:%M) $*" >> $L; env OMP_NUM_THREADS=1 EXTRA=inconsistency "$@" python -u my_build/twibot22_cheap_models.py >> $L 2>&1; echo "=== exit code $? ($*)" >> $L; }
run DESIGN=sample MODEL=xgb_graph PROFILE=1
run DESIGN=sample MODEL=xgb_graph PROFILE=0
run DESIGN=full MODEL=xgb_graph PROFILE=1 SEEDS=0
echo "=== $(date +%H:%M) all done" >> $L
