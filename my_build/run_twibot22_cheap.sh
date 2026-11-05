#!/bin/bash
# TwiBot-22 cheap models, one process per model (G1), exit codes logged (G2). CPU only; runs next to the
# GPU queue (one thread each). Log: my_build/results/twibot22/cheap_models.log
cd "$(dirname "$0")/.." || exit 1
L=my_build/results/twibot22/cheap_models.log
run() { echo "=== $(date +%H:%M) $*" >> $L; env OMP_NUM_THREADS=1 "$@" python -u my_build/twibot22_cheap_models.py >> $L 2>&1; echo "=== exit code $? ($*)" >> $L; }
for D in sample full; do
  [ $D = sample ] && [ -f my_build/results/twibot22/runs/sample/profile_only_s0.npz ] || run DESIGN=$D MODEL=profile_only
  run DESIGN=$D MODEL=flags
  run DESIGN=$D MODEL=links_only
  for P in 0 1; do
    run DESIGN=$D MODEL=mlp_text PROFILE=$P
    run DESIGN=$D MODEL=tree_self PROFILE=$P
    run DESIGN=$D MODEL=xgb_graph PROFILE=$P
  done
done
echo "=== $(date +%H:%M) all done" >> $L
