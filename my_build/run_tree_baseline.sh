#!/bin/bash
# TwiBot-20 tree baseline, settings A then B, CPU only (no GPU job is started or touched).
# Each setting runs MODEL=tree and MODEL=mlp as separate processes (torch and sklearn's
# HistGradientBoosting must not share a process: exit 139, no output), then MODEL=table.
# Every job's output AND exit code go to results/twibot20/<setting>/tree_baseline.log;
# the script stops with a clear message at the first non-zero exit code.
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1 DATASET=twibot20
for setting in A B; do
  log="my_build/results/twibot20/$setting/tree_baseline.log"
  : > "$log"
  for model in tree mlp table; do
    echo "=== $(date '+%F %T') SETTING=$setting MODEL=$model OMP_NUM_THREADS=$OMP_NUM_THREADS" >> "$log"
    SETTING=$setting MODEL=$model python -u my_build/tree_baseline.py >> "$log" 2>&1
    code=$?
    echo "=== exit code $code (SETTING=$setting MODEL=$model)" >> "$log"
    if [ $code -ne 0 ]; then
      echo "STOPPED: SETTING=$setting MODEL=$model exited with code $code. See $log" | tee -a "$log"
      exit $code
    fi
  done
done
echo "all done: exit codes 0 for A and B (tree, mlp, table)"
