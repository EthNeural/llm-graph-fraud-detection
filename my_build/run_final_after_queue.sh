#!/bin/bash
# Fri 6 Nov ~03:25: run the sealed final run once the GPU queue has finished (builder's authorisation, decision 03:20).
# Waits for: the dry run passed, and queue_5nov.sh is no longer running. Then FINAL_RUN=1, exit code logged, commit, push.
cd "$(dirname "$0")/.." || exit 1
R=my_build/results/twibot22; N=reports/overnight_notes_2026-11-05.md
until grep -q "=== exit code 0" $R/final_run_dry.log 2>/dev/null; do
  grep -q "=== exit code [1-9]" $R/final_run_dry.log 2>/dev/null && { echo "- $(date +%H:%M) Final run NOT started: dry run failed." >> $N; exit 1; }
  sleep 60; done
while pgrep -f queue_5nov.sh >/dev/null; do sleep 120; done
echo "- $(date +%H:%M) FINAL RUN started (sealed non-expert users, read once)." >> $N
FINAL_RUN=1 OMP_NUM_THREADS=4 caffeinate -i python -u my_build/twibot22_final_run.py > $R/final_run.log 2>&1; c=$?
echo "=== exit code $c" >> $R/final_run.log
echo "- $(date +%H:%M) FINAL RUN finished, exit $c (final_run.json)." >> $N
git add my_build/twibot22_final_run.py my_build/run_final_after_queue.sh $R/final_run.json $R/final_run.log $R/final_run_dry.json $R/final_run_dry.log $N artifacts/04_decisions_and_reasons.md 2>/dev/null
git commit -qm "TwiBot-22 FINAL RUN on the sealed non-expert users (exit $c), read once

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3 4 5; do git push -q origin main && break; sleep 60; done
exit $c
