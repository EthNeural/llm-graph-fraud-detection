#!/bin/bash
# TwiBot-20 GPU batch (reports/twibot20_todo.md G1-G4), Fri 6 Nov ~04:30. Starts when the TwiBot-22 GPU queue ends.
# G1 LoRA seed 2 for target and dgp (dgp s2 resumes its 4 Nov checkpoint); each followed by step6_rescore (last check),
# so all seeds use the same "chosen" adapter rule (full-validation macro-F1) in twibot20_c1_scores.py (C3).
# G4 matched-text control; G3 dgp_num (numeric profile summaries). Each step: exit code logged, local commit, push.
cd "$(dirname "$0")/.." || exit 1
R=my_build/results/twibot20/A; N=reports/overnight_notes_2026-11-05.md
CO="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
while pgrep -f queue_5nov.sh >/dev/null; do sleep 120; done
sleep 60
E2="env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASET=twibot20 SETTING=A"
step() { L=$1; shift; echo "- $(date +%H:%M) TwiBot-20 GPU: $L started." >> $N
  caffeinate -i $E2 "$@" > $R/$L.log 2>&1; c=$?; echo "=== exit code $c" >> $R/$L.log
  echo "- $(date +%H:%M) TwiBot-20 GPU: $L finished, exit $c." >> $N
  git add $R/$L.log $R/*.json $R/step6_*_s2*.npz $R/g4_* $N my_build/twibot20_g4_matched_text.py my_build/run_twibot20_gpu_batch.sh 2>/dev/null
  git commit -qm "TwiBot-20 GPU batch: $L (exit $c)

$CO" 2>/dev/null; for i in 1 2 3; do git push -q origin main 2>/dev/null && break; sleep 60; done; }
step step6_target_s2 env VARIANT=target LORA_SEED=2 python -u my_build/step6_dgp.py
step step6_rescore_target_s2 env VARIANT=target LORA_SEED=2 python -u my_build/step6_rescore.py
step step6_dgp_s2 env VARIANT=dgp LORA_SEED=2 python -u my_build/step6_dgp.py
step step6_rescore_dgp_s2 env VARIANT=dgp LORA_SEED=2 python -u my_build/step6_rescore.py
step g4_matched_text env VARIANT=target python -u my_build/twibot20_g4_matched_text.py
step step6_dgp_num env VARIANT=dgp_num python -u my_build/step6_dgp.py
echo "- $(date +%H:%M) TwiBot-20 GPU batch done." >> $N
