#!/bin/bash
# GNN expert rotation (blind). Pauses (SIGSTOP) while any LoRA trains, resumes (SIGCONT) when the GPU queue ends.
cd "$(dirname "$0")/.." || exit 1
L=my_build/results/twibot22/expert_rotation_gnn.log; D=my_build/results/twibot22/step6_dgp.log; N=reports/overnight_notes_2026-11-05.md
echo "- $(date +%H:%M) GNN expert rotation (MLP, RGCN, FAGCN; blind) started." >> $N
env THREADS=4 python -u my_build/twibot22_expert_rotation_gnn.py > $L 2>&1 &
P=$!
paused=0
while kill -0 $P 2>/dev/null; do
  if grep -aq "trainable params" $D 2>/dev/null && pgrep -f queue_5nov.sh >/dev/null; then
    [ $paused = 0 ] && kill -STOP $P && paused=1 && echo "- $(date +%H:%M) GNN expert rotation paused (LoRA training)." >> $N
  elif [ $paused = 1 ]; then kill -CONT $P; paused=0; echo "- $(date +%H:%M) GNN expert rotation resumed." >> $N; fi
  sleep 20
done
wait $P; c=$?; echo "=== exit code $c" >> $L
echo "- $(date +%H:%M) GNN expert rotation finished, exit $c (blind)." >> $N
git add my_build/twibot22_expert_rotation_gnn.py my_build/run_twibot22_expert_gnn.sh $L my_build/results/twibot22/expert_rotation_gnn.npz $N
git commit -qm "TwiBot-22 GNN expert rotation, blind (exit $c)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3 4 5; do git push -q origin main && break; sleep 60; done
