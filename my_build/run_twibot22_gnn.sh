#!/bin/bash
# TwiBot-22 GNNs on the full graph (builder, 5 Nov): all 5 + MLP control with seed 0, then the 2 best GNNs
# by VALIDATION macro-F1 with seeds 1 and 2. Exit codes logged (G2). Log: results/twibot22/gnn_full.log
# Runs only while no LoRA trains: starts when DGP's summary phase starts; a watcher pauses (SIGSTOP) the
# GNN when LoRA training starts and resumes it (SIGCONT) when the GPU queue has ended.
cd "$(dirname "$0")/.." || exit 1
L=my_build/results/twibot22/gnn_full.log; D=my_build/results/twibot22/step6_dgp.log
N=reports/overnight_notes_2026-11-05.md
until [ -f $D ] && grep -aq "node summaries" $D; do sleep 30; done
echo "- $(date +%H:%M) GNNs started (DGP summary phase: no LoRA training)." >> $N
run() { echo "=== $(date +%H:%M) $*" >> $L; env THREADS=6 "$@" python -u my_build/twibot22_gnn.py >> $L 2>&1; echo "=== exit code $? ($*)" >> $L; }
( while true; do                                   # pause / resume watcher
    p=$(pgrep -f "python -u my_build/twibot22_gnn.py"); [ -z "$p" ] && ! pgrep -f run_twibot22_gnn.sh >/dev/null && exit
    if grep -aq "trainable params" $D 2>/dev/null && pgrep -f queue_5nov.sh >/dev/null; then
      [ -n "$p" ] && kill -STOP $p 2>/dev/null && [ ! -f /tmp/.gnn_paused ] && touch /tmp/.gnn_paused && echo "- $(date +%H:%M) GNN paused (LoRA training started)." >> $N
    else
      [ -n "$p" ] && kill -CONT $p 2>/dev/null; [ -f /tmp/.gnn_paused ] && rm /tmp/.gnn_paused && echo "- $(date +%H:%M) GNN resumed (GPU queue ended)." >> $N
    fi; sleep 20; done ) &
run MODELS=FAGCN,RGCN,SAGE,GCN,LINKX,MLP SEEDS=0
best=$(python3 -c "
import json; r=json.load(open('my_build/results/twibot22/gnn_full.json'))
g=sorted([k[:-3] for k in r if k.endswith('_s0') and not k.startswith('MLP')], key=lambda m: -r[m+'_s0']['val_macro_f1'])
print(','.join(g[:2]))")
echo "- $(date +%H:%M) GNN seed 0 done; 2 best by validation macro-F1: $best -> seeds 1, 2." >> $N
run MODELS=$best SEEDS=1,2
echo "=== $(date +%H:%M) all done" >> $L
echo "- $(date +%H:%M) GNNs all done." >> $N
git add my_build/twibot22_gnn.py my_build/run_twibot22_gnn.sh $L my_build/results/twibot22/gnn_full.json $N
git commit -qm "TwiBot-22 GNNs on the full graph: 5 + MLP control seed 0, 2 best ($best) seeds 1-2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3 4 5; do git push -q origin main && break; sleep 60; done
