#!/bin/bash
# Overnight finisher (Fri 6 Nov ~01:00): works without the builder and without Claude, until 15:00.
#  1. RGCN reconciliation cell done -> fill it in the meeting brief section 7, commit.
#  2. First DGP comparison done -> append it to the meeting brief as section 9, commit.
#  2b. Second-seed comparison done -> append to the brief, commit.
#  3. GNN expert run done -> read the 986 experts again with FAGCN (expert_read_with_fagcn.json), commit.
#  4. Every 10 min: if local commits are not on GitHub, push (works whenever the internet is back).
cd "$(dirname "$0")/.." || exit 1
B=reports/meeting_brief_2026-11-06.md; R=my_build/results/twibot22; N=reports/overnight_notes_2026-11-05.md
CO="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
# retry: the GPU queue may hold the git lock for a moment
commit() { for i in 1 2 3 4 5 6; do git add "$@" 2>/dev/null && git commit -qm "$MSG

$CO" 2>/dev/null && return 0; sleep 30; done; echo "- $(date +%H:%M) finisher: commit failed ($MSG)" >> $N; }
d1=0; d2=0; d3=0; d4=0
end=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) 15:00" +%s)
while [ "$(date +%s)" -lt "$end" ]; do
  if [ $d1 = 0 ] && python -c "import json,sys; sys.exit(0 if 'ours_verified1/RGCN' in json.load(open('$R/official_protocol_gnn.json')) else 1)" 2>/dev/null; then
    python - <<'P'
import json
r=json.load(open("my_build/results/twibot22/official_protocol_gnn.json"))["ours_verified1/RGCN"]
p="reports/meeting_brief_2026-11-06.md"; s=open(p).read()
s=s.replace("| RGCN | in / out | *running* / 0.836 | – / 0.689 | – / 0.800 |",
 f"| RGCN | in / out | {r['acc_at_0_5']:.3f} / 0.836 | {r['macro_f1_at_val_cut']:.3f} / 0.689 | {r['auroc']:.3f} / 0.800 |")
open(p,"w").write(s)
P
    echo "- $(date +%H:%M) Reconciliation complete (last cell RGCN, our split, verified in); brief section 7 filled automatically." >> $N
    MSG="Reconciliation complete: last cell filled in the meeting brief (auto)"; commit $B $N $R/official_protocol_gnn.json $R/official_protocol.log; d1=1
  fi
  if [ $d2 = 0 ] && grep -q "dgp vs target on the 1,500 subset" $N 2>/dev/null; then
    { echo; echo "## 9. DGP result (added automatically $(date '+%a %H:%M'); 1,500-user test subset, seed 0, paired bootstrap of dgp − target)"; echo;
      echo "DGP = the LLM reading each account's text **plus summaries of its neighbours** (node and path summaries). Same 1,300 training users, same 1,000 validation users as LoRA \`target\`."; echo;
      echo '```'; grep -v -i warn $R/dgp_vs_target.log | cut -c1-300; echo '```'; } >> $B
    MSG="Meeting brief: DGP result section 9 (auto)"; commit $B; d2=1
  fi
  if [ $d4 = 0 ] && grep -q "Comparison (both seeds)" $N 2>/dev/null; then
    { echo; echo "### 9b. DGP vs target with 2 seeds (added automatically $(date '+%a %H:%M'))"; echo;
      echo '```'; grep -v -i warn $R/dgp_vs_target.log | cut -c1-300; echo '```'; } >> $B
    MSG="Meeting brief: DGP vs target with 2 seeds (auto)"; commit $B; d4=1
  fi
  if [ $d3 = 0 ] && [ -f $R/expert_rotation_gnn.npz ]; then
    WITH_FAGCN=1 OMP_NUM_THREADS=1 python my_build/twibot22_expert_read.py 2>&1 | grep -v -i warn > $R/expert_read_with_fagcn.log
    echo "- $(date +%H:%M) Expert read repeated with FAGCN (expert_read_with_fagcn.json):" >> $N
    { echo '```'; grep -E "^FAGCN|^ +diff" $R/expert_read_with_fagcn.log | tail -2; echo '```'; } >> $N
    MSG="Expert read with FAGCN (auto, after the GNN expert run)"; commit my_build/twibot22_expert_read.py $R/expert_read_with_fagcn.json $R/expert_read_with_fagcn.log $N; d3=1
  fi
  git fetch -q origin 2>/dev/null
  [ -n "$(git log origin/main..main --oneline 2>/dev/null)" ] && git push -q origin main 2>/dev/null
  sleep 600
done
