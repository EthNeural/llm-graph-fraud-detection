#!/bin/zsh
# Run the whole pipeline for both settings, then print the comparison table.
#   zsh my_build/run_all.sh          (from the repo root)
# Embeddings and summaries are cached, so re-runs only redo the cheap parts.
set -e
cd "$(dirname "$0")/.."

python my_build/step1_explore.py > /dev/null      # sanity check only
for S in A B; do
  echo "\n################ SETTING $S ################"
  SETTING=$S python my_build/step2_baseline.py
  SETTING=$S python my_build/step3_embed.py
  SETTING=$S python my_build/step4_gnn.py
  SETTING=$S python my_build/step5_llm_reason.py
done
python my_build/compare.py
