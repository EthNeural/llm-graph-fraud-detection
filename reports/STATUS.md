# Live status, updated Fri 6 Nov 2026, ~10:50

## Done
- **TwiBot-22:** everything, including the **sealed final run** (10:20, read once, `final_run.json`): sealed numbers replicate the working set (best: XGB-Graph + profile 0.735 / 0.874 / PR-AUC 0.590). D7: the neighbour-tree gain appears only under weak labels (weak − expert +0.018 [+0.008, +0.027] AUROC). DGP vs `target`, 2 seeds: no reliable gain (0.000 macro-F1, +0.015 AUROC, within seed noise). Expert read incl. FAGCN; D8; leaderboard reconciliation.
- **TwiBot-20 CPU batch:** C1 (headlines on non-verified users), C2 (GNNs with profile: none beats the MLP), C3 (seed-averaged LoRA; dgp − target AUROC +0.033 on non-verified), C4 (crawl-artefact gain disappears without verified users). C5 deferred. REPORT.md updated (R2, R3, R5, R6).
- **Report draft:** `artifacts/08_final_report.md` (~5,900 words; sealed numbers; fact-checked by a sceptic agent, corrections applied). Open: timeline (deferred), [TEAM] sections, TwiBot-20 seed 2.

## Running (unattended)
- **GPU, TwiBot-20 batch** (started 10:14): LoRA `target` seed 2 → rescore → `dgp` seed 2 → rescore → G4 matched-text control → G3 `dgp_num`; ~12 h, done ~22:00–02:00.

## Next
- When the TwiBot-20 batch lands: R1 / R4 edits, report §4.1 seed 2, then the humanized version of the report.
- Then the independent study (BotSim-24; stage 1 done, `reports/independent_study/`).
- Before submission: timeline, team sections, final fact-check.
