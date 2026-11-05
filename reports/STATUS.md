# Live status (TwiBot-22), updated Fri 6 Nov 2026, 00:50

Detail: meeting brief `reports/meeting_brief_2026-11-06.md` (results, reconciliation §7, expert results and D8 §8); run log `reports/overnight_notes_2026-11-05.md`; findings with CIs `reports/report_for_submission.md`; everything in one file `artifacts/Snapshot.md`.

## Running now (unattended; each step pushes itself)

| Where | Job | Expected end |
|---|---|---|
| GPU | **DGP**: path summaries → LoRA training + scoring → dgp vs target (1,500 users) | ~04:00–05:00 |
| GPU | LoRA `target` seed 1 → `dgp` seed 1 → dgp vs target with 2 seeds | ~13:00 |
| CPU | Reconciliation: last cell (RGCN, our split, `verified` in) | ~01:15 |
| CPU | FAGCN expert stage (pauses while a LoRA trains) | ~01:00, or after the GPU queue |

## Done

- Data checks, split, sealed set, rules before scores.
- All models on the working test users (weak labels): text baselines, zero-shot, LoRA, cheap models (sample and 1M), 5 GNNs (FAGCN, RGCN 3 seeds). Best: XGB-Graph + profile, 1M: macro-F1 0.731, AUROC 0.873, PR-AUC 0.580.
- CIs, sceptic review, literature check (Hays et al. verified 6 Nov), Q1–Q3, TROPICAL check, empty-bio flag and with-bio slice.
- Official-code check and reconciliation (all but one cell): the split explains most of the gap; `verified` little; no graph advantage under their protocol.
- **Expert results read early** (builder's decision, Thu 23:50; LoRA, trees, MLP, RGCN): the ranking holds on expert labels (best XGB-Graph + profile, AUROC 0.910); RGCN gains most (+0.067) but stays below the trees; LoRA worsens (−0.099).
- **D8:** empty-bio lift 5.0× under weak labels vs 1.6× under independent human labels (same 458 users; difference +3.4 [+1.2, +6.1]).

## Still to do

- Fri 9:00 tutor meeting.
- Fri afternoon: final run on the sealed non-expert users (headline table, D7), FAGCN expert result, dgp vs target (2 seeds), comparison with TwiBot-20.
- Sat: figures. Sun 8 Nov: report (7 pages).
