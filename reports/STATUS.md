# Live status (TwiBot-22), updated Thu 5 Nov 2026, 18:50

Detailed log with every exit code: `reports/overnight_notes_2026-11-05.md`. Sceptic-checked findings with CIs: `reports/report_for_submission.md` (Outcome section).

## Running now

| Where | Job | Progress | Expected end |
|---|---|---|---|
| GPU | **DGP** (Dual Granularity Prompting), TwiBot-20 scope: neighbour summaries | 15 of 57 chunks (~5.8 min each) | summaries ~23:00, LoRA ~02:30, scored ~03:00 |
| CPU | GNN seeds 1-2 for the 2 best (FAGCN, RGCN) | FAGCN seed 2 | ~19:45 |

Queued on the GPU after DGP (keeps it busy until the builder is up): LoRA `target` seed 1 (~4 h), LoRA `dgp` seed 1 (~3.5 h, summaries cached), then dgp vs target on the 1,500-user subset with both seeds. GNNs pause automatically if any LoRA trains.

## Done today

| Job | Result (working-test users) |
|---|---|
| LoRA `target` (Qwen3-1.7B, 1,300 train, 1 seed) | macro-F1 0.626, AUROC 0.710, PR-AUC 0.312 |
| Zero-shot Qwen (no training) | 0.505 / 0.461 (says "bot" for almost everyone; readout checked, not inverted) |
| Text baselines | TF-IDF+LR 70k train 0.628 / 0.727; MiniLM+MLP 0.607 / 0.684 |
| H10 sample features (830 columns), shortcut rule | no shortcuts |
| Cheap models, sample + 1M, with / without profile, 3 seeds | best: XGB-Graph + profile on 1M, 0.731 / 0.873 / PR-AUC 0.580; profile only 0.701 / 0.837 |
| Expert rotation for LoRA target (5 folds) | done, **blind**: scores saved, read only in the final run |
| GNNs, full 1M graph, seed 0 | see table below |
| Bootstrap CIs + sceptic check | in `report_for_submission.md` |

## GNNs on the full 1M graph (own profile + bio, 7 relations)

| Model | Seeds | Val F1 | Test F1 | AUROC | PR-AUC |
|---|---|---|---|---|---|
| FAGCN | 3 (running) | 0.692 | 0.689 | 0.828 | 0.449 |
| RGCN | 3 (queued) | 0.691 | 0.689 | 0.800 | 0.441 |
| LINKX | 1 | 0.684 | 0.681 | 0.811 | 0.430 |
| SAGE | 1 | 0.683 | 0.684 | 0.810 | 0.448 |
| GCN | 1 | 0.679 | 0.678 | 0.808 | 0.431 |
| MLP control (no edges) | 1 | 0.681 | 0.679 | 0.799 | 0.415 |

Early read: every GNN is within ~0.01 F1 of the MLP control, and all trail the tree models (own profile + bio: 0.720 / 0.860). Seed-0 numbers; seeds 1-2 still running for FAGCN and RGCN.

## Still to do

DEC-lite; expert rotation for trees / MLP; D7 (weak vs expert gains); sealed final run (`FINAL_RUN=1`); TwiBot-20 comparison; TwiBot-20 batch if time allows.
