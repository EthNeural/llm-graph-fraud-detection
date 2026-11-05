# Independent study: results log (BotSim-24, working test users = 291, 100 bots; sealed part unread)

## Stage 1 (CPU, 02:50; `my_build/independent/botsim_stage1.py`, `results/independent/botsim_stage1.json`)
| Model (no leak fields) | Macro-F1 | AUROC | PR-AUC |
|---|---|---|---|
| In-degree probe (untrained; the artefact) | 0.781 | 0.833 | 0.622 |
| Profile + timing tree (18 clean features, 3 seeds) | 0.886 | 0.970 | 0.945 |
| + out-neighbour features (G-out) | 0.940 | 0.988 | 0.978 |
| + in-neighbour features too (G-full) | 0.945 | 0.992 | 0.984 |
| TF-IDF + LR on all text | **1.000** | **1.000** | **1.000** |
| Fusion (profile + cross-fitted text score + G-out) | 0.998 | 1.000 | 1.000 |
| Leak contrast (profile + zero-score share) | 0.996 | 1.000 | 1.000 |
Paired CIs: G-out − profile +0.067 [+0.031, +0.103] macro-F1; **G-full − G-out +0.001 [−0.023, +0.027]** (in-edge artefact adds ~nothing to trees that already see out-neighbours); text − profile +0.124 [+0.086, +0.165]; fusion − text 0.000.

## Stage 1b (02:55; shortcut checks after TF-IDF = 1.000; `botsim_stage1b.json`)
- Comments only (no headline-style posts): macro-F1 0.988, AUROC 0.999.
- Text needed: 1 random comment per user AUROC 0.964; 2 → 0.993; ≥ 5 → 0.999.
- **One comment on its own: AUROC 0.990** (8,074 test comments).
- Untrained rate of published LLM-overused words (Juzek & Ward 2024; Kobak et al. 2025): AUROC **0.844**. Top TF-IDF bot words: tensions, political, accountability, ongoing, highlights, implications, landscape, amid, dynamics, raises, crucial; human words: says, he, people, would, my, you.
- Profile/timing tree driven by comment length, median gap between actions, post length (permutation importance).

**Reading so far.** With exact labels and the vote/score/id/description leaks removed, BotSim-24's GPT-4o-mini bots are **trivially detectable from their text** by a bag-of-words model, even from one comment, contradicting the paper's statement that LLM text is "nearly impossible to distinguish" (their text baselines: Wei et al. F1 50.4, RoBERTa+NN 30.5). Graph features add +0.067 macro-F1 over clean profile/timing but nothing once text is used; the in-edge artefact adds nothing to neighbour-feature trees. Caveat: one generator (GPT-4o-mini), one prompt design; human text was cleaned to resemble GPT text (App. C.1), so this is a lower bound on the style gap only for this dataset.
