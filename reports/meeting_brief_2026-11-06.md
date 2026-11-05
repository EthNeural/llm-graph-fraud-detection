# Meeting brief: Fri 6 Nov 2026, 9:00 (tutor, Macquarie)

Prajan Bhattarai, COMP8851. One page to talk from. Detail: `reports/STATUS.md`, `reports/report_for_submission.md`, `artifacts/Snapshot.md`.

## 1. The project in two sentences
Do LLM-based and graph-based methods help detect bots when the graph is heterophilous (bots camouflage by linking to normal users)? We test it on Instagram, TwiBot-20 and, as the main large test, TwiBot-22 (1M users, 7 relations, 88M tweets), measuring heterophily rather than assuming it.

## 2. What is done
- **Instagram, TwiBot-20:** full pipeline (encoders, 5 GNNs, zero-shot LLM, LoRA fine-tuning, a DGP replication), shortcut and crawl-artefact checks, a 22-item self-review.
- **TwiBot-22:** data checks (shortcuts: only `verified`, removed; heterophily on all 7 relations: adjusted homophily 0.01–0.04); our own random split with a **sealed test set** scored once at the end; every rule written down before any score.
- **Models on TwiBot-22:** text baselines, fine-tuned Qwen3-1.7B (LoRA), cheap models with/without profile on a 100k sample and on all 1M, 5 GNNs on the full 1M graph (2 best with 3 seeds), bootstrap CIs, a sceptic review, a literature check (citations verified).
- **Graph questions from the camouflage literature:** homophily by relation and class (Q1), local heterogeneity (Q2, MAGNET), cross-relation inconsistency (Q3, FRAUDRE), higher-order bot density (TROPICAL).
- **Expert labels (986):** used for training *and* clean evaluation by 5-fold rotation, for the LLM, the trees and the GNNs; run blind, **read early Thu 23:50** (my decision, recorded first; no model or rule changed after): section 8. The empty-bio test (D8) on the experts and on the independent annotation study: section 8.

## 3. Main results (TwiBot-22, working test users, weak labels; macro-F1 / AUROC)

| Model | Score |
|---|---|
| Fine-tuned LLM on text (LoRA, 1,300 train) | 0.626 / 0.710 |
| Profile only (15 fields, tree) | 0.701 / 0.837 |
| Profile + text | 0.713 / 0.848 |
| **+ neighbour features (XGB-Graph, all 1M)** | **0.731 / 0.873** (best) |
| GNNs, full 1M graph (FAGCN, RGCN, SAGE, LINKX, GCN) | 0.678–0.690 / 0.80–0.83, within ~0.01 of the same net without edges (0.679 / 0.799) |

## 4. Findings (each with a paired 95% CI)
1. **Profile beats text** by a wide margin (+0.07 F1 over the LLM, also with the same 1,300 training users). **Not just the empty-bio cue:** on users who *have* a bio (where the empty-bio flag is useless, AUROC 0.500), the profile model still scores AUROC 0.790 (sample) / 0.802 (1M), and XGB-Graph + profile 0.841 (1M).
2. **Text adds a little** on top of profile (+0.01); averaging tweets in *hurts* a text-only model because it blurs the empty-bio cue (40.6% of bots vs 7.8% of humans have no bio). Tested: one empty-bio flag closes the gap.
3. **Neighbours help a little at full scale** (+0.01 F1, +0.05 PR-AUC), more without profile; GNN message passing adds little (heterophily).
4. **Local heterogeneity (MAGNET's question):** neighbour averaging helps users in homophilous neighbourhoods (+0.057 AUROC) and hurts users in heterophilous ones (−0.016). Mao et al. (NeurIPS 2023) predict this.
5. **Higher-order (TROPICAL's question):** bot density around bots *falls* with hop distance (0.116 → 0.059 over 1–4 hops); the Amazon pattern does not transfer to TwiBot-22.
6. **Cross-relation inconsistency (FRAUDRE's question):** 8 label-free features comparing follow vs interaction neighbourhoods add nothing (−0.002 [−0.004, −0.000] macro-F1, all 1M).
7. **The main conflict:** the TwiBot-22 paper ranks graph models first. Our reading of the paper: ranked by **accuracy** on a skewed official split (test 29.4% bots, so all-human = 70.6%); on bot-F1 a non-graph model is best; and the labels were partly produced by **GCN / GAT / R-GCN**, the same families that top the board. **Confirmed in their official code:** BotRGCN, GCN and GAT take `verified` as an input, use the skewed split, train a fixed 200 epochs with no validation selection, and report AUC from 0/1 predictions. A reconciliation run (our models under their protocol, with and without `verified`): section 7. Their repo also has a separate 500-user, 3-expert annotation study that never trained the label-makers: used for D8 (section 8).

## 5. Still to do (before Sunday 8 Nov, report day)
DGP (LLM with neighbour summaries) result early Fri, second LLM seeds by ~13:00; FAGCN expert stage; the sealed final run on the non-expert sealed users (main table, D7: do graph models gain only on weak labels?); TwiBot-20 comparison; figures; report (7 pages).

## 6. Questions for the tutor
1. Report framing: is a mostly *negative* result for graph methods ("profile and simple neighbour features suffice; message passing adds little under measured heterophily") acceptable as the main outcome, if every claim is CI-backed and the leaderboard conflict is explained?
2. Scope for 7 pages: TwiBot-22 as the main dataset, TwiBot-20 and Instagram as shorter supporting sections — OK?
3. Label quality: the 986 expert labels trained the label-makers (not independent); the 500-user annotation study did not. Is it sound to lead the label-quality discussion with the annotation study (small: 44 bots) and use the 986 experts as the training / rotation set?
4. Literature review: MAGNET, FRAUDRE, TROPICAL and the TKDD 2026 paper as the camouflage / heterophily line, plus GADBench, Hays et al. (WWW 2023) and Mao et al. (NeurIPS 2023) — anything missing?
5. Anything specific the marking expects in "Contributions" for a team project where my part is the TwiBot / LLM pipeline?
6. Citation rule (≤ 5 years): may older foundational works (CARE-GNN 2020 for camouflage) be cited briefly?
7. The discriminative intra- / inter-relation model in the camouflage line: is it called DEFEND, and is it the TKDD 2026 paper ("Beyond Smoothing")?

## 7. Leaderboard reconciliation (our models under the official protocol; complete, Fri 00:40)

Official split, our sealed users removed: test 94,889 users, **29.5% bots → calling everyone human scores 0.705 accuracy**. Our split: test 49,951, 14.0% bots → all-human 0.860.

| Model | `verified` in / out | Accuracy @0.5 | Macro-F1 | AUROC |
|---|---|---|---|---|
| **Official split** | | | | |
| Own-feature tree (profile + bio, no neighbours) | in / out | 0.763 / 0.763 | 0.700 / 0.695 | 0.765 / 0.764 |
| XGB-Graph + profile | in / out | 0.769 / 0.769 | 0.689 / 0.691 | 0.758 / 0.758 |
| MLP (no edges) | in / out | 0.627 / 0.644 | 0.683 / 0.674 | 0.745 / 0.734 |
| FAGCN | in / out | 0.603 / 0.625 | 0.667 / 0.678 | 0.731 / 0.745 |
| RGCN | in / out | 0.645 / 0.640 | 0.666 / 0.653 | 0.724 / 0.726 |
| **Our random split** | | | | |
| Own-feature tree (profile + bio, no neighbours) | in / out | 0.878 / – | 0.721 / 0.720 | 0.863 / 0.860 |
| XGB-Graph + profile | in / out | 0.885 / 0.869 | 0.730 / 0.731 | 0.874 / 0.873 |
| MLP (no edges) | in / out | 0.770 / 0.843 | 0.686 / 0.679 | 0.824 / 0.799 |
| FAGCN | in / out | 0.816 / 0.850 | 0.674 / 0.689 | 0.799 / 0.828 |
| RGCN | in / out | 0.814 / 0.836 | 0.678 / 0.689 | 0.797 / 0.800 |

Seed 0 each. All models use the 1M bio-only features (15 profile fields + multilingual-E5 bio). "Out" on our split = the main results.

**What it says**
1. **The split matters far more than `verified`.** AUROC drops by ~0.11 from our random split to the official one, for every model. `verified` moves the trees by ≤ 0.01; for the GNNs it goes both ways (MLP +0.025 AUROC on our split, FAGCN −0.03), within single-seed noise of a few hundredths.
2. **Graph models do not win under their protocol either.** On the official split the tree without neighbours is the best on macro-F1 and AUROC; XGB-Graph is 0.006 higher only on accuracy; FAGCN and RGCN do not beat the MLP without edges (RGCN 0.02 below it without `verified`).
3. **Caveat:** our GNNs use class weights (theirs do not), which trades accuracy for catching bots, so their accuracy at 0.5 is below all-human. Compare GNNs on macro-F1 / AUROC.

**Likely explanation of the leaderboard:** their ranking rests on accuracy on a skewed split, graph models given `verified`, fixed-epoch training without validation selection, and labels partly made by the same graph-model families. Under one common protocol, we do not reproduce a graph advantage.

## 8. Expert labels, read early (986 experts, 105 bots; Thu 23:50, my decision)

Each expert is scored by a model that never saw that expert (5-fold rotation). 95% bootstrap CIs. FAGCN still running.

| Model | AUROC weak-only → after expert stage | Change [95% CI] | Macro-F1 after |
|---|---|---|---|
| LoRA `target` (text) | 0.822 → 0.724 | **−0.099** [−0.164, −0.038] | 0.735 |
| Profile only (tree) | 0.861 → 0.867 | +0.006 [−0.017, +0.030] | 0.736 |
| Profile + text (tree) | 0.879 → 0.899 | +0.020 [+0.000, +0.040] | 0.752 |
| **XGB-Graph + profile** (sample) | 0.883 → **0.910** | +0.027 [+0.009, +0.044] | **0.751** |
| XGB-Graph + profile (1M) | 0.875 → 0.887 | +0.012 [−0.002, +0.028] | 0.714 |
| MLP, no edges (1M) | 0.819 → 0.845 | +0.025 [+0.006, +0.046] | 0.728 |
| RGCN (1M) | 0.799 → 0.866 | **+0.067** [+0.034, +0.104] | 0.725 |
| *The dataset's own weak label as a prediction* | – | – | 0.745 |

- **The ranking holds on expert labels:** profile-based trees with neighbour features best; the GNN gains most from expert labels but stays below the trees; LoRA's ranking worsens (1 epoch on ~789 users disturbs the adapter; its F1 gain comes from the new cut-off).
- **90.5% agreement is not much:** on these 986 users, calling everyone human agrees 89.4%. On bots, the weak label and the experts agree for 56 of the 105 expert bots.

**D8: is "empty bio → bot" a real bot habit or a label-maker habit?** Lift = bot rate among empty-bio users ÷ bot rate among users with a bio.

| Users | Weak labels | Human labels | Weak − human |
|---|---|---|---|
| 986 experts (103 empty bio) | 8.7× [6.4, 12.3] | experts 7.2× [5.3, 10.3] | +1.5 [−1.5, +4.6] |
| 458 annotation-study users (44 bots, 62 empty bio) | 5.0× [3.3, 7.7] | annotators **1.6×** [0.7, 3.1] | **+3.4** [+1.2, +6.1] |

- **The 986 experts cannot settle it:** 800 of their labels trained the label-making models (TwiBot-22 paper §3.2), so agreement is built in. The annotation study (6 researchers, 3 per user) never trained them.
- **Reading:** under independent human labels the empty-bio cue is much weaker; the weak labels overstate it (difference excludes 0). Mostly a label-maker habit, possibly a small real effect (1.6× CI includes 1, sample small). Consistent with Hays et al. (WWW 2023), who warn that benchmark labels are artefacts of collection and labelling (studied TwiBot-20, not TwiBot-22); extending it to TwiBot-22 is, as far as we found, new.
