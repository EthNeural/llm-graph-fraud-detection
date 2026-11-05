# Meeting brief: Fri 6 Nov 2026, 9:00 (tutor, Macquarie)

Prajan Bhattarai, COMP8851. One page to talk from. Detail: `reports/STATUS.md`, `reports/report_for_submission.md`, `artifacts/Snapshot.md`.

## 1. The project in two sentences
Do LLM-based and graph-based methods help detect bots when the graph is heterophilous (bots camouflage by linking to normal users)? We test it on Instagram, TwiBot-20 and, as the main large test, TwiBot-22 (1M users, 7 relations, 88M tweets), measuring heterophily rather than assuming it.

## 2. What is done
- **Instagram, TwiBot-20:** full pipeline (encoders, 5 GNNs, zero-shot LLM, LoRA fine-tuning, a DGP replication), shortcut and crawl-artefact checks, a 22-item self-review.
- **TwiBot-22:** data checks (shortcuts: only `verified`, removed; heterophily on all 7 relations: adjusted homophily 0.01–0.04); our own random split with a **sealed test set** scored once at the end; every rule written down before any score.
- **Models on TwiBot-22:** text baselines, fine-tuned Qwen3-1.7B (LoRA), cheap models with/without profile on a 100k sample and on all 1M, 5 GNNs on the full 1M graph (2 best with 3 seeds), bootstrap CIs, a sceptic review, a literature check (citations verified).
- **Graph questions from the camouflage literature:** homophily by relation and class (Q1), local heterogeneity (Q2, MAGNET), cross-relation inconsistency (Q3, FRAUDRE), higher-order bot density (TROPICAL).
- **Expert labels (986):** used for training *and* clean evaluation by 5-fold rotation, for the LLM, the trees and the GNNs; run blind, read in the final run.

## 3. Main results (TwiBot-22, working test users, weak labels; macro-F1 / AUROC)

| Model | Score |
|---|---|
| Fine-tuned LLM on text (LoRA, 1,300 train) | 0.626 / 0.710 |
| Profile only (15 fields, tree) | 0.701 / 0.837 |
| Profile + text | 0.713 / 0.848 |
| **+ neighbour features (XGB-Graph, all 1M)** | **0.731 / 0.873** (best) |
| GNNs, full 1M graph (FAGCN, RGCN, SAGE, LINKX, GCN) | 0.678–0.690 / 0.80–0.83, within ~0.01 of the same net without edges (0.679 / 0.799) |

## 4. Findings (each with a paired 95% CI)
1. **Profile beats text** by a wide margin (+0.07 F1 over the LLM, also with the same 1,300 training users).
2. **Text adds a little** on top of profile (+0.01); averaging tweets in *hurts* a text-only model because it blurs the empty-bio cue (40.6% of bots vs 7.8% of humans have no bio). Tested: one empty-bio flag closes the gap.
3. **Neighbours help a little at full scale** (+0.01 F1, +0.05 PR-AUC), more without profile; GNN message passing adds little (heterophily).
4. **Local heterogeneity (MAGNET's question):** neighbour averaging helps users in homophilous neighbourhoods (+0.057 AUROC) and hurts users in heterophilous ones (−0.016). Mao et al. (NeurIPS 2023) predict this.
5. **Higher-order (TROPICAL's question):** bot density around bots *falls* with hop distance (0.116 → 0.059 over 1–4 hops); the Amazon pattern does not transfer to TwiBot-22.
6. **Cross-relation inconsistency (FRAUDRE's question):** 8 label-free features comparing follow vs interaction neighbourhoods add nothing (−0.002 [−0.004, −0.000] macro-F1, all 1M).
7. **The main conflict:** the TwiBot-22 paper ranks graph models first. Our reading of the paper: ranked by **accuracy** on a skewed official split (test 29.4% bots, so all-human = 70.6%); on bot-F1 a non-graph model is best; and the labels were partly produced by **GCN / GAT / R-GCN**, the same families that top the board. **Confirmed in their official code:** BotRGCN, GCN and GAT take `verified` as an input, use the skewed split, train a fixed 200 epochs with no validation selection, and report AUC from 0/1 predictions. A reconciliation run (our models under their protocol, with and without `verified`): section 7. Their repo also has a separate 500-user, 3-expert annotation study that never trained the label-makers: a cleaner check of the empty-bio cue for the final run.

## 5. Still to do (before Sunday 8 Nov, report day)
DGP (LLM with neighbour summaries) result early Fri, second LLM seeds by ~13:00; GNN expert stage (FAGCN) overnight; the sealed final run (sealed users + experts, read once); weak-vs-expert check (do graph models gain only on weak labels?); TwiBot-20 comparison; figures; report (7 pages).

## 6. Questions for the tutor
1. Report framing: is a mostly *negative* result for graph methods ("profile and simple neighbour features suffice; message passing adds little under measured heterophily") acceptable as the main outcome, if every claim is CI-backed and the leaderboard conflict is explained?
2. Scope for 7 pages: TwiBot-22 as the main dataset, TwiBot-20 and Instagram as shorter supporting sections — OK?
3. Is the weak-vs-expert-label comparison (986 experts, 5-fold rotation) a reasonable way to address label quality, or is something else expected?
4. Literature review: MAGNET, FRAUDRE, TROPICAL and the TKDD 2026 paper as the camouflage / heterophily line, plus GADBench, Hays et al. (WWW 2023) and Mao et al. (NeurIPS 2023) — anything missing?
5. Anything specific the marking expects in "Contributions" for a team project where my part is the TwiBot / LLM pipeline?

## 7. Leaderboard reconciliation (our models under the official protocol; partial, Thu 22:45)

Official split, our sealed users removed: test 94,889 users, **29.5% bots → calling everyone human scores 0.705 accuracy**.

| Model | `verified` in / out | Accuracy @0.5 | Macro-F1 | AUROC |
|---|---|---|---|---|
| Profile tree (no neighbours) | in / out | 0.763 / 0.763 | 0.700 / 0.695 | 0.765 / 0.764 |
| XGB-Graph + profile | in / out | 0.769 / 0.769 | 0.689 / 0.691 | 0.758 / 0.758 |
| MLP (no edges) | in / out | 0.627 / 0.644 | 0.683 / 0.674 | 0.745 / 0.734 |
| FAGCN | in / out | 0.603 / 0.625 | 0.667 / 0.678 | 0.731 / 0.745 |
| RGCN | in / out | 0.645 / *running* | 0.666 / – | 0.724 / – |
| *For reference: XGB-Graph + profile, our random split* | out | 0.869 | 0.731 | 0.873 |

**What it says so far**
1. **The split matters far more than `verified`.** AUROC drops from 0.873 (our random split) to 0.758 (official split); adding `verified` moves our models by ≤ 0.01.
2. **Graph models do not win under their protocol either.** On macro-F1 and AUROC the tree without neighbours is at least as good as XGB-Graph (XGB-Graph is 0.006 higher on accuracy only); FAGCN and RGCN stay within ~0.01 of the MLP without edges (FAGCN slightly above without `verified`, below with it).
3. **Caveat:** our GNNs use class weights (theirs do not), which trades accuracy for catching bots; GNN accuracy is therefore below all-human. Compare GNNs on macro-F1 / AUROC, not accuracy.

**Likely explanation of the leaderboard:** their ranking rests on accuracy on a skewed split, graph models that were given `verified`, fixed-epoch training, and labels partly made by the same graph-model families. Under one common protocol, we do not reproduce a graph advantage. *(Last cells, RGCN without `verified` and our split with `verified`, added ~00:30.)*
