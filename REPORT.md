# Progress Report

A living report, updated as experiments finish. The newest entries are at the top of each
section. For how everything works, see [TEAM_GUIDE.md](TEAM_GUIDE.md).

**Last updated:** 3 November 2026, 14:15

**Scope.** TwiBot-20 is the main dataset: steps 1–6, settings A and B. Instagram is the secondary dataset, steps 1–5 only: it checks whether the step 1–5 findings hold on a second, multilingual graph. Step 6 is not planned for Instagram. One step 6 run takes about 3–6 h on our Mac, so that time goes to the main dataset.

## Status

| Experiment | TwiBot-20 A | TwiBot-20 B | Instagram A | Instagram B |
|---|---|---|---|---|
| Step 1: homophily | ✅ | ✅ | ✅ | ✅ |
| Step 2: baselines | ✅ | ✅ | ✅ | ✅ |
| Step 3: encoders | ✅ 6 encoders | ✅ 6 encoders | ✅ 3 encoders | ✅ 3 encoders |
| Step 4: MLP / GCN / SAGE / FAGCN | ✅ + bridge graph | ✅ | ✅ | ✅ |
| Step 5: zero-shot reasoner | ✅ | ✅ | ✅ | ✅ |
| Step 6: DGP (LoRA), dgp | ✅ 1 seed | ⏳ | not in scope | not in scope |
| Step 6: DGP (LoRA), target | ✅ 1 seed | ⏳ | not in scope | not in scope |
| FLAG baseline | ⏳ | ⏳ | ⏳ | ⏳ |

✅ done · 🔄 running · ⏳ to do

**Current run (GPU, one job at a time):** step 3 Qwen3-1.7B embeddings (started 3 Nov 15:42), then step 6 `dgp` and `target` on TwiBot-20 B, then rescoring of the epoch-2 LoRA adapters (`step6_rescore.py`). Expected to end about 05:30 on 4 Nov.
- Done today: step 6 `target` on TwiBot-20 A (13:30–15:42, early stop after epoch 2).
- On CPU alongside: shuffled-edge control (step 4), then the setting B bootstrap.

## Latest results (macro-F1, test set)

**Which test set?** Each heading says so. Most tables use the **full test set**: TwiBot-20 1,183 users (A) and 597 (B); Instagram 2,268 (A) and 1,590 (B). One table, "LLM reasoner vs trained models", uses the smaller **step 5 test sample**, because LLM scoring is slow. Its trained rows are also a single seed, so the same model can show a slightly different number there (for example, Qwen2.5 + MLP is 0.766 below and 0.747 there).

### TwiBot-20, encoder ladder (full test set, mean of 3 seeds)

| Encoder | A: MLP | A: GCN | A: SAGE | A: FAGCN | B: MLP | B: GCN | B: SAGE | B: FAGCN |
|---|---|---|---|---|---|---|---|---|
| MiniLM | 0.726 | 0.686 | 0.721 | 0.722 | 0.588 | 0.550 | 0.565 | 0.584 |
| E5-base (English) | 0.760 | 0.700 | 0.740 | 0.733 | 0.564 | 0.589 | 0.567 | 0.542 |
| Twitter-RoBERTa | 0.749 | 0.692 | 0.743 | 0.722 | 0.612 | 0.579 | 0.609 | 0.571 |
| multilingual-E5 | 0.747 | 0.699 | 0.741 | 0.726 | 0.606 | 0.585 | 0.589 | 0.567 |
| Qwen3-Embedding | 0.727 | 0.691 | 0.707 | 0.710 | 0.599 | 0.574 | 0.558 | 0.579 |
| Qwen2.5-1.5B | **0.766** | 0.707 | 0.750 | 0.744 | 0.613 | 0.577 | **0.624** | 0.554 |
| Graph only | – | 0.577 | 0.568 | 0.562 | – | 0.497 | 0.497 | 0.492 |

FAGCN (Bo et al., AAAI 2021; PyG `FAConv`, eps 0.3) is built for heterophily: each edge gets a learned weight in [−1, 1], so a neighbour can be subtracted as well as added. It has two propagation layers between an input and an output linear layer, the same hidden size (128) and about the same parameter count as GCN (197k). It also uses the same learning rate, weight decay, early stopping and seeds. Its own settings (eps 0.3, dropout 0.5) are fixed, not tuned, like every GNN here. In setting B (1:10), FAGCN is below MLP for every encoder (by 0.004–0.058), and its seeds vary more (std up to 0.02).

References for TwiBot-20 A: TF-IDF 0.702; metadata without "verified" 0.755 (not a valid model).

### Instagram (secondary dataset, steps 1–5): full test set; LR one fit, MLP and GNNs mean of 3 seeds

| Encoder | A: LR | A: MLP | A: GCN | A: SAGE | A: FAGCN | B: LR | B: MLP | B: GCN | B: SAGE | B: FAGCN |
|---|---|---|---|---|---|---|---|---|---|---|
| MiniLM | 0.588 | 0.585 | 0.607 | 0.588 | 0.596 | 0.546 | 0.536 | 0.552 | 0.543 | 0.566 |
| multilingual-E5 | **0.612** | 0.605 | 0.606 | 0.602 | **0.613** | 0.563 | 0.561 | 0.555 | 0.572 | 0.560 |
| Qwen2.5-1.5B | 0.608 | 0.607 | 0.609 | 0.604 | 0.606 | **0.578** | 0.575 | 0.549 | 0.562 | 0.534 |
| Graph only | – | – | 0.548 | 0.544 | 0.522 | – | – | 0.500 | 0.498 | 0.494 |

References: TF-IDF (balanced) 0.578 (A), 0.552 (B). On Instagram, every model sits within about 0.03 of LR. No graph model is consistently better or worse than text alone.

### LLM reasoner vs trained models (step 5 test sample, 1 seed)

**Sample sizes:** 400 test users for TwiBot-20 A, 300 for Instagram A and 600 for Instagram B, stratified, the same users for every row in a column. For TwiBot-20 B the "sample" is the whole test set (597). The trained rows here are one run (seed 0), not the mean of 3 seeds used elsewhere.

Two different LLMs appear here. **Step 5 uses Qwen2.5-1.5B-Instruct** (zero-shot only). **Step 6 uses Qwen3-1.7B**, the same family as the DGP paper's Qwen3-8B, both zero-shot and fine-tuned with LoRA. The trained baselines use Qwen2.5-1.5B *embeddings* (step 3) with LR or GCN on top. Every row is scored on the same step 5 test sample.

| Step | Model | TwiBot-20 A | TwiBot-20 B | Instagram A | Instagram B |
|---|---|---|---|---|---|
| 3–4 | Qwen2.5-1.5B embeddings + LR (trained) | **0.749** | **0.633** | 0.607 | 0.546 |
| 3–4 | Qwen2.5-1.5B embeddings + GCN (trained) | 0.698 | 0.573 | **0.623** | **0.562** |
| 5 | Qwen2.5-1.5B-Instruct zero-shot: target only | 0.584 | 0.541 | 0.553 | 0.522 |
| 5 | Qwen2.5-1.5B-Instruct zero-shot: raw neighbours | 0.591 | 0.547 | 0.552 | 0.499 |
| 5 | Qwen2.5-1.5B-Instruct zero-shot: DGP summaries | 0.550 | 0.541 | 0.537 | 0.538 |
| 6 | Qwen3-1.7B zero-shot: target only | 0.590 | ⏳ | not in scope | not in scope |
| 6 | Qwen3-1.7B zero-shot: DGP summaries | 0.562 | ⏳ | not in scope | not in scope |
| 6 | Qwen3-1.7B LoRA: target only | 0.694 | ⏳ | not in scope | not in scope |
| 6 | Qwen3-1.7B LoRA: DGP summaries | 0.726 | ⏳ | not in scope | not in scope |

**Validation samples.** LLM scoring is slow, so the zero-shot rows (steps 5 and 6) pick their cut-off on a stratified validation *sample*: 200 users for TwiBot-20 A, 400 for B, 150 for Instagram A and 300 for Instagram B. The trained models pick theirs on the full validation set. The step 6 LoRA runs use that same sample for early stopping, then pick their cut-off on the full validation set (2,365 users for TwiBot-20 A).

The two "zero-shot: DGP summaries" rows (0.550 and 0.562) differ in the LLM and in how the summaries were made. Step 5 used its own summaries; step 6 used the paper's diffusion trimming and 10-token summaries. So they are not a like-for-like comparison of the two LLMs.

LoRA dgp is one seed (42). On the full TwiBot-20 A test set (1,183 users) it scores 0.711 macro-F1, AUROC 0.794. Every model picks its own cut-off by the same rule: the one that maximises macro-F1 on validation, applied unchanged to test. On the 400-user step 5 sample its AUROC (0.828) is the highest, but that does not hold on the full test set: AUROC 0.794, against 0.822 for Qwen2.5 + LR and 0.840 for MLP. So it is behind on ranking too, not just on the cut-off. Even the best cut-off chosen on test itself (not a valid model) gives only 0.726.

### Where GCN loses: connected vs isolated test users (TwiBot-20 A, full test set, mean of 3 seeds)

| Qwen2.5-1.5B + | Connected (1,096 users) | Isolated (87 users) |
|---|---|---|
| MLP | **0.767** | 0.743 |
| GCN | 0.701 | **0.777** |
| SAGE | 0.752 | 0.734 |
| FAGCN | 0.742 | 0.770 |

GCN's whole loss is on connected users, where neighbours are averaged in. On isolated users GCN has nothing to average, so it acts like an MLP. The isolated group is only 87 users, so that column is noisy.

### How much label signal do the edges carry? (all graphs)

Plain edge homophily is misleading when classes are unbalanced, so this table also gives two measures from Platonov et al. ([arXiv 2209.06177](https://arxiv.org/abs/2209.06177)) that correct for chance. **Adjusted homophily** is 0 when edges join same-label users no more often than chance. **Label informativeness** (0 to 1) says how much a neighbour's label tells you about yours. Both use all labels, test included: they describe the data and are never a model input. From `my_build/graph_stats.py` (`results/graph_stats.json`).

| Graph | Edges | Edge homophily | Chance level | Adjusted homophily | Label informativeness |
|---|---|---|---|---|---|
| TwiBot-20 A, all edges | 16,349 | 0.532 | 0.501 | 0.062 | 0.003 |
| TwiBot-20 A, friend only | 11,819 | 0.508 | 0.515 | −0.015 | 0.000 |
| TwiBot-20 A, follow only | 4,922 | 0.591 | 0.554 | 0.081 | 0.005 |
| TwiBot-20 A, + bridge edges | 46,362 | 0.588 | 0.501 | 0.175 | 0.022 |
| TwiBot-20 B, all edges | 5,274 | 0.879 | 0.878 | 0.014 | 0.000 |
| Instagram A, all edges | 72,005 | 0.586 | 0.551 | 0.078 | 0.005 |
| Instagram B, all edges | 38,193 | 0.861 | 0.856 | 0.033 | 0.002 |

Every graph sits near zero once chance is removed. Setting B's high edge homophily (0.86–0.88) only reflects the 1:10 ratio: almost every edge touches the large class. Friend links on TwiBot-20 carry no label signal at all. The bridge edges raise adjusted homophily to 0.175, yet GCN still gets worse on them (robustness check below).

### TwiBot-20 graph: dropped support users

Our TwiBot-20 graph keeps only the 11,826 labelled users. The benchmark also ships 217,754 unlabelled support users, and its own GCN/GAT code keeps them as neighbours. `my_build/playground_bridges.py` counts what we lost (`results/twibot20/support_bridges.json`).

- 8,082 support users (3.7%) are **bridges**: linked to 2 or more labelled users. 171,896 (79%) link to exactly one, so they connect nothing. The largest hub has only 25 labelled neighbours.
- 786 of our 923 isolated users sit next to a bridge (75 of the 87 isolated test users).

| Graph | Edges | Edge homophily |
|---|---|---|
| Random-edge baseline | – | 0.507 |
| Ours (labelled only) | 16,349 | 0.532 |
| Bridge edges only (labelled–support–labelled, collapsed) | 30,240 | 0.619 |
| Ours + bridge edges | 46,362 | 0.588 |

Homophily here uses all labels, test included: it describes the data and is never a model input. The published TwiBot-20 GCN/GAT scores (F1 0.809 / 0.853) are binary F1 with profile metadata, so they are not comparable with our text-only macro-F1.

### Control: training-set size (TwiBot-20 A)

Step 6 fine-tunes on **1,300** training users (the DGP paper's size); the text-only models above train on all **8,278**. `my_build/control_train_size.py` retrains Qwen2.5-1.5B + LR / MLP on the exact 1,300 users step 6 used. Validation, cut-off and test sets are unchanged. MLP is the mean of 3 seeds.

| Model | Training users | Full test set: macro-F1 | Full test set: AUROC | Step 5 sample: macro-F1 |
|---|---|---|---|---|
| Qwen2.5-1.5B + MLP | 8,278 | **0.766** | **0.840** | 0.754 |
| Qwen2.5-1.5B + LR | 8,278 | 0.756 | 0.822 | 0.749 |
| Qwen2.5-1.5B + MLP | 1,300 | 0.726 | 0.796 | 0.723 |
| Qwen2.5-1.5B + LR | 1,300 | 0.725 | 0.786 | 0.712 |
| Qwen3-1.7B LoRA + DGP summaries (1 seed) | 1,300 | 0.711 | 0.794 | 0.726 |
| Qwen3-1.7B LoRA, target text only (1 seed) | 1,300 | 0.704 | 0.782 | 0.694 |

With the same 1,300 users, the text-only models lose about 0.03–0.04, and the gap to LoRA + DGP shrinks from 0.055 to 0.015 on the full test set. On the step 5 sample LoRA is level with them. The full-train rows reproduce steps 3 and 4 exactly, which checks the setup.

**Learning curve** (Qwen2.5-1.5B embeddings, full test set, MLP mean of 3 seeds; `my_build/bootstrap_ci.py`):

| Training users | 325 | 650 | 1,300 | 2,600 | 5,200 | 8,278 |
|---|---|---|---|---|---|---|
| LR | 0.714 | 0.720 | 0.725 | 0.743 | 0.754 | 0.757 |
| MLP | 0.697 | 0.713 | 0.726 | 0.751 | 0.754 | 0.766 |

LoRA + DGP at 1,300 users (0.711) sits between the 650- and 1,300-user text-only points.

### Are the gaps real? Paired bootstrap 95% CIs (TwiBot-20 A, full test set)

Each pair is scored on the same 2,000 resamples of the 1,183 test users; seeded models count as the mean of their 3 seeds. A CI that excludes 0 means the gap is larger than test-set noise. LoRA is one seed, so its CI covers test-set noise only, not seed noise.

| Comparison (Qwen2.5-1.5B embeddings unless LoRA) | Difference | 95% CI | Excludes 0? |
|---|---|---|---|
| GCN − MLP (all train) | −0.059 | [−0.082, −0.037] | yes |
| FAGCN − MLP (all train) | −0.021 | [−0.039, −0.004] | yes |
| SAGE − MLP (all train) | −0.015 | [−0.031, +0.000] | no (borderline) |
| FAGCN − GCN (all train) | +0.038 | [+0.022, +0.055] | yes |
| MLP: 1,300 − all train | −0.040 | [−0.055, −0.023] | yes |
| LoRA + DGP − MLP (both 1,300) | −0.014 | [−0.041, +0.013] | no |
| LoRA + DGP − LR (both 1,300) | −0.014 | [−0.042, +0.016] | no |
| LoRA + DGP (1,300) − MLP (all train) | −0.054 | [−0.080, −0.027] | yes |

### Control: the same GNNs on shuffled edges (full test set, mean of 3 seeds)

`GRAPH=shuffled` keeps every user's degree but rewires who links to whom at random (`my_build/step4_gnn.py`, `step4_gnn_shuffled.json`). If a GNN scores the same on shuffled edges, the real edges give it nothing.

| Model | TwiBot-20 A: real edges | A: shuffled | TwiBot-20 B: real edges | B: shuffled |
|---|---|---|---|---|
| Qwen2.5 + MLP (no edges) | 0.766 | 0.766 | 0.613 | 0.613 |
| Qwen2.5 + GCN | 0.707 | 0.659 | 0.577 | 0.573 |
| Qwen2.5 + SAGE | 0.750 | 0.751 | 0.624 | 0.634 |
| Qwen2.5 + FAGCN | 0.744 | 0.723 | 0.554 | 0.563 |
| E5-base + GCN | 0.700 | 0.652 | 0.589 | 0.560 |
| E5-base + SAGE | 0.740 | 0.736 | 0.567 | 0.574 |
| E5-base + FAGCN | 0.733 | 0.727 | 0.542 | 0.569 |
| Graph only (GCN) | 0.577 | 0.554 | 0.497 | 0.497 |

- **In A the real edges carry a little signal**: GCN loses 0.05 more on random edges, FAGCN 0.01–0.02, and graph-only drops 0.023. This matches the small positive adjusted homophily (0.062). It is far less than what GCN's averaging costs (−0.059 against MLP).
- **SAGE does not care**: it scores the same on real and random edges, because it can learn to ignore the neighbour average.
- **In B the real edges are no better than random** (within about one standard deviation everywhere), as adjusted homophily 0.014 predicts.

### Robustness check: step 4 on the bridge graph (TwiBot-20 A, full test set)

Step 4 rerun with the bridge edges added (`GRAPH=bridges`, `results/twibot20/A/step4_gnn_bridges.json`): 92,724 edge rows instead of 32,698, and 137 isolated users instead of 923. **This is a robustness check only. Steps 5 and 6 and all tables above stay on the labelled-only graph.** "Isolated" still means isolated in the labelled-only graph, so the groups are the same users in both runs. Macro-F1, mean of 3 seeds:

| Encoder | MLP | GCN: original → bridges | SAGE: original → bridges | FAGCN: original → bridges |
|---|---|---|---|---|
| MiniLM | 0.726 | 0.686 → 0.651 (−0.035) | 0.721 → 0.725 (+0.005) | 0.722 → 0.716 (−0.007) |
| E5-base (English) | 0.760 | 0.700 → 0.671 (−0.029) | 0.740 → 0.746 (+0.006) | 0.733 → 0.744 (+0.010) |
| Twitter-RoBERTa | 0.749 | 0.692 → 0.670 (−0.021) | 0.743 → 0.741 (−0.002) | 0.722 → 0.730 (+0.007) |
| multilingual-E5 | 0.747 | 0.699 → 0.676 (−0.023) | 0.741 → 0.741 (0.000) | 0.726 → 0.735 (+0.008) |
| Qwen3-Embedding | 0.727 | 0.691 → 0.665 (−0.025) | 0.707 → 0.719 (+0.012) | 0.710 → 0.716 (+0.006) |
| Qwen2.5-1.5B | **0.766** | 0.707 → 0.677 (−0.029) | 0.750 → **0.761** (+0.010) | 0.744 → 0.743 (−0.001) |
| Graph only | – | 0.577 → 0.538 | 0.568 → 0.532 | 0.562 → 0.519 |

- MLP is identical in both runs (it ignores edges), which confirms only the graph changed.
- GCN drops on every encoder (std 0.002–0.010), most on users who were isolated before (Qwen2.5: 0.777 → 0.666). Average degree rises from about 2.8 to about 7.8, so GCN's averaging gives each user's own text less weight, and homophily 0.588 is still close to the 0.507 random baseline.
- SAGE, which keeps its own features separate, gains slightly: Qwen2.5 + SAGE comes within 0.005 of MLP.
- FAGCN, which can give an edge a negative weight, holds steady or gains slightly (−0.007 to +0.010). The extra edges do not hurt it the way they hurt GCN.
- Graph-only drops by about 0.04. One hypothesis, not tested: in the labelled-only graph, degree partly reflects a user's position in the dataset's BFS crawl, and the bridge edges blur that.

## Limitations (so far)

- **One seed for step 6** (the DGP paper uses 5), and Qwen3-1.7B instead of the paper's Qwen3-8B. The LoRA CIs cover test-set noise only.
- **LoRA reads less text than the encoders.** Its prompt caps the user's own text at 480 tokens of whole lines, which cuts 81% of TwiBot-20 users short (13.2 of 16.6 lines kept on average, 800-user sample). The encoders read every line. This biases LoRA vs MLP against LoRA; dgp vs target is unaffected (same cap).
- **LoRA and the other models choose their checkpoint differently.** LoRA keeps the lowest validation loss on a 200-user sample; MLP and GNNs keep the best full-validation macro-F1. In both TwiBot-20 A LoRA runs the loss was lowest at epoch 1 while validation AUROC kept rising (dgp 0.748 → 0.772, target 0.765 → 0.781). The epoch-2 adapters are being scored with the full-validation rule (`my_build/step6_rescore.py`).
- **GNN hyperparameters are fixed and shared, not tuned.**
- **Setting B is small and changes the graph.** Its test set has 54 bots; one random bot subsample (seed 42); 29% of users are isolated in B against 7.8% in A (37% of B's test bots). Its paired bootstrap CIs are about ±0.05: GCN − MLP −0.036 [−0.087, +0.012], SAGE − MLP +0.012 [−0.034, +0.058], FAGCN − MLP −0.058 [−0.112, −0.002]. Only the FAGCN gap is outside noise. B's learning curve is flat (MLP 0.62 at 325 users, 0.61 at 4,000), so B cannot separate most models. `results/twibot20/B/bootstrap_ci.json`.
- **TwiBot-20 graph = labelled users only** (see "dropped support users"; bridge check below).
- **Edge signal near zero**, so these results speak to near-random edges, not to strongly heterophilous graphs (see the adjusted-homophily table).

## Findings so far

1. **A good text encoder is where the gains are; the graph adds nothing.** On TwiBot-20 A, Qwen2.5 + MLP (0.766) beats TF-IDF (0.702) by 0.06. The neural encoders span 0.04 (0.726–0.766), and no graph model improves on its encoder's MLP.
2. **GCN hurts under heterophily; a heterophily GNN only recovers the loss.** GCN is below MLP for every encoder on TwiBot-20 A, by 0.04–0.06. FAGCN, built for heterophily, beats GCN on every encoder (by 0.02–0.04) but stays 0.004–0.027 below MLP. So on this graph, no GNN we tried beats ignoring the edges, with the same fixed, untuned hyperparameters for all of them. The GCN and FAGCN gaps are outside test-set noise (paired bootstrap); SAGE's is borderline. SAGE stays within about 0.02 of MLP. In B, Qwen2.5 + SAGE edges past MLP on macro-F1 (0.624 vs 0.613, within about one standard deviation), but not on the threshold-free metrics (AUROC 0.749 vs 0.788, PR-AUC 0.251 vs 0.269). E5-base is the one encoder where GCN beats MLP in B (0.589 vs 0.564, GCN std 0.027). *Robustness:* this is on the labelled-only graph, whose edges are barely above random (homophily 0.532 vs 0.507). Adding the more homophilous bridge edges (0.619) makes GCN worse on every encoder, by 0.02–0.04, while SAGE gains up to 0.012. So the GCN result is not an artifact of dropping the support users. On Instagram, all graph models stay within about 0.03 of text-only LR in both settings.
3. **The edges carry almost no label signal.** Adjusted homophily is −0.015 to 0.08 on every original graph, and label informativeness is at most 0.005. This is consistent with findings 2 and 4: averaging neighbours mostly adds noise. On degree-preserving shuffled edges, GCN loses a further 0.05 in A and nothing in B: the edges carry a little signal in A and none in B (see the shuffled-edge control).
4. **The graph alone is weak:** 0.577 (TwiBot-20 A), and 0.497 in B, near the floor. On the bridge graph it drops further, to 0.538.
5. **The zero-shot LLM reasoner (Qwen2.5-1.5B-Instruct, step 5) loses to trained classifiers** on every dataset and setting. Adding neighbours (raw or summaries) does not reliably help.
6. **Size is not everything.** Twitter-RoBERTa (125M) beats Qwen3-Embedding (0.6B). On Instagram A the encoders are tied (LR 0.588–0.612, MLP 0.585–0.607).
7. **Fine-tuning helps a lot; the gap to text-only models is mostly training data.** On TwiBot-20 A (step 5 sample), Qwen3-1.7B with DGP summaries goes from 0.562 zero-shot to 0.726 with LoRA. Against text-only models trained on all 8,278 users it is behind (full test set: 0.711 vs 0.756 LR, 0.766 MLP). Trained on the same 1,300 users as LoRA, those models drop to 0.725–0.726, so the remaining gap is only 0.014, and its 95% CI [−0.041, +0.013] includes 0 (see "Control: training-set size" and the bootstrap table). One seed; the DGP trimming had little to trim on this sparse graph (mean 0.8–1.8 neighbours kept per 1-hop metapath, limit 8). **Do the DGP neighbour summaries help?** Barely: LoRA with summaries scores 0.711 on the full test set against 0.704 for the same model on the user's own text (AUROC 0.794 vs 0.782); on the step 5 sample 0.726 vs 0.694. Zero-shot, the summaries even hurt (0.562 vs 0.590). A paired bootstrap for dgp vs target is queued.

## Decision log

| Date | Decision | Why |
|---|---|---|
| 3 Nov | Report adjusted homophily and label informativeness next to edge homophily | Plain homophily is inflated by class imbalance (0.88 in setting B, 0.014 adjusted); the standard fix from the heterophily literature |
| 3 Nov | Training-set size control: LR / MLP retrained on step 6's 1,300 users | LoRA trains on 1,300 users and the text-only models on 8,278, which confounds "encoder vs predictor" with data size |
| 3 Nov | Qwen3-1.7B added as a step 3 encoder (same checkpoint as step 6, mean-pooled like Qwen2.5) | Removes the model-version confound between the encoder rows (Qwen2.5) and the LoRA rows (Qwen3) |
| 3 Nov | TwiBot-20 = main dataset (steps 1–6); Instagram = secondary (steps 1–5) | TwiBot-20 has the benchmark split, typed friend/follow relations and real bot labels. Step 6 costs 3–6 h per run, so it goes to the main dataset |
| 3 Nov | Added FAGCN (PyG `FAConv`, eps 0.3) to step 4 | A GNN built for heterophily: signed edge weights can subtract a neighbour. Tests whether finding 2 is about GCN or about graphs |
| 3 Nov | Bridge graph used only as a step 4 robustness check; steps 5 and 6 stay on the labelled-only graph | Keeps all main tables on one graph; the step 4 check already answers whether the graph construction drove finding 2 |
| 3 Nov | Measured the dropped support users (bridges, homophily) before changing any model | Findings 2, 3 and 6 depend on the graph; the benchmark keeps support users as neighbours |
| 2 Nov | Step 6 summaries generated in batches of 4 on TwiBot-20 | At 16, the GPU memory cache grew to 16 GB and the Mac swapped |
| 2 Nov | Step 6 relations treated as undirected | Kept directed, only 38% of users had a friend link; undirected, 93% have a neighbour (matches steps 4–5) |
| 2 Nov | Step 6 LLM = Qwen3-1.7B, not step 5's Qwen2.5-1.5B-Instruct | Same family as the paper's Qwen3-8B; every LLM row names its model |
| 2 Nov | Step 6 = DGP replication with Qwen3-1.7B, 1,300 training users, B=10, K=2, M=8, r=16, lr 1e-4, one seed, no numeric summaries | Closest to the paper that fits a 16 GB Mac; the official code is not public yet |
| 2 Nov | Diffusion features X = multilingual-E5 embeddings | The paper says "raw node features"; our raw features are text |
| 1 Nov | TwiBot-20 graph = labelled users only; the 217,754 support users dropped | No labels and no text needed for them, so prep stays small. The effect on the graph was not checked then (see 3 Nov) |
| 1 Nov | Added TwiBot-20 (official split, bio + 20 tweets) | A second, larger heterophilous dataset with real text |
| 1 Nov | Excluded TwiBot-20 metadata | "Verified" alone gives macro-F1 0.803: a shortcut, not text or graph signal |
| 1 Nov | Encoder ladder: six encoders, everything else fixed | To separate encoder size and type from the graph's effect |
| 30 Oct | Threshold chosen on validation for every model | Scores are not calibrated; 0.5 is badly placed when class 1 is rare |
| 30 Oct | Settings A (natural) and B (1:10, FLAG-style) | Tests both the data as it is and rare fraud |
| 27 Oct | Dropped Elliptic, rebuilt on Instagram | Elliptic has no natural text for an LLM to read |

## Next steps

1. Step 6 on TwiBot-20 A, `VARIANT=target` (running).
2. Step 4 FAGCN on TwiBot-20 B (CPU, about 30 min).
3. Step 6 on TwiBot-20 B (`dgp` and `target`).
4. Run the FLAG baseline on our splits (`results/<setting>/flag.json`).
5. Optional, if time allows: 3 seeds for step 6, and an M ∈ {4, 8, 16} sweep.

## Changelog

- **3 Nov 2026 (16:00):** Shuffled-edge control (A: real edges slightly better than random for GCN/FAGCN; B: no difference; SAGE indifferent). Setting B paired bootstrap: CIs about ±0.05, only FAGCN − MLP excludes 0.
- **3 Nov 2026 (15:45):** Step 6 `target` on TwiBot-20 A: LoRA 0.704 (full test), 0.694 (step 5 sample); zero-shot 0.590. DGP summaries add +0.007 on the full test set.
- **3 Nov 2026 (16:00):** After an independent review: connected/isolated table; Limitations section; fixed ranges in findings 2 and the B FAGCN note; setting B SAGE claim checked against AUROC/PR-AUC; finding 1 reworded. Queued step 6 rescoring of the epoch-2 adapters and a shuffled-edge control.
- **3 Nov 2026 (15:00):** Adjusted homophily and label informativeness for every graph (near 0 everywhere). Learning curve and paired bootstrap CIs on TwiBot-20 A: the LoRA vs MLP gap at equal training size is not significant.
- **3 Nov 2026 (14:30):** FAGCN on TwiBot-20 B; Instagram graph-only GCN/SAGE rows. Rerunning Instagram GCN/SAGE reproduced every earlier number exactly.
- **3 Nov 2026 (14:15):** FAGCN results on TwiBot-20 A (both graphs) and Instagram A and B. Instagram step 4 now has MLP rows and the full MLP / GCN / SAGE / FAGCN table. Corrected the claim that DGP ranks best: on the full test set its AUROC is 0.794, against 0.822 for LR and 0.840 for MLP. Added a note on validation samples.
- **3 Nov 2026 (14:00):** TwiBot-20 made the main dataset, Instagram the secondary one (steps 1–5); every LLM row now names its model. FAGCN added to step 4.
- **3 Nov 2026 (afternoon):** Step 4 robustness check on the bridge graph: GCN drops 0.02–0.04 on every encoder, SAGE gains up to 0.012. Steps 5 and 6 stay on the labelled-only graph. Started step 6 `VARIANT=target` on TwiBot-20 A.
- **3 Nov 2026:** TwiBot-20 A step 6 `dgp` finished (LoRA 0.726 on the step 5 sample, 0.711 on the full test set). Counted the dropped support users: 8,082 bridges; bridge edges are more homophilous (0.619) than our graph (0.532).
- **2 Nov 2026:** TwiBot-20 step 5 finished (A and B). Step 6 (DGP replication) written and started on TwiBot-20 A. Repository created.
