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
| Step 6: DGP (LoRA), target | 🔄 running | ⏳ | not in scope | not in scope |
| FLAG baseline | ⏳ | ⏳ | ⏳ | ⏳ |

✅ done · 🔄 running · ⏳ to do

**Current run:** TwiBot-20 A, `VARIANT=target` (log `results/twibot20/A/step6_target.log`).
- Started 3 Nov ~13:30. No summaries needed, so it goes straight to zero-shot scoring and LoRA training.
- Expected ~2.5–3.5 h (prompts are a median 495 tokens, close to dgp's, so training speed is similar). It resumes from checkpoints if stopped.
- Done earlier: `VARIANT=dgp` (3 Nov 02:40, ~5.7 h) and step 4 on the bridge graph (~12 min).
- On CPU alongside it (done 13:55): step 4 FAGCN on TwiBot-20 A (both graphs), and Instagram A and B MLP + FAGCN rows.

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

FAGCN (Bo et al., AAAI 2021; PyG `FAConv`, eps 0.3) is built for heterophily: each edge gets a learned weight in [−1, 1], so a neighbour can be subtracted as well as added. It has two propagation layers between an input and an output linear layer, the same hidden size (128) and about the same parameter count as GCN (197k). It also uses the same learning rate, weight decay, early stopping and seeds. Its own settings (eps 0.3, dropout 0.5) are fixed, not tuned, like every GNN here. In setting B (1:10), FAGCN is below MLP for every encoder (by 0.006–0.059), and its seeds vary more (std up to 0.02).

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
| 6 | Qwen3-1.7B zero-shot: DGP summaries | 0.562 | ⏳ | not in scope | not in scope |
| 6 | Qwen3-1.7B LoRA: target only | 🔄 | ⏳ | not in scope | not in scope |
| 6 | Qwen3-1.7B LoRA: DGP summaries | 0.726 | ⏳ | not in scope | not in scope |

**Validation samples.** LLM scoring is slow, so the zero-shot rows (steps 5 and 6) pick their cut-off on a stratified validation *sample*: 200 users for TwiBot-20 A, 400 for B, 150 for Instagram A and 300 for Instagram B. The trained models pick theirs on the full validation set. The step 6 LoRA runs use that same sample for early stopping, then pick their cut-off on the full validation set (2,365 users for TwiBot-20 A).

The two "zero-shot: DGP summaries" rows (0.550 and 0.562) differ in the LLM and in how the summaries were made. Step 5 used its own summaries; step 6 used the paper's diffusion trimming and 10-token summaries. So they are not a like-for-like comparison of the two LLMs.

LoRA dgp is one seed (42). On the full TwiBot-20 A test set (1,183 users) it scores 0.711 macro-F1, AUROC 0.794. Every model picks its own cut-off by the same rule: the one that maximises macro-F1 on validation, applied unchanged to test. On the 400-user step 5 sample its AUROC (0.828) is the highest, but that does not hold on the full test set: AUROC 0.794, against 0.822 for Qwen2.5 + LR and 0.840 for MLP. So it is behind on ranking too, not just on the cut-off. Even the best cut-off chosen on test itself (not a valid model) gives only 0.726.

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

With the same 1,300 users, the text-only models lose about 0.03–0.04, and the gap to LoRA + DGP shrinks from 0.055 to 0.015 on the full test set. On the step 5 sample LoRA is level with them. The full-train rows reproduce steps 3 and 4 exactly, which checks the setup.

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

## Findings so far

1. **The encoder matters most.** On TwiBot-20 A, Qwen2.5 + MLP (0.766) beats TF-IDF (0.702) by more than any graph model adds.
2. **GCN hurts under heterophily; a heterophily GNN only recovers the loss.** GCN is below MLP for every encoder on TwiBot-20 A, by 0.04–0.06. FAGCN, built for heterophily, beats GCN on every encoder (by 0.02–0.04) but stays 0.02–0.03 below MLP. So on this graph, no GNN we tried beats ignoring the edges, with the same fixed, untuned hyperparameters for all of them. SAGE stays within about 0.02 of MLP. In B, Qwen2.5 + SAGE edges past MLP (0.624 vs 0.613), but the gap is within about one standard deviation. *Robustness:* this is on the labelled-only graph, whose edges are barely above random (homophily 0.532 vs 0.507). Adding the more homophilous bridge edges (0.619) makes GCN worse on every encoder, by 0.02–0.04, while SAGE gains up to 0.012. So the GCN result is not an artifact of dropping the support users. On Instagram, all graph models stay within about 0.03 of text-only LR in both settings.
3. **The graph alone is weak:** 0.577 (TwiBot-20 A), and 0.497 in B, near the floor. On the bridge graph it drops further, to 0.538.
4. **The zero-shot LLM reasoner (Qwen2.5-1.5B-Instruct, step 5) loses to trained classifiers** on every dataset and setting. Adding neighbours (raw or summaries) does not reliably help.
5. **Size is not everything.** Twitter-RoBERTa (125M) beats Qwen3-Embedding (0.6B). Multilingual-E5 is the best text-only encoder on multilingual Instagram A.
6. **Fine-tuning helps a lot; the gap to text-only models is mostly training data.** On TwiBot-20 A (step 5 sample), Qwen3-1.7B with DGP summaries goes from 0.562 zero-shot to 0.726 with LoRA. Against text-only models trained on all 8,278 users it is behind (full test set: 0.711 vs 0.756 LR, 0.766 MLP). Trained on the same 1,300 users as LoRA, those models drop to 0.725–0.726, so the remaining gap is only 0.015 (see "Control: training-set size"). One seed; the DGP trimming had little to trim on this sparse graph (mean 0.8–1.8 neighbours kept per 1-hop metapath, limit 8). Still open: the `target` variant, to see if the summaries help at all.

## Decision log

| Date | Decision | Why |
|---|---|---|
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

- **3 Nov 2026 (14:30):** FAGCN on TwiBot-20 B; Instagram graph-only GCN/SAGE rows. Rerunning Instagram GCN/SAGE reproduced every earlier number exactly.
- **3 Nov 2026 (14:15):** FAGCN results on TwiBot-20 A (both graphs) and Instagram A and B. Instagram step 4 now has MLP rows and the full MLP / GCN / SAGE / FAGCN table. Corrected the claim that DGP ranks best: on the full test set its AUROC is 0.794, against 0.822 for LR and 0.840 for MLP. Added a note on validation samples.
- **3 Nov 2026 (14:00):** TwiBot-20 made the main dataset, Instagram the secondary one (steps 1–5); every LLM row now names its model. FAGCN added to step 4.
- **3 Nov 2026 (afternoon):** Step 4 robustness check on the bridge graph: GCN drops 0.02–0.04 on every encoder, SAGE gains up to 0.012. Steps 5 and 6 stay on the labelled-only graph. Started step 6 `VARIANT=target` on TwiBot-20 A.
- **3 Nov 2026:** TwiBot-20 A step 6 `dgp` finished (LoRA 0.726 on the step 5 sample, 0.711 on the full test set). Counted the dropped support users: 8,082 bridges; bridge edges are more homophilous (0.619) than our graph (0.532).
- **2 Nov 2026:** TwiBot-20 step 5 finished (A and B). Step 6 (DGP replication) written and started on TwiBot-20 A. Repository created.
