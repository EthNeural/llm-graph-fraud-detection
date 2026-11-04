# Handoff: COMP8851 LLM + graph fraud detection (state at Wed 4 Nov 2026, ~12:40)

This file has three parts:
- **Part 1**: a summary written for you, Claude, so you can pick up the work.
- **Part 2**: the full working report (`REPORT.md`), which has every result, caveat and decision. It was last updated on 4 Nov at 05:40.
- **Part 3**: the DGP replication audit (`reports/dgp_replication_audit.md`).

Where Part 1 and Part 2 disagree, Part 1 is newer.

---

# Part 1: summary

## Who and what
- **User:** Prajan Bhattarai, a Macquarie University student taking COMP8851 (Major Project). Team of 4; supervisor Dr. Venus Haghighi.
- **Topic brief:** "LLM-based approaches for graph fraud detection". The supervisor wants the evaluation to deal with **heterophily** (camouflaged fraud).
- **The user's own build** lives in `my_build/` (Python, PyTorch/PyG, Hugging Face, on a 16 GB M-series Mac with MPS; one GPU job at a time).
  - The team's original Elliptic pipeline (`src/`, `run_all.py`) was dropped on purpose ("Elliptic has no natural text"). Don't steer back to it.
- **Repo:** the user's own GitHub repo, https://github.com/EthNeural/llm-graph-fraud-detection, branch `main`. Only push there.
  - Never touch the teammate Abhisekh's repo (abh123abh/...).

## How the user wants to work
- **Code:** the user asks for complete, well-commented code, then checks the results. They also want to understand and defend everything as their own work.
- **Teaching:** when asked to explain, go one step at a time, in plain language with a concrete data example. End with 2–3 explain-back questions and don't move on until they can explain it back.
- **Naming:** always write **"multilingual-E5"**, never plain "E5". The code key is `e5` and the model is `intfloat/multilingual-e5-base`. English E5-base is a separate encoder, written "E5-base (English)".
- **Two reports:**
  1. `REPORT.md` is the complete working report. Never trim it.
  2. The university submission is a separate document, built at the end.
- **Submission limits:** 7 A4 pages at 12 pt, about 2,200–2,500 words plus 4–5 figures. The lit review is about 400 words.
  - Required template sections: Introduction (for a layperson), Literature Review, Research Methodology (with a timeline vs the original plan and the deviations), Research Outcome (results, analysis, discussion), Individual Contributions.
  - Still to confirm with the unit guide: whether references and an appendix count toward the 7 pages, and the line spacing.
  - An earlier final-report Claude Doc exists: https://claude.ai/artifact/5DYMwRdMuegyWdgfULvwRK
- **Diagrams:** the user wants 4–5 explanatory diagrams in `my_build/figures/`:
  1. the pipeline, steps 1–6, with the 3 LLM roles (encoder, zero-shot predictor, fine-tuned predictor);
  2. DGP step 6: metapaths → diffusion trimming → summaries → prompt;
  3. how TwiBot-20 was crawled and the "who lists whom" artefact;
  4. the graph variants (labelled-only, bridges, shuffled, similar-10);
  5. GCN averaging vs SAGE/FAGCN/skip keeping a user's own features.

  `my_build/figures/class_homophily.png` already exists.

## Deadline
- **All experiments finish by Sat 7 Nov 2026, night.**
- **Sun 8 Nov:** writing only.
- Budget GPU time accordingly; after Friday, prefer CPU-only or short runs.

## Datasets
**TwiBot-20 (main dataset; steps 1–6)**
- 11,826 labelled users (6,589 bots, 55.7%); official split 8,278 / 2,365 / 1,183; 16,349 undirected friend/follow edges between labelled users.
- Text per user = bio + first 20 tweets, capped at 2,000 characters.
- Prepared by `my_build/prep_twibot20.py` into `data/twibot-20/`.
- Profile metadata: "verified" is a documented shortcut (macro-F1 0.803 alone) and is excluded. Five other fields pass a shortcut rule fixed in advance; they form a separate "text + profile" track worth +0.04–0.05.

**Instagram (secondary dataset; steps 1–5)**
- The GLBench graph that FLAG also uses: 11,339 users, 72,005 edges, bios only.
- Class 1 = commercial account (a stand-in for a minority class, not fraud).

**Settings**
- **A:** natural class balance.
- **B:** 1:10 subsample of class 1 (FLAG's protocol).
  - TwiBot-20 B has 54 test bots, so its CIs are about ±0.05.

## Pipeline (`my_build/`)
- `common.py`: loading, splits, `evaluate()` (macro-F1, AUROC, PR-AUC, connected vs isolated). The `DATASET` and `SETTING` environment variables select the data.
- `step1_explore.py`: homophily.
- `step2_baseline.py`: TF-IDF.
- `step3_embed.py`: 7 encoders.
- `step4_gnn.py`: MLP, GCN, SAGE, FAGCN, skip-GCN, LINKX and FLAG-lite. The `GRAPH` variable picks the graph: real, bridges, shuffled, similar10 or synthetic.
- `step5_llm_reason.py`: zero-shot Qwen2.5-1.5B-Instruct.
- `step6_dgp.py`: a DGP replication (Li et al., AAAI 2026) with Qwen3-1.7B LoRA.
  - Variants: `dgp`, `target`, `dgp_num`.
  - Other settings: `LORA_SEED`, `LORA_ADAPTER`.
- `step6_rescore.py`: scores the last-check adapter.
- Controls and analyses: `bootstrap_ci.py`, `control_train_size.py`, `heterophily_sweep.py`, `tune_step4.py`, `tree_baseline.py`, `metadata_check.py`, `link_direction.py`, `edge_similarity.py`, `graph_stats.py`, `class_homophily.py`, `duplicate_text.py`, `clean_text_check.py`.
- Results go to `my_build/results/<dataset>/<setting>/`; `compare.py` prints the tables.

## Headline findings (details in Part 2)
1. **A strong text encoder does the work.** On TwiBot-20 A, Qwen3-1.7B + MLP scores 0.769 and Qwen2.5-1.5B + MLP 0.766, against TF-IDF 0.702.
2. **No GNN beats MLP on TwiBot-20 A.**
   - GCN is −0.059 [−0.082, −0.037]. SAGE, FAGCN, skip-GCN, FLAG-lite and LINKX each lose 0.01–0.02.
   - "Keep your own features" (skip/SAGE/FAGCN eps) recovers most of GCN's loss.
   - On Instagram A, GCN is level with MLP.
3. **Neighbour labels carry almost nothing.** Adjusted homophily is −0.015 to 0.08 and label informativeness ≤ 0.005.
   - Linked users write alike, regardless of label.
   - The real label signal sits in link *direction* (who lists whom). That is a crawl artefact (Hays et al., WWW 2023) and tracks follower count.
   - FLAG's "camouflage" histogram pattern is what chance predicts under 1:10 imbalance.
4. **Zero-shot LLMs lose to trained models.** Fine-tuned LoRA (1,300 users) roughly matches text-only models trained on the same 1,300 users, but stays below them when they use all 8,278.
5. **DGP neighbour summaries vs target-only LoRA, unseeded run 0:** +0.007 [−0.015, +0.030].
   - Under the full-validation checkpoint rule it is −0.001.
   - In A, target LoRA + 3 link flags matches `dgp`.

## NEW since REPORT.md was last updated (not yet in the report; results uncommitted)
1. **Step 6 seed 1 on TwiBot-20 A (`LORA_SEED=1`), full test set.**

   | Variant | Adapter | Macro-F1 | AUROC | PR-AUC | Full-validation macro-F1 |
   |---|---|---|---|---|---|
   | `dgp` | lowest validation loss (epoch 3) | 0.745 | 0.827 | 0.855 | 0.760 |
   | `dgp` | last check (epoch 4) | 0.720 | 0.790 | – | 0.722 |
   | `target` | – | 0.728 | 0.812 | 0.846 | – |

   - **`dgp`:** the full-validation rule keeps the lowest-loss adapter (epoch 3).
   - **`target`:** the last-check rescore was **still running** at 12:40 (PID 48965, `step6_rescore.py`, `VARIANT=target LORA_SEED=1`).
   - **On the step 5 sample:** `dgp` 0.761 and `target` 0.718.
   - **What it means:** in seed 1, `dgp` − `target` is about +0.017 on the full test set, larger than in run 0. Seed 1 `dgp` (0.745) is also the best LoRA so far, but still below MLP on all 8,278 users (0.766).
   - **Status:** seed 2 was in the original plan (the chain is about 11 h for seeds 1 and 2). Check whether it has run (`ls my_build/results/twibot20/A/*s2*`).
   - **Report:** seeds 0–2 as mean ± std under ONE pre-declared checkpoint rule.
2. **Tuned GNNs, TwiBot-20 A** (`tune_step4.py`, Luo et al. 2024-style grid; picked on validation; test, 3 seeds):

   | Model | Tuned | Untuned |
   |---|---|---|
   | MLP | 0.767 | 0.766 |
   | GCN | 0.706 | 0.707 |
   | SAGE | 0.755 | 0.750 |
   | FAGCN | 0.752 | 0.744 |

   Tuning doesn't change the finding: no GNN beats MLP. This removes the "untuned GNN" limitation.
3. **Semi-synthetic heterophily sweep, TwiBot-20 A** (`heterophily_sweep.py`; Qwen2.5 features, edges rewired to a target adjusted homophily). MLP = 0.766 throughout.

   | Adjusted homophily | GCN | SAGE | FAGCN | skip-GCN | LINKX |
   |---|---|---|---|---|---|
   | −0.8 | 0.723 | 0.854 | 0.780 | 0.757 | 0.812 |
   | −0.6 | 0.684 | 0.819 | 0.760 | 0.751 | 0.794 |
   | −0.4 | 0.668 | 0.791 | 0.752 | 0.748 | 0.770 |
   | −0.2 | 0.683 | 0.758 | 0.738 | 0.754 | 0.759 |
   | 0.0 | 0.692 | 0.750 | 0.733 | 0.756 | 0.754 |
   | +0.2 | 0.722 | 0.747 | 0.754 | 0.766 | 0.757 |
   | +0.4 | 0.776 | 0.771 | 0.760 | 0.787 | 0.756 |
   | +0.6 | 0.816 | 0.813 | 0.801 | 0.812 | 0.784 |
   | +0.8 | 0.867 | 0.853 | 0.834 | 0.864 | 0.817 |
   | real (+0.06) | 0.707 | 0.750 | 0.744 | 0.752 | 0.752 |

   - The GNNs work when edges carry label signal, in either direction. GCN needs positive homophily; SAGE and LINKX also exploit strong heterophily.
   - Real TwiBot-20 sits at about 0, the worst region for GCN.
   - This is the key "heterophily" figure for the report.
   - Caveat: the synthetic edges are built from every label, test included, so this is a controlled probe, not a valid model.
   - Separate `GRAPH=synthetic` step 4 runs with Qwen3 features (`step4_gnn_synthetic_h±0.6.json`) agree: at +0.6, SAGE scores 0.820, FAGCN 0.817, skip-GCN 0.813 and LINKX 0.776.
4. **Tree baseline** (`tree_baseline.py`, GADBench-style): the log files are **empty**, so it didn't run or failed. It needs rerunning.
5. **DGP replication audit** (`reports/dgp_replication_audit.md`, Part 3). It compares the code with the paper component by component.
   - Main conclusion: the paper's "DGP > everything" and "GNN > MLP" fail mainly because TwiBot-20's neighbourhoods carry no label signal and are tiny, not because of bugs.
   - Biggest implementation deviations:
     - model selection (200-user validation loss vs the paper's validation AUROC on 1,300 users);
     - no numerical summary (eq. 11).
   - `VARIANT=dgp_num` (NumSumm with the 5 allowed fields) is written and was queued after the seeds.
   - Ranked fixes are in its section 3.

## Open decisions and next steps (by Sat night)
1. **Choose ONE primary LoRA checkpoint rule** before reading any more test scores.
   - Option 1: full-validation macro-F1 (already fixed in `step6_rescore.py`).
   - Option 2: the paper's validation AUROC on 1,300 users.
   - Report the other rule as a sensitivity check.
2. **Finish the seeds.** Collect seed-1 `target` rescoring and seed 2 for both variants; report mean ± std over 3 seeds.
3. **Run `VARIANT=dgp_num`** (about 2.2 h GPU) with a CPU control (MLP + the same neighbour means). Report it with the crawl-artefact caveat.
4. **Optional:** the paper-faithful selection run (audit rank 2, about 8 h).
5. **CPU work:**
   - rerun the tree baseline;
   - GNNs at 1,300 training users (the missing "GNN vs DGP at equal data" row);
   - the setting B training-size control and bootstrap.
6. **Update REPORT.md** with all of the above (Status table, new sections, changelog), then commit and push to EthNeural.
7. **Fri/Sat:** draw the 4–5 diagrams and the sweep figure.
8. **Sun:** build the submission from REPORT.md, following the template and page limit.

---

# Part 2: full working report (REPORT.md, as of 4 Nov 05:40)

# Progress Report

A living report, updated as experiments finish. The newest entries are at the top of each
section. For how everything works, see [TEAM_GUIDE.md](TEAM_GUIDE.md).

**Last updated:** 3 November 2026, 20:15

**Scope.** TwiBot-20 is the main dataset: steps 1–6, settings A and B. Instagram is the secondary dataset, steps 1–5 only: it checks whether the step 1–5 findings hold on a second, multilingual graph. Step 6 is not planned for Instagram. One step 6 run takes about 3–6 h on our Mac, so that time goes to the main dataset.

## Status

| Experiment | TwiBot-20 A | TwiBot-20 B | Instagram A | Instagram B |
|---|---|---|---|---|
| Step 1: homophily | ✅ | ✅ | ✅ | ✅ |
| Step 2: baselines | ✅ | ✅ | ✅ | ✅ |
| Step 3: encoders | ✅ 7 encoders | ✅ 7 encoders | ✅ 3 encoders | ✅ 3 encoders |
| Step 4: MLP / GCN / SAGE / FAGCN | ✅ + bridge graph | ✅ | ✅ | ✅ |
| Step 5: zero-shot reasoner | ✅ | ✅ | ✅ | ✅ |
| Step 6: DGP (LoRA), dgp | ✅ 1 seed | ✅ 1 seed | not in scope | not in scope |
| Step 6: DGP (LoRA), target | ✅ 1 seed | ✅ 1 seed | not in scope | not in scope |
| FLAG-lite and LINKX baselines (step 4) | ✅ | ✅ | ✅ | ✅ |

✅ done · 🔄 running · ⏳ to do

**Current run (GPU, from 4 Nov 04:26):** step 6 LoRA seeds 1 and 2 for `dgp` and `target` on TwiBot-20 A (`LORA_SEED`), each rescored right after; about 11 h in total. CPU alongside: the semi-synthetic heterophily sweep.
- Done today: step 6 `target` on TwiBot-20 A (13:30–15:42); Qwen3-1.7B embeddings (15:42–19:14, 3.5 h).
- On CPU after the chain: the setting B training-size control and bootstrap.

## Data preparation

### Sources and labels
- **TwiBot-20** (Feng et al., CIKM 2021), the TwiBot-22-format release (`TwiBot-20-Format22.zip`), read straight from the zip by `my_build/prep_twibot20.py`. Class 1 = bot. We keep the **11,826 labelled users** and drop the 217,754 unlabelled support users (see "dropped support users").
- **Instagram**, the GLBench version (Li et al., 2024) that FLAG also uses, already prepared as `x_text`, `y` and `edge_index` files; we did not rebuild it. Class 1 = commercial account (business, brand or media), a stand-in for a minority class, not fraud.

### Text per user
- **TwiBot-20:** the bio plus the user's first 20 tweets, one per line (whitespace trimmed, line breaks inside a tweet removed), cut at 2,000 characters. **61.6% of users reach that cap**, so the models see the bio and as many tweets as fit. 12.9% of users have no bio, 1.3% have no tweets, 37 have no text at all.
- **Instagram:** the bio (median 119 characters, maximum 523); 99 users have none.
- **Profile metadata is excluded.** TwiBot-20's "verified" flag alone gives macro-F1 0.803: a collection shortcut (Hays et al., WWW 2023), not text or graph signal.

### Edges
- **TwiBot-20:** friend and follow links between labelled users, stored both ways (undirected), merged into one edge type for steps 4–5; step 6 keeps the two types. 16,349 undirected edges; no self-loops or duplicates.
- **Instagram:** 72,005 undirected follow edges; no self-loops or duplicates.

### Splits and the two settings
| | Users | Class 1 | Edges | Train / val / test |
|---|---|---|---|---|
| TwiBot-20 A (natural) | 11,826 | 6,589 bots (55.7%) | 16,349 | 8,278 / 2,365 / 1,183 (official split) |
| TwiBot-20 B (1:10) | 5,761 | 524 bots | 5,274 | 4,000 / 1,164 / 597 |
| Instagram A (natural) | 11,339 | 4,115 commercial (36.3%) | 72,005 | 6,803 / 2,268 / 2,268 (stratified 60/20/20, seed 42) |
| Instagram B (1:10) | 7,946 | 722 commercial | 38,193 | 4,767 / 1,589 / 1,590 |

**Setting B** keeps every class-0 user and a random sample of class-1 users (seed 42) so that class 1 is 10% of class 0, following FLAG's protocol (§4.1.1), with only the edges whose two ends are kept. Splits are restricted to the kept users. The subsampling removes almost all bot–bot edges on TwiBot-20 (54 rows vs 8,162 in A).

### Encoding (step 3)
TwiBot-20 text is longer than an encoder reads, so each piece (the bio and each tweet) is encoded separately, up to 128 tokens, and the L2-normalised piece vectors are averaged per user. Decoder LLMs (Qwen2.5-1.5B, Qwen3-1.7B) and Twitter-RoBERTa are mean-pooled over tokens. A user with no text gets a zero vector, and two 0/1 flags ("no bio", "no tweets") are added for TwiBot-20, so nobody is dropped. Features are standardised with training-set statistics only.

### Profile metadata: which fields are shortcuts? (TwiBot-20 A, full test set)

We use every profile field unless it is a shortcut, under a rule fixed before looking at results (`my_build/metadata_check.py`): a field is excluded if (a) it is a documented collection or labelling artefact, or (b) one split on it labels a large group almost perfectly (a depth-1 tree leaf with at least 10% of training users that is at least 99% one class).

| Field | Single-field depth-1 tree: macro-F1 / AUROC | Purest large leaf (share of users, purity) | Spearman with in-friend count | Verdict |
|---|---|---|---|---|
| verified | 0.803 / 0.800 | – | – | **shortcut**: documented (Hays et al., WWW 2023, §4.1) |
| followers count | 0.731 / 0.731 | 31.7%, 77.0% | +0.59 | allowed |
| following count | 0.456 / 0.534 | 16.9%, 68.8% | −0.03 | allowed |
| tweet count | 0.594 / 0.597 | 49.9%, 65.8% | +0.38 | allowed |
| listed count | 0.714 / 0.716 | 27.0%, 81.6% | +0.59 | allowed |
| account age | 0.620 / 0.621 | 52.6%, 66.5% | +0.33 | allowed |

| Qwen2.5-1.5B embeddings + | LR macro-F1 (AUROC) | MLP macro-F1 (AUROC), 3 seeds | Gain over text only (paired bootstrap) |
|---|---|---|---|
| nothing (text only) | 0.756 (0.822) | 0.766 (0.840) | – |
| **the 5 allowed fields** | **0.804 (0.874)** | **0.806 (0.885)** | LR +0.048 [+0.027, +0.067]; MLP +0.040 [+0.024, +0.056] |
| all 6 fields incl. verified (reference only) | 0.861 (0.926) | 0.844 (0.921) | +0.10 |
| the 5 allowed fields alone (no text) | 0.748 (0.769) | 0.742 (0.796) | – |

The allowed fields add a real +0.04–0.05 to text. Two caveats: followers and listed count correlate ρ 0.59 with the crawl artefact (in-friend count), so part of their gain is the same collection effect; and every comparison elsewhere in this report (encoders, GNNs, LLMs) is text-only, so these rows are a separate "text + profile" track, not a replacement. The DGP numerical-summary variant (`VARIANT=dgp_num`) uses only the 5 allowed fields.

### Checks on the prepared data
| Check | Result |
|---|---|
| Graph: self-loops, duplicate edges, symmetry | none, none, symmetric (both datasets) |
| Duplicate text across splits (`duplicate_text.py`) | 1 of 1,183 TwiBot-20 test users and 12 of 2,268 Instagram test users share a training user's normalised text; dropping them changes Qwen2.5 + LR by ≤ 0.001 |
| Raw vs cleaned text (`clean_text_check.py`, TF-IDF + LR) | URLs → `URL`, @mentions → `@USER`, retweet prefixes removed: −0.007 to +0.002 macro-F1; deleting URLs and mentions: −0.007 to −0.026. Every CI includes 0, so cleaning would not have helped, and removing URLs and mentions slightly hurts (they carry signal: 93.8% of TwiBot-20 users post URLs, 33.4% of tweets are retweets) |
| Language | 2.8% of TwiBot-20 and 5.6% of Instagram users write mostly non-Latin text; kept, and the multilingual encoder covers it |

## Latest results (macro-F1, test set)

**Which test set?** Each heading says so. Most tables use the **full test set**: TwiBot-20 1,183 users (A) and 597 (B); Instagram 2,268 (A) and 1,590 (B). One table, "LLM reasoner vs trained models", uses the smaller **step 5 test sample**, because LLM scoring is slow. Its trained rows are also a single seed, so the same model can show a slightly different number there (for example, Qwen2.5 + GCN is 0.707 in the ladder and 0.698 there).

### TwiBot-20, encoder ladder (full test set, mean of 3 seeds)

| Encoder | A: MLP | A: GCN | A: SAGE | A: FAGCN | B: MLP | B: GCN | B: SAGE | B: FAGCN |
|---|---|---|---|---|---|---|---|---|
| MiniLM | 0.726 | 0.686 | 0.721 | 0.722 | 0.588 | 0.550 | 0.565 | 0.584 |
| E5-base (English) | 0.760 | 0.700 | 0.740 | 0.733 | 0.564 | 0.589 | 0.567 | 0.542 |
| Twitter-RoBERTa | 0.749 | 0.692 | 0.743 | 0.722 | 0.612 | 0.579 | 0.609 | 0.571 |
| multilingual-E5 | 0.747 | 0.699 | 0.741 | 0.726 | 0.606 | 0.585 | 0.589 | 0.567 |
| Qwen3-Embedding | 0.727 | 0.691 | 0.707 | 0.710 | 0.599 | 0.574 | 0.558 | 0.579 |
| Qwen2.5-1.5B | 0.766 | 0.707 | 0.750 | 0.744 | 0.613 | 0.577 | **0.624** | 0.554 |
| Qwen3-1.7B (the step 6 LLM) | **0.769** | 0.706 | 0.750 | 0.740 | 0.593 | 0.581 | 0.579 | 0.570 |
| Graph only | – | 0.577 | 0.568 | 0.562 | – | 0.497 | 0.497 | 0.492 |

FAGCN (Bo et al., AAAI 2021; PyG `FAConv`, eps 0.3) is built for heterophily: each edge gets a learned weight in [−1, 1], so a neighbour can be subtracted as well as added. It has two propagation layers between an input and an output linear layer, the same hidden size (128) and about the same parameter count as GCN (197k). It also uses the same learning rate, weight decay, early stopping and seeds. Its own settings (eps 0.3, dropout 0.5) are fixed, not tuned, like every GNN here. In setting B (1:10), FAGCN is below MLP for every encoder (by 0.004–0.058), and its seeds vary more (std up to 0.02).

References for TwiBot-20 A: TF-IDF 0.702; metadata without "verified" 0.755 (not a valid model).

### Heterophily and fraud baselines: FLAG-lite and LINKX (full test set, mean of 3 seeds)

Two more graph baselines, run with the same training, seeds and validation cut-off rule as step 4 (`my_build/step4_gnn.py`):
- **FLAG-lite**: FLAG (Yang et al., KDD 2025) without its LLM. Its skip-GNN, `GCN(X, A) + Linear(X)` (eq. 6), on a graph pruned by its semantic-similarity sampling: each user keeps up to its 10 most text-similar neighbours with cosine > 0 (`GRAPH=similar10`, 1-hop). FLAG's own ablation (Table 5) credits these two parts with most of its gain. The pruning drops about 25% of TwiBot-20's edges and 48% of Instagram's.
- **LINKX** (Lim et al., NeurIPS 2021; PyG `LINKX`): a non-homophilous baseline that never averages neighbours. One MLP reads each user's adjacency row, one its features, and a third combines them.

| Encoder + model | TwiBot-20 A | TwiBot-20 B | Instagram A | Instagram B |
|---|---|---|---|---|
| Qwen2.5 + MLP (no edges) | **0.766** | 0.613 | 0.607 | **0.575** |
| Qwen2.5 + GCN | 0.707 | 0.577 | **0.609** | 0.549 |
| Qwen2.5 + skip-GCN | 0.752 | 0.612 | 0.604 | 0.558 |
| Qwen2.5 + GCN, similar-10 graph | 0.719 | 0.585 | 0.607 | 0.562 |
| Qwen2.5 + FLAG-lite (skip-GCN, similar-10) | 0.754 | **0.615** | 0.602 | 0.568 |
| Qwen2.5 + LINKX | 0.752 | 0.607 | 0.598 | 0.562 |
| multilingual-E5 + MLP | **0.747** | 0.606 | 0.605 | 0.561 |
| multilingual-E5 + FLAG-lite | 0.731 | **0.635** | **0.613** | **0.565** |
| multilingual-E5 + LINKX | 0.730 | 0.615 | 0.602 | 0.557 |
| Graph only: GCN / skip-GCN / LINKX | 0.577 / 0.578 / **0.595** | 0.497 / 0.497 / 0.488 | 0.548 / 0.547 / 0.532 | 0.500 / 0.498 / 0.521 |

- **The skip connection does most of the work.** It lifts GCN from 0.707 to 0.752 on TwiBot-20 A, the same "keep your own features" effect as SAGE and FAGCN. Similarity pruning adds about 0.01 to GCN.
- **On TwiBot-20 A neither FLAG-lite (0.754) nor LINKX (0.752) beats MLP (0.766)**, matching every other GNN here. Setting B's best (multilingual-E5 + FLAG-lite 0.635 ± 0.018 vs MLP 0.606) is inside B's ±0.05 noise.
- **LINKX graph-only (0.595) beats GCN graph-only (0.577) in A**: reading who links to whom helps more than averaging. It is still far below the untrained in/out link-direction score (AUROC 0.713 vs LINKX's 0.633), because our adjacency is undirected.

### Instagram (secondary dataset, steps 1–5): full test set; LR one fit, MLP and GNNs mean of 3 seeds

On Instagram, class 1 is **commercial accounts** (business, brand or media) against personal ones: a stand-in for a minority class, not fraud. FLAG uses the same Instagram graph (from GLBench, Li et al. 2024) as a fraud proxy at 1:10, our setting B. Its reported macro-F1 there is 0.473–0.483 for text-embedding GNNs and at most 0.507 for FLAG itself (Table 4), near the 0.476 an all-majority predictor scores; our Qwen2.5 + LR reaches 0.578 in B. Splits, subsample and encoder differ (FLAG uses Sentence-BERT), so this is not a head-to-head comparison. Text alone is much weaker here than on TwiBot-20 (AUROC about 0.63 against 0.84).

| Encoder | A: LR | A: MLP | A: GCN | A: SAGE | A: FAGCN | B: LR | B: MLP | B: GCN | B: SAGE | B: FAGCN |
|---|---|---|---|---|---|---|---|---|---|---|
| MiniLM | 0.588 | 0.585 | 0.607 | 0.588 | 0.596 | 0.546 | 0.536 | 0.552 | 0.543 | 0.566 |
| multilingual-E5 | **0.612** | 0.605 | 0.606 | 0.602 | **0.613** | 0.563 | 0.561 | 0.555 | 0.572 | 0.560 |
| Qwen2.5-1.5B | 0.608 | 0.607 | 0.609 | 0.604 | 0.606 | **0.578** | 0.575 | 0.549 | 0.562 | 0.534 |
| Graph only | – | – | 0.548 | 0.544 | 0.522 | – | – | 0.500 | 0.498 | 0.494 |

References: TF-IDF (balanced) 0.578 (A), 0.552 (B). Every model sits within about 0.03 of LR. Unlike TwiBot-20, GCN is level with or above MLP for all three encoders in setting A (macro-F1 +0.001 to +0.022; AUROC 0.639–0.642 vs 0.609–0.635), and graph-only reaches 0.548. In B no graph model is consistently better or worse.

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
| 6 | Qwen3-1.7B zero-shot: target only | 0.590 | 0.503 | not in scope | not in scope |
| 6 | Qwen3-1.7B zero-shot: DGP summaries | 0.562 | 0.527 | not in scope | not in scope |
| 6 | Qwen3-1.7B LoRA: target only | 0.694 | 0.618 | not in scope | not in scope |
| 6 | Qwen3-1.7B LoRA: DGP summaries | 0.726 | 0.628 | not in scope | not in scope |

**TwiBot-20 B, step 6 `dgp`** (597 test users, 54 bots; the step 5 "sample" is the whole test set; 1 seed): LoRA + DGP scores macro-F1 0.628, AUROC 0.723, PR-AUC 0.296, against Qwen2.5 + LR 0.634 / 0.799 / 0.282 and MLP 0.613 / 0.788 / 0.269 (mean of 3 seeds; the seed-0 run in the step 5 table is 0.633). Paired bootstrap against that 3-seed MLP on all 4,000 training users (`B/bootstrap_ci.json`): macro-F1 +0.015 [−0.051, +0.081], AUROC −0.064 [−0.124, −0.008], PR-AUC +0.027 [−0.080, +0.143]. So in B LoRA is level on macro-F1 and PR-AUC (CIs about ±0.07–0.11 wide) and significantly lower on AUROC. With 54 test bots, B's apparent PR-AUC lead is noise.

**TwiBot-20 B, step 6 `target`**: LoRA on the user's own text scores 0.618 / AUROC 0.683 / PR-AUC 0.188 (zero-shot 0.503), so `dgp` − `target` is +0.010 macro-F1, +0.040 AUROC, +0.108 PR-AUC (paired bootstrap queued). Unlike A, the link-type cue does not account for it: target LoRA + 3 link flags (LR fitted on validation) scores only 0.574 (AUROC 0.687), −0.054 [−0.102, −0.006] below `dgp`, and `dgp` ranks better than `target` within most flag groups (no follow link: AUROC 0.702 vs 0.611; no friend link: 0.718 vs 0.681; has follow link is the exception, 0.770 vs 0.814). So in B the summaries seem to add something beyond "which links exist", but B's edges are themselves label-dependent and its CIs are wide. Validation loss and AUROC agreed in B (best check at epoch 2.0: loss 0.335, AUROC 0.810), unlike A.

**Sample noise.** On 400 users, macro-F1 moves by about ±0.04 from test-set noise alone, and the sample shifts models in different directions against the full test set: LoRA + DGP 0.711 → 0.726, LoRA target 0.704 → 0.694, LR 0.756 → 0.749. So on the sample dgp − target is +0.032; on the full test set it is +0.007. Use the full-test numbers (training-size and bootstrap sections) for any LoRA claim.

**Validation samples.** LLM scoring is slow, so the zero-shot rows (steps 5 and 6) pick their cut-off on a stratified validation *sample*: 200 users for TwiBot-20 A, 400 for B, 150 for Instagram A and 300 for Instagram B. The trained models pick theirs on the full validation set. The step 6 LoRA runs use that same sample for early stopping, then pick their cut-off on the full validation set (2,365 users for TwiBot-20 A).

The two "zero-shot: DGP summaries" rows (0.550 and 0.562) differ in the LLM and in how the summaries were made. Step 5 used its own summaries; step 6 used the paper's diffusion trimming and 10-token summaries. So they are not a like-for-like comparison of the two LLMs.

LoRA dgp is one seed (42). On the full TwiBot-20 A test set (1,183 users) it scores 0.711 macro-F1, AUROC 0.794. Every model picks its own cut-off by the same rule: the one that maximises macro-F1 on validation, applied unchanged to test. On the 400-user step 5 sample its AUROC (0.828) is the highest, but that does not hold on the full test set: AUROC 0.794, against 0.822 for Qwen2.5 + LR and 0.840 for MLP. So it is behind on ranking too, not just on the cut-off. Even the best cut-off chosen on test itself (not a valid model) gives only 0.726.

### LoRA checkpoint rule: both adapters (full test set, 1 seed)

`my_build/step6_rescore.py` scores each run's last-check adapter and applies the text-only models' rule, written into the script before any last-adapter test score existed (commit `fc45a63`, 3 Nov 15:24): keep the adapter with the higher full-validation macro-F1. Only two adapters per run were saved and compared.

| Run | Lowest val loss: epoch, test macro-F1 / AUROC | Last check: epoch, test macro-F1 / AUROC | Full-val macro-F1 (lowest loss / last) | Kept by the rule |
|---|---|---|---|---|
| A `dgp` | 1: 0.711 / 0.794 | 2: **0.727** / 0.815 | 0.718 / 0.743 | last |
| A `target` | 1: 0.704 / 0.782 | 2: **0.728** / 0.812 | 0.718 / 0.737 | last |
| B `dgp` | 2: **0.628** / 0.723 | 3: 0.502 / 0.546 (collapsed) | 0.648 / 0.538 | lowest loss |
| B `target` | 1: 0.618 / 0.683 | 2: **0.644** / 0.739 | 0.638 / 0.644 | last |

**Under this rule the DGP summaries add nothing:** `dgp` − `target` is −0.001 in A (0.727 vs 0.728) and −0.016 in B (0.628 vs 0.644). LoRA on 1,300 users is then level with text-only models on the same 1,300 users (MLP 0.726 on the same draw). **The tables and findings below still use the lowest-validation-loss adapters** until the team decides which rule is primary; CIs for the rule's adapters (`LORA_ADAPTER=chosen` in `bootstrap_ci.py` and `link_direction.py`) are running.

### Where GCN loses: connected vs isolated test users (TwiBot-20 A, full test set, mean of 3 seeds)

| Qwen2.5-1.5B + | Connected (1,096 users) | Isolated (87 users) |
|---|---|---|
| MLP | **0.767** | 0.743 |
| GCN | 0.701 | 0.777 |
| SAGE | 0.752 | 0.734 |
| FAGCN | 0.742 | 0.770 |

GCN's whole loss is on connected users, where neighbours are averaged in. The isolated column is only 87 users (47 bots) and is not reliable: isolated users have no neighbours in either graph, yet GCN scores 0.777 on them when trained on the real edges and 0.720–0.758 (mean 0.738) when trained on five shuffled versions. The difference comes only from weights learned on a different graph.

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

Every graph sits near zero once chance is removed. Setting B's high edge homophily (0.86–0.88) only reflects the 1:10 ratio: almost every edge touches the large class. On TwiBot-20, B's near-zero value is also caused by how B is built: it keeps 8% of bots and only edges whose two ends are both kept, so bot–bot edge rows fall from 8,162 (A) to 54 (B). A bot's neighbours are then 7.8% bots, below B's 9.1% base rate, and isolation becomes tied to the label (41.8% of B's bots are isolated, against 27.6% of its humans; 7.3% vs 8.5% in A). Friend links on TwiBot-20 join same-label users no more often than chance (adjusted −0.015); the label signal in TwiBot-20's links sits elsewhere, in who lists whom (see "who lists whom"). The bridge edges raise adjusted homophily to 0.175, yet GCN still gets worse on them (robustness check below).

### Per-class homophily: is there "camouflage"? (FLAG's Figure 5 test)

FLAG (KDD 2025, Fig. 5) argues its 1:10 fraud datasets show *neighbourhood camouflage*: minority nodes have low per-node homophily, majority nodes high. We ran the same histogram on our four graphs, adding the chance level (each class's share of all edge ends, which is what random edges would give). `my_build/class_homophily.py`, figure `my_build/figures/class_homophily.png`.

| Graph | Minority: mean homophily | Minority: chance | Majority: mean | Majority: chance |
|---|---|---|---|---|
| TwiBot-20 A | 0.543 | 0.484 | 0.581 | 0.516 |
| TwiBot-20 B (1:10) | 0.091 | 0.065 | 0.935 | 0.935 |
| Instagram A | 0.389 | 0.341 | 0.692 | 0.659 |
| Instagram B (1:10) | 0.099 | 0.078 | 0.921 | 0.922 |

Every class sits a few points above chance. The low-minority / high-majority pattern FLAG reads as camouflage appears in both 1:10 settings, but it is almost exactly what chance predicts once one class is 10 times the other; it comes from the imbalance, not from the minority hiding among the majority. On TwiBot-20 A there is no camouflage at all (bots 0.54 vs humans 0.58).

![Per-class homophily](my_build/figures/class_homophily.png)

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
| Qwen2.5-1.5B + MLP | 8,278 | 0.766 | 0.840 | 0.754 |
| Qwen2.5-1.5B + LR | 8,278 | 0.756 | 0.822 | 0.749 |
| Qwen2.5-1.5B + MLP | 1,300 | 0.726 | 0.796 | 0.723 |
| Qwen2.5-1.5B + LR | 1,300 | 0.725 | 0.786 | 0.712 |
| Qwen3-1.7B + MLP (same model as LoRA, used as an encoder) | 8,278 | **0.769** | **0.840** | 0.759 |
| Qwen3-1.7B + LR | 8,278 | 0.765 | 0.826 | 0.759 |
| Qwen3-1.7B + MLP | 1,300 | 0.724 | 0.792 | 0.729 |
| Qwen3-1.7B + LR | 1,300 | 0.717 | 0.795 | 0.713 |
| Qwen3-1.7B LoRA + DGP summaries (1 seed) | 1,300 | 0.711 | 0.794 | 0.726 |
| Qwen3-1.7B LoRA, target text only (1 seed) | 1,300 | 0.704 | 0.782 | 0.694 |

**Same model, two roles.** Qwen3-1.7B used as a frozen encoder with an MLP scores 0.724 on 1,300 users and 0.769 on all users; fine-tuned with LoRA on the same 1,300 users it scores 0.711 (DGP summaries) and 0.704 (target text). So the model version does not explain the gap: in setting A, Qwen2.5 and Qwen3 behave alike as encoders (in B, Qwen3 is lower: LR 0.584 vs 0.634). With the same 1,300 users, the text-only models lose about 0.03–0.04, and the gap to LoRA + DGP shrinks from 0.055 to 0.015 on the full test set. On the step 5 sample LoRA is level with them. The full-train rows reproduce steps 3 and 4 exactly, which checks the setup.

**Which 1,300 users?** The table uses the same draw as step 6. Over 6 random stratified 1,300-user draws (`control_train_size.py`, `DRAWS=6`), LR scores 0.730 ± 0.005 and MLP 0.733 ± 0.006 on the full test set (MLP: 3 seeds per draw). The step 6 draw is the lowest of the six (LR 0.725, MLP 0.726), so the typical gap between LoRA + DGP (0.711) and a text-only model on 1,300 users is about 0.02, not 0.014. LoRA itself was trained on one draw only.

**Learning curve** (Qwen2.5-1.5B embeddings, full test set, MLP mean of 3 seeds; `my_build/bootstrap_ci.py`):

| Training users | 325 | 650 | 1,300 | 2,600 | 5,200 | 8,278 |
|---|---|---|---|---|---|---|
| LR | 0.714 | 0.720 | 0.725 | 0.743 | 0.754 | 0.756 |
| MLP | 0.697 | 0.713 | 0.726 | 0.751 | 0.754 | 0.766 |

LoRA + DGP at 1,300 users (0.711) is below every text-only point from 650 users up, and level with LR trained on 325 users (0.714), a quarter of its data. LoRA target only (0.704) is below them all. Equal training data closes most of the gap, but not all of it.

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
| LoRA target only − MLP (both 1,300) | −0.022 | [−0.047, +0.003] | no |
| LoRA + DGP − LoRA target only | +0.007 | [−0.015, +0.030] | no |

Same test on **Instagram A** (Qwen2.5-1.5B embeddings, 2,268 test users; `results/A/bootstrap_ci.json`): GCN − MLP +0.002 [−0.016, +0.020], SAGE − MLP −0.003 [−0.018, +0.012], FAGCN − MLP −0.002 [−0.017, +0.013]; MLP on 1,300 users − all users −0.035 [−0.051, −0.020]. So on Instagram the graph models are indistinguishable from MLP, while on TwiBot-20 A GCN is clearly below it (−0.059).

### Control: the same GNNs on shuffled edges (full test set, mean of 3 seeds)

`GRAPH=shuffled` keeps every user's degree but rewires who links to whom at random (`my_build/step4_gnn.py`, `step4_gnn_shuffled.json`). If a GNN scores the same on shuffled edges, the real edges give it nothing.

| Model | TwiBot-20 A: real edges | A: shuffled (5 rewirings) | TwiBot-20 B: real edges | B: shuffled (5 rewirings) |
|---|---|---|---|---|
| Qwen2.5 + MLP (no edges) | 0.766 | 0.766 | 0.613 | 0.613 |
| Qwen2.5 + GCN | **0.707** | 0.670 ± 0.009 | 0.577 | 0.588 ± 0.022 |
| Qwen2.5 + SAGE | 0.750 | 0.751 ± 0.006 | 0.624 | 0.615 ± 0.014 |
| Qwen2.5 + FAGCN | **0.744** | 0.725 ± 0.002 | 0.554 | 0.575 ± 0.008 |
| E5-base + GCN | **0.700** | 0.661 ± 0.007 | **0.589** | 0.559 ± 0.004 |
| E5-base + SAGE | 0.740 | 0.731 ± 0.004 | 0.567 | 0.577 ± 0.017 |
| E5-base + FAGCN | 0.733 | 0.738 ± 0.007 | 0.542 | 0.568 ± 0.013 |
| Graph only (GCN) | **0.577** | 0.552 ± 0.012 | 0.497 | 0.497 ± 0.000 |

Each shuffled value is the mean ± std over 5 random rewirings (each itself the mean of 3 model seeds). Bold: real edges more than 2 rewiring std above shuffled.

- **In A, real edges beat random ones for GCN** by 0.037–0.039 (both encoders), and for graph-only by 0.025. FAGCN gains 0.019 with Qwen2.5 and nothing with E5-base; SAGE gains 0–0.009. GCN's gain is a large part of its gap to MLP (−0.059). Label signal cannot explain it (label informativeness 0.003); similar text along real edges can (next table).
- **In B, no consistent direction**: real edges help E5-base + GCN (+0.030), shuffled edges score higher for FAGCN (+0.021–0.026) and Qwen2.5 + GCN. B has almost no bot–bot edges (see the homophily note).

**Why real edges help GCN without carrying labels: linked users write alike.** Mean cosine similarity of centred Qwen2.5-1.5B embeddings (`my_build/edge_similarity.py`; centring removes the common direction that pushes every raw cosine of these vectors to about 0.9):

| TwiBot-20 A | All | Bot–bot | Bot–human | Human–human |
|---|---|---|---|---|
| Real edges | **0.263** | 0.274 | 0.235 | 0.299 |
| Shuffled edges (mean of 5 rewirings) | 0.053 | 0.066 | 0.044 | 0.058 |
| Random pairs | 0.034 | 0.062 | 0.021 | 0.022 |

Real edges join users with similar text whatever their labels, even across bot–human pairs (0.235). So GCN loses less on real edges because averaging similar users blurs a user's own vector less, not because the edges reveal labels. Setting B shows the same pattern (0.336 real vs 0.071 shuffled).

### Who lists whom: a crawl artefact that carries label signal (TwiBot-20 A)

TwiBot-20 stored, for each labelled user, a sample of about 10 accounts they follow (`friend`) and 10 accounts following them (`follow`). So a link between two labelled users means one appears in the other's 10-sample; out-degree inside the labelled graph is capped at 10, and only 1.2% (friend) and 3.8% (follow) of links are mutual. Our pipelines symmetrise these links, and steps 4–5 also merge the two types. `my_build/link_direction.py` (`results/twibot20/A/link_direction.json`) looks at direction:

| Link, as stored | Share with ≥1: bots | humans | Test AUROC of the count |
|---|---|---|---|
| Outgoing friend (user lists someone) | 0.387 | 0.358 | 0.517 |
| Outgoing follow | 0.118 | 0.088 | 0.509 |
| **Incoming friend** (user is in someone's following sample) | 0.388 | **0.634** | 0.306 (0.694 reversed) |
| **Incoming follow** (user is in someone's follower sample) | **0.499** | 0.238 | 0.640 |

- **All the signal is in incoming links**, and it is strong: the untrained score in-follow minus in-friend reaches test AUROC 0.713, above graph-only GCN (0.61). Outgoing links carry almost none.
- **It tracks follower count**, a metadata field we exclude as a shortcut: Spearman ρ between in-friend count and `followers_count` is 0.59 (0.69 among humans), and −0.44 for in-follow.
- **Whether a user's own lists were recorded also depends on the label**: 95.8% of bots have a recorded follow list against 81.4% of humans (friend lists 95.0% vs 88.0%).

We read this as a property of how the dataset was collected (who appeared in whose sample, and how popular they are), not as relational signal a deployed detector would have. It fits Hays et al. (WWW 2023, §4.1): TwiBot-20 was collected "starting from well-known (verified) seed users" by breadth-first traversal, and its verified accounts were labelled human automatically, so a user's position in the crawl is tied to its label. It explains why graph-only models still score 0.577 although neighbour labels are uninformative.

**How much do these cues add to text?** LR on Qwen2.5-1.5B embeddings, full test set, paired bootstrap 95% CIs:

| Features | Macro-F1 | AUROC | Gain over text |
|---|---|---|---|
| Text | 0.756 | 0.822 | – |
| Text + 3 undirected link flags (has friend / follow / friend-of-friend) | 0.770 | 0.833 | +0.013 [−0.003, +0.029] |
| Text + 4 log in/out degrees | 0.776 | 0.845 | +0.019 [+0.001, +0.038] |
| 3 link flags only | 0.633 | 0.655 | |
| 4 log in/out degrees only | 0.660 | 0.717 | |

In setting B the three flags lift LR from 0.634 to 0.656 (+0.023 [+0.002, +0.049], `B/link_direction.json`), but B's edges themselves depend on every user's label through its subsampling, so that number is partly built from the labels.

**Does step 6 `dgp` gain only through this cue? In A, apparently yes.** The `dgp` prompt writes "none" for each missing link type, so it exposes the three undirected flags; `target` does not. Combining the target-only LoRA score with the three flags (an LR fitted on the validation set) scores 0.731 macro-F1 (AUROC 0.803), against 0.711 (0.794) for `dgp`: +0.020 [−0.002, +0.040]. Within groups of users with the same flag, `dgp` is better in some (no friend link: AUROC 0.803 vs 0.775; has follow link: 0.786 vs 0.770) and worse in others (has friend: 0.768 vs 0.777). So in A the summaries' own contribution is not established: a 4-feature LR on top of target LoRA matches `dgp`. In setting B the same check goes the other way (target + flags 0.574 vs `dgp` 0.628, see the setting B step 6 paragraph).

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
- GCN drops on every encoder (std 0.002–0.010), most on users who were isolated before (Qwen2.5: 0.777 → 0.666; the 0.777 starting point is itself unstable, see the connected/isolated note). Average degree rises from about 2.8 to about 7.8, so GCN's averaging gives each user's own text less weight. This happens even though the bridge graph is the only one with clear label signal (adjusted homophily 0.175, label informativeness 0.022).
- SAGE, which keeps its own features separate, gains slightly: Qwen2.5 + SAGE comes within 0.005 of MLP.
- FAGCN holds steady or gains slightly (−0.007 to +0.010). The extra edges do not hurt it the way they hurt GCN.
- Graph-only drops by about 0.04. One hypothesis, not tested: in the labelled-only graph, degree partly reflects a user's position in the dataset's BFS crawl, and the bridge edges blur that.

## Limitations (so far)

- **One seed for step 6 so far, and that run was unseeded on the LoRA side.** The first `dgp` and `target` runs fixed the training users (SEED=42) and batch order, but not the LoRA start weights or dropout (no `torch.manual_seed`), so they cannot be reproduced exactly. Seeded runs (`LORA_SEED=1, 2`, same 1,300 users and summaries) are running on TwiBot-20 A; the unseeded run will be the third. Qwen3-1.7B instead of the paper's Qwen3-8B; the paper uses 5 seeds.
- **LoRA reads less text than the encoders.** Every model works from the same prepared text (bio + tweets, cut at 2,000 characters). The encoders read all of it; the LoRA prompt caps the user's own text at 480 tokens of whole lines, which cuts 81% of TwiBot-20 users short (13.2 of 16.6 lines kept on average, 800-user sample). This biases LoRA vs MLP against LoRA; dgp vs target is unaffected (same cap).
- **LoRA and the other models choose their checkpoint differently.** LoRA keeps the lowest validation loss on a validation sample (200 users in A, 400 in B); MLP and GNNs keep the best full-validation macro-F1. In both TwiBot-20 A runs and in B `target` the loss was lowest at epoch 1 while validation AUROC kept rising (A dgp 0.748 → 0.772, A target 0.765 → 0.781, B target 0.699 → 0.773). In B `dgp` loss and AUROC agreed (best at epoch 2.0), and the run ended on a collapsed check (epoch 3.0: AUROC 0.624). The last-check adapters (epoch 2 in A and B `target`, epoch 3 in B `dgp`) are being scored with the full-validation rule (`my_build/step6_rescore.py`).
- **GNN hyperparameters are fixed and shared, not tuned.**
- **The `dgp` prompt reveals which link types a user has** (its "none" lines), a label cue worth +0.013 to LR in A (CI includes 0) and +0.023 [+0.002, +0.049] in B. In A, target LoRA plus these flags matches `dgp`; in B it does not (see "who lists whom" and the setting B step 6 paragraph). `target` does not see it.
- **TwiBot-20's links reflect how it was collected.** Each user stores a sample of about 10 followers and 10 followings; incoming links track follower count (excluded metadata), and whether a user's own lists were recorded depends on the label (95.8% of bots vs 81.4% of humans for follow lists).
- **Step 5's neighbour ranking uses multilingual-E5 vectors with the two 0/1 "no bio" / "no tweets" flags still attached** (`step5_llm_reason.py:180`), so a shared flag can outweigh text similarity when a user has more than 8 neighbours. Step 6's diffusion drops the flags (`step6_dgp.py:441`).
- **Setting B is small and changes the graph.** Its test set has 54 bots; one random bot subsample (seed 42); 29% of users are isolated in B against 7.8% in A (37% of B's test bots). Its paired bootstrap CIs are about ±0.05: GCN − MLP −0.036 [−0.087, +0.012], SAGE − MLP +0.012 [−0.034, +0.058], FAGCN − MLP −0.058 [−0.112, −0.002]. Only the FAGCN gap is outside noise. B's MLP learning curve is flat and noisy (0.622 at 325 users, 0.613 at 4,000); LR rises from 0.583 to 0.634. B cannot separate most models. B also has almost no bot–bot edges (54 rows), so its "no edge signal" result reflects the subsampling, not rare fraud as such. `results/twibot20/B/bootstrap_ci.json`.
- **No duplicate-text leakage.** 1 of 1,183 TwiBot-20 A test users (12 of 2,268 on Instagram A) has exactly the same normalised text as a training user; dropping them changes Qwen2.5 + LR by at most 0.001 (`my_build/duplicate_text.py`).
- **TwiBot-20 graph = labelled users only** (see "dropped support users"; bridge check below).
- **Edge signal near zero**, so these results speak to near-random edges, not to strongly heterophilous graphs (see the adjusted-homophily table).

## Findings so far

**In one line:** on TwiBot-20 a strong text encoder does the work. Neighbour labels carry almost no signal, so no GNN and no neighbour-summary prompt is significantly better than a text-only model; the label signal that the links do carry comes from how the dataset was collected (who appears in whose follower sample) and tracks follower count. A fine-tuned LLM roughly matches text-only models trained on the same 1,300 users but stays below them when those use all 8,278.

1. **A good text encoder is where the gains are.** On TwiBot-20 A, Qwen3-1.7B + MLP (0.769) and Qwen2.5-1.5B + MLP (0.766) beat TF-IDF (0.702) by about 0.065. The neural encoders span 0.726–0.769, and no graph model improves on its encoder's MLP.
2. **Neighbour averaging (GCN) hurts; models that keep a user's own features lose less; none beats MLP on TwiBot-20.** GCN is 0.036–0.064 below MLP for every encoder in A (Qwen2.5: −0.059, 95% CI [−0.082, −0.037]). SAGE (own features kept separate) is within about 0.02 (−0.015, borderline). FAGCN is 0.003–0.030 below MLP and beats GCN by 0.02–0.04, but it beats GCN by as much on shuffled edges (Qwen2.5 0.725 vs 0.670), so its gain comes from mixing its input back in at every layer (eps), not from signed edge weights. All GNNs use the same fixed, untuned hyperparameters. Two more baselines built for this problem also stay below MLP on TwiBot-20 A: FLAG-lite (FLAG's skip-GNN on its similarity-pruned graph, 0.754) and LINKX (0.752). Even on the bridge graph, the only one with clear label signal (adjusted homophily 0.175), GCN gets worse by 0.02–0.04. On Instagram A, where text alone is weak (AUROC about 0.63), GCN is level with MLP (+0.002, CI [−0.016, +0.020]). This is consistent with averaging hurting when a user's own text is strong and being neutral when it is weak, though the two datasets also differ in language, label meaning and density.
3. **Neighbour labels carry almost no signal; neighbours write alike; who lists whom is a crawl artefact.** Adjusted homophily is −0.015 to 0.08 on every original graph and label informativeness at most 0.005. Real edges join users with similar text whatever their labels (centred cosine 0.26 vs 0.05 for shuffled edges), so on TwiBot-20 A real edges are less harmful than random ones (GCN +0.037 over 5 rewirings) but never better than no edges. The links do carry label signal in their direction: being in other users' follower samples vs their following samples gives an untrained AUROC of 0.713, and it tracks follower count. We read it as an artefact of how TwiBot-20 was collected. Our undirected pipelines discard it.
4. **The graph alone is weak:** 0.577 (TwiBot-20 A) and 0.497 (B). On the bridge graph it drops to 0.538.
5. **The zero-shot LLM reasoner (Qwen2.5-1.5B-Instruct, step 5) loses to trained classifiers on TwiBot-20 A and B and Instagram A.** On Instagram B it roughly ties the weak trained baselines on its 600-user sample (zero-shot DGP summaries 0.538 vs LR 0.546; zero-shot target AUROC 0.567 vs LR 0.524). Adding neighbours (raw or summaries) does not reliably help.
6. **Encoder type matters more than size.** The two decoder LLMs lead (Qwen3-1.7B 0.769, Qwen2.5-1.5B 0.766), then English E5-base (110M, 0.760), ahead of multilingual-E5 (278M, 0.747) and Qwen3-Embedding (0.6B, 0.727). On Instagram A the encoders are tied.
7. **Fine-tuning helps a lot; at equal data the fine-tuned LLM roughly matches text-only models.** Qwen3-1.7B goes from 0.562 zero-shot to 0.726 with LoRA on the step 5 sample (0.711 on the full test set; zero-shot was scored on the sample only). The same model as a frozen encoder + MLP scores 0.724 on the same 1,300 users and 0.769 on all 8,278. Against Qwen2.5 + MLP on 1,300 users the gap is −0.014 (95% CI [−0.041, +0.013]) on the same draw and about −0.02 over 6 draws. LoRA was never trained on more than 1,300 users, so whether more data would close the full-data gap is untested. One seed. **The DGP neighbour summaries add +0.007** over target-only LoRA (0.711 vs 0.704; CI [−0.015, +0.030]), and in A target LoRA plus three link flags matches `dgp` (0.731 vs 0.711, +0.020 [−0.002, +0.040]), so there the summaries' own contribution is not established. In setting B, `dgp` beats `target` on AUROC (0.723 vs 0.683) and target + flags does not catch up (0.574 macro-F1), but B has 54 test bots and label-dependent edges. Zero-shot they hurt (0.562 vs 0.590, step 5 sample).
8. **Setting B separates few models on macro-F1.** Its test set has 54 bots and its macro-F1 CIs are about ±0.05; only FAGCN − MLP excludes 0. On AUROC more gaps are clear: GCN − MLP −0.072 [−0.129, −0.018], FAGCN − MLP −0.141 [−0.196, −0.090], FAGCN − GCN −0.069, and LoRA `dgp` − MLP −0.064. Its subsampling also removes almost every bot–bot edge (54 rows, against 8,162 in A).

## Decision log

| Date | Decision | Why |
|---|---|---|
| 3 Nov | Measure the link-type cue before reading the setting B step 6 result | The `dgp` prompt shows which link types a user has; in B isolation is tied to the label |
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
| 1 Nov | Excluded TwiBot-20 metadata | "Verified" alone gives macro-F1 0.803: a shortcut, not text or graph signal. Hays et al. (WWW 2023, §4.1, Table 2) find the same: a depth-1 decision tree on "verified" reaches accuracy 0.82 / F1 0.86 on TwiBot-20, 0.05 / 0.03 below the best published model, because collection started from verified seed users and "verified accounts were automatically labeled human" |
| 1 Nov | Encoder ladder: six encoders, everything else fixed | To separate encoder size and type from the graph's effect |
| 30 Oct | Threshold chosen on validation for every model | Scores are not calibrated; 0.5 is badly placed when class 1 is rare |
| 30 Oct | Settings A (natural) and B (1:10, FLAG-style) | Tests both the data as it is and rare fraud. FLAG (Yang et al., KDD 2025, §4.1.1) builds its text-rich fraud datasets the same way: the minority class "randomly selected so that the final ratio … is about 1:10" |
| 27 Oct | Dropped Elliptic, rebuilt on Instagram | Elliptic has no natural text for an LLM to read |

## Next steps

1. Running tonight (GPU): Qwen3-1.7B as an encoder (step 3, then LR / MLP / GNNs and the 1,300-user control), step 6 `dgp` and `target` on TwiBot-20 B, then scoring the last-check LoRA adapters.
2. Decide which LoRA checkpoint rule is primary. The full-validation macro-F1 rule was fixed in `step6_rescore.py` (commit `fc45a63`) before the last-check adapters were scored; both are reported (see "LoRA checkpoint rule").
3. FLAG (Yang et al., KDD 2025; no public code): running a FLAG-lite baseline from its two non-LLM components (semantic-similarity top-10 neighbour sampling, skip-GNN `GNN(X) + Linear(X)`), which its own ablation (Table 5) credits with most of the gain. FLAG also selects models by best validation macro-F1 (§4.1.2), the rule `step6_rescore.py` applies.
4. Optional, by Saturday night: more LoRA seeds on TwiBot-20 A (about 2.2 h per run, so 2 seeds × dgp and target fit one night; needs a separate `LORA_SEED` and the seed in the checkpoint name first), a semi-synthetic heterophily sweep (CPU), figures.
5. Low priority: the M ∈ {4, 8, 16} sweep. The limit of 8 binds only for friend-of-friend (mean 5.1 kept in A); 1-hop metapaths keep 0.8–1.8.

## Changelog

- **4 Nov 2026 (05:40):** Metadata shortcut check: only "verified" is a shortcut under the pre-set rule; text + the 5 allowed fields gains +0.04–0.05 macro-F1 (significant). `VARIANT=dgp_num` (DGP's numerical summaries) added and queued after the LoRA seeds.
- **4 Nov 2026 (05:00):** New Data preparation section (sources, text, edges, splits and settings, encoding, checks). Raw vs cleaned text check: cleaning changes TF-IDF + LR by −0.026 to +0.002, all CIs include 0. Noted the 2,000-character text cap (61.6% of TwiBot-20 users).
- **4 Nov 2026 (04:40):** `LORA_SEED` added to step 6; seeds 1 and 2 started on TwiBot-20 A. Duplicate-text check: 1 TwiBot-20 test user and 12 Instagram test users share a training user's text; removing them changes Qwen2.5 + LR by at most 0.001. Heterophily sweep started.
- **4 Nov 2026 (05:30):** FLAG-lite (similar-10 graph + skip-GCN) and LINKX on all four settings: neither beats MLP on TwiBot-20 A; the skip connection recovers most of GCN's loss.
- **4 Nov 2026 (04:40):** Cited Hays et al. (WWW 2023) for the metadata exclusion and the crawl artefact.
- **4 Nov 2026 (04:15):** FLAG (KDD 2025) read: setting B's 1:10 protocol and the validation-F1 model selection both match it; Instagram comparison with its Table 4; FLAG-lite baseline and per-class homophily histograms (FLAG Fig. 5) started.
- **4 Nov 2026 (03:30):** Rescoring done: the full-validation rule keeps the last-check adapter in A (both) and B `target`, and the lowest-loss one in B `dgp`. Under it `dgp` − `target` is −0.001 (A) and −0.016 (B). New section with both adapters; main tables unchanged pending a decision.
- **4 Nov 2026 (01:40):** After review 6 (QA): B MLP baseline quoted as the 3-seed mean the CIs use; checkpoint-rule wording covers B (B target like A; B dgp ended on a collapsed check); "target + flags matches dgp" limited to A; finding 8 adds B's AUROC CIs.
- **4 Nov 2026 (01:20):** Step 6 `target` on TwiBot-20 B: 0.618 / AUROC 0.683 / PR-AUC 0.188 (`dgp` 0.628 / 0.723 / 0.296). In B, target LoRA + link flags does not reach `dgp` (0.574), unlike A.
- **3 Nov 2026 (23:50):** `bootstrap_ci.py` now also gives AUROC and PR-AUC CIs and can use either LoRA adapter (`LORA_ADAPTER`); setting B: LoRA + DGP vs MLP AUROC −0.064 (significant), PR-AUC +0.027 (not).
- **3 Nov 2026 (23:20):** Step 6 `dgp` on TwiBot-20 B: LoRA 0.628 macro-F1 (LR 0.634), AUROC 0.723 (LR 0.799), PR-AUC 0.296 (LR 0.282); zero-shot 0.527.
- **3 Nov 2026 (22:00):** After review 4: TwiBot-20's link signal is in link direction (incoming links), tracks follower count and reflects how the dataset was collected (`link_direction.py`); the cue section, finding 3 and the one-line answer reworded. Target LoRA + 3 link flags matches `dgp` (0.731 vs 0.711). Number fixes (GCN/FAGCN ranges, LR 0.756, B +0.023, zero-shot on the sample only); softened two overclaims.
- **3 Nov 2026 (20:30):** Instagram A paired bootstrap: every GNN − MLP gap is within ±0.02 and includes 0.
- **3 Nov 2026 (20:15):** After review 3: Findings rewritten (one-line answer first; finding 2 credits FAGCN's gain to its input residual, shown by shuffled edges; Instagram contrast; finding 6 by encoder type; finding 7 with the same-model comparison; new finding 8 on setting B). Qwen3-1.7B encoder rows (ladder, training-size control). Link-type cue table (`isolation_cue.py`). Instagram label described; step 5 sample noise noted; stale numbers fixed.
- **3 Nov 2026 (18:30):** Shuffled-edge control over 5 rewirings (A: GCN +0.037–0.039 on real edges; B: no consistent direction). 6 training draws of 1,300 users: MLP 0.733 ± 0.006; the step 6 draw is the lowest, so LoRA's typical gap at equal data is about 0.02.
- **3 Nov 2026 (18:00):** After review 2: corrected the learning-curve, shuffled-edge and setting B sentences against their numbers; isolated column marked unreliable; setting B's bot–bot edge loss documented; new edge text-similarity table (real edges join similar-text users regardless of label). Fixed epoch labels in `step6_rescore.py`. Queued 4 more rewirings, 5 more training draws, and the setting B control and bootstrap.
- **3 Nov 2026 (16:05):** Shuffled-edge control (A: real edges slightly better than random for GCN/FAGCN; B: no difference; SAGE indifferent). Setting B paired bootstrap: CIs about ±0.05, only FAGCN − MLP excludes 0.
- **3 Nov 2026 (16:00):** After an independent review: connected/isolated table; Limitations section; fixed ranges in findings 2 and the B FAGCN note; setting B SAGE claim checked against AUROC/PR-AUC; finding 1 reworded. Queued step 6 rescoring of the epoch-2 adapters and a shuffled-edge control.
- **3 Nov 2026 (15:45):** Step 6 `target` on TwiBot-20 A: LoRA 0.704 (full test), 0.694 (step 5 sample); zero-shot 0.590. DGP summaries add +0.007 on the full test set.
- **3 Nov 2026 (15:00):** Adjusted homophily and label informativeness for every graph (near 0 everywhere). Learning curve and paired bootstrap CIs on TwiBot-20 A: the LoRA vs MLP gap at equal training size is not significant.
- **3 Nov 2026 (14:30):** FAGCN on TwiBot-20 B; Instagram graph-only GCN/SAGE rows. Rerunning Instagram GCN/SAGE reproduced every earlier number exactly.
- **3 Nov 2026 (14:15):** FAGCN results on TwiBot-20 A (both graphs) and Instagram A and B. Instagram step 4 now has MLP rows and the full MLP / GCN / SAGE / FAGCN table. Corrected the claim that DGP ranks best: on the full test set its AUROC is 0.794, against 0.822 for LR and 0.840 for MLP. Added a note on validation samples.
- **3 Nov 2026 (14:00):** TwiBot-20 made the main dataset, Instagram the secondary one (steps 1–5); every LLM row now names its model. FAGCN added to step 4.
- **3 Nov 2026 (afternoon):** Step 4 robustness check on the bridge graph: GCN drops 0.02–0.04 on every encoder, SAGE gains up to 0.012. Steps 5 and 6 stay on the labelled-only graph. Started step 6 `VARIANT=target` on TwiBot-20 A.
- **3 Nov 2026:** TwiBot-20 A step 6 `dgp` finished (LoRA 0.726 on the step 5 sample, 0.711 on the full test set). Counted the dropped support users: 8,082 bridges; bridge edges are more homophilous (0.619) than our graph (0.532).
- **2 Nov 2026:** TwiBot-20 step 5 finished (A and B). Step 6 (DGP replication) written and started on TwiBot-20 A. Repository created.


---

# Part 3: DGP replication audit (reports/dgp_replication_audit.md)

# DGP replication audit (step 6)

Written 4 Nov 2026. Compares `my_build/step6_dgp.py`, `my_build/step6_rescore.py`, `my_build/step5_llm_reason.py` and REPORT.md with Li, Hu, Hooi, He, Chen, "DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs" (AAAI 2026, arXiv 2507.21653).

**Sources checked.** The arXiv HTML (https://arxiv.org/html/2507.21653) was fetched today, and every spec in `reports/dgp_paper_notes.md` matches it. Five points the notes leave out or get only partly right:
- **Metapaths.** The paper never lists its metapath set per dataset. It gives only edge types: Yelp has R-U-R (same user), R-S-R (same product and star rating) and R-T-R (same product and month); Amazon has R-U-R, R-P-R and R-S-R. These are typed relations between nodes of one type, reviews. That is closer to our friend/follow setup than the word "heterogeneous" suggests.
- **Unreported settings.** The paper does not report LoRA alpha, early-stopping patience, the summariser model (only "frozen LLM"), whether the target text is truncated, or which numerical features it uses.
- **Normalisation.** Eq. 7 writes 1/K over K+1 terms, which is probably a typo for 1/(K+1).
- **Budget units.** The text says the summary budget is in tokens; the Figure 5 caption says words.
- **Prompt content.** Eq. 12 puts only the target's *text* in the prompt. The target's own numbers do not appear; only the neighbours' means a_P(v) do.

The abstract page was not fetched separately; the HTML version was enough.

**Compute assumptions.** We have one GPU job at a time on a 16 GB M-series Mac. From the run logs, one Qwen3-1.7B LoRA run on 1,300 users takes about 2.2 h (8.7–9.8 s/step, about 0.8–0.9 h per epoch, early stop after about 2 epochs). Rescoring the full validation and test sets takes about 0.7 h. Regenerating all summaries for one budget takes about 3.3 h on TwiBot-20 A: 8,873 node summaries in about 2.3 h, then 9,030 metapath summaries in about 1 h (`results/twibot20/A/step6_dgp.log`). The current seed chain (from 4 Nov 04:26, about 11 h) holds the GPU until about 15:30 on Wed 4 Nov. That leaves about 75 h of GPU wall-clock before Saturday 7 Nov at night, or about 55–60 h after a safety margin.

---

## 1. Component-by-component table

| # | Paper component | What our code does (file:line) | Faithful / deviation | Why we deviated (if stated) | Likely effect on results | Fixable on the Mac by Saturday? (GPU h) |
|---|---|---|---|---|---|---|
| 1 | **Relations and metapaths** (§3.2, eq. 2–4): typed relations; A_P = product of the A_r; N_P(v) = {u : (A_P)_vu > 0} | Three metapaths: friend, follow, friend-of-friend (`step6_dgp.py:95-98`). Typed edges are re-read from the zip (`:105-117`), symmetrised (`:138`), multiplied and binarised with the diagonal removed (`:142-152`) | **Mostly faithful** in mechanics (eq. 3–4 exactly; removing the self-loop is a sensible addition). **Deviation**: we symmetrise, so edge direction is lost, and we have 3 metapaths against the paper's 3 relations, which have no 2-hop composition | Stated (Decision log, 2 Nov): directed, only 38% of users have an outgoing friend link; undirected, 93% have a neighbour. The paper's same-user/product/time relations are symmetric by construction | Large for what the prompt *can* carry: on TwiBot-20 all edge label signal is in direction (in-follow minus in-friend gives untrained AUROC 0.713; outgoing links about 0.51), and symmetrising throws it away. That signal is a crawl artefact (REPORT "Who lists whom"), so dropping it is defensible | Yes, but not recommended as a main result. Directed metapaths (in/out × friend/follow) need about 1.5 h of new metapath summaries (node summaries are mostly cached) plus 2.2 h LoRA, so **about 4 h**. They would expose the artefact to the LLM |
| 2 | **Node summary** (eq. 5): task-agnostic "Summarize the text within B tokens", frozen LLM, B_node from {10,20,40,80} | `SUMMARY_PROMPT = "Summarize the text within {b} tokens..."` (`:91`), B=10 (`:71`), greedy, `max_new_tokens=B` (`:199-214`), frozen Qwen3-1.7B before LoRA (`:439`, `:452-455`). Input cut to the first 1,000 characters (`SUMMARY_CHARS`, `step5_llm_reason.py:56`, used at `step6_dgp.py:452`) | **Faithful** prompt, budget and frozen model (B=10 is the paper's best). **Deviation**: summariser input truncated to 1,000 characters; B fixed, not grid-searched; summariser is 1.7B, not (presumably) 8B | Stated for the cut: memory (step 5 docstring, "to fit memory"); fixed B: laptop budget (Decision log 2 Nov) | Small to moderate. A 10-token summary of a tweet-heavy user is very lossy, and the first 1,000 characters (bio + first tweets) are what it summarises. B=20 might suit tweet text better than review text, but the paper finds longer budgets dilute | B=20 needs all summaries again (3.3 h) plus LoRA (2.2 h), so **about 5.5 h** per budget point |
| 3 | **MDK trimming** (eq. 6–9): T = D⁻¹A_P, Z(K) = (1/K) Σ_{k=0..K} T^k, h = Z X with X = raw node features; keep top-M by ‖h_u − h_v‖₂; K ∈ {1,2,3}, M ∈ {2,4,8,16} | `diffusion_embeddings` (`:155-164`) computes (X + TX + T²X)/K with K=2 (`:72`); `trim_neighbours` keeps the M=8 (`:73`) nearest by L2 (`:167-180`). X = multilingual-E5 text embeddings with the two 0/1 no-text flags dropped (`:446-448`) | **Faithful** equations, including the literal 1/K. 1/K and 1/(K+1) are the same constant on every node, so **the neighbour ranking is identical** either way and the normalisation cannot matter. **Deviation**: X is a text embedding, not "raw" (numeric) features; K and M fixed | Stated: "our raw features are text" (Decision log 2 Nov; docstring `:22`) | Small. The M=8 limit binds only for friend-of-friend (mean kept 1.8 friend, 0.8 follow, 5.1 friend-of-friend; `step6_dgp.log`), so on 1-hop metapaths MDK keeps every neighbour and does nothing. Using E5 for X makes trimming favour text-similar neighbours, and real TwiBot-20 edges already join similar-text users (centred cosine 0.26) | Possible but low value: a new K or M changes only friend-of-friend items, about 0.5–1 h of metapath summaries plus 2.2 h LoRA, so **about 3 h** per point. X = allowed metadata instead of E5: same cost |
| 4 | **Metapath summary** (eq. 10): Summarize(⊕ kept node summaries; B_meta) | The kept neighbours' node summaries are joined with newlines and summarised again with the same prompt and B=10 (`:456-459`). No neighbours on a metapath gives the literal string "none" (`:222`, `:255`) | **Faithful** (B_meta = B_node = 10, the paper's best region) | – | The "none" lines tell the LLM which link types a user has, a label cue (+0.013 to LR in A, CI includes 0; +0.023 in B). In A, target LoRA plus these three flags matches `dgp` (0.731 vs 0.711). The paper's relations are dense, so "none" is probably rare there | Already controlled for (target + flags LR, 0 GPU). No fix needed, only disclosure |
| 5 | **Numerical summary** (eq. 11, §4.3): a_P(v) = mean of the kept neighbours' numerical/categorical features per metapath, appended to each S_P(v) | **Omitted** (docstring `:26-27`): the prompt has no numbers. `metadata_check.py` now allows 5 non-shortcut fields: log followers, log following, log tweet count, log listed count, account age; `verified` is excluded as a documented shortcut | **Deviation**: this is the paper's "w/o NumSumm" ablation | Stated: all metadata was excluded as a shortcut (`verified` alone gives macro-F1 0.803; Hays et al. 2023). Superseded by `metadata_check.py`, which keeps 5 fields under a rule fixed in advance | Probably moderate. The paper's ablation (Fig. 4) shows removing NumSumm lowers DGP, though less than removing text. On TwiBot-20 the 5 fields add +0.048 [0.027, 0.067] to text LR, but that is the *target's own* metadata, which eq. 12 does not put in the prompt. Neighbour means of followers and listed counts are tied to the crawl artefact (Spearman ρ 0.59 with in-friend count) | **Yes, about 2.2 h** (summaries cached; prompt change only), plus a CPU control (MLP + the same neighbour means) |
| 6 | **Prompt format** (eq. 12): target text ⊕ per-metapath (S_P ⊕ a_P); question not given | `build_prompt` (`:250-256`): "Target account (bio, then recent tweets):", then "Summaries of linked accounts:" with one line per metapath, then "Is this Twitter account a bot ...? Answer Yes or No." (`step5_llm_reason.py:52-53`), in the Qwen3 chat template with `enable_thinking=False` (`:192-195`) | **Faithful** in structure. The wording is ours, since the paper's template is not published | – | Negligible, apart from the "none" cue (row 4) | – |
| 7 | **Target text**: the paper keeps the target's "fine-grained textual details" and states no truncation | `fit_target` keeps whole lines up to `TARGET_TOKENS = 480` (`:84`, `:235-247`); 81% of users are cut (13.2 of 16.6 lines kept); median prompt about 495 (target) / 534 (dgp) tokens | **Deviation** (probably; the paper's reviews are short and likely untruncated) | Stated (Limitations): memory and speed on the Mac | Biases LoRA against MLP, whose encoder reads all 2,000 characters. dgp vs target is unaffected (same cap). Probably worth 0.01–0.02 for LoRA vs MLP, unmeasured | Partly. 960 tokens roughly doubles step time: **about 4.5 h** per run, with a memory risk (7–8 GB now) |
| 8 | **LoRA config**: "all attention layers"; rank ∈ {4,8,16,32}; dropout ∈ {0,0.05,0.1}; alpha not reported | `LORA = r=16, alpha=32, dropout=0.05, q/k/v/o_proj` (`:75-76`), 6.4M trainable (0.37%) | **Faithful** (inside the grid; q/k/v/o is "all attention layers"); fixed, not tuned | Laptop budget (Decision log 2 Nov) | Small. r=16 is a middle grid value | Each grid point is **2.2 h**. Low value |
| 9 | **Learning rate** ∈ {1e-5, 3e-5, 1e-4}; AdamW | `LR = 1e-4` (`:77`), AdamW (`:315`), gradient clipping at 1.0 (`:350`, not in the paper) | **Faithful** (inside the grid); untuned | Budget | Possibly moderate. 1e-4 is the grid's top value, and validation loss was lowest after 1 epoch and then rose while AUROC kept rising (A dgp: 1.512 → 0.588 → 0.618 → 0.682; AUROC 0.720 → 0.772). That is fast over-confidence; 3e-5 might train more smoothly | 3e-5 needs more epochs: **about 3–4 h** |
| 10 | **Batch** 4 | `BATCH = 4` (`:78`), length-sorted within chunks of 100 (`:294-303`) | **Faithful** | – | – | – |
| 11 | **Epochs / early stopping**: up to 10 epochs, early stop on validation loss (patience not reported) | `MAX_EPOCHS = 10`, 2 checks per epoch (`:329`), `PATIENCE = 2` checks = 1 epoch (`:80`, `:383-393`) | **Faithful** in rule; the patience is ours | – | Runs stop after 2 epochs (A) because validation loss on 200 users rises after epoch 1. With the paper's 1,300-user validation set and AUROC-based selection the stop point would probably be later | See row 13 |
| 12 | **Loss** (eq. 13) and **score** (eq. 14) | Full-vocabulary cross-entropy on the last position, Yes/No target, no class weights (`:343-345`); P(Yes) = softmax over the two logits (`:287`) | **Faithful** | – | – | – |
| 13 | **Model selection**: all hyperparameters chosen to maximise **validation AUROC**; checkpoint by early stopping on validation loss | Checkpoint = lowest validation **loss** on a **200-user** stratified validation sample (`:432`, N_VAL from `step5_llm_reason.py:42`; `:383-386`). Cut-off = best macro-F1 on the full validation set (2,365; `:506`). `step6_rescore.py:35-36, 74-86` re-picks between only two adapters (lowest loss, last check) by full-validation macro-F1 | **Deviation**: different selection metric (loss / macro-F1 vs AUROC), much smaller check set, only 2 saved adapters | Stated: speed (validation sample); the rescore rule copies the text-only models' rule (written into the script before any last-adapter test score was seen) | **Large for headline numbers**: the rule changes A `dgp` 0.711 → 0.727 and `target` 0.704 → 0.728, and flips `dgp` − `target` from +0.007 to −0.001. On 200 users validation loss is noisy and disagreed with AUROC in 3 of 4 runs | **Yes**: check on 1,300 validation users (the paper's size), save every check, keep best validation AUROC. Adds about 15 min per check, about 0.5 h per epoch: **about 3.5–4.5 h per run** |
| 14 | **Validation size**: 1,300 (= train size, Table 1) | 200-user sample for checks (A; 400 in B); full 2,365 for the cut-off | **Deviation** | Speed | As row 13 | As row 13 |
| 15 | **Training size**: about 1,300 (Table 1) | `N_TRAIN = 1300`, stratified sample of the 8,278 (`:74`, `:431`) | **Faithful** | – | Makes LoRA comparable to the paper, but not to our text-only baselines on 8,278 users. Hence the training-size control (MLP 0.726 / LR 0.725 on the same 1,300; 0.733 ± 0.006 over 6 draws) | Done (CPU) |
| 16 | **Hyperparameter grid** over B, K, M, rank, dropout, lr: 4·4·3·4·4·3·3 = 6,912 combinations (or fewer, if searched one at a time) | None: one fixed setting (`:71-80`) | **Deviation** | Budget | Unknown sign. The paper's tuned DGP is compared with baselines tuned on their own code. Our untuned DGP faces an MLP that is also untuned, so this is fair internally, but it cannot reproduce the paper's best-case numbers | Full grid **infeasible** (thousands of 2.2 h runs). 2–3 points feasible (section 3) |
| 17 | **Seeds**: 5, mean ± std | 1 unseeded-LoRA run per variant; seeds 1 and 2 running on A (`LORA_SEED`, `:477-481`); batch order is seeded from SEED+epoch (`:297`) | **Deviation** (3 vs 5; A only) | Budget | One-seed differences of ±0.01–0.02 are inside noise; every `dgp` − `target` gap so far is that size | 3 seeds by Wed night (running). Seeds 3–4 for both variants: **about 9 h** more |
| 18 | **Backbone**: Qwen3-8B on 4× A100 80 GB | Qwen3-1.7B fp16 on MPS (`:63`), gradient checkpointing (`:482`) | **Deviation** (same family, about 1/5 the size) | Memory: 8B in fp16 is about 16.4 GB of weights alone, more than the Mac's 16 GB unified memory | Plausibly moderate: a bigger model reads summaries and long text better. But the paper's own 8B target-only LLM is *below* MLP on 3 of 4 datasets, so size alone does not make the LLM win | 8B: **infeasible** in fp16. 4-bit QLoRA: bitsandbytes has no proper MPS support. An MLX-LM port would be a new pipeline (risky; probably at least 8–10 h per run, unverified). Qwen3-4B fp16 (about 8 GB weights) may fit with checkpointing: about 5 h per run, untested |
| 19 | **Baselines**: MLP, GraphSAGE, HGT, ConsisGAD, PMP, GAAP; LLM (target only); TAPE; GraphGPT, HiGPT, InstructGLM | LR, MLP, GCN, SAGE, FAGCN, FLAG-lite, LINKX, TF-IDF, graph-only (step 4); zero-shot target/raw/DGP (steps 5–6); LoRA target-only (the paper's "LLM" baseline) | **Partly faithful**: MLP, SAGE and LLM-target are covered. Missing: HGT (typed relations), the fraud GNNs (ConsisGAD, PMP, GAAP), TAPE, graph-LLMs | Budget; FLAG-lite and LINKX chosen as heterophily/fraud stand-ins | Our GNNs train on 8,278 users with LLM embeddings; the paper's train on about 1,300. No GNN has been run at 1,300, so "GNN vs DGP at equal data" is untested | HGT or relation-typed SAGE on friend/follow, and all GNNs at 1,300 users: **CPU only, 0 GPU h**. Graph-LLM baselines: infeasible |

---

## 2. How our results compare with the paper's

| Paper finding (Table 2) | Ours (TwiBot-20 A, full test) | Agrees? |
|---|---|---|
| GNNs > MLP (GraphSAGE +0.5 to +3.5 F1 over MLP; HGT/ConsisGAD more) | MLP ≥ every GNN on all 7 encoders; GCN −0.059 [−0.082, −0.037]; SAGE −0.015 (borderline); FLAG-lite, LINKX below MLP | No |
| Fine-tuned LLM alone < MLP (3 of 4 datasets on F1; about −1 to −1.5 points) | LoRA target 0.704 vs MLP on the same 1,300 users 0.726: −0.022 [−0.047, +0.003]. Under the full-validation rule: 0.728, level | **Roughly yes** in direction and size, within noise |
| DGP > everything (+5 to +10 F1 over MLP; +4 to +11 over LLM) | `dgp` 0.711 / 0.727 (two checkpoint rules) ≈ `target` 0.704 / 0.728 ≈ MLP at 1,300 users 0.726; below MLP on 8,278 (0.766) | No |

### GNNs ≥ MLP (paper) vs MLP ≥ GNNs (ours): mainly a property of TwiBot-20
- **Dataset.** On TwiBot-20 neighbour labels carry almost nothing: adjusted homophily −0.015 to 0.08, label informativeness ≤ 0.005. Real edges help GCN only relative to shuffled edges (+0.037), never relative to no edges. The paper's R-U-R relation joins reviews by the *same user*, a link that in review-spam data tends to join same-label nodes. The paper does not report homophily for its graphs, so this contrast is an inference, but it is the most likely reason.
- **Dataset.** Our node text is strong (text-only AUROC 0.84), and averaging blurs it. On Instagram, where text is weak, GCN is level with MLP. The paper's MLP is much weaker (AUROC 0.70–0.75 on Yelp/Amazon), so it leaves more room for the graph.
- **Implementation, small.** Our GNNs use fixed, untuned hyperparameters and train on 8,278 users, while the paper's baselines are tuned and train on about 1,300. Our MLP also gets strong LLM embeddings of the full text. The shuffled-edge and bridge-graph controls and the 7-encoder ladder make an implementation bug an unlikely explanation. A 1,300-user GNN run would close the training-size gap (CPU).

### Fine-tuned LLM alone < MLP: we roughly reproduce this
Our gap (−0.022, CI includes 0) matches the paper's 1–1.5 point gap. Two of our deviations push the same way, so the true gap could be smaller:
- the 480-token cap on the target text (row 7);
- the early checkpoint picked on 200-user validation loss (row 13).

With the full-validation macro-F1 rule, LoRA target is level with MLP (0.728).

### DGP > everything (paper) vs DGP ≈ target ≈ MLP (ours)
In rough order of plausibility:
1. **Dataset: the neighbourhood has nothing label-relevant to summarise.** DGP's gain has to come from neighbour content that predicts the target's label. On TwiBot-20 friend links are at chance (adjusted −0.015) and follow links nearly so. The only strong relational signal, link direction, is a crawl artefact that our undirected metapaths discard on purpose. Even a perfect replication would have little to gain here. This is the main reason, and it is not a fault of the implementation.
2. **Dataset: neighbourhoods are tiny.** 1-hop metapaths keep 1.8 (friend) and 0.8 (follow) neighbours on average; about 30% and 55% of targets get "none". MDK and summarisation have little to work on, and the "none" lines mostly pass on link-type flags. In A, those flags alone explain `dgp`'s edge (target + flags 0.731 vs `dgp` 0.711).
3. **Implementation: no NumSumm** (row 5). The paper's ablation says removing it costs performance. Our neighbours' allowed metadata is partly the crawl artefact again (followers/listed counts ρ 0.59 with in-friend count), so any gain would need the same artefact check.
4. **Implementation: model selection** (rows 13–14). The paper picks everything on validation AUROC over 1,300 users; we used validation loss on 200. That choice alone moves each variant by about 0.02, as large as any `dgp` − `target` gap.
5. **Implementation: no grid.** The paper tunes B, K, M, rank, dropout and lr on validation AUROC; tuning can only raise its DGP number, by an unknown amount. Ours is one untuned point.
6. **Implementation: backbone 1.7B vs 8B.** Plausibly limits how much the model gets from 10-token summaries, but it does not explain the dataset-level pattern (point 1), and the paper's own 8B target-only model does not beat MLP.
7. **Implementation: summary input.** The summariser sees only the first 1,000 characters, and 10 tokens is very short for tweet text. Small.
8. **Noise.** One seed so far, with CIs of about ±0.02–0.03 for every LoRA gap.

**Bottom line.** Of the three disagreements, "GNNs > MLP" and "DGP > everything" fail mainly because of TwiBot-20: neighbour labels carry almost nothing, neighbourhoods are tiny, and the only strong relational cue is a crawl artefact. "LLM alone < MLP" is reproduced. Missing NumSumm and the model-selection rule are the implementation deviations most likely to have cost DGP measurable performance. Both can be fixed by Saturday.

---

## 3. Ranked fixes worth doing this week

The GPU is free from about 15:30 on Wed 4 Nov. Budget: about 55–60 GPU h. Each item needs the run to finish and be rescored. Where the paper and our existing rule disagree, report both rules.

| Rank | Fix | Expected value | GPU h |
|---|---|---|---|
| 1 | **Finish seeds 1–2 (running) and fix one checkpoint rule for all runs before seeing their test scores.** Report `dgp` − `target` as the mean ± std over 3 seeds | High: gives the main step 6 claim a seed estimate, which at the moment rests on 1 seed | 0 extra (in the current chain) |
| 2 | **Paper-faithful selection**: validation checks on 1,300 users (the paper's size), adapter saved at every check, keep best validation AUROC; patience 2 checks on validation loss as now, `MAX_EPOCHS=4` cap. Run `dgp` and `target`, 1 seed each | High: removes the two largest procedural deviations (rows 13–14) and settles the checkpoint-rule question with the paper's own metric | about 4 h × 2 = **8 h** |
| 3 | **NumSumm (eq. 11)** with the 5 allowed fields: the neighbours' mean per metapath appended to each summary line (not the target's own fields, per eq. 12). Run `dgp`+NumSumm under rule 2. CPU control: MLP/LR on text + the same neighbour means. Optional second run: `target` + own metadata, to separate "metadata helps" from "neighbour numbers help" | High for faithfulness (restores the last missing DGP component); moderate for results. Must be reported with the crawl-artefact caveat | 4 h (+4 h optional) = **4–8 h** |
| 4 | **Two to three extra seeds under rule 2** for `dgp` and `target`, so the final comparison has 3 seeds under one rule | Medium-high: the paper reports 5 seeds | 4 h × 4 = **16 h** (2 more seeds each) |
| 5 | **Small grid on validation AUROC**: lr 3e-5 (one run), B=20 (summaries again, then one run). Keep whichever wins on validation | Medium: tests whether the fixed point was a poor choice; cannot recover the paper's full tuning | 4 + 5.5 = **about 9.5 h** |
| 6 | **Longer target text** (960 tokens) for `target` and `dgp`, 1 seed | Medium-low: affects LoRA vs MLP, not `dgp` vs `target`; memory risk | about 4.5 h × 2 = **9 h** |
| 7 | CPU-only, run alongside: GNNs (SAGE, FAGCN, FLAG-lite) at the same 1,300 training users; relation-typed SAGE or HGT on friend/follow | Medium: gives the missing "GNN vs DGP at equal data" row and a typed-relation baseline like the paper's HGT | **0 GPU** |
| 8 | Directed metapaths (in/out × friend/follow), disclosed as an artefact analysis | Low for the replication; it would mostly feed the crawl artefact to the LLM | about 4 h |
| – | Changing K or M | Low: M binds only on friend-of-friend | about 3 h each |

A realistic Wed–Sat plan is ranks 1–4 plus one item of rank 5, about 37–41 GPU h after the current chain, which leaves margin for one crash or rerun.

**Infeasible by Saturday:**
- **Qwen3-8B in fp16.** The weights alone are about 16.4 GB, more than the Mac's 16 GB.
- **QLoRA 4-bit 8B through bitsandbytes.** It has no proper MPS support.
- **An MLX port.** It would mean a new training and scoring pipeline. Untested, probably at least 8–10 h per run, too risky this week.
- **The paper's full grid.** Thousands of combinations at 2.2 h or more each.
- **5 seeds for every variant on both A and B under a new rule.** About 40 h or more on top of the plan above.
- **Graph-LLM baselines (GraphGPT, HiGPT, InstructGLM) and TAPE.** Too much new code and compute.

Qwen3-4B in fp16 (about 8 GB weights) might fit with gradient checkpointing at about 5 h per run. It is a possible single extra data point, not part of the plan.

---

## 4. Suggested Methodology wording

> Step 6 is a laptop-scale replication of DGP (Li et al., AAAI 2026), written from the paper's equations because its code is not public. We follow its pipeline: typed metapaths (friend, follow and friend-of-friend on TwiBot-20), diffusion-based neighbour trimming (eq. 6–9, K = 2, M = 8), task-agnostic 10-token node and metapath summaries from the frozen LLM, and LoRA fine-tuning on the first answer token with AdamW, batch size 4, early stopping and about 1,300 training users. The deviations are: Qwen3-1.7B instead of Qwen3-8B, because the 8B model's fp16 weights alone exceed our 16 GB Mac; one fixed hyperparameter setting taken from the paper's grid (r = 16, dropout 0.05, lr 1e-4) instead of a grid search; undirected relations and text (multilingual-E5) embeddings as the diffusion features; the target's text capped at 480 tokens; and checkpoints selected on a 200-user validation sample [update if the 1,300-user, validation-AUROC rule is adopted]. The numerical summary (eq. 11) was omitted in the first runs, which is the paper's "w/o NumSumm" ablation, because TwiBot-20's profile metadata contains a labelling shortcut; [if run: we later add it using five fields that pass a shortcut test fixed in advance]. We report [3] seeds instead of the paper's 5, and compare against our own text-only and GNN baselines trained on the same 1,300 users, not the paper's full baseline set.
