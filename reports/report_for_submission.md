# For the submission (internal)

"Internal" = goes into the submitted COMP8851 research project report (7 pages, template: Introduction, Literature review, Methodology + timeline, Outcome, Contributions). Everything else in the repository is external working material. Items are added here as they are decided; Sunday's writing starts from this file.

## Methodology: timeline

Past work (dates 2026):

| When | Work |
|---|---|
| Sep – Oct | Team pipeline on Elliptic (Bitcoin transactions) |
| 27 Oct | Elliptic dropped (no text for an LLM to read); rebuilt on Instagram (GLBench text graph) |
| 30 Oct | Instagram results; settings A (natural) and B (1:10, FLAG protocol) |
| 1 – 4 Nov | TwiBot-20 as the main dataset: encoder ladder, GNNs (GCN, SAGE, FAGCN, LINKX, FLAG-lite), zero-shot LLM, LoRA fine-tuning and a DGP replication; shortcut and crawl-artefact checks; 22-item self-review |
| 4 Nov | TwiBot-22 data preparation: 1M users, 88.2M tweets, 7 relations; shortcut, heterophily and leakage checks; sealed test set and 100k study sample; all rules recorded before any model |

Planned (estimates, 5 Nov):

| When | Work | Estimate |
|---|---|---|
| Thu 5 Nov, early | Text embeddings (multilingual-E5, MiniLM) for the 100k sample + hubs | ~3.5 h GPU (measured pace for E5) |
| Thu 5 Nov | Cheap models (MLP, trees, XGB-Graph, DEC-lite, baselines), with and without profile, sample and 1M | ~2–4 h CPU |
| Thu 5 Nov | GNNs on the full 1M graph: 1 seed for all 5, then 3 seeds for the 2 best by validation macro-F1 (tables mark 1 vs 3 seeds) | ~4–7 h CPU |
| Thu 5 Nov, 02:41–06:18 | LoRA `target` (measured: 3.6 h) | done |
| Thu 5 Nov – Fri 6 Nov | Expert rotation (~4 h), then DGP at TwiBot-20 scope (1,500-user test subset; ~9.5 h, summaries ~1 s each), then second seeds of `target` and `dgp` (~7 h) | ~20 h GPU |
| Fri 6 Nov, early | Weak → expert LoRA fine-tuning, 5-fold rotation over the 986 experts (blind) | ~3.5–4 h GPU |
| Fri 6 Nov | Confidence intervals, expert rotation for trees / MLP, sealed-set final run, comparison with TwiBot-20 | ~3 h |
| Fri night – Sat 7 Nov | TwiBot-20 follow-up batch (optional), figures | ~12 h GPU + ~4 h |
| Sun 8 Nov | Writing | 1 day |

## Methodology: other items to include (to be filled)

- Datasets and why: Instagram (secondary), TwiBot-20, TwiBot-22 (see `artifacts/04_decisions_and_reasons.md`).
- Heterophily measures: adjusted homophily, label informativeness.
- Shortcut rule and what it excluded (`verified`).
- Expert labels (986, 105 bots): used for training **and** clean evaluation by 5-fold rotation (decided 5 Nov, before any expert score); weak labels match experts for 56 of 101 dataset bots.
- Split, sealed set, sample; metrics (macro-F1, AUROC, PR-AUC, reweighted PR-AUC, bootstrap CIs, non-verified slice).

## Outcome: TwiBot-22 findings so far (5 Nov evening; working-test users, weak labels; sealed set and experts not yet scored)

Paired bootstrap 95% CIs (2,000), seed 0, each model with its own validation cut-off (`my_build/twibot22_ci.py` → `results/twibot22/ci_pairs.json`). Sample = the 5,000 sample working-test users (same as LoRA); 1M = 49,951 working-test users, bio only. Models are HistGradientBoosting trees ("XGB-Graph style"), 3 seeds unless noted; sceptic-checked 5 Nov.

1. **Profile beats every text model.** Profile only (15 fields): macro-F1 0.701, AUROC 0.837. LoRA `target` (Qwen3-1.7B, 1 seed, 1,300 training users): 0.626 / 0.710; difference +0.073 [+0.054, +0.093] F1, +0.128 [+0.106, +0.149] AUROC. With the same 1,300 training users, profile only still gets 0.677 / 0.808. Caveat: much of the profile signal is the empty-bio flag (40.6% of bots vs 7.8% of humans), bio length and follower counts, on weak labels partly made by labelling functions; the expert check (rotation) tests this.
2. **Text adds a little on top of profile.** Tree profile + text − profile: +0.011 [−0.001, +0.025] F1, +0.012 [+0.005, +0.018] AUROC (sample); +0.012 [+0.008, +0.016] / +0.014 [+0.012, +0.016] (1M).
3. **Neighbour features help without profile, barely with it.** XGB-Graph − own features, no profile: +0.005 [−0.010, +0.021] F1 but +0.053 [+0.037, +0.070] AUROC (sample), +0.015 / +0.044 (1M). With profile: −0.001 [−0.012, +0.010] F1, +0.005 [−0.000, +0.010] AUROC (sample); +0.010 [+0.007, +0.013] / +0.012 [+0.011, +0.014] (1M, where PR-AUC +0.054). Best model overall: XGB-Graph with profile on 1M, 0.731 / 0.873 / PR-AUC 0.580.
4. **Tweets add nothing over the bio for a text-only model; averaging them in hurts.** Same 70k training / 5,000 test users: bio + 20 tweets − bio only: −0.052 [−0.074, −0.030] F1, −0.049 [−0.074, −0.026] AUROC. Mechanism checked by slices: on users with a bio the two are equal (AUROC 0.682 vs 0.681); on the 602 users without a bio (46% bots) tweets help (0.613 vs 0.500), but averaging blurs the no-bio group's contrast with everyone else (mean score 0.25 vs 0.12, against 0.46 vs 0.10 for bio only). With profile (which holds the empty-bio flag) the two are equal (0.713 vs 0.712).
5. **Fine-tuned LLM vs cheap text models** (5,000 users): with the same 1,300 training users LoRA beats TF-IDF+LR by +0.050 F1 [CI excludes 0]; TF-IDF+LR on 70k training users ties LoRA (+0.002 [−0.015, +0.019]). Zero-shot Qwen: 0.505 / 0.461 (answers "bot" for nearly everyone; readout checked).
6. Crawl-structure baseline (links only, neighbour counts): 0.563 / 0.672: the graph's shape alone carries some signal.
