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
| Thu 5 Nov | GNNs on the full 1M graph (1 seed for all 5, 3 seeds for the 2 best on validation) | ~4–7 h CPU |
| Thu night – Fri | LLM on the sample: neighbour summaries, LoRA `target` and `dgp` | ~6–8 h GPU |
| Fri 6 Nov | Confidence intervals, sealed-set final run, expert recalibration, comparison with TwiBot-20 | ~3 h |
| Fri night – Sat 7 Nov | TwiBot-20 follow-up batch (optional), figures | ~12 h GPU + ~4 h |
| Sun 8 Nov | Writing | 1 day |

## Methodology: other items to include (to be filled)

- Datasets and why: Instagram (secondary), TwiBot-20, TwiBot-22 (see `artifacts/04_decisions_and_reasons.md`).
- Heterophily measures: adjusted homophily, label informativeness.
- Shortcut rule and what it excluded (`verified`).
- Split, sealed set, sample; metrics (macro-F1, AUROC, PR-AUC, reweighted PR-AUC, bootstrap CIs, non-verified slice).
