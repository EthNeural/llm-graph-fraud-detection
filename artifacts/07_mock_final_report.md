# MOCK final report: for review (Fri 6 Nov 2026, ~02:00)

> **Status of this draft.** Structure follows `uni brief /Assessment 3 – Project Completion (2).docx`. Numbers are from the working test users (weak labels) unless marked; the sealed-set final run is Friday afternoon, so headline numbers will be replaced. **[PENDING]** = result not in yet; **[verify]** = citation not yet checked; **[TEAM]** = needs the other members' input. The **timeline is skipped on purpose** (deferred by the builder; to be added before submission). Length now ~2,600 words + 5 figure slots; target ≤ 7 pages at 12 pt.

---

**Title of the Project:** Large Language Models for Graph-Based Fraud Detection: When Do Text and Links Help Under Camouflage?

**Supervisor:** Dr Venus Haghighi

**Project Team Members:** Prajan Bhattarai [48807567], Mustafa Ugur Atas [48763802], Zeki Kose [48718459], B Abhishek Biswal [48681466]

---

## 1. Introduction

Online platforms are full of fake and automated accounts that spread spam, scams and manipulation. These accounts are hard to catch one by one, but they live inside a network: they follow, mention and retweet other users. Two kinds of tools promise to help. **Graph neural networks (GNNs)** learn from who is connected to whom. **Large language models (LLMs)** read what accounts write. This project asked a simple question: **when we combine the two, does detection actually improve, and when does it not?**

There is a catch that makes the question interesting. Fraudulent accounts often *camouflage* themselves by connecting to ordinary users, so their neighbours look legitimate. Graph methods that assume "you are like your neighbours" can then be misled. We therefore measured how much each dataset's links really reflect the labels before trusting any graph model.

We set out (Assessment 1) to build a prototype that combines LLM-derived text representations with GNN representations and to compare it with traditional machine learning, GNN-only and language-only models under the same splits and metrics. We expected the hybrid to beat each single component.

What we achieved is a complete, reproducible comparison across three datasets, ending on TwiBot-22 (1 million Twitter users, 7 relation types, 88 million tweets). The outcome differs from our expectation in an informative way: **an account's own profile and text carry most of the signal; neighbour information adds a little at large scale; GNN message passing adds almost nothing under the measured camouflage; and the fine-tuned LLM does not beat simpler text models given enough training data.** We also found that the dataset's labels, made largely by automatic rules and models, overstate one profile cue by about three times compared with independent human judges. These results explain why published benchmark rankings that favour graph models do not carry over to a fair protocol.

**[FIGURE 1: pipeline: data → own features (profile, multilingual-E5 text) / neighbour features / LLM (zero-shot, LoRA, DGP) → models → sealed evaluation]**

## 2. Literature Review

**Graph fraud detection under camouflage.** Fraud graphs are often *heterophilous*: fraudsters connect to normal users, so neighbourhoods mix classes. FRAUDRE (Zhang et al., 2021) frames this as feature, topology and relation inconsistency and reports per-relation class homophily above 0.85 for normal nodes and below 0.2 for fraudsters. A line of work from Haghighi and colleagues addresses camouflage directly: MAGNET adapts aggregation to homophilous versus heterophilous neighbourhoods (Haghighi et al., 2024a); TROPICAL uses hypergraph transformers for higher-order relations (Haghighi et al., 2024b; 2025); a discriminative aggregation model separates intra- and inter-relation patterns (Haghighi et al., 2026) **[verify model name]**. Adjusted homophily (Platonov et al., 2023) gives a measure that is comparable across class balances, and Mao et al. (2023) show that GNNs do well on nodes that match the majority structural pattern and worse on the rest.

**Simple baselines and benchmark artefacts.** GADBench (Tang et al., 2023) finds that tree ensembles with neighbour aggregation beat specialised GNNs across anomaly-detection benchmarks. For bot detection specifically, Hays et al. (2023) show that shallow decision trees on a few features come close to state-of-the-art models on most public datasets, and attribute this to simplistic collection and labelling; on TwiBot-20 a one-question tree on the `verified` flag nearly matches the best model.

**LLMs for graph fraud.** Recent work uses LLMs as encoders, reasoners or knowledge sources for graphs: MLED enriches graph fraud detectors with LLM-generated type and relation knowledge (Huang et al., 2025) **[verify]**; GuARD combines text semantics with graph structure for anomaly detection (Pang et al., 2025) **[verify]**; FLAG uses LLM-guided neighbour selection (Yang et al., 2025); DGP prompts an LLM with node and neighbourhood summaries (**[cite DGP; verify]**). On bot detection, Feng et al. (2024) report gains from instruction-tuned LLMs with about 1,000 labelled examples.

**Benchmarks.** TwiBot-22 (Feng et al., 2022) is the largest graph-based bot benchmark; its leaderboard ranks graph models first.

**Where this project fits.** Most of these methods are evaluated on a few fraud datasets, and it is not known whether they transfer to a large social graph with weak labels. We test that transfer on TwiBot-22 with a protocol designed to remove known shortcuts, and ask the camouflage literature's three questions (relation-level homophily, local heterogeneity, cross-relation inconsistency) on it.

## 3. Research Methodology

**3.1 Decomposition.** We split the problem into five sub-problems: (S1) find datasets with real text, real links and fraud-like labels; (S2) measure heterophily and shortcuts before modelling; (S3) build own-feature, neighbour-feature, GNN and LLM models under one protocol; (S4) test whether gains are real (confidence intervals, controls); (S5) check results against human labels.

**3.2 Datasets (S1) and why they changed.** Assessment 1 promised a dataset-agnostic plan. Elliptic was dropped because it has no natural text, so an LLM had nothing to read. The Kaggle Instagram CSV used in Assessment 2 was dropped because it had no real links (a similarity graph had to be built) and no raw text (the LLM read sentences generated from the same numbers). We moved to the GLBench Instagram graph (real bios and follow links, but labels are commercial vs personal), then TwiBot-20 (real bot labels, text and links), and finally **TwiBot-22** as the large test: 1,000,000 users (14.0% bots), 7 relations (following, followers, mention, retweet, reply, quote, like), bios and the 20 most recent tweets per user, and 986 expert-labelled users. Each change followed from the need for real text and real links; none made the results look better.

**3.3 Fair protocol (S2).** Lessons from TwiBot-20 shaped TwiBot-22: every rule was written down with a timestamp before the score it governs.
- **Split:** random stratified 70/20/10 (14% bots in each part). The released split is not random (train 7.8% bots, test 29.4%; all 95,398 verified users in train), so it is reported only for comparison.
- **Sealed test set:** half of the test users (50,937, including all experts) were set aside and scored once at the end. **[PENDING: final run Fri afternoon]**
- **Shortcut rule:** any feature where ≥ 200 training users are ≥ 99% one class is removed. Only `verified` failed (no verified user is a bot); it is never a model input.
- **Heterophily measured, not assumed:** adjusted homophily 0.010–0.024 on all 7 relations.
- **Metrics:** macro-F1 at a cut-off chosen on validation, AUROC, PR-AUC (also reweighted to 5% and 1% bots), paired bootstrap 95% CIs, 3 seeds. Accuracy is reported only for comparison with the benchmark, because calling everyone human already scores 0.860.

**3.4 Models (S3).** All models see the same users and labels.
- *Own features:* 15 profile fields (counts, account age, empty bio, default picture, ...) and multilingual-E5 embeddings of the bio (all 1M) or bio + 20 tweets (a 100k stratified sample).
- *Neighbour features ("XGB-Graph"):* means of neighbours' features per relation and direction, fed to a gradient-boosted tree (the industry-style baseline).
- *GNNs on the full 1M graph:* GCN, GraphSAGE, FAGCN (heterophily-aware), LINKX, R-GCN, and an MLP with the same layers but no edges as the control.
- *LLM (Qwen3-1.7B):* zero-shot, LoRA fine-tuning on the account's text (`target`, 1,300 training users), and DGP: LoRA with node and neighbourhood summaries (TwiBot-20 scope, 1,500-user test subset fixed before scoring).
- *Text baselines:* TF-IDF + logistic regression, MiniLM + MLP.

**3.5 Checks (S4, S5).** Controls for training-set size (the LLM uses 1,300 users; trees are also run on the same 1,300), edges switched off, and slices (users with a bio, non-verified users). Human-label check: the 986 expert labels were used for training *and* evaluation by 5-fold rotation (each expert scored by a model that never saw them), and an independent 500-user annotation study (3 annotators per user) tests whether label cues are real.

**3.6 Timeline.** *Skipped in this mock (deferred by the builder; to be added before submission: planned vs actual, from Assessment 1's task list, Assessment 2's next steps and the decision log).*

**3.7 Difficulties and fallbacks.** Full 1M-user tweet embedding did not fit the laptop's time budget, so tweets are used on a 100k stratified sample and bios on all 1M; full-graph GNNs needed memory-efficient sparse operations (a FAGCN run was killed by memory and rewritten with edge chunks); DGP summaries ran slower than planned, so DGP was run at TwiBot-20 scope rather than on all test users.

## 4. Research Outcome: Results, Analysis and Discussion

**4.1 TwiBot-20 and Instagram (supporting).** On TwiBot-20, a strong text encoder does the work: Qwen3-1.7B embeddings + MLP reach macro-F1 0.769, against 0.702 for TF-IDF. No GNN beats its encoder's MLP; GCN is 0.059 below it (CI [−0.082, −0.037]). Adjusted homophily is −0.015 to 0.08. The label signal the links do carry comes from how the dataset was crawled (who appears in whose follower sample). LoRA fine-tuning matches text models trained on the same 1,300 users, and DGP's neighbour summaries add nothing measurable (0.727 vs 0.728). On Instagram (weak text signal), GCN is level with the MLP.

**4.2 TwiBot-22 main results** (49,951 working-test users; macro-F1 / AUROC / PR-AUC):

| Model | Score |
|---|---|
| Zero-shot Qwen3-1.7B | 0.505 / 0.461 / 0.139 |
| LoRA `target` (text, 1,300 train, 5,000 sample test users) | 0.626 / 0.710 / 0.312 |
| Profile only (15 fields, tree) | 0.707 / 0.846 / 0.508 |
| Profile + bio (tree) | 0.720 / 0.860 / 0.525 |
| **XGB-Graph + profile** | **0.731 / 0.873 / 0.580** |
| GNNs (FAGCN, RGCN 3 seeds; others 1) | 0.678–0.690 / 0.800–0.829 |
| MLP, no edges (GNN control) | 0.679 / 0.799 / 0.415 |
| LoRA DGP vs `target` (1,500 subset) | **[PENDING]** |
| Sealed test set, all models | **[PENDING]** |

**[FIGURE 2: bar chart of the table with CIs]**

**Finding 1: own profile beats text, and not only through one cue.** Profile − LoRA: +0.073 [+0.054, +0.093] macro-F1; with the same 1,300 training users the profile tree still reaches 0.677 / 0.808. On users who have a bio (where an empty-bio flag is useless, AUROC 0.500), the profile model still scores AUROC 0.802.

**Finding 2: text adds a little on top of profile** (+0.012 [+0.008, +0.016], 1M). Averaging tweets into the bio embedding *hurts* a text-only model (−0.052 [−0.074, −0.030]), because it blurs the empty-bio contrast (40.6% of bots vs 7.8% of humans have no bio); one explicit empty-bio flag closes the gap.

**Finding 3: neighbours help a little at full scale; message passing does not.** XGB-Graph − own-feature tree: +0.010 macro-F1, +0.054 PR-AUC (1M). Every GNN stays within ~0.01 of the no-edge MLP.

**Finding 4: the camouflage literature's three questions on TwiBot-22.** *(Q1)* On all 7 relations, bots' neighbours are 4–17% bots against a 14% base rate; humans' are ~93% humans: the FRAUDRE pattern. *(Q2)* Neighbour features help users in homophilous neighbourhoods (+0.057 AUROC) and hurt users in heterophilous ones (−0.016), as Mao et al. predict; these bins use the users' own labels, so they diagnose rather than deploy. *(Q3)* Eight cross-relation inconsistency features add nothing (−0.002 [−0.004, −0.000]). TROPICAL's motivating pattern (bot density rising with hop distance) does not transfer: it falls from 0.116 to 0.059 over 1–4 hops.

**[FIGURE 3: per-relation class homophily (Q1) and gain by local-homophily bin (Q2)]**

**Finding 5: the leaderboard conflict.** The TwiBot-22 benchmark ranks graph models first. Its official code gives BotRGCN, GCN and GAT the `verified` flag, uses the non-random split, trains a fixed 200 epochs and ranks by accuracy. Running our models under that protocol: the split lowers AUROC by ~0.11 for every model; `verified` moves the trees by ≤ 0.01; and still no graph model beats the tree without neighbours (official split, with `verified`: own-feature tree 0.700 / 0.765, XGB-Graph 0.689 / 0.758, FAGCN 0.667 / 0.731, RGCN 0.666 / 0.724, MLP 0.683 / 0.745). We do not reproduce a graph advantage under one common protocol. The benchmark's labels were also partly made by GCN, GAT and R-GCN models, the same families that top it.

**[FIGURE 4: our split vs official split, with and without `verified`]**

**Finding 6: human labels.** On the 986 experts (rotation, 105 bots), the ranking holds: XGB-Graph + profile after the expert stage reaches AUROC 0.910 [0.879, 0.934]; R-GCN gains most from expert labels (+0.067 [+0.034, +0.104]) but stays below the trees; one epoch of LoRA on ~789 experts lowers its AUROC (−0.099). The dataset's 90.5% agreement with experts is barely above calling everyone human (89.4%), and only 56 of 105 expert bots are labelled bots. The 986 expert labels trained the label-making models, so they are not independent. On the independent annotation study (458 users, 44 bots), **empty-bio accounts are 5.0× as likely to be bots under the dataset's labels but only 1.6× [0.7, 3.1] under the human annotators** (difference +3.4 [+1.2, +6.1]). The weak labels overstate the cue. This is consistent with Hays et al.'s warning and, as far as we found, extends it to TwiBot-22.

**[FIGURE 5: empty-bio lift: weak vs expert vs annotators]**

**4.3 Answers to the original research questions.**
- *RQ1 (hybrid vs GNN-only):* adding text to graph models helps, but the gain comes from the text and profile, not from message passing; the best model is a tree on own plus neighbour-averaged features.
- *RQ2 (GNNs vs traditional ML):* trees match or beat GNNs on every dataset.
- *RQ3 (semantic contribution):* text adds +0.01 over profile on TwiBot-22; on TwiBot-20, where profile is not used, a strong encoder is the main gain.
- *RQ4 (fusion strategy vs cost):* concatenating frozen embeddings is cheapest and best; LoRA matches it only at equal training size; DGP's neighbour summaries add nothing on TwiBot-20 **[PENDING TwiBot-22]**.
- *RQ5 (imbalance):* validation cut-offs and PR-AUC reweighted to 5% and 1% bots (best: 0.366 and 0.146).

**4.4 How the problem was revised (P → P′).** The original problem P was "does an LLM–GNN hybrid improve fraud detection?". The evidence moved it to P′: "**under measured camouflage and weak labels, which signals actually carry information, and why do benchmark rankings disagree?**" Sub-problems that did not work as planned: a deep LLM–GNN fusion did not beat simpler fusion; an exploratory oracle test showed that even a perfect per-user "trust your neighbours" switch would gain only +0.003 AUROC on TwiBot-22, so we did not build it.

**Limitations.** Labels are mostly automatic; the sealed-set numbers are pending; the LLM read up to 480 tokens and trained on 1,300 users; tweets were used on a 100k sample; GNN hyperparameters were not tuned.

## 5. Individual Contributions

- **Prajan Bhattarai:** the language-model and TwiBot pipeline: text encoders, LoRA fine-tuning and DGP, the TwiBot-20 and TwiBot-22 data preparation, shortcut and heterophily checks, the sealed-set protocol, the leaderboard reconciliation, the expert-label and annotation-study analysis, and the report draft.
- **Mustafa Ugur Atas:** **[TEAM]**
- **Zeki Kose:** **[TEAM]**
- **B Abhishek Biswal:** **[TEAM]**
- *Role changes:* **[TEAM: how responsibilities were redistributed after Assessment 2]**

## 6. Future Outlook

1. **Better labels.** The largest remaining problem: weak labels overstate a profile cue, and the expert labels are not independent. More independent annotation, targeted by active learning at the users models disagree on, and noise-robust training are the most promising next steps.
2. **Camouflage-aware aggregation has little room here.** Per-node adaptive models (MAGNET-style) are the natural next test, but our oracle switch shows at most +0.003 AUROC of headroom on TwiBot-22, so gains must come from labels and own-account features first.
3. **LLMs with more context and data.** Longer inputs than 480 tokens, more training users and LLM-derived features fed into tree models instead of the LLM deciding alone. **[update after DGP]**
4. **Shift and newer bots.** Every model loses ~0.11 AUROC on the non-random split; time-based splits and post-2022 data (bots writing with LLMs) are needed.

## 7. Bibliography

*(Every entry is from 2021 or later, per the brief. Style to be unified.)*

- Feng, S., et al. (2022). TwiBot-22: Towards graph-based Twitter bot detection. *NeurIPS Datasets and Benchmarks*. arXiv:2206.04564.
- Feng, S., et al. (2024). What does the bot say? Opportunities and risks of large language models in social media bot detection. *ACL*. arXiv:2402.00371.
- Haghighi, V., et al. (2024a). Robust graph learning against camouflaged malicious actors (MAGNET). *WISE 2024*, LNCS. doi:10.1007/978-981-96-0567-5_12.
- Haghighi, V., et al. (2024b). TROPICAL: Transformer-based hypergraph learning for camouflaged fraudster detection. *IEEE ICDM*. doi:10.1109/icdm59182.2024.00019.
- Haghighi, V., et al. (2025). Beyond pairwise relationships: a transformer-based hypergraph learning approach for fraud detection. *Knowledge and Information Systems*, 67, 7987–8022. doi:10.1007/s10115-025-02476-5.
- Haghighi, V., et al. (2026). Beyond smoothing: A discriminative enhanced aggregation graph neural network for camouflaged fraud detection. *ACM TKDD*. doi:10.1145/3801967. **[verify title / model name]**
- Hays, C., Schutzman, Z., Raghavan, M., Walsh, E., & Cresci, S. (2023). Simplistic collection and labeling practices limit the utility of benchmark datasets for Twitter bot detection. *WWW*. arXiv:2301.07015.
- Huang, T., et al. (2025). Can LLMs find fraudsters? Multi-level LLM enhanced graph fraud detection. arXiv:2507.11997. **[verify]**
- Mao, H., et al. (2023). Demystifying structural disparity in graph neural networks: Can one size fit all? *NeurIPS*. arXiv:2306.01323.
- Pang, Y., et al. (2025). GuARD: Effective anomaly detection through a text-rich and graph-informed language model. *KDD*. **[verify]**
- Platonov, O., et al. (2023). Characterizing graph datasets for node classification: Homophily–heterophily dichotomy and beyond. *NeurIPS*. **[verify id]**
- Tang, J., et al. (2023). GADBench: Revisiting and benchmarking supervised graph anomaly detection. *NeurIPS Datasets and Benchmarks*. arXiv:2306.12251.
- Yang, et al. (2025). FLAG. *KDD*. **[full reference; verify]**
- Zhang, G., Wu, J., Yang, J., Beheshti, A., Xue, S., Zhou, C., & Sheng, Q. Z. (2021). FRAUDRE: Fraud detection dual-resistant to graph inconsistency and imbalance. *IEEE ICDM*. doi:10.1109/ICDM51629.2021.00098.
- **[DGP reference; verify]** · **[FAGCN is 2021, LINKX 2021, R-GCN 2018: cite R-GCN only if allowed as a foundational method]**
