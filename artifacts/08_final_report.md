# Final report draft (Fri 6 Nov 2026, ~10:40; sealed final run included)

> **Status.** Built from the reviewed mock (`07_mock_final_report.md`) after the TwiBot-22 sealed final run (Fri 10:20, read once). Length target ~7,000 words excluding the bibliography, Arial 11 pt; one submission per group, due 11:55 PM Sun 8 Nov. Still open: the timeline (§3.8, skipped on purpose; to add before submission), and TwiBot-20 `dgp_num` (running). Numbers come from the result files named in `reports/report_for_submission.md`.

---

**Title of the Project:** Large Language Models for Graph-Based Fraud Detection: When Do Text and Links Help Under Camouflage?

**Supervisor:** Dr Venus Haghighi

**Project Team Members:** Prajan Bhattarai [48807567], Mustafa Ugur Atas [48763802], Zeki Kose [48718459], B Abhishek Biswal [48681466]

---

## 1. Introduction

Online platforms are full of fake and automated accounts. They push spam and scams, inflate popularity and spread manipulation. Platforms need to find them quickly and at scale, and they cannot inspect every account by hand.

Two families of methods promise to help. The first looks at **connections**: an account does not exist in isolation, it follows others, is followed, mentions, retweets and replies. Graph neural networks (GNNs) learn from these connections by letting each account "listen" to its neighbours. The second looks at **language**: what an account writes in its bio and its posts. Large language models (LLMs) are now very good at reading text, and they can also be asked to reason about an account and its neighbourhood. Combining the two seems natural: the graph shows who an account associates with, and the LLM shows what it says.

There is a catch. Fraudulent accounts often **camouflage** themselves by connecting to ordinary users, so that their neighbourhoods look legitimate. A graph method that assumes "you are like your neighbours" is then pulled the wrong way: it may decide that a bot is human because its neighbours are human. Graphs where connected accounts tend to have *different* labels are called **heterophilous**, and much recent research tries to build graph models that cope with them.

This project set out to answer a practical question: **when an LLM and a graph model are combined to detect fraudulent accounts, does detection actually improve, and if not, why not?** In our project initiation report we planned a hybrid system that joins LLM-derived text representations with GNN representations, and a controlled comparison against traditional machine learning, GNN-only and language-only models under the same data splits and metrics. We expected the hybrid to beat each component on its own.

We carried this out on three datasets, ending with **TwiBot-22**, the largest public graph benchmark for bot detection: one million Twitter users, seven types of connection and 88 million tweets. Before trusting any graph model we measured how much the connections actually reflect the labels, and we designed the evaluation so that known shortcuts could not inflate the scores.

The results differ from what we expected, and in an informative way. **An account's own profile and text carry most of the signal. Information from neighbours adds a little at large scale. GNN message passing adds almost nothing under the camouflage we measured. A fine-tuned LLM does not beat simpler text models once those have enough training data.** We also found that the dataset's labels, which were made mostly by automatic rules and models, overstate one profile cue (an empty bio) about three times compared with independent human judges. Together these findings explain why a published benchmark ranking that puts graph models first does not hold under a common, fair protocol. The project's contribution is therefore not a new model that wins, but a careful answer to *where* text and links help, *where* they do not, and *why* existing benchmarks give a different picture.

**[FIGURE 1: the pipeline. Data → own features (15 profile fields, multilingual-E5 text embeddings) | neighbour features (per-relation averages) | LLM (zero-shot, LoRA fine-tuning, DGP prompting with neighbour summaries) → models (trees, GNNs, LLM) → evaluation on the development test set, the final test set and human-labelled users]**

## 2. Literature Review

**2.1 Graph fraud detection under camouflage.** Graph-based fraud detection treats users, accounts or transactions as nodes and their interactions as edges, and asks whether the network reveals fraud that individual records hide. A recurring problem is that fraud graphs are heterophilous: fraudsters connect to legitimate users, so neighbourhoods mix classes and averaging over neighbours blurs exactly the signal a detector needs. FRAUDRE (Zhang et al., 2021) describes this as three kinds of inconsistency caused by camouflage (in features, in topology and across relations) together with class imbalance, and reports per-relation class homophily above 0.85 for normal nodes and below 0.2 for fraudsters on its benchmarks. Xu et al. (2024) revisit graph fraud detection through heterophily and the graph spectrum and show that standard aggregation is poorly matched to fraud graphs.

A line of work by Haghighi and colleagues targets camouflage directly. MAGNET (Haghighi et al., 2024) labels each neighbourhood as homophily- or heterophily-dominated and aggregates it through separate adaptive channels, combined across relations, so that a camouflaged node is not simply averaged into its benign surroundings. DEFEND (Haghighi et al., 2026) adds discriminative mechanisms at two levels: within each relation it classifies neighbourhoods as homophily- or heterophily-based and aggregates them through separate channels, and across relations it exploits the observation that fraudsters avoid forming consistent cross-relation interactions while benign users form them. These methods share one premise: camouflage leaves traces that the right aggregation can recover.

Two analysis tools frame our measurements. Adjusted homophily (Platonov et al., 2023) corrects edge homophily for class imbalance, so that 0 means "no more same-label links than chance" whatever the class balance. Mao et al. (2023) show that GNNs perform well on nodes whose local structure matches the graph's majority pattern and poorly on the minority pattern: a node-level, not graph-level, view of heterophily.

**2.2 Simple baselines and benchmark artefacts.** Several studies question whether complex graph models are needed. GADBench (Tang et al., 2023) benchmarks supervised graph anomaly detection and finds that tree ensembles with neighbour aggregation outperform most specialised GNNs. For social bots specifically, Hays et al. (2023) show that shallow decision trees on a handful of features come within ten points of state-of-the-art accuracy on most public bot datasets; on TwiBot-20 a single-question tree on the `verified` flag nearly matches the best model. They attribute this to simplistic collection and labelling, and warn that the labels "typically taken as ground truth may have errors" and that human annotators themselves can be biased. Their study covers TwiBot-20 and ten older datasets, not TwiBot-22.

**2.3 LLMs and graphs for fraud.** LLMs enter graph learning in several roles: as encoders that turn text into node features, as predictors prompted with a node and its neighbourhood, and as sources of knowledge that enrich a graph model. In fraud detection, MLED uses an LLM to generate type-level and relation-level knowledge that is injected into existing graph fraud detectors (Huang et al., 2025); GuARD combines text semantics with graph structure for anomaly detection on text-rich graphs (Pang et al., 2025); L2IR uses an LLM to infer the intent behind connections so that misleading links contribute less (Guo et al., 2026); FraudCoT trains an LLM and a GNN together with graph-aware reasoning (Li et al., 2026b). FLAG samples neighbours by semantic similarity and enhances nodes with an LLM to handle neighbourhood camouflage (Yang et al., 2025). Dual-Granularity Prompting (DGP; Li et al., 2026a) keeps a node's own text in full and condenses its neighbours into coarse summaries, so that the prompt stays short and the target's own signal is not drowned out. For bot detection, Feng et al. (2024) report that instruction-tuned LLMs trained on about 1,000 examples can outperform earlier detectors, and that combining modalities helps.

**2.4 Benchmarks.** TwiBot-20 (Feng et al., 2021) introduced a graph-based bot benchmark with crowd-sourced labels; TwiBot-22 (Feng et al., 2022) scaled this to one million users and multiple relation types and labelled users by weak supervision guided by 1,000 expert annotations. Its leaderboard ranks graph-based methods first.

**2.5 Where this project fits.** The camouflage methods above are evaluated on a small number of fraud datasets (YelpChi, Amazon, financial graphs), and the LLM–graph fraud methods mostly on text-rich benchmarks with clean labels. It is not known whether either transfers to a large social graph with weak labels. The simple-baseline studies suggest caution but do not cover TwiBot-22. Our project fills this gap with a controlled comparison on TwiBot-22 under a protocol that removes known shortcuts, and by asking the camouflage literature's own questions (relation-level homophily, local heterogeneity, cross-relation inconsistency, higher-order structure) on it, rather than assuming their answers transfer.

## 3. Research Methodology

**3.1 Decomposition.** We split the overall problem into five sub-problems, each with its own check:

- **S1, data:** find datasets with real text, real links and fraud-like labels.
- **S2, measurement before modelling:** measure heterophily and find shortcut features before any model is scored.
- **S3, models under one protocol:** build own-feature, neighbour-feature, GNN and LLM models that see the same users, labels and metrics.
- **S4, are the gaps real?:** confidence intervals, matched-data controls and edge-off controls.
- **S5, are the labels right?:** check results against human labels.

The sub-problems fit together as a pipeline (Figure 1): S1 and S2 fix the data and the rules, S3 produces the scores, and S4 and S5 decide which differences can be believed.

**3.2 Datasets and why they changed (S1).** The initiation report promised a dataset-agnostic plan: any public fraud dataset that can be represented as a graph. We started with the Elliptic Bitcoin transaction graph. It supported a full pipeline, but it has no natural text, so an LLM could only read sentences generated from the same numbers the other models already had. In the second phase we moved to a Kaggle dataset of fake and authentic Instagram accounts (65,326 accounts, 17 numerical features). It had labels but no real links (we had to build an artificial similarity graph) and no raw text (again, sentences generated from numbers). We therefore moved to the GLBench Instagram graph, which has real bios and real follow links but labels commercial against personal accounts rather than fraud; we kept it as a secondary dataset. TwiBot-20 then gave real bot labels, real text and real links with an official split. Finally, **TwiBot-22** became the large test:

- **Users:** 1,000,000, of which 14.0% are labelled bots.
- **Relations:** 7 (following, followers, mention, retweet, reply, quote, like).
- **Text:** a bio and up to the 20 most recent tweets per user, from 88 million tweets.
- **Human labels:** 986 expert-labelled users, plus a separate 500-user annotation study.

Each change followed from the need for real text and real links. None made our results look better: the final answer is mostly that graph methods add little.

**3.3 A fair protocol (S2).** On TwiBot-20 we learned that decisions made after seeing test scores can bias a comparison. On TwiBot-22 every rule was written down, with a timestamp in a decision log, before the score it governs.

- **Split.** The released TwiBot-22 split is not random: it consists of three contiguous blocks with 7.8% bots in training but 29.4% in test, and all 95,398 verified users sit in training. We used a random stratified 70/20/10 split with 14.0% bots in every part, and report the released split only for comparison.
- **Two test sets.** Because we ran many experiments, we split the TwiBot-22 test users into two halves. The **development test set** (49,951 users) was used while building and comparing models. The **final test set** (50,937 users, incl. the 986 experts) was never looked at until every choice was fixed, and was then scored once. If the two agree, our findings are not luck from testing the same users too often.
- **Shortcut rule.** A feature is removed if at least 200 training users have a value where at least 99% are one class. Only `verified` failed: no verified user is a bot. It is never a model input; a guard in the code stops any script that tries to use it, or the labels, as a feature.
- **Heterophily measured, not assumed.** Adjusted homophily and label informativeness on every relation, before any model.
- **Metrics.** Macro-F1 at a cut-off chosen on validation; AUROC; PR-AUC, also reweighted to 5% and 1% bots to mimic rarer fraud; paired bootstrap 95% confidence intervals (2,000 resamples); three seeds. Accuracy is reported only to compare with the benchmark, because calling every account human already scores 0.860 on our test set.

**3.4 Features and models (S3).**

- *Own features:* 15 profile fields (follower and following counts, account age, empty bio, default profile picture, tweet count and others) and text embeddings from multilingual-E5. Bios are embedded for all one million users; bio plus 20 tweets for a 100,000-user stratified sample. This is because embedding 20 tweets for every user did not fit the time available.
- *Neighbour features ("XGB-Graph"):* for each relation and direction, the mean of the neighbours' own features. These are fed with the user's own features to a gradient-boosted tree, the style of model used in industry.
- *GNNs on the full one-million-user graph:* GCN, GraphSAGE, FAGCN (Bo et al., 2021; it can subtract a neighbour's signal and is designed for heterophily), LINKX (Lim et al., 2021; it reads the adjacency pattern instead of averaging) and R-GCN (one weight matrix per relation). As a control, an MLP with the same layers and inputs but no edges: any GNN gain over it is the gain from the links.
- *LLM (Qwen3-1.7B):* zero-shot prompting; LoRA fine-tuning on the account's own text (`target`, 1,300 training users, following the DGP paper's setting); and DGP, the same fine-tuning with summaries of the account's neighbours added to the prompt. DGP's summaries take about a second each, so DGP was run on a 1,500-user stratified test subset fixed before scoring, and compared with `target` on the same users.
- *Text baselines:* TF-IDF with logistic regression, and MiniLM embeddings with an MLP.

**3.5 Controls (S4).** Training-set size: the LLM sees 1,300 training users, so the trees and text baselines were also trained on exactly those 1,300. Edge controls: the MLP without edges for every GNN, and on TwiBot-20, the same GNNs on randomly rewired edges that keep each user's degree. Slices: users with a bio, users without, and non-verified users.

**3.6 Human-label checks (S5).** TwiBot-22's labels were produced by weak supervision: 8 hand-written rules (including `verified` and spam keywords), 3 feature-based classifiers and 4 neural classifiers (MLP, GCN, GAT, R-GCN), combined by Snorkel and an MLP. The classifiers were trained on 800 of the 1,000 expert labels. We used the 986 released expert labels in two ways. First, by 5-fold rotation each model gets a second, expert-label training stage on four folds and is scored on the fifth, so every expert is scored once by a model that never saw them. Second, as clean evaluation labels. Because the experts trained the label-makers, they are not independent of the weak labels. We therefore also used the dataset's separate annotation study: 500 users, each judged by 3 researchers on a five-point scale. We took the median and kept "likely" or "definitely" judgements, dropping "not sure". These annotators never trained the label-makers.

**3.7 The camouflage questions.** Following the literature, we asked four questions of TwiBot-22:

- **Q1:** class homophily per relation and direction.
- **Q2:** local heterogeneity. Each user's share of training-labelled neighbours with the same label, and where neighbour features help or hurt by that share.
- **Q3:** cross-relation inconsistency. Eight label-free features comparing a user's follow neighbourhood with their interaction neighbourhood.

**3.8 Timeline.** The plan below comes from the next steps listed in the second report (submitted 20 Sep 2026).

| Period | Planned | Actual | Why it changed |
|---|---|---|---|
| 21 Sep – late Sep | Threshold calibration; fusion for GraphSAGE and GAT; repeated seeds | Reviewed the second report's results | Instagram's made-up text profiles could not test the LLM question |
| Early – mid Oct | Find a dataset with real text | Literature on camouflage and heterophily; dataset search; TwiBot-20/22 chosen | Real text and a real follow graph were needed |TwiBot-20 pipeline: LoRA, DGP, robustness checks | — |
| Late Oct | Semantic neighbour filtering | Research questions refined; evaluation protocol designed; first versions of the pipeline | — |TwiBot-22 at scale; shortcut, heterophily and label checks; final test set scored once | TwiBot-20 results needed confirming at scale | 
| 2 – 8 Nov | Write-up | Seed and control runs; final report | — |

Of the planned items, repeated seeds, semantic neighbour filtering (FLAG-lite) and real-language data were done; threshold calibration and GraphSAGE/GAT fusion were replaced by the move to TwiBot.

**3.9 Difficulties and fallbacks.** Several sub-problems did not go as planned.

- **Tweets.** Embedding 20 tweets for all one million users did not fit, so tweets were used on a 100,000-user sample and bios on everyone.
- **Memory.** Full-graph GNNs on 9 million edges exceeded laptop memory. A FAGCN run was killed and rewritten to process edges in checkpointed chunks.
- **DGP speed.** DGP's neighbour summaries ran slower than estimated while other jobs shared the processor, so DGP was scoped to the TwiBot-20 setting.
- **The language model's readout.** A zero-shot AUROC below 0.5 suggested an inverted readout. We checked it directly: it was not inverted, the model simply answers "bot" for nearly every account.
- **The released split.** Its skew made it unsuitable as the main evaluation. Rather than drop it, we used it in a reconciliation experiment (§4.6).

## 4. Research Outcome: Results, Analysis and Discussion

**4.1 Supporting datasets: TwiBot-20 and Instagram.** TwiBot-20 (11,826 users; setting A keeps the natural class balance, setting B subsamples bots to 1:10) showed the pattern that TwiBot-22 later confirmed at scale.

- **A strong text encoder does the work.** Qwen3-1.7B embeddings with an MLP reach macro-F1 0.769 against 0.702 for TF-IDF.
- **No GNN beats its encoder's MLP.** GCN is 0.059 below it (95% CI [−0.082, −0.037]). Models that keep a user's own features separate (GraphSAGE, FAGCN, a skip-connected GCN) lose less, but none wins. Two methods built for heterophily, FLAG-lite (FLAG's skip-GNN on a similarity-pruned graph, without the LLM) and LINKX, score 0.754 and 0.752 against the MLP's 0.766.
- **The links carry almost no label signal.** Adjusted homophily is −0.015 to 0.08. Rewiring the edges at random shows that real edges are better than random ones for GCN (+0.037), but never better than no edges. Linked users write alike whatever their labels (centred text similarity 0.26 on real edges against 0.05 on shuffled ones), so averaging over real neighbours blurs a user less than averaging over random ones.
- **The label signal the links do carry is a crawl artefact.** It is in which of another user's two follower samples an account appears, and it tracks follower count. Adding these in/out counts to a tree helps on all test users (+0.021 [+0.002, +0.040] macro-F1) but not once the verified users, all human, are removed (−0.008 [−0.033, +0.018]).
- **LoRA fine-tuning roughly matches text models trained on the same 1,300 users.** Averaged over three seeds, with each adapter chosen by full-validation macro-F1 like the other models, LoRA scores 0.734 (`target`) and 0.740 (`dgp`) against 0.726 for an MLP on the same 1,300 users, and stays below text models trained on all 8,278 users (`dgp` − MLP on all users −0.026 [−0.046, −0.004]). Reading less text is not the reason: the same encoder reading exactly LoRA's 480 tokens scores 0.762 against 0.766 with all users and 0.722 against 0.726 with 1,300.
- **DGP's neighbour summaries help ranking a little.** Over three seeds, `dgp` − `target` is +0.006 [−0.011, +0.022] macro-F1 on all test users, and +0.023 [+0.006, +0.039] AUROC on the non-verified users.
- **The findings hold without verified users and with profile fields.** On non-verified test users GCN stays 0.052 [0.025, 0.080] below the MLP. With the five allowed profile fields added, as on TwiBot-22, the MLP reaches 0.808 and every GNN stays below it on AUROC (by 0.018 to 0.078).

On the Instagram graph, where text alone is weak (AUROC about 0.63), GCN is level with the MLP (+0.002 [−0.016, +0.020]), consistent with neighbour averaging hurting most when a user's own text is strong.

**4.2 TwiBot-22: main results.** Table 1 gives the results on the final test set (49,951 users once the experts are excluded, scored once). Sample = the ~5,000 sample users with bio and tweets; 1M = all users, bios only. The language models (†) were scored on the development test set only.

| Model | Sample | 1M |
|---|---|---|
| Zero-shot Qwen3-1.7B † | 0.505 / 0.461 / 0.139 | – |
| LoRA `target` (text, 1,300 train) † | 0.626 / 0.710 / 0.312 | – |
| TF-IDF + LR (70,000 train) † | 0.628 / 0.727 / 0.307 | – |
| Links only (neighbour counts) | 0.563 / 0.683 / 0.219 | 0.562 / 0.680 / 0.219 |
| Profile only (15 fields) | 0.705 / 0.841 / 0.514 | 0.709 / 0.847 / 0.515 |
| Profile + text (tree) | 0.723 / 0.858 / 0.546 | 0.724 / 0.862 / 0.537 |
| XGB-Graph, no profile | 0.645 / 0.772 / 0.403 | 0.699 / 0.817 / 0.458 |
| **XGB-Graph + profile** | 0.728 / 0.860 / 0.573 | **0.735 / 0.874 / 0.590** |
| GNNs: GCN / SAGE / LINKX (1 seed), R-GCN / FAGCN (3 seeds) | – | 0.681–0.695 / 0.803–0.830 |
| MLP, no edges (GNN control) | – | 0.683 / 0.800 / 0.425 |
| LoRA `target` / `dgp`, 1,500-user subset, mean of 2 seeds † | 0.618 / 0.699 / 0.332 · 0.618 / 0.714 / 0.343 | – |

*Table 1. TwiBot-22 final-test-set results, macro-F1 / AUROC / PR-AUC; random guessing gives AUROC 0.5 and PR-AUC 0.14. Macro-F1 and AUROC are within 0.016 of the development-test-set values (PR-AUC within 0.042), so the results are not an artefact of repeated testing.*

**[FIGURE 2: Table 1 as a bar chart with 95% confidence intervals]**

The best model, XGB-Graph with profile on all one million users, reaches PR-AUC 0.590 on the final test set, about four times random. When bots are made rarer by reweighting, PR-AUC falls to 0.378 at 5% bots and 0.156 at 1%. That is a reminder of how hard detection becomes at realistic fraud rates.

**4.3 What carries the signal.**

**Finding 1: the profile beats text, and not only through one cue.** The 15 profile fields beat every text model. Profile minus LoRA: +0.073 [+0.054, +0.093] macro-F1 and +0.128 AUROC. The gap is not because the trees saw more data: trained on LoRA's exact 1,300 users, the profile tree still reaches 0.677 / 0.808. Much of the profile's strength comes from one cue: 40.6% of bots have an empty bio, against 7.8% of humans. But the model is not stuck on it. Given 15 fields instead of the empty-bio flag alone, AUROC rises from 0.658 to 0.837. On users who *have* a bio, where the flag is useless (AUROC 0.500), the profile model still scores 0.790 on the sample and 0.802 on all users.

**Finding 2: text adds a little on top of the profile.** On the final test set, profile plus text minus profile is +0.015 [+0.003, +0.028] macro-F1 on the sample and +0.016 [+0.012, +0.020] on all users. A surprise: for a text-only model, adding 20 tweets to the bio *hurts*, −0.052 [−0.074, −0.030] on the same users. Slices explain why. On users with a bio, the two are equal (AUROC 0.682 against 0.681). Averaging tweets into one vector blurs the sharp contrast between empty and non-empty bios. We tested this directly: adding a single empty-bio flag to the bio-plus-tweets embedding closes the gap (0.672 against 0.673).

**Finding 3: the fine-tuned LLM helps only when data is scarce, and its neighbour summaries add nothing.** DGP's prompt with neighbour summaries scores the same as the account's own text on a 1,500-user subset: over two seeds the macro-F1 difference is 0.000 (−0.004 and +0.005) and the AUROC difference +0.015 (−0.001 and +0.030 [−0.000, +0.058]), smaller than the seed-to-seed change of `target` itself (0.037 AUROC). That is what heterophily predicts: the neighbours' text says little about the account. With the same 1,300 training users, LoRA beats TF-IDF by +0.050 macro-F1 (the CI excludes 0). With 70,000 training users, TF-IDF ties LoRA (+0.002 [−0.015, +0.019]). The zero-shot model's raw answer is "bot" for about 99% of accounts; even with a cut-off tuned on validation it ranks no better than chance (AUROC 0.461 on the 5,000 users).

**Finding 3b (exploratory): combining the LLM with the graph model adds nothing.** We stacked the fine-tuned LLM's score with each tree's score, fitting the weights on the 1,000 validation users. The fit gives the LLM a weight of about −0.02 to −0.05 against about 1.0 for the tree, and no gain on test users: XGB-Graph plus the LLM minus XGB-Graph is −0.005 [−0.013, +0.004] macro-F1, with no gain in any slice (users without a bio, without tweets, or where the tree is uncertain), with DGP instead of `target`, or on the experts. The reason is that the LLM is weaker than a tree on multilingual-E5 text features even on its own ground (AUROC 0.618 against 0.687 on users with a bio), and its score overlaps the text tree's (Spearman 0.49). With 1,300 training users and 480 tokens, it learns a weaker version of what the text features already give.

**4.4 What the links add.**

**Finding 4: neighbour features help a little at full scale; message passing does not.** On the final test set, adding neighbour averages to the own-feature tree gives +0.012 [+0.009, +0.016] macro-F1 and +0.054 [+0.048, +0.060] PR-AUC on all users. Without the profile the gain is larger (+0.043 AUROC), because neighbours partly stand in for missing profile information. GNNs are a different story: all five stay within about 0.01 macro-F1 of the MLP without edges and well below the trees on the same features. The best GNN, FAGCN, beats the MLP by +0.011 [+0.008, +0.014] macro-F1 and +0.029 AUROC, yet still scores 0.830 AUROC against 0.874 for XGB-Graph.

**Finding 5: the camouflage questions, answered on TwiBot-22.**

- **(Q1) Bots hide among humans on every relation.** Bots' neighbours are 4–17% bots against a 14% base rate; humans' neighbours are about 93% humans. Adjusted homophily is 0.010–0.024. This is the shape FRAUDRE reports for fraud graphs.
- **(Q2) Neighbours help some users and hurt others.** Neighbour features help users whose neighbours mostly share their label: +0.057 [+0.045, +0.071] AUROC (27,002 users, 1% bots). They hurt users whose neighbours mostly do not: −0.016 [−0.022, −0.009] (4,197 users, 77% bots). R-GCN loses 0.052 in the homophilous group, while FAGCN gains slightly in every group. This is the node-level pattern Mao et al. (2023) predict, and the question MAGNET addresses. One caution: these groups are defined using each user's own label, so they diagnose where models fail but cannot be used to make predictions. The same caution applies to the heterophily bands in our second-phase report.
- **(Q3) Relations do not disagree usefully.** DEFEND's premise is that fraudsters avoid consistent cross-relation interactions. Our eight features comparing each user's follow neighbourhood with their interaction neighbourhood test that premise directly, and they do not help: −0.002 [−0.004, −0.000] macro-F1 on all users (a tiny decrease), and no change on the sample.

**[FIGURE 3: (a) per-relation share of bot neighbours for bots and humans against the 14% base rate (Q1); (b) gain from neighbour features by local-homophily group (Q2)]**

**Finding 6 (exploratory): little room for adaptive neighbour trust.** Because Q2 shows neighbours hurting a minority of users, we asked how much a per-user switch could gain if it knew exactly whom to trust. This test was designed after seeing Q2, so it is exploratory. An oracle switch uses the own-feature tree for users whose true local homophily is below 0.5 (8.8% of users) and XGB-Graph for everyone else. It gains only +0.003 [+0.002, +0.004] AUROC and +0.006 macro-F1 over XGB-Graph. A real switch would have to guess the groups from predicted labels and could only do worse, so we did not build it.

**4.5 Are the labels telling the truth?**

**Finding 7: the model ranking holds on expert labels.** On the 986 experts (105 bots), each scored by a model that never saw them:

| Model | AUROC, weak labels only → after expert stage | Change [95% CI] |
|---|---|---|
| LoRA `target` (text) | 0.822 → 0.724 | −0.099 [−0.164, −0.038] |
| Profile only (tree) | 0.861 → 0.867 | +0.006 [−0.017, +0.030] |
| Profile + text (tree) | 0.879 → 0.899 | +0.020 [+0.000, +0.040] |
| XGB-Graph + profile (sample) | 0.883 → **0.910** | +0.027 [+0.009, +0.044] |
| MLP, no edges | 0.819 → 0.845 | +0.025 [+0.006, +0.046] |
| R-GCN | 0.799 → 0.866 | +0.067 [+0.034, +0.104] |
| FAGCN | 0.846 → 0.841 | −0.005 [−0.017, +0.005] |

*Table 2. Weak-label training vs an added expert-label stage, scored on the 986 expert users.*

Neighbour-feature trees remain best. R-GCN gains most from clean labels but stays below the trees; FAGCN does not change. One epoch of LoRA on about 789 experts lowers its ranking quality, though its macro-F1 rises through the new cut-off. The dataset's own labels agree with the experts 90.5% of the time, as its paper reports. But that is barely above the 89.4% achieved by calling everyone human: only 56 of the 105 expert bots are labelled bots.

**Finding 7b (D7): the neighbour-tree gain appears only under the weak labels.** If graph models partly re-learn the label-making GNNs, their gain should shrink under expert labels. For XGB-Graph minus the own-feature tree on all users, the AUROC gain is +0.012 [+0.011, +0.014] under the weak labels (final test set) but −0.005 [−0.015, +0.004] under the expert labels; the difference, +0.018 [+0.008, +0.027], excludes zero. FAGCN's gain over the MLP holds under both (+0.029 and +0.026 [+0.003, +0.053]). So the small neighbour-feature gain at full scale is at least partly an artefact of how the labels were made.

**Finding 8: the empty-bio cue is mostly a label-maker habit.** If empty bios are a genuine bot habit, independent humans should see the same link as the dataset's labels. We measured the *lift*: the bot rate among empty-bio users divided by the bot rate among users with a bio.

| Users | Under the dataset's labels | Under human labels | Difference |
|---|---|---|---|
| 986 experts | 8.7× [6.4, 12.3] | 7.2× [5.3, 10.3] (experts) | +1.5 [−1.5, +4.6] |
| 458 annotation-study users | 5.0× [3.3, 7.7] | **1.6× [0.7, 3.1]** (annotators) | **+3.4 [+1.2, +6.1]** |

*Table 3. Empty-bio lift. Final-test-set non-expert users were removed from the annotation study (21 users) and "not sure" judgements dropped (21).*

The experts agree with the weak labels, but they trained the label-makers, so their agreement is built in. The independent annotators see a much weaker link, and the difference excludes zero. The annotation study is small (44 bots), and its 1.6× could still hide a modest real effect. But the dataset's labels clearly overstate the cue. This is consistent with Hays et al.'s warning about benchmark labels and, as far as we found, extends it to TwiBot-22.

**[FIGURE 4: empty-bio lift under weak, expert and independent labels, with 95% CIs]**

**Finding 8b (exploratory): what every model gets wrong.** On the 5,000 test users we looked at the accounts on which XGB-Graph, the profile + text tree, the LLM and FAGCN all agree and are wrong. Of 700 bots, 230 (33%) are missed by all four; of 4,300 humans, 95 are flagged by all four. The groups split by activity, not by bot-ness. Bots caught by every model have an empty bio (95%), often a default image (40%), very few followers (5th percentile), and half have no tweets; the humans flagged by every model look the same (87% empty bio, 7th percentile). The bots every model misses look like ordinary active users (4% empty bio, all with tweets, 44th follower percentile). In other words, the detectors find dormant-looking accounts, whichever label they carry, and miss active bots. On the experts, about half of these shared "errors" are label errors: 17 of the 32 dataset "bots" missed by every model are human according to the experts, and 15 of the 33 dataset "humans" flagged by every model are bots.

**4.6 Why the benchmark ranks graph models first.** The TwiBot-22 leaderboard ranks graph models such as BotRGCN at the top. We read the official code. BotRGCN, GCN and GAT take `verified` as an input, use the skewed released split, train for a fixed 200 epochs with no validation-based selection, report AUC computed from hard 0/1 predictions, and are ranked by accuracy. On bot-F1, a random forest (0.587) already beats BotRGCN (0.575) on the leaderboard itself.

We reran our own models under their protocol, with and without `verified` (Table 4).

| Model | Official split: macro-F1 / AUROC (`verified` in / out) | Our split (in / out) |
|---|---|---|
| Own-feature tree | 0.700 / 0.765 · 0.695 / 0.764 | 0.721 / 0.863 · 0.720 / 0.860 |
| XGB-Graph + profile | 0.689 / 0.758 · 0.691 / 0.758 | 0.730 / 0.874 · 0.731 / 0.873 |
| MLP, no edges | 0.683 / 0.745 · 0.674 / 0.734 | 0.686 / 0.824 · 0.679 / 0.799 |
| FAGCN | 0.667 / 0.731 · 0.678 / 0.745 | 0.674 / 0.799 · 0.689 / 0.828 |
| R-GCN | 0.666 / 0.724 · 0.653 / 0.726 | 0.678 / 0.797 · 0.689 / 0.800 |

*Table 4. Reconciliation, seed 0. On the official split 29.5% of test users are bots, so calling everyone human scores accuracy 0.705.*

Three things follow:

1. **The split matters far more than `verified`.** Every model loses 0.07–0.12 AUROC on the released split (0.12 for XGB-Graph). Our trees move by at most 0.01 with `verified`.
2. **Under their protocol, still no graph model beats the tree without neighbours.**
3. **The leaderboard's labels were partly produced by GCN, GAT and R-GCN models**, the same families that top it.

We therefore read the leaderboard's graph advantage as a product of the protocol (accuracy on a skewed split, a shortcut feature, labels partly made by graph models), not as evidence that message passing helps under camouflage.

**[FIGURE 5: AUROC on our split vs the official split, with and without `verified`]**

**4.7 Answers to the original research questions.**

- **RQ1, does an LLM–GNN hybrid beat GNN-only?** Adding text to graph models helps, but the gain comes from the text and profile, not from message passing. The best model is a tree on the user's own features plus neighbour averages, and adding the fine-tuned LLM's score to it adds nothing (Finding 3b).
- **RQ2, do GNNs capture fraud better than traditional machine learning?** No. Trees match or beat GNNs on all three datasets, as in our Elliptic phase (random forest F1 0.816 against GraphSAGE 0.333).
- **RQ3, what does semantic information add?** On TwiBot-22, about +0.01 macro-F1 over the profile. On TwiBot-20, without profile fields, a strong encoder is the main source of accuracy.
- **RQ4, which fusion strategy, at what cost?** Joining frozen text embeddings with other features is the cheapest and best. LoRA fine-tuning matches it only at equal training size. DGP's neighbour summaries add nothing on TwiBot-22 (two seeds: 0.000 macro-F1, +0.015 AUROC, within seed noise) and only a little ranking quality on TwiBot-20; semantic neighbour filtering (FLAG-lite) does not beat the MLP.
- **RQ5, robustness to imbalance.** Validation cut-offs, class weights and reweighted PR-AUC keep the comparison honest. All methods degrade sharply as bots become rarer.

**4.8 How the problem was revised (P → P′).** The original problem, P, was: *does an LLM–GNN hybrid improve fraud detection?* Our evidence reshaped it into P′: *under measured camouflage and weak labels, which signals actually carry information, and why do benchmark rankings disagree?*

The deep LLM–GNN fusion we planned did not outperform simple fusion, and the adaptive-trust idea had little headroom. What we learned from these attempts, that the labels and the protocol shape the answer as much as the model does, led to the human-label and reconciliation experiments that became central results.

**4.9 Limitations.**

- **Labels:** TwiBot-22's labels are mostly automatic, and our clean label sets are small (105 and 44 bots).
- **Final test set:** it covers the tree and graph models; the language models were scored on the development test set only.
- **The LLM:** it read up to 480 tokens, trained on 1,300 users and has 1.7 billion parameters. Larger models or more data might do better.
- **Tweets:** used on a 100,000-user sample only.
- **Tuning:** GNN hyperparameters were not tuned, and some cells have a single seed.
- **Generalisation:** the findings describe Twitter bots in 2022 and may not carry over to other fraud graphs.

## 5. Individual Contributions

All four members took part in planning, weekly discussion of results, and review of the final report. The main responsibilities were divided as follows.

| Member | Main responsibilities in the final phase |
|---|---|
| **Prajan Bhattarai** (lead; major share) | Designed and built the TwiBot-20 and TwiBot-22 pipeline end to end: data preparation, text encoders, LoRA fine-tuning of Qwen3-1.7B and the DGP replication. Ran the GNN, tree and XGB-Graph experiments and the camouflage questions. Carried out the leaderboard reconciliation and the expert-label and annotation-study analysis. |
| **Mustafa Ugur Atas** | Shortcut and heterophily checks: the `verified` and crawl-artefact shortcut tests, and the heterophily measurements (adjusted and local homophily). Literature review on bot detection, heterophily and camouflage (Section 2), limited to work from 2021 onwards. |
| **Zeki Kose** | Evaluation and robustness: the evaluation design (splits, validation cut-off, paired bootstrap confidence intervals), the multi-seed robustness checks (three seeds for the LoRA and GNN comparisons), the TwiBot-20 results without verified users, and the final-test-set scoring. |
| **B Abhishek Biswal** | LLM–graph fusion and error analysis: built the stacking model that combines the language-model and graph-tree scores, and analysed which bots and humans every model gets wrong. Report: wrote and consolidated the final report from the team's results, prepared the figures and diagrams and the presentation, and carried over the lessons from the Instagram phase of the second report. |

*How responsibilities changed after the second report:* the second report used an Instagram dataset with a constructed similarity graph. When the project moved to TwiBot-20 and TwiBot-22, which have real follow graphs and real text, the language-model owner (Prajan) took over the main experimental pipeline, because the new datasets centred on text and LLM fine-tuning. The other members moved from building separate model components to owning one part of the evaluation each: the shortcut and heterophily checks (Mustafa), evaluation and robustness (Zeki) and the fusion, error analysis and report (Abhishek).

## 6. Future Outlook

Several parts of the problem remain open beyond this project.

1. **Better labels are the most important next step.** The dataset's labels overstate at least one profile cue, and its expert labels are not independent of the label-makers. Clean labels at scale are expensive, and human judges are imperfect too. The most promising approach is targeted annotation: use active learning to send humans the accounts on which models disagree most. Pair it with training methods that tolerate label noise, and report results on independent labels as standard.
2. **Camouflage-aware aggregation has limited headroom on this data.** Adaptive methods such as MAGNET are the natural next models to test under our protocol, but our oracle experiment caps the gain from per-user neighbour trust at about +0.003 AUROC on TwiBot-22. Effort is better spent first on labels and on the accounts' own features. Adaptive aggregation may matter more on graphs with stronger camouflage structure.
3. **LLMs with more context and more data.** Our LLM read 480 tokens and trained on 1,300 users. Longer inputs, more training users and LLM-derived features fed into tree models, rather than the LLM deciding alone, are promising and cheap to test. DGP's neighbour summaries did not help on TwiBot-22, so richer LLM use of the account's own text is the more promising direction.
4. **Shift over time and newer bots.** Every model loses 0.07–0.12 AUROC on a non-random split, and TwiBot-22 predates bots that write with LLMs. Time-based splits and newer data are needed before any detector is deployed.

## 7. Bibliography

*(All entries 2021 or later; older methods are described in the text, not cited. Each entry checked on arXiv, Crossref or the publisher on 6 Nov 2026. Style to be unified.)*

- Bo, D., Wang, X., Shi, C., & Shen, H. (2021). Beyond low-frequency information in graph convolutional networks (FAGCN). *AAAI 2021*. arXiv:2101.00797.
- Feng, S., Wan, H., Wang, N., Li, J., & Luo, M. (2021). TwiBot-20: A comprehensive Twitter bot detection benchmark. *CIKM 2021*. arXiv:2106.13088.
- Feng, S., Tan, Z., Wan, H., Wang, N., Chen, Z., Zhang, B., et al. (2022). TwiBot-22: Towards graph-based Twitter bot detection. *NeurIPS 2022 Datasets and Benchmarks*. arXiv:2206.04564.
- Feng, S., Wan, H., Wang, N., et al. (2024). What does the bot say? Opportunities and risks of large language models in social media bot detection. *ACL 2024*. arXiv:2402.00371.
- Guo, J., Weng, Z., Liu, Y., Qiao, Y., & Li, M. (2026). L2IR: Revealing latent intent in graph fraud detection. arXiv:2605.26040.
- Haghighi, V., Shabani, N., Soltani, B., Yao, L., Sheng, Q. Z., Yang, J., & Beheshti, A. (2024). Robust graph learning against camouflaged malicious actors. *WISE 2024*, LNCS, pp. 146–161. doi:10.1007/978-981-96-0567-5_12.
- Haghighi, V., Soltani, B., Yao, L., Wu, J., Yang, J., & Sheng, Q. Z. (2026). Beyond smoothing: A discriminative enhanced aggregation graph neural network for camouflaged fraud detection (DEFEND). *ACM Transactions on Knowledge Discovery from Data*, 20, 1–33. doi:10.1145/3801967.
- Hays, C., Schutzman, Z., Raghavan, M., Walsh, E., & Cresci, S. (2023). Simplistic collection and labeling practices limit the utility of benchmark datasets for Twitter bot detection. *WWW 2023*. arXiv:2301.07015.
- Huang, T., Wang, Y., Li, Q., He, C., & Gao, J. (2025). Can LLMs find fraudsters? Multi-level LLM enhanced graph fraud detection. arXiv:2507.11997.
- Li, Y., Hu, J., Hooi, B., He, B., & Chen, C. (2026a). DGP: A dual-granularity prompting framework for fraud detection with graph-enhanced LLMs. *AAAI 2026*, 40(18), 15171–15179. doi:10.1609/aaai.v40i18.38541. arXiv:2507.21653.
- Li, Y., Hu, J., Hooi, B., He, B., & Chen, C. (2026b). Autonomous chain-of-thought distillation for graph-based fraud detection. arXiv:2601.22949.
- Lim, D., Hohne, F., Li, X., Huang, S. L., Gupta, V., Bhalerao, O., & Lim, S.-N. (2021). Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods (LINKX). *NeurIPS 2021*. arXiv:2110.14446.
- Mao, H., Chen, Z., Jin, W., Han, H., Ma, Y., Zhao, T., Shah, N., & Tang, J. (2023). Demystifying structural disparity in graph neural networks: Can one size fit all? *NeurIPS 2023*. arXiv:2306.01323.
- Pang, Y., Chen, B., Zhang, F., Rao, Y., Kharlamov, E., & Tang, J. (2025). GuARD: Effective anomaly detection through a text-rich and graph-informed language model. *KDD 2025*, pp. 2222–2233. arXiv:2412.03930.
- Platonov, O., Kuznedelev, D., Babenko, A., & Prokhorenkova, L. (2023). Characterizing graph datasets for node classification: Homophily–heterophily dichotomy and beyond. *NeurIPS 2023*. arXiv:2209.06177.
- Tang, J., Hua, F., Gao, Z., Zhao, P., & Li, J. (2023). GADBench: Revisiting and benchmarking supervised graph anomaly detection. *NeurIPS 2023 Datasets and Benchmarks*. arXiv:2306.12251.
- Xu, F., Wang, N., Wu, H., Wen, X., Zhao, X., & Wan, H. (2024). Revisiting graph-based fraud detection in sight of heterophily and spectrum. *AAAI 2024*. arXiv:2312.06441.
- Yang, C., Liu, H., Wang, D., Zhang, Z., Yang, C., & Shi, C. (2025). FLAG: Fraud detection with LLM-enhanced graph neural network. *KDD 2025*, pp. 5150–5160. doi:10.1145/3711896.3737220.
- Zhang, G., Wu, J., Yang, J., Beheshti, A., Xue, S., Zhou, C., & Sheng, Q. Z. (2021). FRAUDRE: Fraud detection dual-resistant to graph inconsistency and imbalance. *IEEE ICDM 2021*. doi:10.1109/ICDM51629.2021.00098.
