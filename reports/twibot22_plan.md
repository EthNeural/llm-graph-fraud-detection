# TwiBot-22 plan

Status as of Wed 4 Nov 2026, 14:20. This is a working plan: decisions marked **open** are not fixed yet. Every decision below is fixed **before** any TwiBot-22 test score is seen (lesson from TwiBot-20, see `twibot20_lessons_for_twibot22.md`).

## Why TwiBot-22

- It is a second, much larger Twitter bot graph: 1,000,000 labelled users, 14% bots, 14 relation types.
- On TwiBot-20, text did the work and no GNN beat an MLP. The TwiBot-22 paper reports the opposite pattern: text-only RoBERTa reaches bot F1 20.5, BotRGCN 57.5.
- So TwiBot-22 tests whether the graph helps when text is weak.

## Approach: cut the computation, not the graph

Industry and the benchmarks keep the whole graph and subsample labels, not structure (`graph_slicing_research.md`).

1. **Cheap models on the full 1M graph (CPU).**
   - Features: bio embeddings (all 1M, GPU, running) and the allowed profile fields (no `verified`).
   - Per-relation neighbour statistics: count, mean, max; 1 and 2 hops; in and out.
   - Models: MLP, XGB-Graph (GADBench recipe: own features + mean 1-hop + mean 2-hop), SIGN-style precomputed propagation, full-batch GCN / SAGE / LINKX.
2. **LLM steps on a stratified random sample at the natural 14% bot share (GPU).**
   - Each sampled user's neighbours come from the full graph.
   - LLM embeddings and DGP with LoRA.
3. **One shared test set** for every model: macro-F1, AUROC, AUPRC at the natural bot share, with bootstrap CIs.
4. **DEC-lite** (Facebook, USENIX Security 2021) as an addition. Neighbour-aggregate features into gradient-boosted trees, plus noisy-then-clean training on the 1M weak labels and the 986 expert labels. **Open:** confirm before building.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Split | Our stratified split, seed 42. The official split is reported as an extra | Official `split.csv` is not random: train 7.8% bots, val 28%, test 29.4%. All 95k verified users sit in train. Median followers 1,593 (train) vs 244 (test) |
| Users for the graph models | All 1M | Slicing removes edges: the paper's 10k sub-communities keep only 290 to 7,900 follow edges each |
| Users for the LLM steps | Stratified random sample, 14% bots. **Open:** size | LLM cost |
| Relations | **following, followers, mention, retweet, reply** (fixed 4 Nov 14:25, before any model score). Quote, like and co-list are measured and reported, not used as model edges | The 5 with the widest coverage: share of humans with an incoming edge is 56%, 43%, 41%, 25%, 20%; quote 10%, like 18%. All have near-zero label signal (table below), so coverage, not homophily, decides. Co-list (61M pairs) is too dense for the laptop |
| Metadata | Allowed: profile fields except `verified` | `verified` is a labelling function and a shortcut: 0 of 95,398 verified users are bots |
| Text | Bio, plus the 20 most recent tweets if the tweet files arrive | Same as TwiBot-20 |
| Metrics | Macro-F1, AUROC, AUPRC (+ accuracy and bot F1 on the official split, for comparison with published work) | Always-human scores 70.6% accuracy on the official test set |
| Second test set | The 986 expert-labelled users, reported both overall and for non-verified users only. **Open:** headline or check | Weak labels: dataset "bot" labels agree with experts for only 56 of 101 users |

## Data checks done

- **Profile shortcut rule:** only `verified` fails. Empty bio covers 40.6% of bots vs 7.8% of humans, and is allowed by the rule but reported.
- **Bio text is weak:** TF-IDF on users with a bio gives macro-F1 0.561 (AUROC 0.609).
- **Follow graph on all 1M users:** adjusted homophily 0.019, label informativeness 0.0007. In-degree of `following` ranks humans above bots (AUROC 0.65), a crawl artefact as on TwiBot-20.
- **Expert labels (986):** not a random sample. 39.6% are verified vs 9.5% overall. 14 of the paper's 1,000 are missing.

## Relation heterophily (all 1M users; `my_build/twibot22_relations.py`)

| Relation | Edges (undirected) | Edge homophily | Chance | Adjusted homophily | Label informativeness | Has incoming edge: bots / humans | In-degree AUROC for "bot" |
|---|---|---|---|---|---|---|---|
| following | 2,614,941 | 0.879 | 0.876 | 0.022 | 0.0009 | 0.318 / 0.562 | 0.353 |
| followers | 1,111,113 | 0.830 | 0.826 | 0.026 | 0.0010 | 0.434 / 0.431 | 0.492 |
| mention | 1,455,155 | 0.910 | 0.908 | 0.017 | 0.0007 | 0.205 / 0.414 | 0.384 |
| retweet | 653,086 | 0.912 | 0.910 | 0.023 | 0.0012 | 0.087 / 0.251 | 0.416 |
| reply | 362,470 | 0.911 | 0.909 | 0.018 | 0.0008 | 0.081 / 0.204 | 0.437 |
| quote | 170,942 | 0.928 | 0.926 | 0.024 | 0.0015 | 0.025 / 0.100 | 0.463 |
| like | 338,922 | 0.870 | 0.868 | 0.011 | 0.0002 | 0.065 / 0.179 | 0.442 |
| co-list | 61,415,493 | 0.923 | 0.920 | 0.038 | 0.0034 | – | – |

Read: every relation is near random for labels (adjusted homophily 0.011–0.038). But **whether a user receives an edge at all** depends on the label: humans are 1.8–4x more likely to have one. That is the same crawl-artefact shape as TwiBot-20.

**The graph is star-shaped.** Only 10,039 users (1.0%) have any outgoing edge in these seven relations. They are 14.5% bots, and none is in the official val split (5,711 train, 4,328 test). For the other 99% of users, every neighbour is one of those ~10k crawled users. So out-degree AUROC is about 0.50 simply because 99% of users have out-degree 0.

**Tweets:** 93.4% of users have at least one collected tweet (82.4% of bots, 95.2% of humans). The median is 53 among users with tweets, 51 over all users.

**Our split, fixed 4 Nov 14:22** (`split_ours.npy`): 699,309 / 199,803 / 100,888 users, 14.0% bots in each part; all 986 expert users are in test.

**Feature table** (`twibot22_features.py`): 814 features per user. That is 47 base features (15 profile fields + 32 bio PCA dims), plus per relation and direction the log count and mean of the neighbours' base features, plus 1- and 2-hop union means.

## Next

1. `edge.csv`, then the 8-relation heterophily table (`twibot22_relations.py`); choose 5 relations.
2. Fix the remaining script issues from the lessons report: per-design output folders; `DATASET=twibot22` in `common.py`; TwiBot-22 field names and dates in `metadata_check.py` and `step6_dgp.py`.
3. Neighbour-feature build, then XGB-Graph and MLP on 1M users. Run trees with `OMP_NUM_THREADS=1` or without torch (TwiBot-20's tree baseline crashed silently).
4. Draw the LLM sample; GPU runs.
5. Resume the paused TwiBot-20 GPU queue (LoRA seed 2, `dgp_num`) after the TwiBot-22 experiments.

## Licence

TwiBot-22 is CC BY-NC-ND 4.0: no data files are committed to this repository.
