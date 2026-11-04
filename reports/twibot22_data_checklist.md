# TwiBot-22 data-processing checklist

Written Wed 4 Nov 2026, 19:50, before any TwiBot-22 model score. Built from `twibot20_lessons_for_twibot22.md` (the TwiBot-20 mistakes) and `twibot22_plan.md`. Nothing below the line "Freeze" may change after the first test score.

Status: ✅ done and checked · 🟡 partly done · ⬜ to do · ➖ not needed (with the reason)

## A. Raw data

| # | Item | Status | Note |
|---|---|---|---|
| A1 | `user.json`, `label.csv`, `split.csv`, `edge.csv` complete | ✅ | `edge.csv` 6,642,968,601 bytes; `user.json` matched byte for byte against a second download |
| A2 | 9 tweet files, each complete (ends with `]`) | 🟡 | 8 of 9 in (0, 2–8, 10.0–12.5 GB each). `tweet_1` downloading (ETA ~20:10) |
| A3 | Tweet count per file and in total, against the paper's ~88M | ⬜ | Count while extracting (C1) |
| A4 | No dataset file in git (CC BY-NC-ND 4.0) | ✅ | `data/` is git-ignored |

## B. Labels and split

| # | Item | Status | Note |
|---|---|---|---|
| B1 | Our split: stratified 70/20/10, seed 42, 14.0% bots in each part | ✅ | `split_ours.npy` |
| B2 | All 986 expert-labelled users forced into test | ✅ | |
| B3 | Official split kept for comparison only (non-random: all 95k verified users in train) | ✅ | |
| B4 | **Untouched confirmation test set**, scored once at the end (lesson 22). Carve it out **before** drawing the sample (B5) | ⬜ | Proposal: split our test part in two, half for working comparisons and half kept sealed |
| B5 | **Sample for text and LLM steps**: 100k stratified random (natural 14% bots) + all 986 experts; seed 42; drawn from our split, so each sampled user keeps its train/val/test role. The 10,039 hub users get tweet embeddings **only as neighbour-feature providers**; they are not added as training or test users unless the random draw picks them (otherwise hubs would be ~10% of the sample instead of ~1%) | ⬜ | Must be fixed before any score |
| B6 | Weak labels documented: dataset "bot" matches experts for 56 of 101 | ✅ | |
| B7 | **Only train-split labels are ever used for training**, including DEC's stage 1 ("1M weak labels" in the plan means the train part only) | ⬜ | Write into every training script |
| B8 | **Expert users have one role: test.** DEC stage 2 (fine-tune on experts with 5-fold rotation) is the one exception: there each expert is scored only in the fold where it was held out, and that run is reported separately | ⬜ | |

## C. Tweets

| # | Item | Status | Note |
|---|---|---|---|
| C1 | Extraction counts: tweets read, kept, regex misses, JSON decode errors, per file (lesson T2) | ⬜ | Fix `twibot22_tweets.py` to log these |
| C2 | **Most recent by date, not first met in file order** (T3). The current script stops at 200 per user in file order | ⬜ | Keep a per-user top-N by `created_at` across all 9 files; parse dates with `pd.to_datetime` (X6) |
| C3 | Wanted users = sample (B5), not the sub-communities (T4) | ⬜ | |
| C4 | Users with 0 tweets: count by class; text = bio only, with a flag | ⬜ | Plan's earlier check (from `post` edges): 82.4% of bots and 95.2% of humans have ≥1 tweet. Re-check on the files |
| C5 | **Text rules**: N = 20 most recent; retweets kept or dropped?; character cap; URLs and @-mentions normalised? | ⬜ | Decide and write down before embedding. TwiBot-20 used bio + 20 tweets |
| C6 | Language mix by class (multilingual-E5 handles non-English; MiniLM does not) | ⬜ | |
| C7 | Duplicate and near-duplicate tweets and bios **across train and test** (copy-paste bot farms leak between splits) | ⬜ | 3,780 duplicate-bio groups overall, 281 of them pure-bot |
| C8 | Tweet dates within the collection window; no tweets after 15 Mar 2022 | ⬜ | |

## D. Profile fields

| # | Item | Status | Note |
|---|---|---|---|
| D1 | Shortcut rule run; only `verified` fails, so it is excluded | ✅ | |
| D2 | Shortcut rule re-run on **our** split's train (it was run on the official split) | ✅ | `twibot22_shortcuts.py`: only `verified` fails. 17 profile fields + 28 graph cues (has-edge, degree per relation) |
| D3 | Leaf-size rule uses a count (≥200 users), not 10% (P2) | ✅ | Plus a stricter every-cut-off tail scan: 5 tiny pure-human tails (≤1.2% of users: very high followers, listed, tweets, following; oldest accounts). The followers and listed tails are 80–90% verified and not pure without them; the tweet-count, following and age tails stay ~99% human but cover ≤0.3%. Reported, not excluded |
| D4 | **Labelling-function leakage**: spam keywords in bios and tweets, by class (the labels were made partly from them) | ⬜ | A model that finds these is partly re-learning the labelling machine |
| D5 | Account age parsed from ISO dates, not zero for everyone (X3) | ✅ | `twibot22_profile_check.py` uses `pd.to_datetime`. The TwiBot-20 parsers in `metadata_check.py` and `step6_dgp.py` are still wrong for TwiBot-22 |
| D7 | **Labels were partly made by graph models** (GCN, GAT, R-GCN were labelling functions). Graph models may re-learn the labeller. Check: compare each graph model's gain on weak-label test users vs on expert-labelled users | ⬜ | Planned as DEC question 3; applies to every graph model |
| D6 | "Empty bio" flag: strong label cue (40.6% of bots, 7.8% of humans). Report a flag-only baseline and a with-bio slice | ⬜ | |

## E. Graph

| # | Item | Status | Note |
|---|---|---|---|
| E1 | Relations fixed: following, followers, mention, retweet, reply | ✅ | Fixed 4 Nov 14:25 |
| E2 | Homophily, adjusted homophily, label informativeness for all 8 relations | ✅ | All near zero (0.011–0.038) |
| E3 | Star-shape check: only 10,039 users have outgoing edges | ✅ | |
| E4 | Degree AUROC on **test users only** (R3); out-degree caps and reciprocity (R5) | ⬜ | Quick |
| E5 | Train–test neighbour leakage: share of test users whose neighbours are same-label train users, vs chance | ⬜ | Quick |
| E6 | co-list counts use int8, so they can overflow above 127 shared lists (R2) | ➖ | co-list is reported only and never used as model edges. Note it in the report |
| E7 | Relation names in `edge.csv` all present and non-empty (R4) | ✅ | Every relation in the heterophily table has edges |

## F. Features and embeddings

| # | Item | Status | Note |
|---|---|---|---|
| F1 | Standardisation and PCA fitted on train only | ✅ | `twibot22_features.py` |
| F2 | Rows kept in `label.csv` order everywhere; any join by id, never by position (X2) | 🟡 | True for the new scripts; `prep_twibot22.py` sorts by id, so it must reindex |
| F3 | Output folders per design (sample, text recipe), so no stale cache is reused (X1, lesson 20) | ⬜ | Do before the first tweet embedding |
| F4 | Bio-only and bio + tweets embeddings saved as separate feature sets (E3) | ⬜ | |
| F5 | Neighbour features use neighbours' features only, never their labels | ✅ | |
| F6 | **Headline embedder fixed in advance**. Proposed (needs the builder's OK): multilingual-E5 on bio + tweets. MiniLM is reported as a check, never chosen by test score | ⬜ | Write into the plan |
| F7 | Users with no neighbours in a direction get a mean of 0 (= the train mean after standardisation); the separate log-count column (0) tells models it is missing | ✅ | `twibot22_features.py` `mean_agg`. GNNs need the same flag |

## G. Run hygiene

| # | Item | Status | Note |
|---|---|---|---|
| G1 | `OMP_NUM_THREADS=1` and no torch import with sklearn boosting (the TwiBot-20 silent segfault) | ⬜ | In the CPU model scripts |
| G2 | Every launcher logs each job's exit code | ⬜ | |
| G3 | Selection rule, seeds (3), metrics (macro-F1, AUROC, AUPRC, bootstrap CIs) written into the scripts | ⬜ | Plan step 1 |
| G4 | Every result also scored on non-verified test users and with-bio users | ⬜ | |
| G5 | **No artificial setting B on TwiBot-22.** The natural 14% bot share is already within graph-fraud benchmark rates (1.3–21.8% in GADBench), and TwiBot-20's 1:10 subsample removed 8,108 of 8,162 bot–bot edges. For rarer rates, report **PR-AUC with test users reweighted to 5% and 1% bots** (bots weighted so they make up 5% or 1% of the test weight; graph, training and cut-off unchanged). Wording in the report: "fraud-like rare class", not "fraud" | ⬜ | Evaluation only; add to the shared metrics code with G3 |

## Freeze

When every row above is ✅ or ➖, commit this file. Only then run the first model.

## Order for tonight and tomorrow morning

1. `tweet_1` lands → A2.
2. Quick checks, ~30 min CPU: D2, D3, E4, E5. Then B4 (sealed set) before B5 (sample).
3. Fix the tweet script (C1–C3, C8), draw the sample (B5), extract tweets (~30–40 min).
4. Tweet checks: C4, C6, C7, D4. Decide the text rules (C5).
5. F3, then a 5-minute embedding timing test, then the full embedding.
6. G1–G3, freeze, first model runs.
