# TwiBot-22 data-processing checklist

Written Wed 4 Nov 2026, 19:50, before any TwiBot-22 model score. Built from `twibot20_lessons_for_twibot22.md` (the TwiBot-20 mistakes) and `twibot22_plan.md`. Nothing below the line "Freeze" may change after the first test score.

Status: ✅ done and checked · 🟡 partly done · ⬜ to do · ➖ not needed (with the reason)

## A. Raw data

| # | Item | Status | Note |
|---|---|---|---|
| A1 | `user.json`, `label.csv`, `split.csv`, `edge.csv` complete | ✅ | `edge.csv` 6,642,968,601 bytes; `user.json` matched byte for byte against a second download |
| A2 | 9 tweet files, each complete (ends with `]`) | ✅ | All 9 in `tweets/` (10.0–12.5 GB, 104.4 GB total), each ends with `}]`. Decode errors are counted during extraction (C1) |
| A3 | Tweet count per file and in total, against the paper's ~88M | ✅ | 88,217,457 tweets = exactly the `post` edges in `edge.csv` (`tweets_top20_summary.json`) |
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
| C1 | Extraction counts: tweets read, kept, regex misses, JSON decode errors, per file (lesson T2) | ✅ | `twibot22_tweets_all.py`: 0 author misses, 0 date misses, 0 decode errors in all 9 files |
| C2 | **Most recent by date, not first met in file order** (T3) | ✅ | Two passes: index every tweet's author and date, select the 20 latest per user across all 9 files (deterministic ties), then decode. 17,838,598 tweets for 933,872 users |
| C3 | Wanted users = all 1M labelled users (top 20 each by date); the sample and hubs are subsets of that file | ✅ | Changed 4 Nov 20:58 (F6). All 88.2M tweet authors are labelled users |
| C4 | Users with 0 tweets: count by class; text = bio only, with a flag | ✅ | On the files: 17.6% of bots vs 4.8% of humans have 0 tweets. A strong cue like empty bio: allowed, reported (D6) |
| C5 | **Text rules**: N = 20 most recent; retweets kept or dropped?; character cap; URLs and @-mentions normalised? | ⬜ | Decide and write down before embedding. TwiBot-20 used bio + 20 tweets |
| C6 | Language mix by class (multilingual-E5 handles non-English; MiniLM does not) | ⬜ | |
| C7 | Duplicate and near-duplicate tweets and bios **across train and test** (copy-paste bot farms leak between splits) | ⬜ | 3,780 duplicate-bio groups overall, 281 of them pure-bot |
| C8 | Tweet dates within the collection window; no tweets after 15 Mar 2022 | ✅ | 2006-03-21 to 2022-03-15 01:16 UTC; 0 after the collection end; all timestamps +00:00 |

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
| E4 | Degree AUROC on held-out users (R3); out-degree caps and reciprocity (R5) | ✅ | `twibot22_graph_checks.py`, on **val** users (no test label read): in-degree AUROC 0.36–0.49, out-degree 0.50; no cap (max out-degree 907); reciprocity 0.3–1% |
| E5 | Train-to-held-out neighbour leakage vs chance | ✅ | On val users: a bot's train neighbours are 11.6% bots (chance 14.0%), a human's 92.5% human (chance 86.0%). Edges do not give away bot labels |
| E6 | co-list counts use int8, so they can overflow above 127 shared lists (R2) | ➖ | co-list is reported only and never used as model edges. Note it in the report |
| E7 | Relation names in `edge.csv` all present and non-empty (R4) | ✅ | Every relation in the heterophily table has edges |

## F. Features and embeddings

| # | Item | Status | Note |
|---|---|---|---|
| F1 | Standardisation and PCA fitted on train only | ✅ | `twibot22_features.py` |
| F2 | Rows kept in `label.csv` order everywhere; any join by id, never by position (X2) | ✅ | `twibot22_graph_checks.py`: all artefacts 1M rows; the 124,231 zero embedding rows equal the 124,231 empty bios exactly. Keep this assertion in every new script. Earlier note: | True for the new scripts; `prep_twibot22.py` sorts by id, so it must reindex |
| F3 | Output folders per design (sample, text recipe), so no stale cache is reused (X1, lesson 20) | ⬜ | Do before the first tweet embedding |
| F4 | Bio-only and bio + tweets embeddings saved as separate feature sets (E3) | ⬜ | |
| F5 | Neighbour features use neighbours' features only, never their labels | ✅ | |
| F6 | **Headline text model and comparison, fixed by the builder 4 Nov 20:58, before any score (replaces the 20:38 rule).** Option B: headline text model = multilingual-E5. **Headline comparison = the 100k stratified sample** (natural 14% bots), text = bio + 20 most recent tweets, neighbour features from the full 1M graph (hub users' tweets embedded as neighbour providers). **Full-graph models on all 1M users use bio only** (reported as the full-scale check). Option A (bio + tweets for all 1M) is run afterwards as an extra check if time allows; it never replaces the headline. MiniLM = check; Qwen2.5-1.5B on the sample only. Before any tweet embedding: a 10k-user timing test, the estimate, and the builder's go; not while a TwiBot-20 LoRA seed run is on the GPU. Report limit: "with tweets" results are for a 100k random sample, not all 1M users | ✅ decided | Tweet extraction covers all 1M users (top 20 each), so it serves B and A |
| F7 | Users with no neighbours in a direction get a mean of 0 (= the train mean after standardisation); the separate log-count column (0) tells models it is missing | ✅ | `twibot22_features.py` `mean_agg`. GNNs need the same flag |

## G. Run hygiene

| # | Item | Status | Note |
|---|---|---|---|
| G1 | `OMP_NUM_THREADS=1` and no torch import with sklearn boosting (the TwiBot-20 silent segfault) | ⬜ | In the CPU model scripts |
| G2 | Every launcher logs each job's exit code | ⬜ | |
| G3 | Selection rule, seeds (3), metrics (macro-F1, AUROC, AUPRC, bootstrap CIs) written into the scripts | ⬜ | Plan step 1 |
| G4 | Every result also scored on non-verified test users and with-bio users | ⬜ | |
| G5 | **No artificial setting B on TwiBot-22.** The natural 14% bot share is already within graph-fraud benchmark rates (1.3–21.8% in GADBench), and TwiBot-20's 1:10 subsample removed 8,108 of 8,162 bot–bot edges. For rarer rates, report **PR-AUC with test users reweighted to 5% and 1% bots** (bots weighted so they make up 5% or 1% of the test weight; graph, training and cut-off unchanged). Wording in the report: "fraud-like rare class", not "fraud" | ⬜ | Evaluation only; add to the shared metrics code with G3 |

## H. Added after the step-list review and its sceptic pass (4 Nov, 20:28)

| # | Item | Status | Note |
|---|---|---|---|
| H1 | Hub users (10,039 with an outgoing edge) get bio + tweet embeddings as neighbour-feature providers; flagged, not added as study users | ⬜ | Tweets come from the all-1M extraction (C3) |
| H2 | **The 814-feature table uses MiniLM bio PCA** (`twibot22_features.py`). Record the embedder in `features_info.json`; if multilingual-E5 is the headline (F6), rebuild the table with it (~10 min) | ⬜ | Depends on F6 |
| H3 | **Model-input contract**, written before any run: which feature blocks each model gets (profile, bio, tweets, degrees, neighbour means), and whether quote and like are neighbour features (the features use 7 relations; the plan's model edges are 5) | ⬜ | |
| H4 | Tweet script checks each file's completeness itself (ends with `]`, size stable) before marking it done (lesson T1) | ✅ | `complete()` asserts the closing `]` before indexing |
| H5 | Parse tweet dates with `pd.to_datetime(utc=True)` before "most recent" (X6), checked on a sample | ✅ | Selection uses parsed seconds; checked against a full JSON parse of a 600 MB slice (532,984 tweets, all authors and dates match) |
| H6 | Results naming that cannot overwrite: one folder per design and seed; seeds fixed per model (lessons 18–20) | ⬜ | With F3 |
| H7 | GPU-time estimate for the LLM sample (embedding + LoRA) and a pre-set fallback sample size if it does not fit by Sat | ⬜ | Before B5 is final |
| H8 | Sealed users: ids saved to disk before the sample draw; every selection script asserts it never reads their labels | ⬜ | With B4 |
| H9 | Extraction time | ✅ | 7 min in total (index 2.5 min at ~900 MB/s, select 1 min, decode 4 min) |

Simplified (descriptive only, not gates): C6 language mix, C8 date range (folded into the C1 counts), C4 zero-tweet counts. G5 reweighted PR-AUC is computed from saved scores after the runs; it changes no model.

## Freeze

When every row above is ✅ or ➖, commit this file. Only then run the first model.

## Order for tonight and tomorrow morning

1. `tweet_1` lands → A2.
2. Quick checks, ~30 min CPU: D2, D3, E4, E5. Then B4 (sealed set) before B5 (sample).
3. Fix the tweet script (C1–C3, C8), draw the sample (B5), extract tweets (~30–40 min).
4. Tweet checks: C4, C6, C7, D4. Decide the text rules (C5).
5. F3, then a 5-minute embedding timing test, then the full embedding.
6. G1–G3, freeze, first model runs.
