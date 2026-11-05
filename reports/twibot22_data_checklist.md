# TwiBot-22 data-processing checklist

Written Wed 4 Nov 2026, 19:50, before any TwiBot-22 model score. Built from `twibot20_lessons_for_twibot22.md` (the TwiBot-20 mistakes) and `twibot22_plan.md`. Rules recorded here are kept; any later change is written down with its reason and time.

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
| B4 | **Sealed test set** (decided by the builder 4 Nov ~22:05: 50% of test + all experts) | ✅ | `twibot22_sealed_sample.py` → `roles.npy`: test_work 49,951 (14.0% bots), test_sealed 50,937 (49,951 + 986 experts). Drawn before the sample |
| B5 | **100k study sample** | ✅ | `sample_idx.npy`: 100,000 drawn stratified by role and label (70k train, 20k val, 5k test_work, 5k sealed; 14.0% bots each) + the 986 experts = 100,986 users |
| B6 | Weak labels documented: dataset "bot" matches experts for 56 of 101 | ✅ | |
| B7 | **Only train-split labels are ever used for training**, including DEC's stage 1 ("1M weak labels" in the plan means the train part only) | ⬜ | Write into every training script |
| B8 | **Expert users are never in train / val / working test.** All 986 sit in test_sealed. **Changed by the builder 5 Nov ~13:20:** they are no longer used only for a final score; they are also used for training, by 5-fold rotation (G7), so each expert is scored once by a model that never saw it | ✅ | All 986 in test_sealed; rotation in G7 |

## C. Tweets

| # | Item | Status | Note |
|---|---|---|---|
| C1 | Extraction counts: tweets read, kept, regex misses, JSON decode errors, per file (lesson T2) | ✅ | `twibot22_tweets_all.py`: 0 author misses, 0 date misses, 0 decode errors in all 9 files |
| C2 | **Most recent by date, not first met in file order** (T3) | ✅ | Two passes: index every tweet's author and date, select the 20 latest per user across all 9 files (deterministic ties), then decode. 17,838,598 tweets for 933,872 users |
| C3 | Wanted users = all 1M labelled users (top 20 each by date); the sample and hubs are subsets of that file | ✅ | Changed 4 Nov 20:58 (F6). All 88.2M tweet authors are labelled users |
| C4 | Users with 0 tweets: count by class; text = bio only, with a flag | ✅ | On the files: 17.6% of bots vs 4.8% of humans have 0 tweets. A strong cue like empty bio: allowed, reported (D6) |
| C5 | **Text rules** (decided by the builder 4 Nov ~22:45, before any score) | ✅ | (1) **Retweets kept**, as TwiBot-20 (`prep_twibot20.py` has no filter); train share 33.6% of bots' vs 36.4% of humans' tweets. (2) **No total length cap**: all 20 tweets, each piece (bio, each tweet) ≤ 128 tokens, pieces averaged per user as TwiBot-20's `step3_embed.py`; TwiBot-20's 2,000-character cap was a file-size convenience and would cut 70.7% of TwiBot-22 users. (3) **Links and @mentions kept raw**, as TwiBot-20 (its cleaning check: no gain; deleting slightly hurt); tweets with a link 46.0% bots vs 44.1% humans. Difference to report: TwiBot-20 took the first 20 tweets in file order, TwiBot-22 the 20 most recent |
| C6 | Language mix by class | ✅ | `twibot22_tweet_checks.py`: main language English for 78.4% of bots and 74.4% of humans, then Spanish (5.5% / 7.0%). Descriptive |
| C7 | Duplicate tweets across train and val | ✅ | 0.56% of val users have an exact original tweet (≥30 chars) also posted by a train user. For those 157 val bots the copy writers are 31% bots (chance 14%): some bot-to-bot copying, touching ~0.5% of users. Reported, not excluded |
| C8 | Tweet dates within the collection window; no tweets after 15 Mar 2022 | ✅ | 2006-03-21 to 2022-03-15 01:16 UTC; 0 after the collection end; all timestamps +00:00 |

## D. Profile fields

| # | Item | Status | Note |
|---|---|---|---|
| D1 | Shortcut rule run; only `verified` fails, so it is excluded | ✅ | |
| D2 | Shortcut rule re-run on **our** split's train (it was run on the official split) | ✅ | `twibot22_shortcuts.py`: only `verified` fails. 17 profile fields + 28 graph cues (has-edge, degree per relation) |
| D3 | Leaf-size rule uses a count (≥200 users), not 10% (P2) | ✅ | Plus a stricter every-cut-off tail scan: 5 tiny pure-human tails (≤1.2% of users: very high followers, listed, tweets, following; oldest accounts). The followers and listed tails are 80–90% verified and not pure without them; the tweet-count, following and age tails stay ~99% human but cover ≤0.3%. Reported, not excluded |
| D4 | **Labelling-function leakage**: spam keywords in bios and tweets, by class | ✅ | 43,469 words used by ≥200 train users checked: **no bot-pure word**. 8 human-pure words, each 200–244 users (topic communities, e.g. French politics, `#ucustrike`). Apps: 3 newsroom tools (SocialFlow, SocialNewsDesk, Echobox) ~99% human, 400–715 users each. Tiny; reported, not excluded (same treatment as the D3 tails) |
| D5 | Account age parsed from ISO dates, not zero for everyone (X3) | ✅ | `twibot22_profile_check.py` uses `pd.to_datetime`. The TwiBot-20 parsers in `metadata_check.py` and `step6_dgp.py` are still wrong for TwiBot-22 |
| D7 | **Labels were partly made by graph models** (GCN, GAT, R-GCN were labelling functions). Graph models may re-learn the labeller. Check: compare each graph model's gain on weak-label test users vs on expert-labelled users | ⬜ | Planned as DEC question 3; applies to every graph model |
| D8 | **Empty-bio cue: real bot habit (A) or label-maker habit (B)?** Planned by the builder 5 Nov ~20:15, before any expert number; run in the final run only. On the same 986 expert users: bot rate among empty-bio users and among users with a bio, and the **lift** (empty ÷ with bio), under the weak labels and under the expert labels; bootstrap 95% CIs (2,000) on each lift and on their difference. A predicts a similar lift under both sources; B predicts the lift shrinks towards 1 under expert labels. Caveats to report: (1) the expert labels are only partly independent: the paper's 7 voting models were trained on 800 of the 1,000 expert-labelled users, and which 800 is not released; (2) experts are also people looking at the profile, so a lift that survives supports A but does not prove it; (3) only ~100–150 empty-bio experts, so a null result means 'not enough data'. **Second source, added 5 Nov ~20:40 (before any number):** the official repo's annotation study, 500 TwiBot-22 users × 3 experts on 0–4 (`data/twibot-22-raw/annotation_study/`); only 1 overlaps the 986 and none trained the voters, so caveat (1) does not apply. Fixed rule: bot if the median rating ≥ 3, human if ≤ 1, dropped if 2 (not sure); same lift comparison against the weak labels on those users (labels vs labels, no model, so their train / val roles do not matter); read in the final run. Reason: the profile's lead over text rests largely on this cue (findings 1, 4) | ⬜ | Final run (`FINAL_RUN=1`) |
| D6 | "Empty bio" and "no tweets" flags: strong label cues (empty bio 40.6% of bots vs 7.8% of humans; no tweets 17.6% vs 4.8%). Report a flag-only baseline and with-bio / with-tweets slices | ⬜ | |

## E. Graph

| # | Item | Status | Note |
|---|---|---|---|
| E1 | Relations fixed: all 7 (following, followers, mention, retweet, reply, quote, like) | ✅ | 5 fixed at 14:25; changed to 7 by the builder at ~21:30, before any score (H3) |
| E2 | Homophily, adjusted homophily, label informativeness for all 8 relations | ✅ | All near zero (0.011–0.038) |
| E3 | Star-shape check: only 10,039 users have outgoing edges | ✅ | |
| E4 | Degree AUROC on held-out users (R3); out-degree caps and reciprocity (R5) | ✅ | `twibot22_graph_checks.py`, on **val** users (no test label read): in-degree AUROC 0.36–0.49, out-degree 0.50; no cap (max out-degree 907); reciprocity 0.3–1% |
| E5 | Train-to-held-out neighbour leakage vs chance | ✅ | On val users: a bot's train neighbours are 11.6% bots (chance 14.0%), a human's 92.5% human (chance 86.0%). Edges do not give away bot labels |
| E6 | co-list counts use int8, so they can overflow above 127 shared lists (R2) | ➖ | co-list is reported only and never used as model edges. Note it in the report |
| E7 | Relation names in `edge.csv` all present and non-empty (R4) | ✅ | Every relation in the heterophily table has edges |
| E8 | **Q1. Homophily by relation, direction and class** (builder, 5 Nov ~19:20, fixed before any number): for each of the 7 relations and each direction, P(neighbour is a bot \| node is a bot) and P(neighbour is a bot \| node is a human) against the 14% base rate, plus adjusted homophily per direction. Labels of train + val users only (no test, no sealed) | ⬜ | `twibot22_relation_checks.py` |
| E9 | **Q2. Local heterogeneity** (fixed ~19:20): each user's local homophily = share of his TRAIN-labelled neighbours with his own label, on the union graph and per relation. (a) Distribution on val users (deciles, share above 0.5, share with no train neighbour). (b) Where graph models gain: on the working test users, FAGCN, RGCN vs the MLP control (same network, no edges) and XGB-Graph vs own-feature tree (1M, with profile), per local-homophily bin (no train neighbour, [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1]); AUROC and macro-F1 per bin. A test user's own label is used only to slice scores (as in heterophily papers), never as an input | ⬜ | `twibot22_local_homophily.py` |
| E10 | **Q3. Cross-relation inconsistency** (fixed ~19:20): label-free features per user from two relation groups, follow (following + followers) and interaction (mention, retweet, reply, quote, like), both directions: cosine distance between the groups' mean neighbour bio embedding (PCA 32), distance between their mean neighbour profiles, own-vs-group cosine for each group and its difference, log degree ratio, flags for an empty group. Test: (a) each feature alone, val AUROC; (b) XGB-Graph + these features vs XGB-Graph, sample (with and without profile, 3 seeds) and 1M (with profile, seed 0), paired bootstrap CI. Shortcut rule run on them first | ⬜ | `twibot22_relation_checks.py`, `EXTRA=inconsistency` |
| E11 | **Leaderboard reconciliation** (builder, 5 Nov ~19:50, fixed before any number). Why: the TwiBot-22 paper ranks graph models first (accuracy on the official split); we find graph ≈ no graph. 2 × 2 design: split {ours (random stratified), official `split.csv` (train 7.8% / val 28.0% / test 29.4% bots)} × `verified` {out, in as an input column}. Ours / out = the existing results. Models: own-feature tree and XGB-Graph (with profile, 1M bio-only table), MLP control, FAGCN, RGCN (same settings as the main GNNs, seed 0). Metrics: accuracy and bot-class F1 at a 0.5 probability threshold (the paper's convention) and at our validation cut-off, plus macro-F1 and AUROC. **Our sealed users (incl. experts) are removed from every official part** (train, val and test), so the seal holds; `verified` enters only these replication models, never the main ones (explicit override, `REPLICATION=1`). GNNs first (they must finish before DGP's LoRA starts) | ⬜ | `twibot22_official_protocol.py` |

## F. Features and embeddings

| # | Item | Status | Note |
|---|---|---|---|
| F1 | Standardisation and PCA fitted on train only | ✅ | `twibot22_features.py` |
| F2 | Rows kept in `label.csv` order everywhere; any join by id, never by position (X2) | ✅ | `twibot22_graph_checks.py`: all artefacts 1M rows; the 124,231 zero embedding rows equal the 124,231 empty bios exactly. Keep this assertion in every new script. `prep_twibot22.py` (superseded) sorts by id and would need a reindex if reused |
| F3 | Output folders per design (sample, text recipe), so no stale cache is reused (X1, lesson 20) | ✅ | `twibot22_embed_text.py` writes `data/twibot-22/sample110k_<enc>_bio20tw_raw/{emb,ids,user_ids}.npy`, asserts the row order after saving, refuses to overwrite. Timing test (10k users): 213 pieces/s, 2.7 h estimated for 110,057 users. Full run started 4 Nov 22:56 with the builder's go (E5, then MiniLM) |
| F4 | Bio-only and bio + tweets embeddings saved as separate feature sets (E3) | ✅ | Bio only: `emb_bio_{minilm,e5}_all.npy` (1M). Bio + 20 tweets: `data/twibot-22/sample110k_{e5,minilm}_bio20tw_raw/` (110,057 users; E5 153.5 min, MiniLM 37.7 min, both exit 0; row order asserted; 3,344 users without text = zero rows) |
| F5 | Neighbour features use neighbours' features only, never their labels | ✅ | |
| F6 | **Headline text model and comparison, fixed by the builder 4 Nov 20:58, before any score (replaces the 20:38 rule).** Option B: headline text model = multilingual-E5. **Headline comparison = the 100k stratified sample** (natural 14% bots), text = bio + 20 most recent tweets, neighbour features from the full 1M graph (hub users' tweets embedded as neighbour providers). **Full-graph models on all 1M users use bio only** (reported as the full-scale check). Option A (bio + tweets for all 1M) is run afterwards as an extra check if time allows; it never replaces the headline. MiniLM = check; Qwen2.5-1.5B on the sample only. Before any tweet embedding: a 10k-user timing test, the estimate, and the builder's go; not while a TwiBot-20 LoRA seed run is on the GPU. Report limit: "with tweets" results are for a 100k random sample, not all 1M users | ✅ decided | Tweet extraction covers all 1M users (top 20 each), so it serves B and A |
| F7 | Users with no neighbours in a direction get a mean of 0 (= the train mean after standardisation); the separate log-count column (0) tells models it is missing | ✅ | `twibot22_features.py` `mean_agg`. GNNs need the same flag |

## G. Run hygiene

| # | Item | Status | Note |
|---|---|---|---|
| G1 | `OMP_NUM_THREADS=1` and no torch import with sklearn boosting (the TwiBot-20 silent segfault) | ⬜ | In the CPU model scripts |
| G2 | Every launcher logs each job's exit code | ⬜ | |
| G7 | **Weak → expert fine-tuning with 5-fold rotation** (builder, 5 Nov ~13:20, replacing the 4 Nov light version; reason: dataset "bot" matches experts for only 56 of 101, so weak-label models learn the labelling rules; using all experts for training would leave no clean test). Folds: StratifiedKFold(5, shuffle, seed 42) on the expert label (~789 train / ~197 held out, 84 / 21 bots). **LoRA target:** start from the best target adapter (validation), 1 epoch on the 4 training folds, step 6 settings, no selection; macro-F1 cut-off chosen on the fold model's own training folds (`twibot22_expert_rotation_llm.py`, queued after DGP, ~3.5-4 h GPU, **blind**: writes scores, prints no metric). **Trees / MLP:** the same folds, a real second training stage (continue / refit on experts), CPU. Every model also scored on the same 986 experts weak-only, for the paired comparison. Read once, in the final run. Fixed before any expert score. Also answers D7 | ⬜ | Replaces DEC stage 4 |
| G8 | **The LLM reads the same text as the encoders it is compared with** (same pieces, same token limit), so "frozen encoder vs fine-tuned LLM" is like for like. Lesson from TwiBot-20 (LoRA 480 tokens vs encoders 2,000 characters; external feedback #2, 5 Nov) | ⬜ | Set when the LLM step is configured |
| G6 | **Label and `verified` out of every model input** (step 3, builder 4 Nov ~21:40) | ✅ | `twibot22_guard.py`: `users.parquet` now has profile fields only (no verified, no label/y/split); verified in `eval_only/verified.parquet` (95,398), read only for the non-verified slice and checks; `check_columns()` stops any model input named verified/label/y/split (tested). Shortcut results identical after the move. `withheld` is 59 users, not 0 (comment fixed); near-constant, left out |
| G3 | Selection rule, seeds (3), metrics (macro-F1, AUROC, AUPRC, bootstrap CIs) written into the scripts | ✅ | `twibot22_metrics.py`: val-only cut-off and selection, seeds 0–2, paired bootstrap CIs (2,000), self-tested |
| G4 | Every result also scored on non-verified test users and with-bio users | ✅ | Slices in `twibot22_metrics.slices()`: all, non-verified, with bio, with tweets; optionally restricted to the 100k sample |
| G5 | **No artificial setting B on TwiBot-22.** The natural 14% bot share is already within graph-fraud benchmark rates (1.3–21.8% in GADBench), and TwiBot-20's 1:10 subsample removed 8,108 of 8,162 bot–bot edges. For rarer rates, report **PR-AUC with test users reweighted to 5% and 1% bots** (bots weighted so they make up 5% or 1% of the test weight; graph, training and cut-off unchanged). Wording in the report: "fraud-like rare class", not "fraud" | ✅ | In `twibot22_metrics.py`; self-test: a random score gives 0.14 / 0.05 / 0.01 as expected |

## H. Added after the step-list review and its sceptic pass (4 Nov, 20:28)

| # | Item | Status | Note |
|---|---|---|---|
| H1 | Hub users (10,039 with an outgoing edge) get bio + tweet embeddings as neighbour-feature providers; flagged, not added as study users | ✅ | `hub_idx.npy`; 968 are also in the random sample. Users to embed (sample ∪ hubs): 110,057 |
| H2 | Feature table with the headline bio encoder | ✅ | `features_full_e5.npz` (multilingual-E5 bio PCA, 7 relations, 814 columns; split reused from file; `bio_encoder` recorded). Shortcut rule rerun on it: 0 of 814 fail. The MiniLM table `features_full.npz` is kept as the check |
| H3 | **Model-input contract, step 1 fixed by the builder 4 Nov ~21:25, before any score.** Profile fields (all except `verified`) are kept as inputs. **Cheap models** (MLP, trees, XGB-Graph, DEC-lite, links-only and profile-only baselines): run **both** versions, with profile and without profile (text + graph only). **Expensive models** (GNNs on 1M users, LLM + LoRA): run **with profile** only, so no information is left out (LLM: profile as numbers in the prompt, as TwiBot-20's `dgp_num`). The "what does the graph add without profile" question is answered by the cheap models. If TwiBot-22 finishes early: rerun TwiBot-20's GNN and LLM with profile too, so the two datasets match. **Step 2 (4 Nov ~21:30): all 7 relations** (following, followers, mention, retweet, reply, quote, like) as model edges and neighbour features; co-list reported only | ✅ | Every new tweet feature passes the shortcut rule before use |
| H4 | Tweet script checks each file's completeness itself (ends with `]`, size stable) before marking it done (lesson T1) | ✅ | `complete()` asserts the closing `]` before indexing |
| H5 | Parse tweet dates with `pd.to_datetime(utc=True)` before "most recent" (X6), checked on a sample | ✅ | Selection uses parsed seconds; checked against a full JSON parse of a 600 MB slice (532,984 tweets, all authors and dates match) |
| H6 | Results naming that cannot overwrite: one folder per design and seed; seeds fixed per model (lessons 18–20) | ✅ | `save_scores()` writes `runs/<design>/<model>_s<seed>.npz` and refuses to overwrite |
| H7 | **LLM budget, decided by the builder 5 Nov ~00:50:** Qwen3-1.7B, **1,300 training users**, same LoRA settings as TwiBot-20 (B = 10, K = 2, M = 8, r = 16, lr 1e-4). Order tonight: smoke test of step 6 on 20 users (5 min); if it passes, after E5 finishes: summaries → LoRA `target` → LoRA `dgp`, then the MiniLM embedding (cheap models only). Each sub-step checkpointed and logged. No GNNs on CPU while LoRA trains. If the smoke test fails: nothing is launched; the failure is written to the overnight notes | ✅ decided | **Details decided 5 Nov ~02:15:** (1) adapter saved at every check, chosen by macro-F1 on a fixed 2,000-user stratified validation subset; if one check takes > 10 min in the smoke test, use 1,000 users or check less often. (2) An MLP on exactly LoRA's 480-token text is added (G8). (3) Profile in the LLM: decided later; `target` and `dgp` run text-only as on TwiBot-20. (4) DGP relation types: 2 groups, follow = following + followers, interaction = mention + retweet + reply + quote + like (6 path types, as TwiBot-20's 2 relations). (5) MiniLM done; the LLM runs tonight. (6) GNNs (changed by the builder 5 Nov ~02:35, replaces "all 5 × 3 seeds"): all 5 GNNs with 1 seed each, then the 2 best by validation macro-F1 with 3 seeds each; never all 5 × 3. Results tables mark each GNN as 1 seed or 3 seeds. Thursday daytime on CPU, not during LoRA. LoRA training users: 1,300 drawn from the sample's train part, stratified, seed 42; tested on the sample's working-test users only |
| H7b | **DGP scope (builder, 5 Nov ~13:45):** 1,300 train, the same 1,000 validation users as `target`, a 1,500-user stratified test subset of test_work (seed 42, saved and committed before scoring: `results/twibot22/A/dgp_test_subset_1500.npy`). dgp vs target on that subset (`twibot22_dgp_vs_target.py`, paired bootstrap). Reason: summaries ~1 s each; 5.6 h instead of 8.6 h. GPU queue: zero-shot rescore → rotation smoke test → expert rotation → DGP → target seed 1 → dgp seed 1 | ✅ decided | `step6_dgp.py` `DGP_TEST` |
| H8 | Sealed users: ids saved to disk before the sample draw; every scoring script refuses them before the final run | ✅ | `twibot22_guard.score_mask()` returns test_work unless `FINAL_RUN=1`; `assert_not_sealed()` stops otherwise (tested) |
| H10 | **Sample neighbour features from bio + tweet embeddings** (headline, F6). Checked 5 Nov 00:50: 98.8% of sample users have all their 1-hop neighbours embedded (59% of links, because a few sample hubs link out widely), but only 18% of 2-hop neighbours are. So: **1-hop** per-relation and union text means from the multilingual-E5 bio + tweet embeddings, averaged over embedded neighbours, plus a "share of neighbours embedded" column; **2-hop** stays on the bio-only embedding (all 1M users have it); profile columns unchanged. Own file next to the embeddings; shortcut rule rerun on it | ✅ | `twibot22_features_sample.py` → `features_sample_e5tw.npz`: 100,986 × 830 (profile 255, text 544, graph 31 columns; text PCA explained 0.51; 106,713 of 110,057 embedded users have text). Shortcut rule rerun: **none** (strongest single column: own log followers, val AUROC 0.740) |
| H9 | Extraction time | ✅ | 7 min in total (index 2.5 min at ~900 MB/s, select 1 min, decode 4 min) |

Simplified (descriptive only, not gates): C6 language mix, C8 date range (folded into the C1 counts), C4 zero-tweet counts. G5 reweighted PR-AUC is computed from saved scores after the runs; it changes no model.

## Freeze (dropped)

No separate freeze commit (builder, 5 Nov). Decisions are recorded with their times in this file and the git history; models start once the rows they need are done.

## Order for tonight and tomorrow morning

1. `tweet_1` lands → A2.
2. Quick checks, ~30 min CPU: D2, D3, E4, E5. Then B4 (sealed set) before B5 (sample).
3. Fix the tweet script (C1–C3, C8), draw the sample (B5), extract tweets (~30–40 min).
4. Tweet checks: C4, C6, C7, D4. Decide the text rules (C5).
5. F3, then a 5-minute embedding timing test, then the full embedding.
6. G1–G3, freeze, first model runs.
