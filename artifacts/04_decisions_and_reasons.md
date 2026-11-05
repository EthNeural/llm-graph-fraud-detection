# Decisions and reasons: Instagram, TwiBot-20, TwiBot-22

Written Wed 4 Nov 2026, evening, so that anyone picking up this project (a teammate, the supervisor, a new Claude session) can see **what was decided, when, and why**, without reading every chat. It covers all three datasets in the order they were used. Detailed numbers live in `REPORT.md` (Instagram, TwiBot-20) and `reports/twibot22_data_checklist.md` (TwiBot-22); this file is the reasoning.

**The project in one paragraph.** COMP8851 (Macquarie). Question: do LLM-based and graph-based methods help detect bots or fraud-like accounts when the graph is **heterophilous**, that is, when bad accounts mostly connect to good ones, so a user's neighbours do not share its label? The supervisor asked for heterophily to be measured, not assumed. We compare text-only models, graph models (GNNs and neighbour-feature trees) and LLMs (zero-shot reasoning and LoRA fine-tuning, including a replication of the DGP method) on the same users.

---

## 1. Instagram (27 Oct – 3 Nov): the first text graph

| Decision | Why |
|---|---|
| **Dropped Elliptic** (27 Oct), the team's original Bitcoin dataset | Elliptic has no natural text, so an LLM has nothing to read. The project is about LLMs on graphs |
| **Instagram from GLBench** (Li et al., 2024), used as prepared | A real text graph (bios) with follow edges, and the same version FLAG (KDD 2025) uses, so results can be set next to a published fraud-detection paper |
| **Class 1 = commercial account**, treated as "a stand-in for a minority class, not fraud" | Instagram has no fraud labels; saying "fraud" would overclaim |
| **Two settings: A (natural, 36% commercial) and B (1:10, FLAG-style)** | A tests the data as it is; B copies FLAG's protocol for a rare class (§4.1.1) |
| **Cut-off chosen on validation for every model** (30 Oct) | Scores are not calibrated, and 0.5 is a bad cut-off when the class is rare |
| **Became the secondary dataset** (3 Nov), steps 1–5 only | TwiBot-20 has real bot labels, a benchmark split and typed relations. The LLM fine-tuning step costs 3–6 GPU hours per run, so it went to the main dataset |

**What Instagram showed.** Text is weak there (AUROC about 0.63), and every model sits within about 0.03 macro-F1 of a logistic regression on the text embeddings. GCN is level with the text-only MLP (+0.002, CI includes 0). So on Instagram the graph neither helps nor hurts.

---

## 2. TwiBot-20 (1 – 4 Nov): the main dataset

### Data decisions

| Date | Decision | Why |
|---|---|---|
| 1 Nov | **Added TwiBot-20** (official split; text = bio + 20 tweets, capped at 2,000 characters) | A real Twitter bot dataset with text and a benchmark split |
| 1 Nov | **Kept only the 11,826 labelled users**; dropped 217,754 unlabelled "support" users | They have no labels and no text in our use; keeps preparation small. Checked later (3 Nov): they bridge 8,082 users, so a robustness run on the bridge graph was added |
| 1 Nov | **Excluded all profile metadata** | `verified` alone gave macro-F1 0.803. Hays et al. (WWW 2023) explain why: the crawl started from verified accounts and "verified accounts were automatically labeled human". The other fields were excluded with it, without a check (our own mistake #1, see `reports/twibot20_lessons_for_twibot22.md`) |
| 4 Nov | **Shortcut rule written down and run**: only `verified` fails; the 5 other fields add +0.04–0.05 | Fixed the 1 Nov over-caution. Profile fields became a separate "text + profile" track because every other comparison was already text-only |
| 1 Nov | **Encoder ladder**: six text encoders, everything else fixed | Separates the effect of the encoder from the effect of the graph |
| 2 Nov | **DGP replication** with Qwen3-1.7B, 1,300 training users, LoRA r = 16 | The closest to the paper that fits a 16 GB Mac; the paper's code is not public |
| 3 Nov | **TwiBot-20 = main dataset, Instagram = secondary** | See section 1 |
| 3 Nov | **Report adjusted homophily and label informativeness**, not only edge homophily | Plain homophily is inflated by class imbalance (0.88 vs 0.014 adjusted in setting B) |
| 3 Nov | **Training-size control**: text models retrained on the LLM's 1,300 users | The LLM trained on 1,300 users and text models on 8,278; that confounded "LLM vs encoder" with data size |
| 3 Nov | **Added FAGCN, FLAG-lite, LINKX** | GNNs designed for heterophily; tests whether "graphs do not help" is about GCN or about the data |

### What TwiBot-20 showed

1. **A good text encoder does the work**: Qwen2.5 + MLP 0.766, Qwen3 + MLP 0.769 macro-F1, against TF-IDF 0.702.
2. **No graph model beats the text-only MLP.** GCN is 0.04–0.06 below it; FAGCN, FLAG-lite and LINKX stay below too.
3. **Edges carry almost no label signal** (adjusted homophily −0.015 to 0.08). The signal links do carry is a **crawl artefact**: who appears in whose follower sample, which tracks follower count.
4. **The fine-tuned LLM roughly matches text models at equal data** (1,300 users), and stays below them when those use all 8,278.
5. **Verified users inflate every score**: all 326 verified test users are human. Without them, target LoRA drops from 0.704 to 0.641 macro-F1 (confirmed from raw data on 4 Nov, `check_verified_subset.py`).
6. **Trees do not beat the MLP** (GADBench's "trees win" does not hold here), and the crawl-artefact gain for trees (+0.025) disappears on non-verified users (−0.009). Tree baseline rerun 4 Nov after fixing a silent crash (torch and sklearn clashing over OpenMP).

### Mistakes we made, and why TwiBot-22 is done differently

The 22-item review (`reports/twibot20_lessons_for_twibot22.md`) found, among others:
- rules (checkpoint choice) set **after** some test scores were seen;
- the first LLM runs had **no seed**;
- **verified users never removed from the test set**;
- the tree baseline **crashed silently**;
- results files that could be **overwritten**.

None of these changed the main conclusion, but they shifted numbers by 0.02–0.05. TwiBot-22's whole approach (checklist first, rules fixed before any score, sealed test set) comes from this list.

---

## 3. TwiBot-22 (4 Nov onwards): the second, larger test

**Why TwiBot-22.**
- It has 1M users and 14% bots, which is fraud-like and within the 1.3–21.8% range of the GADBench fraud benchmarks.
- Its text is weak: bio word counts give macro-F1 0.561 on users with a bio.
- So it tests whether the graph helps when the text cannot do the work.
- Facebook reports 3.8–5.2% abusive accounts (DEC paper), so 14% is not "real-world fraud". The report calls it a **fraud-like rare class**.

### Fixed before any model score (times are git commit times, 4 Nov)

| Time | Decision | Why |
|---|---|---|
| 14:22 | **Our own split**: stratified 70/20/10, seed 42, all 986 expert users in test | The official split is not random: train has 7.8% bots, test 29.4%, and all 95k verified users sit in train |
| 14:25 → ~21:30 | **Relations: all 7** (following, followers, mention, retweet, reply, quote, like); co-list reported only | First 5 by coverage; changed to 7 under "don't miss information". Quote and like add links for only 8,856 users, at small cost. Co-list (61M pairs) is too dense for the laptop |
| afternoon | **Keep the whole 1M graph for cheap models; use a sample for expensive work** | Industry and benchmarks cut the computation, not the graph (`reports/graph_slicing_research.md`). Cutting a subgraph is what broke TwiBot-20's setting B: it removed 8,108 of 8,162 bot–bot edges |
| evening | **No artificial setting B** | The natural 14% is already rare. For rarer rates we instead report PR-AUC with the test set reweighted to 5% and 1% bots (evaluation only) |
| 16:29 | **Stopped the all-1M Qwen bio embedding** (26% done, ~9 h left) | It would have used the GPU for a weak, interim input (bio only), when the final text is bio + tweets |
| 20:38 → 20:58 | **Option B is the headline**: multilingual-E5 on bio + 20 most recent tweets, on a **100k stratified sample**, neighbours from the full graph. Full-graph 1M models use bio only. Option A (tweets for all 1M) later, as an extra, never the headline | Follows Facebook DEC (cheap features on everyone, expensive work on samples) and TwiBot-20 (same text for every compared model). Avoids a 10–15 GPU-hour risk before the TwiBot-20 queue on Thursday night. The 100k sample has ~10× TwiBot-20's test size, so it is less noisy. Report limit: "with tweets" results are for a random sample |
| evening | **Profile fields kept** (all except `verified`). Cheap models: with and without profile. Expensive models (GNNs, LLM): with profile only | TwiBot-20's text-only main track was partly an accident of order. Profile fields are real signal (follower count alone: AUROC 0.733). Running the cheap models both ways still shows what the graph adds without profile. If time allows, rerun TwiBot-20's GNNs and LLM with profile so the two datasets match |
| ~21:40 | **`verified` and the labels moved out of the profile file** into `eval_only/`; every model script stops if it sees them (`twibot22_guard.py`) | Prevents an accidental shortcut: on TwiBot-20, `verified` alone scored 0.803. The profile file also held the label columns, which was a worse risk |
| ~22:05 | **Sealed test set: 50% of test plus all 986 experts**, scored once at the end | Like Kaggle's private leaderboard, it stops choices drifting toward the test set. The experts are the cleanest labels we have (dataset "bot" labels match them only 56 of 101 times), so they stay untouched until the end |
| ~22:45 | **Text rules**: retweets kept; no total length cap (all 20 tweets, each piece ≤ 128 tokens, averaged per user); links and @mentions kept raw | All three follow TwiBot-20, except the cap: there it was a file-size convenience and here it would cut 70.7% of users to ~15 tweets. TwiBot-20's cleaning check found placeholders gave no gain and deleting links slightly hurt. One difference to report: TwiBot-22 uses the 20 most recent tweets, TwiBot-20 the first 20 in file order |
| ~22:20 | ~~**Weak → expert recalibration, light version**~~ (replaced 5 Nov ~14:00, next row), at the very end: trees and MLP keep their weights; a small logistic regression on their score + a few profile fields is fitted on the experts with 5-fold rotation | Facebook DEC trains on noisy labels then clean ones (30M, then 240k). We have only 986 clean labels (105 bots): too few for full fine-tuning, enough for recalibration. Rotation keeps every expert's score clean, so the seal holds. A check, not a headline |
| 5 Nov ~14:00 | **Weak → expert fine-tuning with 5-fold rotation**, for LoRA `target` (1 epoch from its best adapter, ~789 experts per fold) and for trees / MLP (a real second training stage). Every model is also scored weak-only on the same 986 experts | The builder judged that sealing all 986 expert labels for one final score wastes them: the weak labels disagree with the experts too often (56 of 101) for weak-label training to learn the expert view. Training on all experts would leave no clean test, so they are rotated: each expert trains 4 folds and is scored once by a model that never saw it, and all 105 expert bots stay in the evaluation. Measured timings (5 Nov) put the LLM rotation at ~3.5–4 h, so the 4 Nov "too costly" reason no longer holds. The rotation runs blind; scores are read once, in the final run. Fixed before any expert score |
| 5 Nov ~15:00 | **DGP at TwiBot-20 scope**: 1,300 training users (same settings), the same 1,000 validation-check users as `target`, and a **1,500-user stratified test subset** of the working test users (seed 42, `dgp_test_subset_1500.npy`, committed before any scoring). **dgp vs target is compared on that same subset**; `target` keeps its 5,000-user score for the other comparisons | Time: LLM summaries run at ~1 s each (0.94 s measured). Full scope (5,000 test users) needs 26,674 node + 13,126 path summaries, ~8.6 h; the 1,500 subset needs 18,234 + 6,815, ~5.6 h (most summaries belong to the 2,300 train and validation users' neighbours). TwiBot-20 scored DGP on a test of similar size. The subset scores `target` at 0.632 / 0.718 vs 0.626 / 0.710 on all 5,000, so it is representative |
| ~22:05 | **100k sample**: stratified by role and label, seed 42, plus the experts (100,986 users); 10,039 hub users embedded as neighbour providers only | The hubs are the only users with outgoing links, so everyone's neighbour features come from them. Adding them as study users would make them ~10% of the sample instead of ~1% |

### What the checks found (no model run yet)

- **Shortcuts:** only `verified` (0 of 95,398 verified users are bots), across 17 profile fields, 28 graph cues and all 814 feature columns.
- **Graph:** near-zero label signal on all 7 relations (adjusted homophily 0.011–0.038). A bot's neighbours are 11.6% bots, below the 14% chance level, which is heterophily. Only 10,039 users (1%) have any outgoing link, so the graph is star-shaped.
- **Strong cues, allowed but reported:** empty bio (40.6% of bots vs 7.8% of humans); no tweets (17.6% vs 4.8%); default profile picture (13.0% vs 1.7%).
- **Data integrity:**
  - 88,217,457 tweets read, exactly the "posted" links in `edge.csv`;
  - 0 decode errors;
  - no tweets after the collection end;
  - every file in `label.csv` row order.

---

## 4. Rules that now apply to everything

1. **Question every claim.** A low-cost sceptic agent checks new claims; several of our own statements were corrected this way and the corrections are recorded.
2. **Decide before scoring.** Split, sample, metrics, model choice and cut-offs are written down and committed before any test number.
3. **Honest metrics.** Macro-F1, AUROC and PR-AUC with bootstrap confidence intervals, plus a non-verified slice. Accuracy alone hides an "always human" model.
4. **No shortcuts in the inputs**, checked by a fixed rule, and guarded in code.
5. **Before any job longer than 30 minutes**, say in one line what it gives and why the report needs it; ask if the value is weak.
6. **No dataset files in the repository** (TwiBot-22 is CC BY-NC-ND).

## 5. Still open (4 Nov, late evening)

- Text rules for the tweets: keep retweets? length cap? (checklist C5)
- Tweet checks: copy-paste tweets across train and test, spam keywords, posting app (C6, C7, D4).
- Feature rebuild with multilingual-E5 bios, then re-check shortcuts (H2).
- Shared scoring module (G3, G5), then the freeze and the first model runs.
- The builder's go after a 10k-user timing test, before embedding tweets.
