# TwiBot-20 lessons for TwiBot-22

Written 4 Nov 2026 (read-only audit; no code, REPORT.md or results changed). **[V]** = I checked the file, line or number myself. **[S]** = suspicion, not verified.

## 1. What went wrong or was found late on TwiBot-20

| # | Issue | Evidence | Found | Impact |
|---|---|---|---|---|
| 1 | **Metadata handled late.** "verified" was excluded on 1 Nov, along with *all* other profile fields. The shortcut rule was written down only on 4 Nov | `metadata_check.py:4-10`; Decision log 1 Nov; changelog 4 Nov 05:40 [V] | 3 days late | The 5 allowed fields add +0.04 to +0.05 (CI excludes 0). DGP's NumSumm was left out until `dgp_num`. The rule's "fixed before looking" holds for the 5 fields, not for verified (its 0.803 was known on 1 Nov) |
| 2 | **Verified users were never removed from test (new).** 326 of 1,183 test users (27.6%) are verified, and every one is human. On the 857 non-verified test users, target LoRA scores macro-F1 0.641 / AUROC 0.754, against 0.704 / 0.782 overall. `dgp` scores 0.649 / 0.754 | CPU check on `metadata.csv` + `step6_*_scores.npz` [V] | Not in REPORT | Every headline partly rewards spotting a verified-style (official) account in the *text*. The base rate also shifts (75% bots in that subset). LR, MLP and GNNs were not re-checked |
| 3 | Support users dropped without checking | Decision log 1 Nov; "dropped support users" [V] | 3 Nov | 8,082 bridges; 75 of 87 isolated test users sit next to one. Needed a robustness rerun |
| 4 | **Typed edges merged in prep, then needed again** | `prep_twibot20.py:71-75` merges everything that is not `post`; `step6_dgp.py:112-124` re-reads the zip from a hard-coded `~/Downloads` path [V] | 2 Nov | Extra code path. The 4 Nov metadata and link scripts all depend on `edge_typed.npz` |
| 5 | **Crawl artefact in link direction** | `link_direction.py`; review 4 [V] | 3 Nov 22:00, after steps 4-6 had run | In-follow minus in-friend gives AUROC 0.713. Graph-only "signal" and DGP's "none" lines are artefact, not relational signal. `structure_features` (`step4_gnn.py:194-201`) feeds the degree, and so this artefact, to graph-only models |
| 6 | **The "none" isolation cue sits in both prompts** | `step6_dgp.py:286` ("none"); also `step5_llm_reason.py:127` ("This account has no connections") [V] | Step 6: 3 Nov 20:15. Step 5's version never flagged | In A, target + 3 flags matches `dgp` (0.731 vs 0.711) |
| 7 | **First LoRA runs had no LoRA seed** | `step6_dgp.py:514-517` seeds only if `LORA_SEED` is set [V] | 4 Nov 04:40 | Not reproducible. **Seed spread is large:** best-loss `dgp` scores 0.711 (unseeded) vs **0.745** (seed 1). Under the val-F1 rule: 0.727 vs 0.745; target 0.728 vs 0.736 (`step6_rescore.json`) [V]. Seed noise (about 0.02-0.03) exceeds every dgp − target gap |
| 8 | **Checkpoint rule set after the first test results (forking-paths risk)** | Rule committed `fc45a63` 3 Nov 15:24 [V]. The best-loss test scores were already known (`dgp` 0.711 on 3 Nov). Main tables are still not switched (REPORT "LoRA checkpoint rule") | 3-4 Nov | Moves each variant by about 0.02 and flips the sign of dgp − target. Only 2 adapters are saved (`step6_dgp.py:416-418`), so the rule picks between 2 of the checks |
| 9 | Step 5 neighbour ranking uses flagged, non-unit vectors | `step5_llm_reason.py:180` → `common.py:92` appends the 2 flags. The comment at `:68` says "emb is unit-length", but piece averages (`step3_embed.py:116-122`) are not unit length [V] | Review 3 | A shared "no text" flag can outrank text similarity |
| 10 | LoRA text cap of 480 tokens; encoders read 2,000 characters | `step6_dgp.py:91, 242-254` [V] | Review 1 (3 Nov 15:17) | 81% of users are cut. Biased against LoRA in LoRA vs MLP |
| 11 | Early stopping on a 200-user validation sample | `step5_llm_reason.py:42`, `step6_dgp.py:465` [V] | Review 1 | Validation loss disagreed with AUROC in 3 of 4 runs |
| 12 | Training-size confound (1,300 vs 8,278) | Decision log 3 Nov; `control_train_size.py` [V] | 3 Nov | The gap shrinks from 0.055 to about 0.02. The step 6 draw was the *lowest* of 6 |
| 13 | Headline comparisons made on the 400-user test sample | REPORT "Sample noise" [V] | Review 3 | ±0.04 noise. dgp − target is +0.032 on the sample but +0.007 on the full test set |
| 14 | Setting B removes bot–bot edges | `common.py:68-77` (induced subgraph) [V]; 8,162 → 54 rows | Review 2 | B's "no edge signal" and its isolation cue (41.8% vs 27.6%) are built by the subsampling |
| 15 | Isolated-user column unreliable | REPORT table; 87 users [V] | Review 2 | GCN 0.777 vs 0.738 on identical inputs |
| 16 | Hyperparameters untuned until late | `tune_step4.py`; `step4_tuned.json` [V] | 4 Nov 06:04 | Small: MLP 0.766 → 0.768, SAGE 0.750 → 0.755, FAGCN 0.744 → 0.753, GCN 0.707 → 0.706. The findings hold, but this should have been run on day 1 |
| 17 | **Empty tree-baseline logs: silent segfault** | The launcher (`…/9fc4dd83…/scratchpad/cpu_tune_tree.sh`) ran `tree_baseline.py` twice at 06:04 and logged "cpu jobs done" without the exit codes. No `tree_baseline.json` exists. I reproduced it: importing torch + torch_geometric, then fitting `HistGradientBoostingClassifier` on 8,000×130 data, **exits with code 139 (SIGSEGV) and prints nothing**; without torch, or with `OMP_NUM_THREADS=1`, it runs [V]. It is an OpenMP (libomp) clash in the anaconda env. `tree_baseline.py:28` imports torch through `step4_gnn` | Today | The GADBench baseline is missing. The chain hid the failure |
| 18 | Analysis scripts read only the unseeded LoRA scores | `bootstrap_ci.py:39-40`, `link_direction.py:43-44` hard-code `step6_Qwen3-1.7B_{variant}_scores.npz` [V] | – | Seeds 1-2 never enter the CIs or the target + flags check unless the scripts change |
| 19 | Results files overwritten or merged inconsistently | `step3_embed.py:184` overwrites `step3_text_only.json`, so a `SKIP_ENCODERS` run drops rows. `step6_dgp.py:448-453` merges by name, so the shared "same sample" baseline rows are recomputed by every run. The current `git diff` of `step6_lora.json` / `step6_rescore.json` only reorders rows and adds seed rows [V] | – | No numbers lost so far [V], but they are easy to lose |
| 20 | Stale caches keyed only by name | `step3_embed.py:127` (`emb_<name>.npy` exists → skip); `common.py:120` (`splits.npz` exists → reuse); `step6_dgp.py:100` metapath summaries keyed by B only, not K or M [V] | – | Harmless on TwiBot-20. **Dangerous on TwiBot-22**, where prep reruns per subset (section 3) |
| 21 | Metrics and thresholds differ between scripts | Early stopping uses argmax macro-F1 (`step4_gnn.py:159-160`); reporting uses a validation-tuned threshold (`common.py:132-137`). LR = one fit; MLP = mean of 3 seeds; LoRA = 1 seed [V] | – | Comparable within the report, but each table has to say which applies |
| 22 | Many comparisons on one test set | Dozens of pairwise CIs on the same 1,183 users, with no correction or held-out confirmation set [V] | – | Some "significant" gaps will be chance |

Checked and fine [V]: standardisation and PCA are fitted on train only (`step4_gnn.py:288`, `step3_embed.py:147`, `tree_baseline.py:53`); thresholds are chosen on validation; there are no duplicate user ids; only 1 duplicate-text test user; synthetic graphs are labelled as diagnostics.

## 2. Pre-registration checklist for TwiBot-22 (fix before any test number)

**Write these down and commit them before the first test score.**
1. **Population and slicing.** Choose between full 1M + stratified LLM sample with ego-nets (the `graph_slicing_research.md` recommendation) and one sub-community. State which sub-community and why (not "sub6" by default). Sub-communities overlap (8,940 pairwise overlaps, 91,719 union [V]), so they are not independent folds.
2. **Split.** Pick a primary split and a secondary one. Our random stratified 70/20/10 re-split. The official split is also non-random in *composition*: **all 95k verified users are in train** (13.6% of train, 0% of val and test), and median followers are 1,593 in train vs 244 in test [V]. Report the official split only as secondary.
3. **Test set size.** Avoid a 1,000-user test (10% of 10k), which repeats lesson 13. Keep one **untouched confirmation test set**, scored once at the end.
4. **Relations.** Fix the list and keep each relation typed and directed on disk: following, followers, mention, retweet, reply (quote, like, co_list in or out?). Fix what the GNN sees: undirected merged, or typed.
5. **Text.** Fields (bio only vs bio + N tweets), N, ordering ("most recent"), retweets kept or not, character cap, and the LoRA token cap (match the encoder's text or report the gap).
6. **Metadata.** Allowed fields (verified is excluded). A separate "text + profile" track, as on TwiBot-20.
7. **Shortcut slices.** Report each headline also on **non-verified test users** and on **users with a bio**.
8. **Model selection.** One rule for every model: full-validation macro-F1 (or AUROC, as the DGP paper uses), checked at every check, adapter saved at every check, chosen before any test score.
9. **Seeds and runs.** LoRA seeds 1, 2, 3 (no unseeded runs). Tabular and GNN models: 3 seeds. Fix the training-set size for LoRA and give every baseline the same users.
10. **Metrics.** Macro-F1 (validation threshold), AUROC and PR-AUC, all three every time. Paired bootstrap 95% CIs over test users, averaging over seeds, and seed std reported separately.
11. **Hyperparameters.** The `tune_step4.py` grid from day 1, the same budget for every model.

**Data checks to run first (train labels only where a model could see the result):**
- Shortcut rule per field, on the **training split of the split you will use**, with an absolute leaf size instead of 10% (see P2 below).
- **Labelling-function leakage.** Spam-keyword hits in bios and tweets, by class. "Bio empty" alone gives macro-F1 0.682 overall [V]: a zero bio vector is close to a label.
- Duplicate or near-duplicate bios and tweets **across your split**. Overall there are 3,780 duplicate-bio groups, 281 of them pure-bot [V]; sub6 has 16 [V].
- Per relation: share of bots vs humans with ≥1 edge (isolation–label dependence); in-degree and out-degree AUROC **on test users**; out-degree caps and reciprocity (crawl design); whether users have any collected tweets, by class (interaction edges exist only for those users).
- Per relation, on the slice actually used: adjusted homophily, label informativeness, and per-class homophily vs chance. Recompute after slicing; induced subgraphs change them, as in setting B.
- Train–test neighbour leakage: share of test users whose neighbours are train users of the same label, vs chance.

## 3. Review of the new TwiBot-22 scripts

**`twibot22_relations.py`**
- R1 `:112-114`: it assumes `membership` is list → user. If the direction is reversed, `uidx(d)` returns −1 for every row and `co_list` comes out silently empty. Assert both ends resolve for a sample of rows. [S]
- R2 `:160-161`: `B.T @ B` on int8 data. Counts overflow above 127 shared lists, and the product grows with the *square* of each list's size, so one 10k-member list gives 10⁸ pairs (memory). Use int32, cap or drop large lists, or skip co_list. [V code; S on size]
- R3 `:201-208`: degree AUROC is computed on all users, not test users as on TwiBot-20 (`link_direction.py:80`). Add a test-only version. `followers` in-degree has the opposite meaning to `following` in-degree, so label it. [V]
- R4 `:56-57, 110`: only relation names present in the file get rows. Assert that `retweeted`, `quoted`, `replied`, `mentioned`, `like` and `post` are all non-zero, so a renamed relation cannot leave an empty graph. [V]
- R5: there are no out-degree cap, reciprocity or "has tweets by class" checks, so lesson 5 is not yet covered. [V]

**`twibot22_profile_check.py`**
- P1 `:13` vs `:152`: the docstring says 70/10/20, the code uses 80/20, and prep uses 70/20/10 (`prep_twibot22.py:90-91`). The shortcut rule is therefore tested on a different train set from the models'. Import prep's split. [V]
- P2 `:108`: the 10% minimum leaf share. Verified is 9.5% of all users and **4.9-7.4% in sub-communities 0-4** [V], so rule (b) alone misses the known shortcut there; only rule (a) catches it. Use a count (e.g. ≥200 users) or report purity at every size. [V]
- P3 `:111-112`: a fixed 0.5 threshold, unlike `metadata_check.py` (validation threshold). On the official split, verified scores F1 0.227 / AUROC 0.5 only because test has **no** verified users, and the AUROC fallback hides that. Report each field's share in test. [V]
- P4: there is no spam-keyword check, although the dataset's labelling functions used spam keywords. [V]

**`twibot22_tweets.py`**
- T1 `:83-85, 111, 117`: any `tweet_*.json` present is processed and then marked done. A file still being extracted from the zips is read truncated and **never revisited**. Check file completeness (it ends with `]`, the size is stable) before marking it done. [V code; S on how the files arrive]
- T2 `:90, 96-99`: the author regex looks only at the first 200 characters, and decode errors are skipped silently. Count both and log them. [V]
- T3 `:30, 94`: it keeps the first 50 tweets *met*, in file order. Prep's "20 most recent" (`prep_twibot22.py:67-69`) is then the most recent among an arbitrary 50, and it depends on which files had arrived. Keep all tweets for the subset, or select by `created_at`. [V]
- T4 `:47`: wanted users = union of sub-communities only. That does not support the full-graph or ego-net plan. [V]

**`prep_twibot22.py`**
- X1 `:96-102`: outputs go to fixed names in `data/twibot-22/`, with no SUBSET, RELATIONS or N_TWEETS in the path. Combined with the `emb_<name>.npy` and `splits.npz` caches (lesson 20), rerunning prep for another subset makes steps 3-6 **silently reuse stale embeddings and splits for different users**. Write to `data/twibot-22/<subset>_<config-hash>/` and put the hash in `RESULTS_DIR`. [V]
- X2 `:55`: users are sorted by id string. `twibot22_embed_bios.py:33` writes rows in `label.csv` order. Any bridge between them must reindex by `ids.npy`, never by position. [V]
- X3 `:103-104` + `metadata_check.py:41-44` + `step6_dgp.py:266-268`: TwiBot-22 uses `followers`, not `followers_count`, and `created_at` is ISO (`'2020-01-16 02:02:55+00:00'` [V]). The reused parsers use the format `"%a %b %d …"` with `errors="coerce"`, so account age becomes **0 for every user, silently** (or the script crashes with a KeyError). [V]
- X4 `:2`: "steps 2-6 run unchanged" is not true. `common.py:32-34` accepts only instagram and twibot20; `step6_dgp.py:102-105` hard-codes friend and follow; `link_direction.py` and `metadata_check.py` use `in-friend`. [V]
- X5 `:39, 53`: default `sub6` and sub-communities only. That contradicts the slicing report's ranking (sub-communities "secondary only"). Sub6 has 1,120 verified users, all human, who will land in the re-split test set (lesson 2). [V]
- X6 `:67`: sorting by the `created_at` string works only for ISO strings. Parse it with `pd.to_datetime`. [S: tweet format not yet seen]
- X7 `:89-91`: the test set has 1,000 users (lesson 13). [V]
- Good: typed, directed relations are saved (`edge_typed.npz`); the split is documented; profile metadata is kept out of the inputs.

**`twibot22_embed_bios.py`**
- E1 `:33-35, 44-45`: empty bios get a zero vector. With 40.6% of bots vs 7.8% of humans empty [V], the zero vector is a strong label proxy once it is standardised. Report a "bio-empty flag only" baseline and a with-bio slice. [V]
- E2 `:16`: setting `DATASET=twibot20` to reuse step 3 also makes `common` create TwiBot-20 result directories, and it ties TwiBot-22 to TwiBot-20 constants. That is fragile, but harmless today. [V]
- E3: bio only (one 128-token piece), whereas TwiBot-20 averaged bio + tweet pieces. Name it as a separate feature set. [V]

**Launchers.** Log `$?` after *every* job (lesson 17). Set `OMP_NUM_THREADS=1`, or keep torch out of the process, for any sklearn boosting. [V]

## 4. Top 10, in priority order

1. **Namespace prep outputs and caches by subset and config** (X1, lesson 20). Without this, results can be silently wrong.
2. **Pre-register split, slice, relations, text and selection rule** (section 2), and commit it before the first test score.
3. **Score every headline also on non-verified test users and with-bio users** (lesson 2, E1). Run the same check on TwiBot-20 for LR, MLP and GNNs.
4. **Labelling-function leakage check**: spam keywords and empty bio, by class, on train.
5. **Save the LoRA adapter at every check, select by the full-validation rule, use seeds 1-3, no unseeded runs** (lessons 7, 8, 11). Fix `bootstrap_ci.py` and `link_direction.py` to read seeded files (lesson 18).
6. **Fix the metadata parsers for TwiBot-22's field names and ISO dates** (X3) before `metadata_check` or `dgp_num`.
7. **Per-relation crawl checks on test users**: isolation by class, in/out-degree AUROC, caps and reciprocity, has-tweets by class (R3, R5).
8. **Make the tweet extraction safe**: completeness check, counted misses, no first-50 cap (T1-T3).
9. **A larger test set plus an untouched confirmation set** (X7, lesson 22).
10. **Tuning grid and the tree baseline from day 1**, with `OMP_NUM_THREADS=1` and exit codes logged (lessons 16, 17).
