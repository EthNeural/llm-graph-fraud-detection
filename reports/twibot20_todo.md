# TwiBot-20: everything still to do, in one place

Written Thu 5 Nov 2026, ~00:30. It collects every open TwiBot-20 item from `twibot20_lessons_for_twibot22.md` (the 22 lessons), `twibot22_plan.md` (step 8), the TwiBot-22 checklist (H3: profile track), and the external feedback (`artifacts/google_feedback.txt`, reviewed 5 Nov). Items are grouped so they can run **in one go**: one GPU queue, one CPU batch, one round of report edits. ✅ = done, ⬜ = to do.

**When (decided by the builder 5 Nov ~00:40):** TwiBot-22 is finished first (its checklist, models, sealed set), then compared with TwiBot-20's current results, and only **then** is this whole batch run. Target: TwiBot-22 done by Fri 6 Nov evening, this batch Fri night to Sat. If time runs short, GPU jobs run in this order: G1 + G2 (seeds, adapter rule), G4 (matched text), G3 (`dgp_num`). **If time runs out, TwiBot-20's current results are used as they are** (builder, 5 Nov) and compared with TwiBot-22; the report then states the open caveats from this list (adapter rule, text cap, setting B on macro-F1, verified slice for non-LLM models).

## Already done (4 Nov)

- ✅ Tree baseline rerun after the silent crash (`run_tree_baseline.sh`, A and B, exit codes logged), with PR-AUC and a non-verified slice. Lesson 17.
- ✅ Verified-subset check from raw data: target LoRA 0.704 → 0.641 on non-verified test users (`check_verified_subset.py`). Lesson 2, LoRA part.

## Batch 1: GPU queue (~12 h, after TwiBot-22)

Run in one queue, each job logs its exit code and writes to its own named file (lessons 19-20).

| # | Job | Why | Source |
|---|---|---|---|
| G1 | **LoRA seed 2** for `target` and `dgp` (A), seeds 1 and 2 so no headline rests on the unseeded run | Seed noise (0.02-0.03) is larger than every dgp − target gap | lessons 7, 8; plan step 8 |
| G2 | In those runs: **save the adapter at every check** and select by **full-validation macro-F1** (not the 200-user validation loss) | Same selection rule as the MLP; Google feedback #1; lessons 8, 11 | feedback #1 |
| G3 | **`dgp_num`** (DGP with numeric profile summaries, the 5 allowed fields) | The profile track for the LLM, matching TwiBot-22's "expensive models with profile" (H3) | plan step 8; H3 |
| G4 | **Matched-text control**: re-embed TwiBot-20 with the encoders reading exactly LoRA's 480-token strings (Qwen tokenizer), then MLP / LR on them | LoRA reads 480 tokens, encoders up to 2,000 characters; 81% of users are cut for LoRA. Equal text answers "frozen encoder vs fine-tuning" fairly | lesson 10; feedback #2 |

Check before queueing: the step 6 launcher accepts the selection rule and saves every check (`step6_dgp.py:416-418` saved only 2 adapters).

## Batch 2: CPU (can run while the GPU queue runs)

| # | Job | Why | Source |
|---|---|---|---|
| C1 | **Save test scores** for LR, MLP and the GNNs (they are retrained inside `bootstrap_ci.py` and never saved), then score them on **non-verified test users** | Every TwiBot-20 headline must also be shown without the 326 verified (all human) test users | lesson 2; top-10 #3 |
| C2 | **GNNs with profile** (the 5 allowed fields added to the input): GCN, SAGE, FAGCN, LINKX | Match TwiBot-22's design (H3), so the two datasets compare like for like | H3 |
| C3 | `bootstrap_ci.py` and `link_direction.py`: read the **seeded** LoRA score files (seeds 1, 2) and average over seeds | They hard-code the unseeded file | lesson 18 |
| C4 ✅ 6 Nov 04:45: all +0.021 [+0.002, +0.040] F1; non-verified −0.008 [−0.033, +0.018] (gain is from verified users) | **Bootstrap CIs on the tree baseline's crawl-artefact result** (+0.025 on all test users, −0.009 on non-verified) plus a sceptic check | New finding from 4 Nov; must be tested before it goes in the report | 4 Nov tree rerun |
| C5 | Step 5 neighbour ranking: use unit-length vectors without the 2 flags | A shared "no text" flag can outrank text similarity | lesson 9 |

## Batch 3: report edits (`REPORT.md`, after batches 1-2)

| # | Edit | Source |
|---|---|---|
| R1 | Main LoRA tables use the **full-validation adapter rule**, with both adapters shown and a note that the rule was fixed after some test scores were known | feedback #1; lesson 8 |
| R2 | Setting B finding: **lead with AUROC and PR-AUC** (macro-F1 cannot separate models with 54 test bots). On AUROC the gaps are clear and favour text models (GCN − MLP −0.072, FAGCN − MLP −0.141, LoRA dgp − MLP −0.064). Word it as "B cannot separate models on macro-F1", not "no difference" | feedback #3; finding 8 |
| R3 | Every headline also on non-verified test users (from C1) | lesson 2 |
| R4 | Text-cap and training-size caveats next to every "encoder vs LoRA" claim, with the G4 result | lessons 10, 12 |
| R5 | Note: TwiBot-20 used each user's first 20 tweets in file order; TwiBot-22 uses the 20 most recent | TwiBot-22 checklist C5 |
| R6 | Note: many comparisons on one test set and no sealed set on TwiBot-20 (cannot be fixed now; TwiBot-22 has one) | lesson 22 |

## Not doing (and why)

- Raising LoRA's token cap to 2,000 characters: too much GPU memory for a 16 GB Mac; G4 equalises from the other side instead.
- A sealed test set for TwiBot-20 after the fact: every test number has already been seen.
