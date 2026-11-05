# Overnight notes, Thu 5 Nov 2026 (external)

- 02:20 Step 6 smoke test started (DATASET=twibot22, VARIANT=dgp, SMOKE=1: 20 train, 50 val, 20 test users).
- 02:24 Finding: LLM node summaries run at ~1 s each (316 s per chunk of 320 prompts). The full `dgp` run would need summaries for tens of thousands of neighbours (8,300 targets × 3 paths × up to 8) plus ~25k path summaries: an estimated 20-30 h. Not possible tonight.
- Plan agreed for tonight: if the smoke test passes, run LoRA `target` only (no summaries needed). Validation checks on 2,000 users unless the smoke test shows one check would take > 10 min, then 1,000. `dgp` waits for the builder's decision on a smaller scope. No GNNs during LoRA.
- 02:41 Smoke test PASSED. Timings:
```
SMOKE TEST PASSED (end to end). Timings:
  node_summaries    20.28 min for 1,300 items (0.94 s each)
  path_summaries     1.69 min for 219 items (0.46 s each)
  fine_tune          2.05 min for 20 items (6.16 s each)
  test_scoring       0.25 min for 20 items (0.75 s each)
  val check         0.59 min for 50 users -> 23.6 min for 2,000 (limit 10 min: OVER, use 1,000 users or fewer checks)
```
- 02:41 Starting LoRA target (VAL_CHECK=1000).
- 02:42 Note on the validation-check rule: a 2,000-user check was measured at 23.6 min (> 10 min), so the builder's remedy was applied: 1,000 users. Extrapolated, a 1,000-user check takes ~11.8 min, just above 10 min; kept at 1,000 because that is the remedy the builder named. Checks run twice per epoch. Smoke numbers for planning the rest: training ~9.5 s per step of 4 users (~51 min per epoch of 1,300 users), scoring ~0.75 s per user (5,000 test users ~1 h), node summaries 0.94 s each, path summaries 0.46 s each. Estimated LoRA target total: ~4-6 h, done around 07:00-09:00 (estimate).
- 06:18 LoRA target finished, exit code 0. Last lines:
```
  {'check': 2, 'epoch': 1.4984615384615385, 'val_loss': 0.732, 'val_macro_f1': 0.6217, 'val_auroc': np.float64(0.704), 'check_minutes': 9.14}
  {'check': 3, 'epoch': 2.0, 'val_loss': 0.5291, 'val_macro_f1': 0.6162, 'val_auroc': np.float64(0.6669), 'check_minutes': 9.33}
  {'check': 4, 'epoch': 2.4984615384615383, 'val_loss': 0.749, 'val_macro_f1': 0.6196, 'val_auroc': np.float64(0.6773), 'check_minutes': 9.28}

                                                      

                                                              
Qwen3-1.7B LoRA target (full test) acc 0.795  macroF1 0.626  F1_1 0.374  AUROC 0.710  PR-AUC 0.312  | conn 0.625  iso 0.620

Same test sample as step 5:
Qwen3-1.7B LoRA target acc 0.797  macroF1 0.621  F1_1 0.362  AUROC 0.743  PR-AUC 0.341  | conn 0.617  iso 0.617
Qwen3-1.7B zero-shot target acc 0.810  macroF1 0.484  F1_1 0.073  AUROC 0.392  PR-AUC 0.118  | conn 0.492  iso 0.450

Saved -> /Users/pj/Desktop/Macquarie - Sem 4/PROJECT/Graph-based-fraud-detection-system/my_build/results/twibot22/A/step6_lora.json
=== exit code 0
```
- 13:14 Step 1 text baselines (TF-IDF+LR, MiniLM+MLP) exit 0.
```
LoRA target (rescored here, cut-off on its 1,000 val users): {'n': 5000, 'bots': 700, 'macro_f1': 0.6258268349978193, 'auroc': 0.7097636212624584, 'pr_auc': 0.3117658616534659, 'pr_auc_at_5pct_bots': 0.1355174285272691, 'pr_auc_at_1pct_bots': 0.03046781241349008}
tfidf_lr_full70k {'n': 5000, 'bots': 700, 'macro_f1': 0.6277243150773428, 'auroc': 0.7270598006644519, 'pr_auc': 0.3073773846262937, 'pr_auc_at_5pct_bots': 0.12980747527796996, 'pr_auc_at_1pct_bots': 0.02842308125742991, 'minus_lora_target': {'macro_f1': [0.0019872451382913375, -0.015143776910713867
minilm_mlp_full70k seed 0 {'n': 5000, 'bots': 700, 'macro_f1': 0.6031720237640975, 'auroc': 0.6809059800664452, 'pr_auc': 0.2934477378951583, 'pr_auc_at_5pct_bots': 0.12741901585936968, 'pr_auc_at_1pct_bots': 0.02970204458157622, 'minus_lora_target': {'macro_f1': [-0.022378653266130984, -0.041911960
minilm_mlp_full70k seed 1 {'n': 5000, 'bots': 700, 'macro_f1': 0.6109970612124651, 'auroc': 0.6813441860465117, 'pr_auc': 0.2829647125808818, 'pr_auc_at_5pct_bots': 0.11877874499942107, 'pr_auc_at_1pct_bots': 0.02698765283912572, 'minus_lora_target': {'macro_f1': [-0.01487286365178505, -0.0342863075
minilm_mlp_full70k seed 2 {'n': 5000, 'bots': 700, 'macro_f1': 0.6053591516146144, 'auroc': 0.6888003322259136, 'pr_auc': 0.2960759059142913, 'pr_auc_at_5pct_bots': 0.12879045136999392, 'pr_auc_at_1pct_bots': 0.031110635826162577, 'minus_lora_target': {'macro_f1': [-0.02030092577907114, -0.040413601
tfidf_lr_matched1300 {'n': 5000, 'bots': 700, 'macro_f1': 0.575765747047815, 'auroc': 0.6401501661129569, 'pr_auc': 0.24264754711067998, 'pr_auc_at_5pct_bots': 0.09704941491671179, 'pr_auc_at_1pct_bots': 0.02062665496190425, 'minus_lora_target': {'macro_f1': [-0.04998096775492642, -0.066778336624091
minilm_mlp_matched1300 seed 0 {'n': 5000, 'bots': 700, 'macro_f1': 0.5765818013683321, 'auroc': 0.6097279069767442, 'pr_auc': 0.2345723383104145, 'pr_auc_at_5pct_bots': 0.09487997780513902, 'pr_auc_at_1pct_bots': 0.020384570398131886, 'minus_lora_target': {'macro_f1': [-0.0494326581914654, -0.069605
minilm_mlp_matched1300 seed 1 {'n': 5000, 'bots': 700, 'macro_f1': 0.580203269995581, 'auroc': 0.6146441860465116, 'pr_auc': 0.2337058576688688, 'pr_auc_at_5pct_bots': 0.09374388210797924, 'pr_auc_at_1pct_bots': 0.020003088637714796, 'minus_lora_target': {'macro_f1': [-0.045652491024263726, -0.06441
minilm_mlp_matched1300 seed 2 {'n': 5000, 'bots': 700, 'macro_f1': 0.5653162281911631, 'auroc': 0.5840106312292359, 'pr_auc': 0.21650790163396474, 'pr_auc_at_5pct_bots': 0.08713150412001633, 'pr_auc_at_1pct_bots': 0.021664731098819928, 'minus_lora_target': {'macro_f1': [-0.06043881552551307, -0.0790
Saved /Users/pj/Desktop/Macquarie - Sem 4/PROJECT/Graph-based-fraud-detection-system/my_build/results/twibot22/text_baselines.json
=== exit code 0
```
- 13:22 Builder decision: experts used for training by 5-fold rotation (G7, B8). LoRA rotation queued after DGP (blind).
- 13:47 Builder decision: DGP at TwiBot-20 scope (1,300 train, 1,000 val, 1,500 stratified test subset, seed 42, committed before scoring); dgp vs target on that subset. Reason: summaries ~1 s each (5.6 h vs 8.6 h). Queue: zero-shot -> rotation smoke -> rotation -> DGP -> target s1 -> dgp s1 (until ~13:00 Fri).
- 14:00 Zero-shot on 5,000 test users: done (readout checked: not inverted).
```
zero-shot target, 5,000 working-test users: {'n': 5000, 'bots': 700, 'macro_f1': 0.504552682080655, 'auroc': 0.4612513289036545, 'pr_auc': 0.13935416785982876, 'pr_auc_at_5pct_bots': 0.04986199351296926, 'pr_auc_at_1pct_bots': 0.009982694280251808}
same with 1 - p (sign check only, not a result): {'n': 5000, 'bots': 700, 'macro_f1': 0.5531060914221793, 'auroc': 0.5387486710963454, 'pr_auc': 0.19786860249547744, 'pr_auc_at_5pct_bots': 0.07743622133900813, 'pr_auc_at_1pct_bots': 0.016326891613152955}
Saved /Users/pj/Desktop/Macquarie - Sem 4/PROJECT/Graph-based-fraud-detection-system/my_build/results/twibot22/A/step6_Qwen3-1.7B_target_zeroshot_full.npz
=== exit status (from log) 0
```
- 14:01 Rotation smoke test: PASSED.
```
                                                                 

                                                            
fold 0 saved (no metric printed: blind)
ROTATION SMOKE TEST PASSED (1.0 min for 20 train + 10 held out)
=== exit code 0
```
- 14:01 Expert rotation (LoRA target, 5 folds, blind) started.
- 17:14 Expert rotation finished, exit 0 (blind: scores saved, no metric looked at).
```
                                                              
fold 4 saved (no metric printed: blind)
Saved /Users/pj/Desktop/Macquarie - Sem 4/PROJECT/Graph-based-fraud-detection-system/my_build/results/twibot22/A/expert_rotation_target.npz (blind: read in the final run)
=== exit code 0
```
- 17:14 DGP started (TwiBot-20 scope: 1,300 train, 1,000 val, 1,500 test subset). Est. ~9.5 h (5.6 h summaries).
- 17:15 GNNs started (DGP summary phase: no LoRA training).
- 18:17 GNN seed 0 done; 2 best by validation macro-F1: FAGCN,RGCN -> seeds 1, 2.
- 18:21 Check: bio-only tree on the SAME sample users (70k train / 5,000 test): text-only F1 0.673 AUROC 0.761 vs bio + tweets 0.618 / 0.712; with profile 0.712 / 0.846 vs 0.713 / 0.848. Averaging tweets in hides the empty-bio cue; with profile they are equal. To be checked by the sceptic before the report.
- 18:28 CIs for key pairs done; sceptic check done (verdicts: profile-vs-LoRA needed a matched 1,300 run -> profile 0.677 / 0.808, still ahead; empty-bio mechanism confirmed by slices). Findings in report_for_submission.md.
- 19:57 GNNs all done.
- 20:38 GNN expert rotation (MLP, RGCN, FAGCN; blind) started.
- 19:45–22:45 E11 leaderboard reconciliation (REPLICATION=1): official split with/without `verified` done for trees and MLP / FAGCN / RGCN (RGCN without `verified` running); our split with `verified` running. Exit 0 for each finished cell.
- 21:21 Q3 full 1M (XGB-Graph + profile + inconsistency features, seed 0) done, exit 0: 0.729 / 0.872 / PR-AUC 0.579; paired CI vs without: F1 −0.002 [−0.004, −0.000], AUROC +0.000 (`q3_ci_full.json`). Q3 adds nothing.
- 22:43 GNN expert rotation: MLP done (13 min), RGCN done (112 min), saved blind; FAGCN running.
- 22:45 DGP node summaries 49 / 57 (~7 min per chunk, CPU shared with 3 jobs).
- 23:50 Builder's decision: expert rotation scores read early (decision line committed first, 851a08b). Read: 986 experts only, plus the TwiBot-22 annotation study without its 21 sealed non-expert users. Script `twibot22_expert_read.py`, output `expert_read_early.json` / `.log` (84d8da2). FAGCN left out (still running).
  - Expert stage vs weak-only (AUROC): trees +0.006 to +0.027, MLP +0.025, RGCN +0.067 [+0.034, +0.104], LoRA −0.099 [−0.164, −0.038] (its macro-F1 +0.040). Best after the expert stage: XGB-Graph + profile (sample) 0.751 / 0.910 / 0.599. The dataset's weak label itself: macro-F1 0.745 on the experts.
  - D8: lift on the experts, weak 8.74 [6.39, 12.33] vs expert 7.22 [5.31, 10.30] (difference +1.52 [−1.50, +4.64]); on the annotation study (458 users, 44 bots), weak 5.01 [3.30, 7.67] vs annotation 1.64 [0.70, 3.06] (difference +3.36 [+1.20, +6.14]).
  - Paper (arXiv 2206.04564v6, `research documents/`): the 1,000 experts are 17 researchers of the dataset's own group, and their labels (800 of them) trained the label-making classifiers, so expert and weak labels are not independent. The annotation study (6 researchers, most of whom have published on bot detection; 3 per user; whether they are from the same group is not stated) never trained the label-makers: the independent check.
- 00:40 Reconciliation complete (E11, all cells, exit 0): RGCN our split + verified 0.678 / 0.797.
- 00:46 Reconciliation complete (last cell RGCN, our split, verified in); brief section 7 filled automatically.
- 00:58 GNN expert rotation finished, exit 0 (blind).
- 01:08 Expert read repeated with FAGCN (expert_read_with_fagcn.json):
```
FAGCN (1M)                       expert    0.734 [0.685, 0.780]    0.841 [0.800, 0.881]    0.537 [0.441, 0.631]   
                                 diff      +0.013 [-0.011, +0.036]  -0.005 [-0.017, +0.005]  +0.012 [-0.009, +0.035] 
```
- 01:19 Exploratory pilot step 0 (oracle neighbour-trust switch), exit 0: switch − XGB-Graph AUROC +0.0030 [+0.0021, +0.0040] (A, <0.5, 8.8% switched), +0.0032 [+0.0023, +0.0042] (B, <0.25); macro-F1 +0.005–0.006; PR-AUC +0.005–0.006. Below the +0.005 AUROC go/no-go: full pilot NOT built (rule fixed at 01:05).
- 04:11 TwiBot-20 C4 (crawl-artefact CIs, tree baseline, 3 seeds averaged, exit 0): + in/out degrees vs own+neighbour+degree: all test macro-F1 +0.021 [+0.002, +0.040], AUROC +0.031 [+0.019, +0.043]; NON-VERIFIED macro-F1 -0.008 [-0.033, +0.018], AUROC +0.009 [-0.007, +0.026]. The artefact's gain comes from the verified (all-human) users. (c4_crawl_artefact_ci.json)
- 05:00 TwiBot-20 C1 + C3 done, exit 0 (A and B; `c1_nonverified.json`, scores in `c1_scores/`). A, non-verified test users (857): MLP 0.687 / 0.791; GCN − MLP F1 −0.052 [−0.080, −0.025] (holds without verified users); SAGE − MLP −0.004 [−0.024, +0.016]; FAGCN − MLP −0.012 [−0.033, +0.009]. LoRA (C3: seeds unseeded + s1, adapter chosen by full-validation macro-F1): target 0.732 / 0.814 all, dgp 0.736 / 0.821 all; **dgp − target non-verified AUROC +0.033 [+0.014, +0.051]** (F1 +0.014 [−0.008, +0.036]); dgp − MLP(1,300) F1 +0.010 [−0.011, +0.032]. B (54 test bots): all CIs wide; FAGCN − MLP F1 −0.059 [−0.112, −0.002].
- 05:25 TwiBot-20 C2 done, exit 0 (GNNs with the 5 allowed profile fields + Qwen2.5 text, 3 seeds; c2_gnn_profile.json). A: MLP 0.808 / 0.885 (non-verified 0.702 / 0.798); GNN − MLP macro-F1: GCN −0.069 [−0.093, −0.045], SAGE −0.012 [−0.029, +0.005], FAGCN −0.035 [−0.053, −0.015], LINKX −0.009 [−0.025, +0.007]; every GNN below the MLP on AUROC (−0.018 to −0.078). Same on non-verified users. B: no GNN above MLP; AUROC gaps −0.03 to −0.10. Like-for-like with TwiBot-22: message passing adds nothing with profile either.
- 04:44 DGP finished, exit 0.
```

                                                      

                                                            
Qwen3-1.7B LoRA dgp (test subset 1,500) acc 0.809  macroF1 0.628  F1_1 0.369  AUROC 0.717  PR-AUC 0.346  | conn 0.636  iso 0.593

Same test sample as step 5:
Qwen3-1.7B LoRA dgp acc 0.809  macroF1 0.628  F1_1 0.369  AUROC 0.717  PR-AUC 0.346  | conn 0.636  iso 0.593
Qwen3-1.7B zero-shot dgp acc 0.793  macroF1 0.530  F1_1 0.179  AUROC 0.558  PR-AUC 0.182  | conn 0.515  iso 0.532

Saved -> /Users/pj/Desktop/Macquarie - Sem 4/PROJECT/Graph-based-fraud-detection-system/my_build/results/twibot22/A/step6_lora.json
=== exit code 0
```
- 04:44 dgp vs target on the 1,500 subset:
```
target {'macro_f1': 0.632, 'auroc': 0.718, 'pr_auc': 0.346, 'pr_auc_at_5pct_bots': 0.162, 'pr_auc_at_1pct_bots': 0.039}
dgp {'macro_f1': 0.628, 'auroc': 0.717, 'pr_auc': 0.346, 'pr_auc_at_5pct_bots': 0.173, 'pr_auc_at_1pct_bots': 0.064}
dgp - target {'macro_f1': [-0.004065591283265327, -0.02657503243871165, 0.01788914041000687], 'auroc': [-0.0012136139822521192, -0.03139959867318264, 0.02795168622629283]}
Saved dgp_vs_target.json
```
- 04:44 LoRA target seed 1 started (seed noise was 0.02-0.03 on TwiBot-20, larger than dgp - target gaps).
- 08:07 target seed 1 finished, exit 0. dgp seed 1 started.
- 10:13 dgp seed 1 finished, exit 0. Comparison (both seeds):
```
target {'macro_f1': 0.632, 'auroc': 0.718, 'pr_auc': 0.346, 'pr_auc_at_5pct_bots': 0.162, 'pr_auc_at_1pct_bots': 0.039}
dgp {'macro_f1': 0.628, 'auroc': 0.717, 'pr_auc': 0.346, 'pr_auc_at_5pct_bots': 0.173, 'pr_auc_at_1pct_bots': 0.064}
dgp - target {'macro_f1': [-0.004065591283265327, -0.02657503243871165, 0.01788914041000687], 'auroc': [-0.0012136139822521192, -0.03139959867318264, 0.02795168622629283]}
target_s1 {'macro_f1': 0.603, 'auroc': 0.681, 'pr_auc': 0.318, 'pr_auc_at_5pct_bots': 0.147, 'pr_auc_at_1pct_bots': 0.035}
dgp_s1 {'macro_f1': 0.607, 'auroc': 0.711, 'pr_auc': 0.34, 'pr_auc_at_5pct_bots': 0.173, 'pr_auc_at_1pct_bots': 0.07}
dgp - target_s1 {'macro_f1': [0.00480505930740957, -0.021281293668412575, 0.030982996624176386], 'auroc': [0.030273526254619158, -0.00013424285555028995, 0.05838599714281657]}
Saved dgp_vs_target.json
```
