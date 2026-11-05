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
