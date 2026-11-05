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
