# Overnight notes, Thu 5 Nov 2026 (external)

- 02:20 Step 6 smoke test started (DATASET=twibot22, VARIANT=dgp, SMOKE=1: 20 train, 50 val, 20 test users).
- 02:24 Finding: LLM node summaries run at ~1 s each (316 s per chunk of 320 prompts). The full `dgp` run would need summaries for tens of thousands of neighbours (8,300 targets × 3 paths × up to 8) plus ~25k path summaries: an estimated 20-30 h. Not possible tonight.
- Plan agreed for tonight: if the smoke test passes, run LoRA `target` only (no summaries needed). Validation checks on 2,000 users unless the smoke test shows one check would take > 10 min, then 1,000. `dgp` waits for the builder's decision on a smaller scope. No GNNs during LoRA.
