# Progress Report

A living report, updated as experiments finish. The newest entries are at the top of each
section. For how everything works, see [TEAM_GUIDE.md](TEAM_GUIDE.md).

**Last updated:** 2 November 2026, 21:15

## Status

| Experiment | Instagram A | Instagram B | TwiBot-20 A | TwiBot-20 B |
|---|---|---|---|---|
| Step 1: homophily | ✅ | ✅ | ✅ | ✅ |
| Step 2: baselines | ✅ | ✅ | ✅ | ✅ |
| Step 3: encoders | ✅ 3 encoders | ✅ 3 encoders | ✅ 6 encoders | ✅ 6 encoders |
| Step 4: MLP / GCN / SAGE | ✅ (no MLP rows) | ✅ (no MLP rows) | ✅ | ✅ |
| Step 5: zero-shot reasoner | ✅ | ✅ | ✅ | ✅ |
| Step 6: DGP (LoRA), dgp | ⏳ | ⏳ | 🔄 running | ⏳ |
| Step 6: DGP (LoRA), target | ⏳ | ⏳ | ⏳ next | ⏳ |
| FLAG baseline | ⏳ | ⏳ | ⏳ | ⏳ |

✅ done · 🔄 running · ⏳ to do

**Current run:** TwiBot-20 A, `VARIANT=dgp`.
- Started 2 Nov 20:58; it is writing the 10-token node summaries (8,873).
- Summaries should finish around 23:20. Then come about 9,000 metapath summaries (~40 min), zero-shot scoring, and LoRA training (~2.4 h per epoch, ~3 epochs).
- Expected to need a second night; it resumes from checkpoints.

## Latest results (macro-F1, test set)

### TwiBot-20, encoder ladder (mean of 3 seeds)

| Encoder | A: MLP | A: GCN | A: SAGE | B: MLP | B: GCN | B: SAGE |
|---|---|---|---|---|---|---|
| MiniLM | 0.726 | 0.686 | 0.721 | 0.588 | 0.550 | 0.565 |
| E5-base (English) | 0.760 | 0.700 | 0.740 | 0.564 | 0.589 | 0.567 |
| Twitter-RoBERTa | 0.749 | 0.692 | 0.743 | 0.612 | 0.579 | 0.609 |
| multilingual-E5 | 0.747 | 0.699 | 0.741 | 0.606 | 0.585 | 0.589 |
| Qwen3-Embedding | 0.727 | 0.691 | 0.707 | 0.599 | 0.574 | 0.558 |
| Qwen2.5-1.5B | **0.766** | 0.707 | 0.750 | 0.613 | 0.577 | **0.624** |
| Graph only | – | 0.577 | 0.568 | – | 0.497 | 0.497 |

References for TwiBot-20 A: TF-IDF 0.702; metadata without "verified" 0.755 (not a valid model).

### Instagram (text-only LR vs GNN)

| Model | Setting A | Setting B |
|---|---|---|
| TF-IDF (balanced) | 0.578 | 0.552 |
| MiniLM + LR | 0.588 | 0.546 |
| multilingual-E5 + LR | **0.612** | 0.563 |
| Qwen2.5 + LR | 0.608 | **0.578** |
| MiniLM + GCN | 0.607 | 0.552 |
| multilingual-E5 + SAGE | 0.602 | 0.572 |
| Qwen2.5 + GCN | 0.609 | 0.549 |

### LLM reasoner vs trained models (same stratified test sample)

| Model | TwiBot-20 A | TwiBot-20 B | Instagram A | Instagram B |
|---|---|---|---|---|
| Qwen2.5 + LR (trained) | **0.749** | **0.633** | 0.607 | 0.546 |
| Qwen2.5 + GCN (trained) | 0.698 | 0.573 | **0.623** | **0.562** |
| LLM zero-shot: target | 0.584 | 0.541 | 0.553 | 0.522 |
| LLM zero-shot: raw neighbours | 0.591 | 0.547 | 0.552 | 0.499 |
| LLM zero-shot: DGP summaries | 0.550 | 0.541 | 0.537 | 0.538 |
| Qwen3-1.7B LoRA: target | – | – | – | – |
| Qwen3-1.7B LoRA: dgp | – | – | – | – |

## Findings so far

1. **The encoder matters most.** On TwiBot-20 A, Qwen2.5 + MLP (0.766) beats TF-IDF (0.702) by more than any graph model adds.
2. **GCN hurts under heterophily.** It is below MLP for every encoder on TwiBot-20 A, by 0.04–0.06. SAGE stays within about 0.02 of MLP. In B, Qwen2.5 + SAGE edges past MLP (0.624 vs 0.613), but the gap is within about one standard deviation.
3. **The graph alone is weak:** 0.577 (TwiBot-20 A), and 0.497 in B, near the floor.
4. **The zero-shot LLM reasoner loses to trained classifiers** on every dataset and setting. Adding neighbours (raw or summaries) does not reliably help.
5. **Size is not everything.** Twitter-RoBERTa (125M) beats Qwen3-Embedding (0.6B). Multilingual-E5 is the best text-only encoder on multilingual Instagram A.
6. **Open question for step 6:** does fine-tuning close the gap to Qwen2.5 + LR, and do neighbour summaries help once the model is trained?

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2 Nov | Step 6 summaries generated in batches of 4 on TwiBot-20 | At 16, the GPU memory cache grew to 16 GB and the Mac swapped |
| 2 Nov | Step 6 relations treated as undirected | Kept directed, only 38% of users had a friend link; undirected, 93% have a neighbour (matches steps 4–5) |
| 2 Nov | Step 6 = DGP replication with Qwen3-1.7B, 1,300 training users, B=10, K=2, M=8, r=16, lr 1e-4, one seed, no numeric summaries | Closest to the paper that fits a 16 GB Mac; the official code is not public yet |
| 2 Nov | Diffusion features X = multilingual-E5 embeddings | The paper says "raw node features"; our raw features are text |
| 1 Nov | Added TwiBot-20 (official split, bio + 20 tweets) | A second, larger heterophilous dataset with real text |
| 1 Nov | Excluded TwiBot-20 metadata | "Verified" alone gives macro-F1 0.803: a shortcut, not text or graph signal |
| 1 Nov | Encoder ladder: six encoders, everything else fixed | To separate encoder size and type from the graph's effect |
| 30 Oct | Threshold chosen on validation for every model | Scores are not calibrated; 0.5 is badly placed when class 1 is rare |
| 30 Oct | Settings A (natural) and B (1:10, FLAG-style) | Tests both the data as it is and rare fraud |
| 27 Oct | Dropped Elliptic, rebuilt on Instagram | Elliptic has no natural text for an LLM to read |

## Next steps

1. Finish step 6 on TwiBot-20 A: dgp, then target.
2. Step 6 on TwiBot-20 B, then Instagram A and B.
3. Run the FLAG baseline on our splits (`results/<setting>/flag.json`).
4. Add MLP rows to Instagram step 4 so it matches the TwiBot-20 ladder.
5. Optional, if time allows: 3 seeds for step 6, and an M ∈ {4, 8, 16} sweep.

## Changelog

- **2 Nov 2026:** TwiBot-20 step 5 finished (A and B). Step 6 (DGP replication) written and started on TwiBot-20 A. Repository created.
