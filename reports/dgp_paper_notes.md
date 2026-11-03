# DGP paper: key specifications (from arXiv 2507.21653v1, read 4 Nov 2026)

Li, Hu, Hooi, He, Chen. "DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs". AAAI 2026, arXiv 2507.21653. Code: not public (repo README says pending industry approval).

## Method
- Heterogeneous graph; node features = text + numerical/categorical (§3.1).
- Metapaths P = r1∘…∘rL, A_P = A_r1 … A_rL, N_P(v) = {u : (A_P)_vu > 0} (eq. 2–4).
- **Node summary** (eq. 5): task-agnostic "Summarize the text within B tokens", frozen LLM, budget B_node.
- **Diffusion trimming (MDK)** (eq. 6–9): T_P = D_P⁻¹ A_P, Z_P(K) = (1/K) Σ_{k=0..K} T_P^k, h = Z_P(K) X with X = raw node features; keep top-M neighbours by ‖h_u − h_v‖₂.
- **Metapath summary** (eq. 10): summarise the concatenated node summaries of the kept neighbours, budget B_meta.
- **Numerical summary** (eq. 11, §4.3): a_P(v) = mean of the kept neighbours' numerical/categorical features along each metapath.
- **Prompt** (eq. 12): target's full text ⊕ for each metapath (S_P(v) ⊕ a_P(v)).
- **Fine-tuning** (eq. 13): cross-entropy on the first generated token (Yes/No); LoRA on all attention layers; AdamW.
- **Score** (eq. 14): softmax over logit_Yes, logit_No.
- Task-aware summary prompts HURT (Table 3); best budget about 10 tokens (Fig. 5).

## Setup (§5.1)
- Datasets: YelpReviews, AmazonVideo (public), E-Commerce, LifeService (ByteDance, private). Train/val/test ≈ 1,300 / 1,300 / large test (Table 1).
- Baselines: MLP, GraphSAGE, HGT, ConsisGAD, PMP, GAAP; Qwen3-8B fine-tuned on the target node alone ("LLM"); TAPE; GraphGPT, HiGPT, InstructGLM. Official code for all.
- **Grid search** on validation: B_node, B_meta ∈ {10, 20, 40, 80}; K ∈ {1, 2, 3}; M ∈ {2, 4, 8, 16}; LoRA rank ∈ {4, 8, 16, 32}; LoRA dropout ∈ {0, 0.05, 0.1}; lr ∈ {1e-5, 3e-5, 1e-4}. Batch 4, up to 10 epochs, early stopping on validation loss. **All hyperparameters chosen to maximise validation AUROC.**
- Qwen3-8B, 4× A100 80 GB. Macro-F1, AUROC, AUPRC; **mean ± std over 5 seeds**.

## Key results (Table 2, %)
| Dataset | MLP F1 / AUROC | LLM (target only) F1 / AUROC | DGP F1 / AUROC |
|---|---|---|---|
| Yelp | 62.09 / 75.00 | 60.79 / 71.18 | 69.07 / 84.28 |
| Amazon | 61.74 / 70.47 | 59.90 / 71.78 | 66.91 / 77.32 |
| E-Commerce | 65.21 / 71.02 | 63.87 / 67.97 | 75.01 / 82.74 |
| LifeService | 87.98 / 95.49 | 89.19 / 96.40 | 93.73 / 98.04 |

Paper's own observations: GNNs generally beat MLP; standalone fine-tuned LLMs generally underperform MLP; DGP beats all. Ablation (Fig. 4): removing TextSumm, NumSumm, MDK or PathSumm each lowers performance; text matters more than numbers.
