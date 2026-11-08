# Independent study: decision log (rules before scores)

| Time (6 Nov) | Decision | Reason |
|---|---|---|
| 02:06 | Dataset BotSim-24 (exact labels, LLM-written bots, Reddit reply graph); fox8-23 as a real-world second set if its download completes | Tests the main study's open problems: weak labels, LLM-written bots, graph artefacts |
| 02:20 | Leaks excluded from every model: post score / upvote_ratio / num_comments, comment_score, ids, description presence and text, character_setting | Each is 100% one class for bots (simulation artefacts) |
| 02:35 | **Graph variants, all reported:** (G-full) both directions, as in the paper; (G-out) out-edges only = whom a user replies to; (G-in-only) in-degree alone as a "leak probe". The headline graph result is G-out; G-full minus G-out measures what the in-edge artefact adds | In-edges encode the simulation's missing human→bot replies (AUROC 0.871 untrained) |
| 02:35 | **Models:** clean profile/timing tree; text: TF-IDF + LR, multilingual-E5 / MiniLM embeddings + LR/MLP, zero-shot LLM-text detector (Binoculars-style with the local Qwen2.5-1.5B / -Instruct pair), LoRA Qwen3-1.7B; graph: neighbour-feature tree (XGB-Graph style), GCN / SAGE / FAGCN / RGCN (3 edge types) vs the MLP with the same inputs, on G-out and G-full; fusion: own features + text + G-out neighbour features | One model per role named in the project brief (encoder, reasoner/detector, fine-tuned LLM, graph, fusion) |
| 02:35 | **Protocol:** split fixed at 02:30 (train 1,744 / val 581 / test_work 291 / test_sealed 291); cut-off on validation; macro-F1, AUROC, PR-AUC; paired bootstrap 2,000; 3 seeds for trained models; sealed part read once at the end; GPU work only after the main GPU queue; nothing written into the main report | Same rules as the main study |
| 02:35 | **Primary questions:** (Q1) with exact labels and leaks removed, which signal carries detection: profile/timing, text, or graph? (Q2) how much of the reported GNN advantage is the in-edge artefact (G-full − G-out)? (Q3) can an LLM detect LLM-written bots zero-shot, and does fine-tuning help? (Q4) does fusing text with G-out neighbours beat the best single source? | Mirrors RQ1–RQ5 of the main study under clean labels |

## Stage 2 (decided Sat 7 Nov ~15:45, before any score): zero-shot machine-text detection
- Question: without any labels, do published zero-shot detectors flag BotSim-24's GPT-4o-mini bots? (Stage 1: supervised TF-IDF = 1.000, so the open question is the label-free setting a new bot generator would create.)
- Scores per comment (comments only; posts are human news headlines = genre confound), first 128 tokens, comments < 5 tokens skipped:
  - Binoculars (Hans et al., ICML 2024): observer Qwen2.5-1.5B, performer Qwen2.5-1.5B-Instruct (shared tokenizer); B = log PPL(performer) / log X-PPL(observer->performer); LOW = machine.
  - Fast-DetectGPT analytic (Bao et al., ICLR 2024): sampling = observer, scoring = performer; d per comment (sum form); HIGH = machine.
  - Plain log-perplexity under the performer (LOW = machine) as the simple baseline.
- User score = mean over the user's comments. Untrained AUROC (sign fixed in advance as above) on working-test users (291), and on train+val users as a larger check. Comment-level AUROC on working-test comments.
- One trained row: LR on [Binoculars, Fast-DetectGPT, log-PPL, focal-word rate] fitted on train users, cut-off on val, scored on working test.
- Sealed BotSim users are never scored (skipped before any model call).
