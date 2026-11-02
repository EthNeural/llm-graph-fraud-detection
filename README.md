# LLM + Graph Fraud Detection under Heterophily

How much do large language models help graph-based fraud detection when the graph is
**heterophilous**, that is, when fraudulent accounts mostly connect to normal ones?

This repository is my part of the COMP8851 major project (Macquarie University). The topic is
LLM-based approaches for graph fraud detection, and the supervisor asked for an evaluation
that accounts for heterophily and camouflaged fraud.

It compares three ways of combining an LLM with a graph, all on the same splits and metrics:

| LLM role | What it does | Step |
|---|---|---|
| **Encoder** | The LLM turns each user's text into a vector; a classifier or GNN decides | 3, 4 |
| **Zero-shot reasoner** | The LLM reads a prompt (target + neighbour summaries) and answers Yes/No | 5 |
| **Fine-tuned reasoner** | The same prompts, with the LLM fine-tuned with LoRA (a DGP replication) | 6 |

## Datasets

| | Instagram | TwiBot-20 |
|---|---|---|
| Users | 11,339 | 11,826 labelled |
| Class 1 ("fraud-like") | Commercial accounts, 36.3% | Bots, 55.7% |
| Text | Bio | Bio + up to 20 tweets |
| Edges | 72,005 undirected | friend 11,890 + follow 5,018 |
| Split | Stratified 60/20/20 | Official split |
| Class-insensitive homophily | 0.078 (near random) | 0.104 |

Two settings for each dataset:
- **Setting A:** the natural class ratio and all edges.
- **Setting B:** a FLAG-style 1:10 ratio. All class 0 is kept, plus a random 10% as many class 1 users, on the induced subgraph.

**TwiBot-20 uses text + graph only.** The profile metadata is a shortcut: "verified" alone
reaches macro-F1 0.803, because only 1 of 6,589 bots is verified. It is reported as a reference
line only.

The data is not included in this repository. TwiBot-20 is available from its authors for
research use; `my_build/prep_twibot20.py` converts the official zip into the format used here.

## Pipeline (`my_build/`)

| File | What it does |
|---|---|
| `common.py` | Loading, fixed splits, the shared `evaluate()` and validation-tuned threshold. Settings and datasets are switched with `SETTING=A/B` and `DATASET=instagram/twibot20` |
| `step1_explore.py` | Dataset statistics and four homophily measures |
| `step2_baseline.py` | Majority class and TF-IDF + logistic regression |
| `step3_embed.py` | LLM encoders: MiniLM, E5-base (English), Twitter-RoBERTa, multilingual-E5, Qwen3-Embedding, Qwen2.5-1.5B |
| `step4_gnn.py` | MLP (text-only control), GCN and GraphSAGE on each encoder, plus graph-only; 3 seeds |
| `step5_llm_reason.py` | Zero-shot Qwen2.5-1.5B-Instruct reasoner: target-only, raw neighbours, and DGP-style neighbour summaries |
| `step6_dgp.py` | DGP replication: typed metapaths, Markov-diffusion neighbour trimming, bi-level 10-token summaries, LoRA fine-tuning of Qwen3-1.7B |
| `compare.py` | Every result for both settings in one table |
| `study_notes.html` | Study notes covering every concept in the pipeline |

Evaluation rules, the same in every step:
- The same saved splits.
- Thresholds chosen on validation for the best macro-F1.
- Test scored once.
- Metrics reported: macro-F1 (main), class-1 F1, AUROC and PR-AUC, with connected and isolated test users reported separately.

## Results so far (macro-F1, test set)

**TwiBot-20, Setting A** (text-only MLP vs adding the graph, mean of 3 seeds)

| Encoder | Text only (MLP) | + GCN | + SAGE |
|---|---|---|---|
| MiniLM | 0.726 | 0.686 | 0.721 |
| E5-base (English) | 0.760 | 0.700 | 0.740 |
| Twitter-RoBERTa | 0.749 | 0.692 | 0.743 |
| multilingual-E5 | 0.747 | 0.699 | 0.741 |
| Qwen3-Embedding | 0.727 | 0.691 | 0.707 |
| Qwen2.5-1.5B | **0.766** | 0.707 | 0.750 |
| Graph only | | 0.577 | 0.568 |

**LLM reasoner vs trained models** (same stratified test sample)

| Model | TwiBot-20 A | TwiBot-20 B | Instagram A | Instagram B |
|---|---|---|---|---|
| Qwen2.5 + LR (trained) | **0.749** | **0.633** | **0.607** | **0.546** |
| Qwen2.5 + GCN (trained) | 0.698 | 0.573 | 0.623 | 0.562 |
| LLM zero-shot, target only | 0.584 | 0.541 | 0.553 | 0.522 |
| LLM zero-shot, DGP summaries | 0.550 | 0.541 | 0.537 | 0.538 |

The step 6 fine-tuned rows will be added when those runs finish.

### Findings

1. **A better text encoder is the biggest gain.** Going from TF-IDF (0.70) to Qwen2.5 + MLP (0.766) on TwiBot-20 A is worth more than anything the graph adds.
2. **GCN hurts under heterophily.** It is below the text-only MLP for every encoder on TwiBot-20 A. GraphSAGE, which keeps a node's own features separate from its neighbours', loses much less.
3. **The graph alone is weak:** 0.577 on TwiBot-20 A, and about 0.50 in Setting B.
4. **A zero-shot LLM reasoner loses to simple trained classifiers.** Adding neighbour summaries does not help when most neighbours belong to the other class.
5. **Size is not everything.** The 125M tweet-trained Twitter-RoBERTa beats the 0.6B Qwen3-Embedding.

## Running it

```bash
pip install -r my_build/requirements.txt
python my_build/step1_explore.py
SETTING=A python my_build/step2_baseline.py            # also step3, step4, step5
DATASET=twibot20 SETTING=B python my_build/step4_gnn.py
caffeinate -ims env DATASET=twibot20 SETTING=A VARIANT=dgp python -u my_build/step6_dgp.py
python my_build/compare.py
```

Everything was run on a MacBook Pro (Apple M4, 16 GB) using the `mps` GPU backend. The slow
parts (embeddings, LLM summaries, LoRA checkpoints) are cached and resume after an
interruption. Cached embeddings (~400 MB) and LoRA checkpoints are not in the repository;
`step3_embed.py` regenerates the embeddings.

## References

- Li, Hu, Hooi, He, Chen. *DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs.* AAAI 2026. [arXiv:2507.21653](https://arxiv.org/abs/2507.21653)
- Yang et al. *FLAG: Fraud Detection with LLM-enhanced Graph Neural Network.* KDD 2025.
- Lim et al. *Large Scale Learning on Non-Homophilous Graphs.* NeurIPS 2021 (class-insensitive homophily).
- Feng et al. *TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark.* CIKM 2021.

Author: Prajan Bhattarai
