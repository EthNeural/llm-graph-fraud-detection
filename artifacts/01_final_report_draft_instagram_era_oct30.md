# COMP8851 Final Report – LLM-Based Fraud Detection on Heterophilous Graphs

Oct 30, 2026 · @Prajan

**Title of the Project:** LLM-Based Fraud Detection on Heterophilous Graphs: Detecting Camouflaged Commercial Accounts on Instagram

**Supervisor:** Dr. Venus Haghighi

**Project Team Members:** Prajan Bhattarai \[Student ID\], \[Name\] \[Student ID\], \[Name\] \[Student ID\], \[Name\] \[Student ID\]

## 1. Introduction

Our main finding is that, on a network where fraud-like accounts hide among normal ones, a large language model reading each account's own text is the strongest signal we found. Adding the network structure on top did not help, and with a standard graph model it hurt the detection of the minority class.

**The problem.** Online platforms must spot accounts that behave differently from ordinary users: fraudsters, bots or undisclosed commercial accounts. Two sources of evidence are usually available. The first is *text*, such as a profile description. The second is *connections*: who follows, pays or replies to whom. Graph Neural Networks (GNNs) are the standard tool for learning from connections. They work by letting each account borrow information from its neighbours.

**Why it matters.** That borrowing assumes neighbours are alike, a property called *homophily*. Fraudsters deliberately break it. They connect to many normal accounts so they look normal, a tactic called *camouflage*, which makes their neighbourhoods *heterophilous* (mostly unlike them). A GNN then pulls a fraudster's profile towards "normal", which is exactly the wrong direction. Large language models (LLMs) offer a way around this: they can read the text attached to each account and judge it on its own merits.

**What we set out to do.** The project brief asked how LLM reasoning and semantics can be combined with graph structure for fraud detection. Because fraudsters camouflage themselves by linking to benign users (Dou et al., 2020), any such combination has to be evaluated under heterophily. We therefore asked three questions:

1. Do LLM text representations detect fraud-like accounts better than a simple word-count baseline?
2. Does adding the graph, through a GNN, improve on text alone when the graph is heterophilous?
3. Can an LLM reason directly over an account and a summary of its neighbours (dual-granularity prompting), and does summarising neighbours protect it from camouflage?

**How we did it.** We used a public Instagram network of 11,339 users with profile bios and follow links. Following FLAG (Yang et al., 2025), commercial accounts stand in for the fraud class. We first measured how heterophilous the network is. We then built one reproducible pipeline that tests the LLM in two roles: as an *encoder* that turns bios into vectors for a classifier or GNN, and as a *reasoner* that reads a prompt and decides. Every model was trained and tested on identical data splits, in two settings: the natural class ratio (36% commercial) and FLAG's 1:10 ratio (9% commercial), which mimics how rare real fraud is.

**Expected versus achieved.** Our Assessment 2 plan used the Elliptic Bitcoin dataset and expected LLM-derived text to improve a graph model. We changed dataset (Section 3.4): Elliptic has no real text, and an evaluation on it could not address heterophily. On Instagram, the expected gain from the graph did not appear. A good LLM encoder improved on the baseline in both settings (macro-F1 0.612 against 0.578, and 0.578 against 0.552). GNNs added nothing in the natural setting and reduced commercial-class F1 from 0.238 to 0.186 in the 1:10 setting. We consider this a useful, if unflattering, result. It shows *when* graph structure stops helping, and why heterophily-aware designs matter.

## 2. Literature Review

Three lines of research meet in this project: graph-based fraud detection, learning on heterophilous graphs, and the use of LLMs on text-attributed graphs. The gap we address is that LLM+GNN fraud methods rarely separate how much of their gain comes from text versus structure, and are rarely evaluated with threshold-independent, minority-class metrics under measured heterophily.

### 2.1 Graph-based fraud detection

Fraud detection moved from rules and classical machine learning to graph models because fraud is relational (Motie & Raahemi, 2024). GNNs such as GCN (Kipf & Welling, 2017) and GraphSAGE (Hamilton et al., 2017) learn a node's representation by aggregating its neighbours. Weber et al. (2019) introduced the Elliptic Bitcoin benchmark and found that a Random Forest on node features beat a GCN. Our own Assessment 2 experiments reproduced this.

### 2.2 Camouflage and heterophily

Fraudsters camouflage themselves by connecting to benign users, so their neighbourhoods are dominated by the opposite class (Dou et al., 2020). Standard GNNs assume homophily, so aggregation dilutes the fraudster's own signal. Fraud-specific GNNs counter this by re-selecting neighbours (CARE-GNN; Dou et al., 2020), using spectral filters suited to anomalies (BWGNN; Tang et al., 2022; Gao et al., 2023), or modelling higher-order relations with hypergraph transformers (TROPICAL; Haghighi et al., 2024). The general heterophily literature shows that keeping a node's own representation separate from its neighbours' is a key design choice (Zhu et al., 2020). GraphSAGE's separate self and neighbour weights are an instance of this. Lim et al. (2021) showed that edge homophily is misleading under class imbalance and proposed a *class-insensitive* measure that only counts homophily above chance. We use it in Section 4.

### 2.3 LLMs on text-attributed graphs

Chen et al. (2024) and Jin et al. (2024) group LLM-on-graph methods by the LLM's role:

| LLM role | What the LLM does | Example | Covered here |
| --- | --- | --- | --- |
| Encoder | Turns node text into vectors used by a classifier or GNN | Sentence-BERT features (Reimers & Gurevych, 2019) | Yes (Steps 3–4) |
| Enhancer / augmenter | Generates new text or features that feed a GNN | TAPE explanations (He et al., 2024); FLAG discriminative text | Only as the FLAG baseline |
| Predictor / reasoner | Reads a prompt built from the node and graph, and outputs the label | InstructGLM; DGP (Li et al., 2025) | Yes (Step 5) |

Two recent fraud-specific works anchor this project. **FLAG** (Yang et al., 2025) combines semantic-similarity neighbour sampling, LLM-extracted "discriminative text" and a skip-connected GNN. It reports average gains of 3.14% in macro-F1 and 6.97% in AUC, and it built the Instagram fraud benchmark we use by treating commercial accounts as the minority class at a 1:10 ratio. **DGP** (Li et al., 2025) uses the LLM as the classifier. It keeps the target node's text in full and summarises its neighbours, so the prompt fits the LLM's input limit and the target is not drowned out. Luo et al. (2025) illustrate a recurring evaluation weakness: 0.98 accuracy alongside a fraud-class recall of 0.05.

### 2.4 Where this work fits

FLAG's published Instagram F1-macro scores (0.47–0.51) sit close to the 0.476 that a model predicting "normal" for everyone scores at 1:10. This suggests the default decision threshold, not the representation, limits those numbers. No study we found compares, on the same splits, (a) text-only LLM encoders, (b) the same encoders with homophilous and heterophily-aware GNNs, and (c) an LLM reasoner with raw versus summarised neighbours, while also reporting the graph's measured heterophily. This project provides that controlled comparison, using small open LLMs that run on a laptop so that it can be reproduced.

## 3. Research Methodology

We split the main question into seven sub-problems and solved each as one stage of a single scripted pipeline. Every stage reads the same fixed data split and writes its results in the same format, so all models are compared on identical terms.

### 3.1 Sub-problems

| # | Sub-problem | Approach | Code |
| --- | --- | --- | --- |
| SP1 | How heterophilous is the graph? | Edge homophily, per-class homophily h\_k, class-insensitive homophily (Lim et al., 2021), isolated-node count | `step1_explore.py` |
| SP2 | How do we compare models fairly? | One stratified 60/20/20 train/validation/test split (seed 42), one metric function, decision thresholds chosen on validation | `common.py` |
| SP3 | What does text alone achieve? | Majority class and TF-IDF with logistic regression (baselines); then three LLM encoders with logistic regression | `step2_baseline.py`, `step3_embed.py` |
| SP4 | Does the graph add anything? | GCN (mixes self and neighbours) and GraphSAGE (keeps them separate) on each LLM's embeddings | `step4_gnn.py` |
| SP5 | Can an LLM reason over the graph? | Zero-shot LLM classifier with three prompts: target only, plus raw neighbour bios, plus summarised neighbours (DGP-style) | `step5_llm_reason.py` |
| SP6 | Does it hold when fraud is rare? | Re-run everything at FLAG's 1:10 ratio (Setting B) as well as the natural ratio (Setting A) | `common.py` |
| SP7 | How do we compare to published work? | Consolidated tables per setting, plus FLAG's published Instagram results as a reference | `compare.py` |

&#91;embedded content: Figure 1 · pipeline of sub-problems SP1–SP7\]

The data passes through the setting choice and the fixed protocol; the four experiment branches then run on identical splits, and the encoder vectors from SP3 also feed the GNNs in SP4.

### 3.2 Data and settings

The Instagram graph has 11,339 users, each with a profile bio, a label and follow links (144,010 directed edge rows, stored in both directions). Label 1 is a commercial account, which we treat as the fraud class following FLAG. The most label-specific words support this reading: *we, official, products, magazine, news, click* lean commercial, while personal email domains and city names lean normal.

|  | Setting A (natural) | Setting B (FLAG-style 1:10) |
| --- | --- | --- |
| Normal users | 7,224 | 7,224 |
| Commercial users | 4,115 (36.3%) | 722 (9.1%), sampled with seed 42 |
| Edge rows | 144,010 | 76,386 (edges between kept users only) |
| Isolated users | 1,838 | 1,407 |

Subsampling removes edges as well as users, so we recompute homophily for Setting B (Section 4.1).

### 3.3 Models and protocol

- **Text encoders (LLM as encoder).** MiniLM-L6 (22M parameters, English, 384 dimensions), multilingual-E5-base (278M, 768 dimensions, "query:" prefix) and Qwen2.5-1.5B (a decoder LLM; the last hidden layer mean-pooled over tokens, 1,536 dimensions). Each bio is encoded once and cached; Setting B reuses the same vectors.
- **Text-only classifier.** Logistic regression with balanced class weights on standardised embeddings. The regularisation strength is chosen on validation.
- **GNNs.** Two layers, 128 hidden units, dropout 0.5, Adam (learning rate 0.01, weight decay 5e-4), class-weighted cross-entropy, up to 300 epochs with early stopping on validation macro-F1 (patience 50). Three seeds each; we report mean ± standard deviation.
- **LLM reasoner.** Qwen2.5-1.5B-Instruct, zero-shot. It is asked: *"Is this Instagram account a business, brand or media account (rather than a personal account)? Answer Yes or No."* The score is P(Yes) from the first answer token. Neighbours are the eight with the most similar bios (cosine on multilingual-E5 embeddings), which follows FLAG's semantic-similarity sampling. DGP summaries use a task-agnostic prompt with a 32-token budget per neighbour. Because the LLM is slow, it is scored on stratified samples (Setting A: 300 test and 150 validation users; Setting B: 600 and 300). The trained models are re-scored on the same users.
- **Metrics.** Macro-F1 (primary), class-1 F1, ROC-AUC, PR-AUC, and macro-F1 separately for connected and isolated users. Accuracy is not used: always predicting "normal" scores 63.7% in Setting A and 90.9% in Setting B while catching no commercial accounts.
- **Leakage control.** The TF-IDF vocabulary, scaling statistics, regularisation, early stopping and decision thresholds are all fitted on train or validation only. The test set is used once.

### 3.4 Timeline: planned versus actual

| Period | Planned (Assessment 2, Elliptic) | What actually happened |
| --- | --- | --- |
| 22 Sep – 5 Oct 2026 | Reconcile LLaMA fusion; sample-based explanations (SP8) | \[Team to confirm\] |
| 6 – 19 Oct 2026 | FLAG-style neighbour sampling on Elliptic (SP9); close the gap to Random Forest | Review of the plan: evaluation must address heterophily and camouflaged fraud |
| 27 Oct 2026 | (not planned) | Pivot to the Instagram text-attributed graph; heterophily measured (SP1) |
| 30 Oct 2026 | Report consolidation | Pipeline SP2–SP7 built and run for both settings; FLAG comparison |

### 3.5 Difficulties and fallbacks

- **Dataset change.** Elliptic has no native text, so our earlier pipeline synthesised text from numeric features, and its evaluation did not address heterophily. Fallback: a real text-attributed, measurably heterophilous graph (Instagram).
- **Gated model.** Llama-3.2-1B needs licence approval. Fallback: the open Qwen2.5-1.5B.
- **Hardware.** On a 16 GB laptop, encoding with the 1.5B model stalled through memory swapping. Halving the batch size to 16 fixed it; encoding then took about 15 minutes.
- **Empty bios.** The 99 users with empty bios produced invalid (NaN) embeddings, because zero tokens give a division by zero in mean pooling. Fallback: a placeholder text, "\[empty bio\]".
- **Threshold artefact.** With a fixed 0.5 cut-off, models looked worse than TF-IDF in Setting B despite higher AUC. All models now tune their threshold on validation. This also shrank the apparent graph gain in Setting A from about +0.02 to zero.
- **LLM cost.** Prompts with raw neighbour bios are about five times longer than summarised ones. Fallback: stratified evaluation samples.
- **Unresolved.** Full FLAG (Gemma-9B with LoRA fine-tuning on A100 GPUs) is beyond our hardware, so we compare against its published numbers (Section 4.4). LLM-as-augmenter methods and newer LLMs (Qwen3-Embedding, Qwen3.5) remain future work.

## 4. Research Outcome – Results, Analysis, and Discussion

A good LLM encoder on the bio alone was the best model in both settings. GNNs added nothing at the natural ratio and hurt the commercial class at 1:10. Summarising neighbours helped the LLM reasoner only at 1:10, where heterophily is extreme.

### 4.1 The graph is close to random for the fraud class (SP1, SP6)

| Measure | Setting A | Setting B |
| --- | --- | --- |
| Edge homophily | 0.586 | 0.861 |
| h\_k, normal users | 0.686 | 0.925 |
| h\_k, commercial users | 0.392 | 0.109 |
| Commercial share of users (chance level) | 0.363 | 0.091 |
| Class-insensitive homophily (0 = random, 1 = perfect) | 0.078 | 0.033 |

Commercial users' neighbours are commercial at almost exactly the chance rate (0.392 against 0.363, and 0.109 against 0.091). This is the camouflage pattern: heterophilous fraud nodes beside homophilous normal ones. In Setting B, edge homophily rises to 0.861 only because nine in ten users are normal. The class-insensitive score falls to 0.033, which shows the graph is even less informative than in Setting A.

### 4.2 Text encoders and GNNs (SP3, SP4)

Test set, full split. GNN rows are means of three seeds; the standard deviation of macro-F1 is 0.001–0.004 in Setting A and 0.005–0.016 in Setting B.

**Setting A (natural ratio, 36% commercial)**

| Model | Macro-F1 | Class-1 F1 | ROC-AUC | PR-AUC | Connected | Isolated |
| --- | --- | --- | --- | --- | --- | --- |
| Majority class | 0.389 | 0.000 | – | – | 0.395 | 0.359 |
| TF-IDF + LR | 0.578 | 0.422 | 0.592 | 0.501 | 0.578 | 0.564 |
| MiniLM + LR | 0.588 | 0.442 | 0.609 | 0.503 | 0.587 | 0.586 |
| Multilingual-**E5 + LR** | **0.612** | 0.469 | 0.643 | 0.550 | 0.607 | **0.623** |
| Qwen + LR | 0.608 | 0.460 | **0.644** | **0.552** | 0.604 | 0.617 |
| MiniLM + GCN | 0.607 | 0.489 | 0.639 | 0.513 | 0.604 | 0.581 |
| MiniLM + SAGE | 0.588 | 0.449 | 0.616 | 0.485 | 0.576 | 0.581 |
| Multilingual-E5 + GCN | 0.606 | **0.493** | 0.641 | 0.520 | 0.600 | 0.600 |
| Multilingual-E5 + SAGE | 0.602 | 0.481 | 0.633 | 0.523 | 0.596 | 0.595 |
| Qwen + GCN | 0.609 | 0.478 | 0.642 | 0.533 | **0.606** | 0.590 |
| Qwen + SAGE | 0.604 | 0.465 | 0.640 | 0.546 | 0.595 | 0.612 |

**Setting B (FLAG-style 1:10, 9% commercial)**

| Model | Macro-F1 | Class-1 F1 | ROC-AUC | PR-AUC | Connected | Isolated |
| --- | --- | --- | --- | --- | --- | --- |
| Majority class | 0.476 | 0.000 | – | – | 0.477 | 0.473 |
| TF-IDF + LR | 0.552 | 0.188 | 0.603 | 0.155 | 0.545 | 0.571 |
| MiniLM + LR | 0.546 | 0.167 | 0.607 | 0.141 | 0.542 | 0.558 |
| Multilingual-E5 + LR | 0.563 | 0.219 | 0.627 | **0.193** | 0.556 | 0.584 |
| **Qwen + LR** | **0.578** | **0.238** | 0.606 | 0.188 | **0.567** | **0.609** |
| MiniLM + GCN | 0.552 | 0.194 | 0.616 | 0.154 | 0.552 | 0.530 |
| MiniLM + SAGE | 0.542 | 0.180 | 0.602 | 0.161 | 0.542 | 0.523 |
| Multilingual-E5 + GCN | 0.555 | 0.201 | 0.605 | 0.156 | 0.551 | 0.553 |
| Multilingual-E5 + SAGE | 0.572 | 0.228 | 0.619 | 0.177 | 0.564 | 0.579 |
| Qwen + GCN | 0.549 | 0.186 | **0.629** | 0.168 | 0.530 | 0.565 |
| Qwen + SAGE | 0.562 | 0.213 | 0.609 | 0.176 | 0.554 | 0.562 |

Four findings follow.

1. **LLM encoders beat the baseline.** The best encoder improves macro-F1 over TF-IDF by 0.034 in Setting A and 0.026 in Setting B, and PR-AUC by about 0.05 and 0.04. Multilingual-E5 clearly beats the English-only MiniLM, which matters because many bios are in Spanish, Portuguese or Dutch.
2. **The graph does not help.** No GNN beats the best text-only model in either setting. In Setting A the best GNN ties text alone (0.609 against 0.612).
3. **GCN hurts the camouflaged class.** In Setting B, adding a GCN to Qwen lowers macro-F1 from 0.578 to 0.549 and class-1 F1 from 0.238 to 0.186. This matches Section 4.1: averaging a commercial user with neighbours who are 89% normal washes out its signal.
4. **Heterophily-aware aggregation limits the damage.** In Setting B, GraphSAGE beats GCN with multilingual-E5 (0.572 against 0.555) and with Qwen (0.562 against 0.549), consistent with Zhu et al. (2020). In Setting A, where heterophily is milder, GCN and SAGE are within 0.02 of each other.

### 4.3 LLM as reasoner (SP5)

These rows use stratified test samples (Setting A: 300 users; Setting B: 600 users, about 55 of them commercial), so they are compared only with the trained models re-scored on the same users.

| Model (same test sample) | A: Macro-F1 | A: Class-1 F1 | A: PR-AUC | B: Macro-F1 | B: Class-1 F1 | B: PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| LLM, target bio only | 0.553 | **0.486** | 0.461 | 0.522 | 0.122 | 0.124 |
| LLM + raw neighbour bios | 0.552 | 0.366 | 0.495 | 0.499 | 0.057 | 0.127 |
| LLM + summarised neighbours (DGP) | 0.537 | 0.348 | 0.478 | 0.538 | 0.162 | 0.123 |
| Qwen + LR (trained) | 0.607 | 0.463 | 0.572 | 0.546 | 0.177 | **0.157** |
| Qwen + GCN (trained) | **0.623** | 0.484 | **0.576** | **0.562** | **0.198** | 0.149 |

- The zero-shot reasoner trails the trained models in both settings. A 1.5B model without training cannot match classifiers trained on thousands of labelled users.
- Adding raw neighbour bios lowers commercial-class F1 in both settings (0.486 to 0.366 in A; 0.122 to 0.057 in B). The neighbours' text drowns out the target.
- Summarising neighbours (DGP) does not fix this in Setting A, but in Setting B it recovers commercial-class F1 from 0.057 to 0.162, the best of the three prompts. This supports DGP's claim where heterophily is extreme. With about 55 positives the estimate is noisy, and DGP's ROC-AUC there is the lowest of the three (0.531), so we treat it as suggestive.
- Summarised prompts are about five times shorter than raw ones, which also makes the reasoner far cheaper to run.

### 4.4 Comparison with FLAG (SP7)

FLAG reports Instagram results only at 1:10, which corresponds to our Setting B. We could not run FLAG itself, so the comparison below uses its published numbers (Yang et al., 2025, Table 4).

| Source | Model | Macro-F1 | ROC-AUC |
| --- | --- | --- | --- |
| FLAG paper | GCN + text (Sentence-BERT) | 0.473 | 0.557 |
| FLAG paper | Best F1: GAT + FLAG\* (fine-tuned) | 0.507 | 0.560 |
| FLAG paper | Best AUC: PMP + FLAG\* (fine-tuned) | 0.498 | 0.577 |
| This project | MiniLM (a Sentence-BERT model) + GCN | 0.552 | 0.616 |
| This project | Qwen + LR (text only) | 0.578 | 0.606 |
| This project | Qwen + GCN | 0.549 | 0.629 |

On ROC-AUC, which does not depend on a threshold, our simple encoder models (0.61–0.63) match or exceed FLAG's published Instagram results (0.56–0.58). The macro-F1 gap is mostly a protocol effect: many of FLAG's F1 values sit at the always-normal level, while we tune each model's threshold on validation. The two studies also use different random 1:10 subsamples, splits, run counts and language models. The comparison is therefore indicative, not head-to-head.

### 4.5 Answers to the research questions

- **Q1, LLM text versus a word-count baseline:** yes. LLM encoders win in both settings, and a multilingual or larger model helps most.
- **Q2, does the graph help under heterophily:** no. At best it ties text alone; a plain GCN hurts the minority class when camouflage is extreme; heterophily-aware aggregation (SAGE) reduces the harm.
- **Q3, does neighbour summarisation help an LLM reasoner:** only partly. Raw neighbours always hurt; summaries recover part of the loss at 1:10 but not at the natural ratio. The zero-shot reasoner stays behind trained models.

### 4.6 Second dataset check: a metadata shortcut in TwiBot-20

On TwiBot-20, a bot-detection benchmark we are adding as a second dataset (Feng et al., 2021), one profile field nearly solves the task by itself. A logistic regression that sees only whether an account is verified scores macro-F1 0.803 on the official test split, because 1 of 6,589 bots is verified against 2,963 of 5,237 humans (57%). Median followers are 724 for bots and 27,322 for humans.

Metadata-only models, official split, natural ratio (1,183 test users, 54% bots):

| Features | Macro-F1 | Bot F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- |
| None (majority class) | 0.351 | – | – | – |
| Verified only | 0.803 | 0.855 | 0.800 | 0.747 |
| Log(followers) only | 0.727 | 0.777 | 0.753 | 0.707 |
| Verified + log(followers) | 0.804 | 0.854 | 0.820 | 0.760 |

At the 1:10 ratio (bots as the minority), the same two fields still reach ROC-AUC 0.802, with macro-F1 0.576 and PR-AUC 0.198.

This is a shortcut: the model learns "verified or famous means human", not how bots write or connect. Any model given these fields would score well without showing whether text or graph structure helps. We therefore exclude all profile metadata (verification, follower and following counts, account age) on TwiBot-20 and use only text (bio and recent tweets) and the follow graph. The scores above are reported as a reference for what a non-text, non-graph shortcut achieves.

### 4.7 Deviations and limitations

We changed the original question, "does LLM text improve a GNN on Elliptic?", into the question P′: "when does graph structure help an LLM-based detector, given measured heterophily?" Elliptic offered no real text and no heterophily evaluation, so P could not test what happens under camouflage. Limitations: commercial accounts are a proxy for fraud; we used one dataset; LLM-reasoner results come from samples; models are small (up to 1.5B parameters) and not fine-tuned; FLAG is compared through published numbers only. The planned explainability output (SP8 in Assessment 2) was not carried over to the new dataset.

## 5. Individual Contributions

Each member must confirm their own entry before submission; entries marked \[to complete\] should not be filled in on another member's behalf.

| Member | Key tasks and responsibilities | Changes over time |
| --- | --- | --- |
| Prajan Bhattarai | Instagram rebuild (`my_build/`): heterophily analysis, evaluation protocol, LLM encoders, GNN experiments, LLM reasoner with neighbour summarisation, Setting A/B protocol, FLAG comparison; report drafting \[to confirm\] | Moved from the Elliptic pipeline to the Instagram rebuild after supervisor feedback (27 Oct 2026) |
| Mustafa Ugur Atas | \[to complete\] | \[to complete\] |
| Zeki Kose | \[to complete\] | \[to complete\] |
| Abhishek Biswal | \[to complete\] | \[to complete\] |

Earlier shared work (Assessment 2, Elliptic): data pipeline and chronological split, classical baselines, GCN/GraphSAGE/GAT training, text-profile synthesis, MiniLM and LLaMA encoding, fusion classifier, results consolidation and literature review. \[Team to allocate these tasks to members.\]

## References

APA 7th edition. Details should be checked against the original sources before submission.

- Chen, Z., Mao, H., Li, H., Jin, W., Wen, H., Wei, X., Wang, S., Yin, D., Fan, W., Liu, H., & Tang, J. (2024). Exploring the potential of large language models (LLMs) in learning on graphs. *ACM SIGKDD Explorations Newsletter, 25*(2), 42–61.
- Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). Enhancing graph neural network-based fraud detectors against camouflaged fraudsters. In *Proceedings of the 29th ACM International Conference on Information & Knowledge Management* (pp. 315–324).
- Feng, S., Wan, H., Wang, N., Li, J., & Luo, M. (2021). TwiBot-20: A comprehensive Twitter bot detection benchmark. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (pp. 4485–4494).
- Gao, Y., Wang, X., He, X., Liu, Z., Feng, H., & Zhang, Y. (2023). Addressing heterophily in graph anomaly detection: A perspective of graph spectrum. In *Proceedings of the ACM Web Conference 2023* (pp. 1528–1538).
- Haghighi, V., Soltani, B., Shabani, N., Wu, J., Zhang, Y., Yao, L., Sheng, Q. Z., & Yang, J. (2024). TROPICAL: Transformer-based hypergraph learning for camouflaged fraudster detection. In *2024 IEEE International Conference on Data Mining* (pp. 121–130). https://doi.org/10.1109/ICDM59182.2024.00019
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. In *Advances in Neural Information Processing Systems 30*.
- He, X., Bresson, X., Laurent, T., & Hooi, B. (2024). Harnessing explanations: LLM-to-LM interpreter for enhanced text-attributed graph representation learning. In *International Conference on Learning Representations*.
- &#91;Instagram dataset source – to confirm; FLAG cites it as its reference 25. Possibly Huang, X., et al. (2024). Can GNN be good adapter for LLMs? In *Proceedings of the ACM Web Conference 2024* (pp. 893–904).\]
- Jin, B., Liu, G., Han, C., Jiang, M., Ji, H., & Han, J. (2024). Large language models on graphs: A comprehensive survey. *IEEE Transactions on Knowledge and Data Engineering*.
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In *International Conference on Learning Representations*.
- Li, Y., Hu, J., Hooi, B., He, B., & Chen, C. (2025). DGP: A dual-granularity prompting framework for fraud detection with graph-enhanced LLMs. arXiv. https://arxiv.org/abs/2507.21653
- Lim, D., Hohne, F., Li, X., Huang, S. L., Gupta, V., Bhalerao, O., & Lim, S.-N. (2021). Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods. In *Advances in Neural Information Processing Systems 34*.
- Luo, R., Wang, N., & Zhu, X. (2025). Fraud detection and risk assessment of online payment transactions on e-commerce platforms based on LLM and GCN frameworks. arXiv. https://arxiv.org/abs/2509.09928
- Motie, S., & Raahemi, B. (2024). Financial fraud detection using graph neural networks: A systematic review. *Expert Systems with Applications, 240*, 122156.
- Qwen Team. (2024). Qwen2.5 technical report. arXiv. https://arxiv.org/abs/2412.15115
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*.
- Tang, J., Li, J., Gao, Z., & Li, J. (2022). Rethinking graph neural networks for anomaly detection. In *Proceedings of the 39th International Conference on Machine Learning* (PMLR 162).
- Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F. (2024). Multilingual E5 text embeddings: A technical report. arXiv. https://arxiv.org/abs/2402.05672
- Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics. arXiv. https://arxiv.org/abs/1908.02591
- Yang, C., Liu, H., Wang, D., Zhang, Z., Yang, C., & Shi, C. (2025). FLAG: Fraud detection with LLM-enhanced graph neural network. In *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2*. https://doi.org/10.1145/3711896.3737220
- Zhu, J., Yan, Y., Zhao, L., Heimann, M., Akoglu, L., & Koutra, D. (2020). Beyond homophily in graph neural networks: Current limitations and effective designs. In *Advances in Neural Information Processing Systems 33*.
