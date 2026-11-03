# Literature scout: LLM + graph fraud/bot detection under heterophily

Prepared 3 Nov 2026 for the COMP8851 project (TwiBot-20 main, Instagram secondary).
Every entry below was checked against a page opened during this search (URL given). Where the opened page did not show a fact (for example the venue), it is marked **UNVERIFIED** or the source of that fact is named.

---

## 0. Three things found that change the project's own text

1. **DGP venue is confirmed: AAAI 2026** (AAAI-40, vol. 40 no. 18, pp. 15171–15179, according to the search index; the AAAI OJS page blocked the fetch). The official repo **github.com/Xtra-Computing/DGP is public but has only a README**: "Pending approval from our industry partner before public release." So REPORT.md's "the official code is not public yet" is still correct. Cite the repo and its status.
2. **FLAG = "FLAG: Fraud Detection with LLM-enhanced Graph Neural Network"**, Chengdong Yang, Hongrui Liu, Daixin Wang, Zhiqiang Zhang, Cheng Yang, Chuan Shi, **KDD 2025**, DOI 10.1145/3711896.3737220. It is an *LLM-as-enhancer* method: (a) semantic-similarity neighbour sampling, (b) LLM-based node enhancement, with an LLM fine-tuned to pull out label-relevant text. Reported gains: +3.14% macro-F1 and +6.97% AUC on public datasets, plus +0.9% KS in Alipay deployment. **No public code** (AI4Risk awesome list gives no code link). **I could not find any source saying FLAG uses a 1:10 class ratio, or Instagram.** Unless the team has the FLAG PDF showing it, "FLAG-style 1:10" is UNVERIFIED. Rename it "rare-class setting (1:10)", or cite the exact FLAG section.
3. **The DGP authors' own follow-up moves away from LLM-as-predictor.** FraudCoT (Li, Hu, Hooi, He, Chen, arXiv 2601.22949, Jan 2026) uses LLM-generated graph-aware reasoning to *enrich node text for a GNN*. Its stated motivation: LLM-only models "treat nodes independently and cannot propagate multi-hop evidence," and on-the-fly LLM reasoning is expensive. FLAG is one of its baselines and does poorly there (macro-F1 57–62 vs MLP 57–81). This supports the project's reading that the LLM is better used as an encoder.

---

## 1. Heterophily GNNs and evaluation critiques

| Paper | Venue | URL opened | Claim (1–2 lines) |
|---|---|---|---|
| Kipf & Welling, *Semi-Supervised Classification with GCNs* | ICLR 2017 | https://arxiv.org/abs/1609.02907 | Anchor GCN: neighbourhood averaging with self-loops, built for (homophilous) citation graphs. |
| Hamilton et al., *Inductive Representation Learning on Large Graphs* (GraphSAGE) | NIPS 2017 | https://arxiv.org/abs/1706.02216 | Learns aggregators over sampled neighbourhoods; keeps a separate self weight, i.e. ego/neighbour separation. |
| Zhu et al., *Beyond Homophily in GNNs: Current Limitations and Effective Designs* (H2GCN) | NeurIPS 2020 | https://arxiv.org/abs/2006.11468 | Many GNNs fail under heterophily and "some are outperformed by MLPs." Three helpful designs: ego/neighbour separation, higher-order neighbourhoods, combining intermediate layers. |
| Chien et al., *Adaptive Universal Generalized PageRank GNN* (GPR-GNN) | ICLR 2021 | https://arxiv.org/abs/2006.07988 | Learns signed GPR propagation weights, so it adapts to homophily or heterophily and limits over-smoothing. |
| Bo et al., *Beyond Low-frequency Information in GCNs* (FAGCN) | AAAI 2021 | https://arxiv.org/abs/2101.00797 | A self-gating mechanism mixes low- and high-frequency signals (signed edge weights). Claims to reduce over-smoothing and beat SOTA on six networks. |
| Lim et al., *Large Scale Learning on Non-Homophilous Graphs* (LINKX) | NeurIPS 2021 | https://arxiv.org/abs/2110.14446 | New large non-homophilous benchmarks; scalable GNNs degrade on them. A simple method (LINKX: separate MLPs on features and adjacency) is SOTA. |
| Ma et al., *Is Homophily a Necessity for GNNs?* | ICLR 2022 (https://iclr.cc/virtual/2022/poster/6329 via search) | https://arxiv.org/abs/2106.06134 | A plain GCN can beat heterophily-specific models on some heterophilous benchmarks. "Good" heterophily exists, where neighbour label distributions are class-distinguishable. |
| Li et al., *Finding Global Homophily in GNNs When Meeting Heterophily* (GloGNN) | ICML 2022 | https://arxiv.org/abs/2205.07308 | Aggregates from all nodes through a learned closed-form coefficient matrix. Linear time; best of 11 methods on 15 datasets. |
| Luan et al., *Revisiting Heterophily for GNNs* (ACM-GNN) | NeurIPS 2022 | https://arxiv.org/abs/2210.07606 | Edge-label homophily metrics are incomplete. Adaptive Channel Mixing (aggregation / diversification / identity channels per node) improves baselines on 10 benchmarks. |
| Platonov et al., *A critical look at the evaluation of GNNs under heterophily: Are we really making progress?* | ICLR 2023 (https://iclr.cc/virtual/2023/poster/11466) | https://arxiv.org/abs/2302.11640 ; HTML read | Squirrel/Chameleon contain duplicate nodes (train–test leakage). On five new heterophilous graphs, **standard GNNs win 13 of 15 top-3 places**, and models with ego/neighbour separation (SAGE, GAT-sep, GT-sep) are best. Some specialised models (H2GCN, GPR-GNN, FAGCN, GloGNN were tested) fall below the graph-agnostic ResNet. Binary sets use ROC-AUC. |
| Platonov et al., *Characterizing Graph Datasets for Node Classification: Homophily–Heterophily Dichotomy and Beyond* | arXiv 2209.06177 (NeurIPS 2023 per common citation: **venue UNVERIFIED**, page shows none) | https://arxiv.org/abs/2209.06177 ; HTML read | Edge homophily depends on the number of classes and their balance. Proposes **adjusted homophily** h_adj = (h_edge − Σ_k p̄(k)²)/(1 − Σ_k p̄(k)²), with p̄(k) = D_k/2\|E\| (degree-weighted), and **label informativeness (LI)**. Example: *questions* has h_edge 0.84 but h_adj 0.02. LI tracks GNN accuracy better than homophily does. |
| Luo et al., *Classic GNNs are Strong Baselines* | NeurIPS 2024 | https://arxiv.org/abs/2406.08993 | With light tuning (normalisation, dropout, residuals, depth), GCN/GAT/SAGE match or beat Graph Transformers on 17 of 18 datasets. Claimed gains often come from under-tuned baselines. |
| Shchur et al., *Pitfalls of GNN Evaluation* | arXiv 1811.05868 (R2L workshop, NeurIPS 2018: **UNVERIFIED**) | https://arxiv.org/abs/1811.05868 | Different splits give "dramatically different rankings." With fair tuning, simpler GNNs beat sophisticated ones. |

**Does a strong MLP beat GNNs under heterophily?** It depends on the graph, and the literature says so:
- H2GCN reports MLP > several GNNs on heterophilous graphs.
- Platonov et al. find that on *their* heterophilous graphs, separated GNNs (SAGE etc.) beat the graph-agnostic ResNet, sometimes by a lot (tolokers +10.8 ROC-AUC, questions +8.5). Specialised heterophily models are not better than classic ones.
- What decides it is whether neighbour labels are *informative* (LI, "good heterophily" in Ma et al.), not raw h_edge.

The project's TwiBot-20 graph has h_edge 0.532 against a 0.507 chance level. That makes h_adj ≈ (0.532−0.507)/(1−0.507) ≈ **0.05**: almost no label signal on the edges. So MLP ≥ GNN is the expected outcome, not a surprise. (This is my approximation, assuming the "random-edge baseline" equals Σp̄²; compute it exactly.)

---

## 2. Graph fraud / anomaly detection under heterophily; Twitter bot detection

### 2a. Graph fraud / anomaly detection

| Paper | Venue | URL opened | Claim |
|---|---|---|---|
| Dou et al., *Enhancing GNN-based Fraud Detectors against Camouflaged Fraudsters* (CARE-GNN) | CIKM 2020 | https://arxiv.org/abs/2008.08692 | Names *feature* and *relation camouflage*. Label-aware similarity neighbour selection with an RL-chosen threshold, plus cross-relation aggregation; tested on YelpChi/Amazon. |
| Liu et al., *Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection* (PC-GNN) | WWW 2021, pp. 3168–3177 | https://github.com/PonderLY/PC-GNN | Label-balanced node/edge sampler for mini-batches plus a neighbourhood sampler that picks same-class-like neighbours, for imbalanced fraud graphs. |
| Tang et al., *Rethinking GNNs for Anomaly Detection* (BWGNN) | ICML 2022 | https://arxiv.org/abs/2205.15508 | Anomalies cause a "right-shift" of spectral energy to high frequencies. Beta-wavelet band-pass filters (BWGNN). |
| Gao et al., *Addressing Heterophily in Graph Anomaly Detection: A Perspective of Graph Spectrum* (GHRN) | WWW 2023 | https://github.com/blacksingular/GHRN | Heterophily correlates with high graph frequency; prunes inter-class edges using a high-pass indicator. |
| Wang et al., *Label Information Enhanced Fraud Detection against Low Homophily in Graphs* (GAGA) | WWW 2023 | https://arxiv.org/abs/2302.10407 | Transformer with *group aggregation* that feeds training labels into neighbourhood features. Up to +24.39% over graph fraud detectors. |
| Tang et al., *GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection* | NeurIPS 2023 D&B | https://arxiv.org/abs/2306.12251 | 29 models on 10 datasets: "**tree ensembles with simple neighborhood aggregation can outperform the latest GNNs tailored for the GAD task**." |
| Huang et al., *Can LLMs Find Fraudsters? Multi-level LLM Enhanced Graph Fraud Detection* (MLED) | ACM MM 2025 | https://arxiv.org/abs/2507.11997 | LLM-extracted type-level and relation-level knowledge added to existing graph fraud detectors; SOTA on four datasets. |
| Pang et al., *GuARD: Effective Anomaly Detection through a Text-Rich and Graph-Informed Language Model* | KDD 2025 | https://arxiv.org/abs/2412.03930 | Combines structural features with small-LM semantic embeddings in multi-turn instruction tuning; covers social bots among its anomaly types. Beats graph-based and LLM-based detectors on four datasets with ~5× speed-up over long-context LLMs. |
| Yang et al., *FLAG* | KDD 2025 | https://chatpaper.com/pt/paper/175073 (aggregator; ACM DL returned 403) | See §0.2. LLM-enhanced GNN; semantic-similarity neighbour sampling against camouflage. Datasets not named on the page I could open: **UNVERIFIED**. |
| Li et al., *DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs* | AAAI 2026 | https://arxiv.org/abs/2507.21653 ; HTML read; https://github.com/Xtra-Computing/DGP | Keeps fine-grained target text and compresses metapath neighbours into coarse summaries (bi-level text abstraction plus numeric statistics), fine-tuning Qwen3-8B with LoRA. Data: Yelp/Amazon/E-Commerce/LifeService with ~1,300 training nodes each, 5 seeds. Best AUPRC on all four (Yelp 48.87 vs ConsisGAD 42.11). Macro-F1 vs MLP: Yelp 69.07 vs 62.09; target-only LLM 60.79 is *below* MLP. Tuned on validation AUROC; M ∈ {2,4,8,16}, K ∈ {1,2,3}, budget ~10 best. **The paper does not say how the macro-F1 threshold was chosen.** |
| Li et al., *Autonomous Chain-of-Thought Distillation for Graph-Based Fraud Detection* (FraudCoT) | arXiv 2601.22949 (2026, no venue) | https://arxiv.org/abs/2601.22949 ; HTML read | Same DGP group. LLM chain-of-thought enriches node text for a GNN, with asymmetric co-training (up to 1,066× throughput) and +8.8% AUPRC. Baselines include FLAG and TAPE, not DGP. |
| Pan et al., *CAMERA: Adapting to Semantic Camouflage in Unsupervised TAG Fraud Detection* | IJCAI 2026 | https://arxiv.org/abs/2605.20032 | Fraudsters imitate benign *text* ("semantic camouflage"). A mixture-of-experts over fraud cues, unsupervised. |
| Zuo et al., *Let Relations Speak: An End-to-End LLM-GNN Soft Prompt Framework for Fraud Detection* (LGSPF) | arXiv 2605.28524 (2026) | https://arxiv.org/abs/2605.28524 | A GNN encoder turns multi-relational topology into soft-prompt tokens for an LLM, trained end to end. |
| AI4Risk, *awesome-graph-based-fraud-detection* (curated list) | GitHub | https://github.com/AI4Risk/awesome-graph-based-fraud-detection | Confirms DGP (AAAI 2026), FLAG (KDD 2025), MLED (MM 2025), GuARD (KDD 2025), BotSay (ACL 2024) and LMBot (WSDM 2024); none of these entries lists code. |

### 2b. Twitter bot detection

| Paper | Venue | URL opened | Claim |
|---|---|---|---|
| Feng et al., *TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark* | CIKM 2021 | https://arxiv.org/abs/2106.13088 | 229,573 users (11,826 labelled, the rest support users), 33.5M tweets. Existing methods fall short of the scores claimed in their own papers. |
| Feng et al., *BotRGCN: Twitter Bot Detection with Relational GCNs* | ASONAM 2021 | https://arxiv.org/abs/2106.13092 | R-GCN over typed follow relations with text and property features; beats baselines on TwiBot-20. |
| Feng et al., *Heterogeneity-aware Twitter Bot Detection with Relational Graph Transformers* (RGT) | AAAI 2022 | https://arxiv.org/abs/2109.02927 | Models relation *and* influence heterogeneity with relational graph transformers plus semantic attention. (About heterogeneity, not heterophily.) |
| Feng et al., *TwiBot-22: Towards Graph-Based Twitter Bot Detection* | NeurIPS 2022 D&B | https://arxiv.org/abs/2206.04564 ; HTML read | 35 baselines on 9 datasets. "Graph-based approaches are generally more effective than feature-based or text-based." TwiBot-20 accuracy: BotRGCN 85.8, RGT 86.6. **Removing the graph** costs BotRGCN 3.1 points and RGT 4.0 on TwiBot-20. These models use metadata and the full support-user graph. |
| Hays et al., *Simplistic Collection and Labeling Practices Limit the Utility of Benchmark Datasets for Twitter Bot Detection* | WWW 2023 (DOI 10.1145/3543507.3583214) | https://arxiv.org/abs/2301.07015 | Shallow decision trees on a few features come close to SOTA on most bot datasets. High scores reflect how each dataset was collected and labelled, not real bot–human differences. **This is the citation that justifies excluding "verified".** |
| Wu et al., *BotSCL: Heterophily-aware Social Bot Detection with Supervised Contrastive Learning* | ICPR 2024 (arXiv comment) | https://arxiv.org/abs/2306.07478 | Neighbour aggregation across heterophilous edges blurs bots and humans and raises false negatives; proposes supervised contrastive learning with a channel-wise encoder. (Search snippet, not the abstract: bots are heterophilous on *follower* edges and more homophilous on *following* edges in TwiBot-20.) |
| Liu et al., *BotMoE: Twitter Bot Detection with Community-Aware Mixtures of Modal-Specific Experts* | SIGIR 2023 | https://arxiv.org/abs/2304.06280 | Separate metadata / text / graph experts with community-aware routing; SOTA on three benchmarks, more robust to evasive bots. |
| Cai et al., *LMBot: Distilling Graph Knowledge into Language Model for Graph-less Deployment* | WSDM 2024 (per search and AI4Risk; arXiv page shows no venue) | https://arxiv.org/abs/2306.17408 | Iterative GNN→LM distillation, so a text-only LM does inference without the graph; SOTA on four datasets. |
| Feng et al., *What Does the Bot Say? Opportunities and Risks of LLMs in Social Media Bot Detection* | ACL 2024 | https://arxiv.org/abs/2402.00371 ; HTML read | A mixture of LLM experts (metadata / text / meta+text / structure). **Instruction tuning on 1,000 users** beats SOTA by up to 9.1%. LLM-guided manipulation cuts existing detectors by up to 29.6%. On TwiBot-20 (instruction-tuned ChatGPT, acc/F1): text-only 0.767/0.791, structure 0.890/0.904, ensemble 0.899/0.915. The metadata expert *includes "verified"* and alone reaches 0.812 accuracy, above text-only. |

---

## 3. LLMs on graphs

| Paper | Venue | URL opened | Claim |
|---|---|---|---|
| Chen et al., *Exploring the Potential of LLMs in Learning on Graphs* | SIGKDD Explorations 2023/24 | https://arxiv.org/abs/2307.03393 ; HTML read | Two pipelines: **LLMs-as-Enhancers** and **LLMs-as-Predictors**. Obs 3: sentence-embedding models (e5) + GNN are a strong baseline. Obs 4: scaling to LLaMA-7B does *not* close the gap, so the pre-training objective matters more than size. Obs 11: zero-shot LLMs lag GNNs on most sets. Obs 16: **structure-aware prompts can hurt under heterophily**. Warns about test-data leakage into LLM pre-training. |
| Li et al., *A Survey of Graph Meets Large Language Model* | IJCAI 2024 Survey | https://arxiv.org/abs/2311.12399 | Taxonomy by LLM role: **enhancer, predictor, alignment component**. |
| Jin et al., *Large Language Models on Graphs: A Comprehensive Survey* | TKDE 2024 | https://arxiv.org/abs/2312.02783 | Scenarios (pure / text-attributed / text-paired graphs) × techniques (**LLM as Predictor, Encoder, Aligner**). |
| He et al., *Harnessing Explanations: LLM-to-LM Interpreter* (TAPE) | ICLR 2024 | https://arxiv.org/abs/2305.19523 | Zero-shot LLM predictions and explanations, turned into LM features for a GNN; SOTA on Cora/PubMed/ogbn-arxiv. |
| Tang et al., *GraphGPT: Graph Instruction Tuning for LLMs* | SIGIR 2024 | https://arxiv.org/abs/2310.13023 | Text–graph grounding plus a projector with two-stage instruction tuning; better zero-shot and supervised generalisation. |
| Ye et al., *Language is All a Graph Needs* (InstructGLM) | EACL 2024 | https://arxiv.org/abs/2308.07134 | Describes multi-hop structure in natural-language instructions; an instruction-tuned LLM beats GNNs on arxiv/Cora/PubMed. |
| Chen et al., *LLaGA: Large Language and Graph Assistant* | ICML 2024 (PMLR 235) | https://proceedings.mlr.press/v235/chen24bh.html | Structure-aware node sequences projected into token space; one model covers four datasets and three tasks. |
| Huang et al., *Can GNN be Good Adapter for LLMs?* (GraphAdapter) | WWW 2024 | https://arxiv.org/abs/2402.12984 ; HTML read | GNN adapter on a frozen LLM trained by next-token prediction. **Source of the Instagram dataset**: 11,339 users, commercial vs normal, profile text, ROC-AUC. Reported ROC-AUC: GNN on frozen PLM 0.60–0.63; GraphAdapter (Llama 2) 0.651. So Instagram is hard in their paper too, which matches the project's ~0.61 macro-F1. |
| Yan et al., *A Comprehensive Study on Text-attributed Graphs: Benchmarking and Rethinking* (CS-TAG) | NeurIPS 2023 D&B | https://proceedings.neurips.cc/paper_files/paper/2023/hash/37d00f567a18b478065f1a91b95622a0-Abstract.html | Large TAG benchmark suite; compares PLMs, GNNs, PLM–GNN co-training and topological pre-training. |
| Li et al., *GLBench: A Comprehensive Benchmark for Graph with LLMs* | NeurIPS 2024 D&B | https://nips.cc/virtual/2024/poster/97881 ; https://arxiv.org/abs/2407.07457 | Supervised: **LLM-as-enhancer most robust**. LLM-as-predictor is less effective, with "uncontrollable output." No clear scaling laws. |
| Wu et al., *When Do LLMs Help With Node Classification? A Comprehensive Analysis* (LLMNodeBed) | ICML 2025 | https://arxiv.org/abs/2502.00829 ; HTML read | 2,700+ models. LLM gains are large in semi-supervised settings and **marginal in supervised ones**. LLM-as-Encoder has the best cost/accuracy. Fine-tuned LLM-as-predictor does best only **with plenty of labels** and is unstable with few. **LLM-as-Encoder beats LMs by ~10% on heterophilic graphs.** Neighbour summaries help mainly on homophilic graphs. |
| Wu et al., *Exploring the Potential of LLMs for Heterophilic Graphs* (LLM4HeG) | NAACL 2025 | https://arxiv.org/abs/2408.14134 | A fine-tuned LLM edge discriminator labels edges homophilic or heterophilic and reweights GNN messages; distilled to small models. |
| Wang & Cheng, *Language Model-Enhanced Message Passing for Heterophilic Graph Learning* (LEMP4HG) | arXiv 2505.19762 (seen in search results only; **UNVERIFIED**, abstract not opened) | (search result) | An LM analyses paired node texts to guide message passing; 16 datasets. |

---

## 4. Evaluation practice

| Paper | Venue | URL opened | Claim |
|---|---|---|---|
| Saito & Rehmsmeier, *The Precision-Recall Plot Is More Informative than the ROC Plot … on Imbalanced Datasets* | PLOS ONE 2015 | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432 | ROC curves stay the same as imbalance changes, but PR curves move. PR is more informative when positives are rare. |
| Lipton, Elkan, Narayanaswamy, *Thresholding Classifiers to Maximize F1 Score* | arXiv 1402.1892 (ECML-PKDD 2014: **UNVERIFIED**) | https://arxiv.org/abs/1402.1892 | For calibrated scores, the F1-optimal threshold is half the optimal F1; an uninformative classifier maximises F1 by predicting all positive. Supports tuning the threshold on validation, not using 0.5. |
| Bouthillier et al., *Accounting for Variance in ML Benchmarks* | MLSys 2021 | https://arxiv.org/abs/2103.03098 | Variance from data sampling, initialisation and hyperparameters "impact markedly the results." Randomise several sources of variation when comparing. |
| Dror et al., *The Hitchhiker's Guide to Testing Statistical Significance in NLP* | ACL 2018 | https://aclanthology.org/P18-1128/ | A practical protocol for choosing significance tests (the page did not show which tests). |
| Berg-Kirkpatrick et al., *An Empirical Investigation of Statistical Significance in NLP* | EMNLP-CoNLL 2012 | https://aclanthology.org/D12-1091/ | Classic reference for the paired bootstrap. (The page showed metadata only, so the claim detail is **UNVERIFIED** from the page.) |
| Kapoor & Narayanan, *Leakage and the Reproducibility Crisis in ML-based Science* | arXiv 2207.07048 (Patterns 2023: **UNVERIFIED**) | https://arxiv.org/abs/2207.07048 | Leakage hit 329 papers across 17 fields. In civil-war prediction, complex ML did no better than LR once leakage was removed. Proposes "model info sheets." |
| Hays et al. 2023 (above), Platonov et al. 2023 (duplicates leakage), Chen et al. 2023 (LLM pre-training leakage) | | | The three leakage types relevant here: dataset shortcuts, duplicate nodes, memorised benchmarks. |

---

## 5. Where the project agrees or disagrees with the literature

| Project finding | Literature | Verdict |
|---|---|---|
| **MLP ≥ every GNN on TwiBot-20 (labelled-only graph, h_edge 0.532 vs 0.507 chance)** | H2GCN: MLP beats many GNNs under heterophily. Platonov 2023: separated GNNs usually beat ResNet *when edges carry label information*. Ma et al.: it depends on whether the heterophily is "good". Platonov 2209.06177: use h_adj and LI. | **Agrees.** With h_adj ≈ 0.05 the edges carry almost no label signal, so a GNN cannot be expected to help. Frame it as "consistent with LI/h_adj theory," not as a new discovery. |
| **SAGE ≈ MLP, GCN clearly worse** | Platonov 2023 and H2GCN: ego/neighbour separation is the most useful design under heterophily. | **Agrees** exactly. SAGE separates; GCN mixes the ego into the average. |
| **FAGCN beats GCN but not MLP** | Platonov 2023: specialised heterophily models (FAGCN included) are rarely better than classic separated GNNs; some fall below ResNet. | **Agrees.** |
| **GCN gets worse on the denser, more homophilous bridge graph; SAGE/FAGCN do not** | H2GCN / Ma et al.: GCN's averaging hurts when neighbour labels are mixed. The bridge graph is still low (h_adj roughly 0.2 by my estimate). | **Plausible and consistent.** Report h_adj/LI for both graphs to back it up. |
| **Graph-only features are weak (0.577)** | TwiBot-22 graph ablation: removing the graph costs BotRGCN/RGT 3–4 points. BotSay structure expert 0.904 F1 on TwiBot-20. | **Apparent disagreement**, explained by setup. Those models use metadata, typed R-GCN relations and the **full graph with 217k support users**, and the BotSay structure expert puts neighbour info into the prompt (it also uses metadata). This is the strongest place to discuss "graph construction matters." The bridge check is a partial answer. |
| **Encoder choice matters most; size is not everything (Twitter-RoBERTa > Qwen3-Embedding)** | Chen 2023 Obs 3–4: sentence-embedding models are strong and scaling LLaMA does not close the gap. LLMNodeBed: LLM encoders help most on heterophilic graphs. | **Agrees.** |
| **Zero-shot reasoner loses; adding neighbours does not reliably help** | Chen 2023 Obs 11 and 16 (structure prompts can hurt under heterophily); GLBench (predictor less effective); LLMNodeBed (summaries help mainly on homophilic graphs). | **Agrees** strongly. |
| **LoRA + DGP 0.711 < MLP; with 1,300 users MLP drops to 0.726, so most of the gap is training size** | LLMNodeBed: LLM-as-predictor needs many labels and LLM gains are marginal when labels are plentiful. BotSay: 1,000 instruction-tuning examples suffice (but with metadata and ChatGPT). **DGP paper: with ~1,300 training nodes DGP beats MLP by 4.8–9.8 macro-F1.** | **Partly agrees** with LLMNodeBed and GLBench, **disagrees** with DGP's own result. Likely reasons: (i) 1.7B vs 8B model; (ii) very sparse metapaths (0.8–1.8 neighbours kept), so the summaries add little; (iii) h_adj ≈ 0 here, while DGP's gain comes from neighbour summaries; (iv) our MLP uses a strong LLM embedding of the same text, whereas DGP's Yelp MLP used handcrafted features; (v) one seed vs five. The `target` variant run is the key ablation: DGP's own Table 2 has target-only LLM below MLP. |
| **Excluding "verified" (alone 0.803)** | Hays et al. 2023: simple rules on a few features nearly match SOTA because of collection/labelling artefacts. BotSay's metadata expert (with verified) beats its text expert. | **Agrees** and is well justified. Make it a headline methodological point. |
| **Instagram: all models within ~0.03 of LR** | GraphAdapter: ROC-AUC 0.60–0.65 for every method on Instagram. | **Agrees.** Instagram is intrinsically hard; cite GraphAdapter as its source. |

---

## 6. How to make the project better (ranked by value per hour)

Time budget: today Tue 3 Nov to Sat 7 Nov night. One GPU job at a time; the CPU work below can run beside the GPU job.

| # | What to do | Motivated by | Expected effect on conclusions | Cost (16 GB Mac) | Fits by Sat? |
|---|---|---|---|---|---|
| 1 | **Report adjusted homophily (h_adj) and label informativeness (LI)** for the labelled-only, bridge-only and combined graphs, per relation (friend / follow), for TwiBot-20 and Instagram. step1 already has class-insensitive homophily (Lim et al.), so this adds two functions. | Platonov et al. 2209.06177; Ma et al. 2022 | Turns "barely above random" into a standard, comparable number (~0.05). Explains *why* MLP ≥ GNN and why GCN suffers. The per-relation split tests BotSCL's follower/following claim. | < 15 min CPU | Yes |
| 2 | **Paired bootstrap 95% CIs on the test set** for the headline comparisons: MLP vs SAGE vs FAGCN vs GCN (Qwen2.5), LoRA-DGP vs LoRA-target vs MLP-1300. Also show seed std next to each mean. step4 does not save per-user scores, so add a save and rerun (it is cheap). step6 scores are already in npz. | Bouthillier et al. 2021; Berg-Kirkpatrick et al. 2012; Dror et al. 2018 | Tells you which gaps are real. LoRA vs MLP-1300 (0.015) is probably *not* significant on 1,183 users, which supports "the gap is training size." SAGE vs MLP in B (0.624 vs 0.613) likely is not either. | ~30–60 min CPU (rerun step4 for 1–2 encoders) plus a 10-line bootstrap | Yes |
| 3 | **Make PR-AUC a co-primary metric in setting B** (already computed in `evaluate`); keep macro-F1 at the validation-chosen threshold. Add one sentence giving the PR-AUC chance level (≈0.09 in B). | Saito & Rehmsmeier 2015; DGP uses AUPRC as its headline | Aligns with DGP's metric, so the replication compares like for like. May change the ranking of LoRA vs MLP in B. | 0 compute (tables) | Yes |
| 4 | **Learning curve**: Qwen2.5 + LR/MLP at 325 / 650 / 1,300 / 2,600 / 8,278 training users (3 seeds), with the LoRA point overlaid. | LLMNodeBed (predictor needs labels); BotSay (1k examples); control_train_size already exists | Turns the strongest new finding ("most of the gap is data") into a figure. Shows where MLP crosses LoRA, if it does. | ~20–40 min CPU | Yes |
| 5 | **GADBench-style baseline**: gradient-boosted trees (sklearn HistGradientBoosting) on [x, mean-of-neighbours x, degree], with x = PCA-64 of the Qwen2.5 embedding; also MLP on the same concatenation. | GADBench 2023 (tree ensembles with neighbour aggregation beat GAD GNNs); LINKX | A cheap, well-cited "graph as features" baseline. If it ≈ MLP, the graph carries no extra signal; if it beats SAGE, aggregation helps once kept separate. Either way it strengthens finding 2. | ~30 min CPU | Yes |
| 6 | **Small, equal tuning budget for all step 4 models** (same grid for each: lr {1e-3, 1e-2}, hidden {64, 256}, dropout {0.2, 0.5}, FAGCN eps {0.1, 0.3, 0.5}), Qwen2.5 encoder only, selected on validation macro-F1. | Luo et al. 2024 (classic GNNs strong when tuned); Shchur et al. 2018 | Removes the main reviewer objection that the GNNs were untuned. Expected: rankings unchanged and gaps narrower. If MLP still wins, the claim is much stronger. | ~2–3 h CPU | Yes (start Wed) |
| 7 | **Relation-typed GNN (R-GCN on friend vs follow)** plus a GCN with ego concatenation (H2GCN-style skip). PyG `RGCNConv`. | BotRGCN (TwiBot standard); H2GCN / Platonov separation; BotSCL (relations differ in heterophily) | Answers "would the TwiBot-standard GNN behave differently?" Closes the gap with TwiBot-22's graph-ablation finding. | ~45 min CPU | Yes |
| 8 | **LoRA seeds**: 2 more seeds of the better step-6 variant on TwiBot-20 A (DGP uses 5 seeds). | Bouthillier 2021; DGP protocol | The 0.711 vs 0.726 comparison currently rests on one run; it needs a std. | 2 × ~3–6 h GPU | Probably 1–2 seeds; schedule overnight Wed/Thu |
| 9 | **"FLAG-inspired" baseline instead of full FLAG**: no code exists and LLM fine-tuning for node enhancement is not feasible. Implement FLAG's *semantic-similarity neighbour sampling* (keep the top-k neighbours by embedding cosine, k ∈ {2, 4}) before SAGE/GCN. Name it honestly. | FLAG KDD 2025; CARE-GNN / PC-GNN (similarity-based neighbour selection) | Fills the "FLAG baseline ⏳" row with something defensible. Tests whether pruning dissimilar neighbours rescues GCN under heterophily. | ~45 min CPU | Yes |
| 10 | **Leakage / duplicate check**: hash bios and tweets and count exact or near duplicates across train/val/test (bot farms repeat text), then report test macro-F1 with duplicates removed. | Platonov 2023 (duplicate-node leakage); Hays 2023; Kapoor & Narayanan 2022 | If many test bots share text with training bots, all text models are inflated; better to find out before the markers do. Likely a small effect, but it protects the conclusions. | ~30 min CPU | Yes |

**Recommended GPU queue:** current `target` run → step 6 B `dgp` (or LoRA seed 2 on A if B is less important to the story) → one more seed overnight Fri. Everything in rows 1–7, 9 and 10 runs on CPU in parallel. Skip the M ∈ {4, 8, 16} sweep: with 0.8–1.8 neighbours kept per metapath, M = 8 is rarely binding, so the sweep would show nothing. Say this in the report instead.

**Not worth doing by Saturday:** GraphGPT/LLaGA/InstructGLM reproduction, GloGNN/ACM-GNN, rebuilding the graph with all support users for steps 5–6, Qwen3-8B.

---

## 7. Mapping to the report template

### Literature Review (5 paragraphs)

1. **Graph fraud and bot detection, and why heterophily matters.** Start from camouflage: CARE-GNN, PC-GNN, then spectral views (BWGNN, GHRN) and label-aware aggregation (GAGA). In bots, TwiBot-20/22 established graph methods (BotRGCN, RGT, BotMoE) and found graphs help. BotSCL shows heterophilous bot edges hurt aggregation. *Gap seed:* these results come with metadata and full support graphs.
2. **Heterophily GNNs and the evaluation critique.** H2GCN, GPR-GNN, FAGCN, GloGNN, ACM-GNN proposed specialised designs. Platonov 2023 (ICLR), Ma et al. 2022 and Luo et al. 2024 showed classic, separated GNNs are as good when evaluation is fair. Platonov 2209.06177 gives h_adj and LI; GADBench shows simple aggregation plus trees is a strong baseline. *Point:* whether a graph helps is an empirical, per-graph question, and it should be measured with h_adj/LI.
3. **LLMs on graphs: three roles.** Use the taxonomies of Chen et al. 2023, Li et al. 2024 (IJCAI) and Jin et al. 2024 (TKDE). Enhancer/encoder: TAPE, GraphAdapter, LLM4HeG. Predictor: InstructGLM, GraphGPT, LLaGA. Benchmarks (CS-TAG, GLBench, LLMNodeBed) find enhancers/encoders most robust and predictors data-hungry. LLMNodeBed is the only one that looks at heterophily, and finds encoders help most there.
4. **LLM-based fraud and bot detection (2024–2026).** BotSay (instruction-tuned LLM experts, but uses metadata), LMBot, GuARD, FLAG (enhancer), MLED, DGP (predictor with dual-granularity prompts, AAAI 2026), FraudCoT (the DGP authors returning to enhancer + GNN), CAMERA (semantic camouflage).
5. **Evaluation pitfalls and the gap.** Hays 2023 on bot-dataset shortcuts, Platonov on duplicates, Chen et al. on LLM pre-training leakage, Saito & Rehmsmeier on imbalance metrics, Bouthillier on variance. **The gap (a real one):** no study puts the *three LLM roles* (encoder + MLP/GNN, zero-shot predictor, fine-tuned graph-prompted predictor such as DGP) on the *same heterophilous text graph*, with metadata shortcuts removed, a matched training-size control, and one validation-threshold protocol. DGP compares against GNN baselines but not against a strong LLM-embedding MLP trained on the same number of labels. LLMNodeBed covers heterophily but not fraud or bots, and not the 1:10 rare-class setting. The project fills exactly this gap.

### Research Methodology: what the literature justifies

- **Text-only features, metadata excluded:** Hays et al. 2023 (shortcuts from collection and labelling); BotSay's metadata expert includes "verified". Report the shortcut separately, as REPORT.md already does.
- **Official TwiBot-20 split:** Feng et al. 2021; Shchur et al. 2018 (splits change rankings), so fix the splits across all models.
- **Encoder ladder with a fixed downstream model:** Chen et al. 2023 Obs 3–4 (objective vs size); LLMNodeBed (LLM-as-encoder).
- **MLP as the key control, plus GCN / SAGE / FAGCN:** H2GCN (MLP beats GNNs under heterophily); Platonov 2023 (separation; specialised vs classic); FAGCN as the representative signed-propagation model. Add h_adj and LI (Platonov 2209.06177) as the homophily measure.
- **Same hyperparameters for all GNNs, or one equal tuning budget:** Luo et al. 2024; Shchur et al. 2018.
- **Zero-shot reasoner with target / raw neighbours / summaries:** Chen et al. 2023 (Obs 15–16); GLBench.
- **DGP replication with ~1,300 training nodes, metapaths, budget ~10, M = 8, LoRA:** DGP paper's own protocol (Table 1 sizes; hyperparameter ranges). State the deviations: 1.7B vs 8B, 1 seed vs 5, threshold rule not given by the paper.
- **Training-size control:** motivated by LLMNodeBed's supervised vs semi-supervised finding and DGP's 1,300-node regime.
- **Validation-chosen threshold for macro-F1:** Lipton et al. 2014.
- **Macro-F1 plus AUROC plus PR-AUC; setting B at 1:10:** Saito & Rehmsmeier 2015; DGP's AUPRC headline; GADBench (AUPRC/AUROC as standard for GAD). Do not call 1:10 "FLAG-style" unless the FLAG paper is checked.
- **Seeds, std and bootstrap CIs:** Bouthillier et al. 2021; Berg-Kirkpatrick et al. 2012; Dror et al. 2018.
- **Bridge-graph robustness check:** TwiBot-20/22 use support users as neighbours; this answers the "graph construction" objection.

---

## Sources (all opened during this search unless noted)

- https://arxiv.org/abs/1609.02907 (GCN)
- https://arxiv.org/abs/1706.02216 (GraphSAGE)
- https://arxiv.org/abs/2006.11468 (H2GCN)
- https://arxiv.org/abs/2006.07988 (GPR-GNN)
- https://arxiv.org/abs/2101.00797 (FAGCN)
- https://arxiv.org/abs/2110.14446 (LINKX)
- https://arxiv.org/abs/2106.06134 (Ma et al.; ICLR 2022 via https://iclr.cc/virtual/2022/poster/6329 search result)
- https://arxiv.org/abs/2205.07308 (GloGNN)
- https://arxiv.org/abs/2210.07606 (ACM-GNN)
- https://arxiv.org/abs/2302.11640 and https://arxiv.org/html/2302.11640 (Platonov ICLR 2023; venue via https://iclr.cc/virtual/2023/poster/11466 search result)
- https://arxiv.org/abs/2209.06177 and https://arxiv.org/html/2209.06177 (adjusted homophily, LI)
- https://arxiv.org/abs/2406.08993 (Classic GNNs strong baselines)
- https://arxiv.org/abs/1811.05868 (Pitfalls of GNN evaluation)
- https://arxiv.org/abs/2008.08692 (CARE-GNN)
- https://github.com/PonderLY/PC-GNN (PC-GNN)
- https://arxiv.org/abs/2205.15508 (BWGNN)
- https://github.com/blacksingular/GHRN (GHRN)
- https://arxiv.org/abs/2302.10407 (GAGA)
- https://arxiv.org/abs/2306.12251 (GADBench)
- https://arxiv.org/abs/2507.11997 (MLED)
- https://arxiv.org/abs/2412.03930 (GuARD)
- https://chatpaper.com/pt/paper/175073 (FLAG abstract, aggregator); https://dl.acm.org/doi/10.1145/3711896.3737220 (403, not readable)
- https://github.com/AI4Risk/awesome-graph-based-fraud-detection (curated list)
- https://arxiv.org/abs/2507.21653 and https://arxiv.org/html/2507.21653 (DGP); https://github.com/Xtra-Computing/DGP (README only); AAAI 2026 venue via search result https://ojs.aaai.org/index.php/AAAI/article/view/38541 (page blocked)
- https://arxiv.org/abs/2601.22949 and https://arxiv.org/html/2601.22949v1 (FraudCoT)
- https://arxiv.org/abs/2605.20032 (CAMERA)
- https://arxiv.org/abs/2605.28524 (LGSPF)
- https://arxiv.org/abs/2106.13088 (TwiBot-20)
- https://arxiv.org/abs/2106.13092 (BotRGCN)
- https://arxiv.org/abs/2109.02927 (RGT)
- https://arxiv.org/abs/2206.04564 and https://arxiv.org/html/2206.04564 (TwiBot-22)
- https://arxiv.org/abs/2301.07015 (Hays et al.)
- https://arxiv.org/abs/2306.07478 (BotSCL)
- https://arxiv.org/abs/2304.06280 (BotMoE)
- https://arxiv.org/abs/2306.17408 (LMBot)
- https://arxiv.org/abs/2402.00371 and https://arxiv.org/html/2402.00371 (What Does the Bot Say?)
- https://arxiv.org/abs/2307.03393 and https://arxiv.org/html/2307.03393 (Chen et al.)
- https://arxiv.org/abs/2311.12399 (Li et al. survey)
- https://arxiv.org/abs/2312.02783 (Jin et al. survey)
- https://arxiv.org/abs/2305.19523 (TAPE)
- https://arxiv.org/abs/2310.13023 (GraphGPT)
- https://arxiv.org/abs/2308.07134 (InstructGLM)
- https://arxiv.org/abs/2402.08170 and https://proceedings.mlr.press/v235/chen24bh.html (LLaGA)
- https://arxiv.org/abs/2402.12984 and https://arxiv.org/html/2402.12984 (GraphAdapter; Instagram dataset)
- https://proceedings.neurips.cc/paper_files/paper/2023/hash/37d00f567a18b478065f1a91b95622a0-Abstract.html (CS-TAG)
- https://arxiv.org/abs/2407.07457 and https://nips.cc/virtual/2024/poster/97881 (GLBench)
- https://arxiv.org/abs/2502.00829 and https://arxiv.org/html/2502.00829 (LLMNodeBed)
- https://arxiv.org/abs/2408.14134 (LLM4HeG)
- https://arxiv.org/html/2505.19762v1 (LEMP4HG, search result only, not opened: UNVERIFIED)
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432 (Saito & Rehmsmeier)
- https://arxiv.org/abs/1402.1892 (Lipton et al.)
- https://arxiv.org/abs/2103.03098 (Bouthillier et al.)
- https://aclanthology.org/P18-1128/ (Dror et al.)
- https://aclanthology.org/D12-1091/ (Berg-Kirkpatrick et al.)
- https://arxiv.org/abs/2207.07048 (Kapoor & Narayanan)
