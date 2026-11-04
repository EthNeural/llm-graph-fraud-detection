# Slicing a 1M-node graph: what researchers and industry do, and what we should do with TwiBot-22

*COMP8851 research note, 4 Nov 2026. Notation: **[V]** = verified (number or quote from a source I opened, listed at the end). **[I]** = my inference or back-of-envelope arithmetic, not taken from a source.*

---

## Executive summary

1. **Industry mostly does not slice the graph. It slices the computation.** Production systems keep the whole graph and sample a small neighbourhood around each target node (GraphSAGE/PinSAGE, LinkedIn's LiGNN, Ant's AGL, AWS's k-hop inference). Or they precompute graph aggregates once and feed them to a cheap model (Facebook's DEC with over 20,000 aggregated neighbour features, Uber's RGCN scores fed into a GBDT, GADBench's RF/XGB-Graph). The *training labels* are what get subsampled, never the *neighbourhoods*. [V for each system, see section 3]
2. **Every graph-slicing method moves the quantities we study.** Random-node induced subgraphs go almost empty. BFS and random walks over-sample hubs. Cluster/community slices skew the label mix. Balanced slices change the chance level of edge homophily. [V for the mechanisms, I for how big the effect is on TwiBot-22]
3. **TwiBot-22's own baselines trained on the full 1M-user graph, full batch.** The official BotRGCN `train.py` does one forward pass over all nodes and edges per epoch (the `NeighborLoader` import is never used), on 2080 Ti GPUs with 11 GB [V]. So 1M nodes is not a problem in itself. The LLM embedding cost is.
4. **Recommendation (ranked):** (1) run every non-LLM baseline on the **full 1M graph**, using cheap features and precomputed propagation (SGC/SIGN-style) plus graph-feature GBDT, on CPU. Then run the LLM steps on a **stratified random sample of target users at natural 14% prevalence**, and build each sampled user's graph context from its ego-net in the *full* graph. (2) Only if (1) breaks, build a 20-50k "seed users + 1-hop ego-net" slice. (3) Use the paper's 10 sub-communities only as a secondary generalisation check, never as the main graph testbed.

---

## 1. Graph sampling methods and their known biases

**Leskovec & Faloutsos, "Sampling from Large Graphs" (KDD 2006)** compared 10 samplers on 5 graphs of 3k-76k nodes. They measured only *structural* properties (degree, connected components, hop-plot, clustering), not labels [V]:

- **Random node (RN) and the induced subgraph.** This sampler "does not retain power-law degree distribution" (citing earlier work). At small sample sizes RN and RNN "are far off" on clustering [V]. *Arithmetic for us [I]:* a uniform 2% node sample (20k of 1M) keeps each edge with probability about 0.02² = 0.0004. That leaves roughly 1,500 of the ~3.7M following/follower edges, which is the same near-emptiness we measured in the paper's sub-communities.
- **Random edge (RE/RNE/HYB).** The resulting graphs "will be very sparsely connected... large diameter and will not respect community structure". These samplers were the worst on average (D-statistic 0.37-0.44) [V].
- **Random walk (RW), random jump (RJ), forest fire (FF).** These scored best on average (RW 0.202, FF 0.205). But RW/RJ/RDN "are biased towards high degree nodes and densely connected parts of the graph… the tail of the degree distribution is over estimated" [V]. Their RW used a 0.15 restart probability, and their best scale-down FF setting used p_f ≥ 0.6, where "every fire eventually floods the connected component" [V]. The authors conclude that "there is no single perfect answer to graph sampling" [V].
- **BFS/snowball.** Gjoka et al. (INFOCOM 2010, Facebook crawl) show that BFS and simple RW are biased toward high-degree nodes, with low-degree nodes "under-represented by two orders of magnitude" under BFS. Metropolis-Hastings RW (MHRW) or re-weighted RW (RWRW) correct this and recover a uniform stationary distribution [V].

**Why this matters for TwiBot-22 specifically [I].** In our data, humans have higher in-degree than bots (in-degree AUROC 0.65). So any degree-biased sampler (BFS, snowball, plain RW, forest fire) will (a) pull the bot share below 14% and (b) make degree an even stronger label shortcut. That is the same crawl artefact we already flagged, now amplified by our own sampling. The dataset itself was built by a BFS from @NeurIPSConf that sampled 6 of up to 1,000 followers/followees per user under a "diversity-aware" rule [V]. We should not stack a second BFS on top of it.

**Training-time samplers (they sample the computation, not the dataset):**
- **GraphSAGE** (NeurIPS 2017): fixed-fanout neighbour sampling, inductive [V].
- **PinSAGE** (KDD 2018): random-walk visit counts choose the top-T "important" neighbours. It ran on 3B nodes / 18B edges and 7.5B training examples, with MapReduce inference to embed every node. The importance-based neighbourhoods gave "46% performance gain" offline over k-hop neighbourhoods [V].
- **Cluster-GCN** (KDD 2019): METIS partitions. It trained a 3-layer GCN on Amazon2M (2M nodes / 61M edges) in 2.2 GB, against 11.2 GB for VR-GCN [V]. The authors warn that "the distribution of a cluster could be different from the original data set". Most cluster batches had "low label entropy". Their fix is to merge several random clusters per batch, which restores between-cluster links [V]. *This is the community-slicing bias, stated by the method's own authors.*
- **GraphSAINT** (ICLR 2020): samples node, edge or random-walk subgraphs, then corrects the bias with normalisation coefficients α (aggregator) and λ (loss). These are estimated by pre-sampling and counting how often each node and edge appears. In the authors' words, "a sampler that preserves connectivity… will almost inevitably introduce bias" [V]. The lesson for us [I]: if you sample non-uniformly, re-weight. A fixed slice with no re-weighting gets no such correction.

**Precompute-once propagation (no sampling at all):**
- **SGC** (ICML 2019) removes the nonlinearities, so a GCN becomes a fixed low-pass filter plus a linear classifier, with "up to two orders of magnitude speedup" [V].
- **SIGN** (2020) precomputes A_k·X for several operators (adjacency powers, PPR, triangle-based) "once", then trains an MLP. On ogbn-products, an epoch takes 1.04 s against 2.89 s (GraphSAINT) and 13.34 s (Cluster-GCN), and inference takes 2.86 s against about 93-95 s. It scaled to ogbn-papers100M (110M nodes, 1.5B edges) [V].

For heterophily, separate per-relation and per-direction operators (in-following, out-following, interactions) are a natural fit [I]. They are close to FAGCN/LINKX-style "don't just average neighbours".

## 2. How fraud/bot benchmarks handle size

| Benchmark | Size [V] | What they do about size [V] |
|---|---|---|
| **GADBench** (Tang et al., NeurIPS 2023 D&B) | 10 datasets, up to T-Social 5,781,065 nodes / 73.1M edges (3.0% anomalies). DGraph-Fin 3,700,550 / 4.3M (1.3%). T-Finance 39,357 / 21.2M (4.6%). Elliptic 203,769 (9.8%). YelpChi 45,954 (14.5%). Amazon 11,944 (9.5%) | **No graph subsampling.** They *subsample labels*: the semi-supervised setting uses **100 labels (20 anomalous + 80 normal)** for every dataset, over 10 random splits. The fully-supervised setting uses existing splits or 40/20/40. One 48 GB A6000. Methods that took more than a day were marked OOT. Headline finding: tree ensembles with neighbour aggregation (RF-/XGB-Graph) beat specialised GNNs (+2.0 AUROC fully-supervised, +2.8 semi-supervised). |
| **DGraph** (Huang et al., NeurIPS 2022 D&B) | 3,700,550 nodes, 4,300,999 edges. 1,225,601 labelled (1.27% fraud). **2,474,949 unlabelled "background" nodes** | Keeps the background nodes on purpose. As background nodes are removed, GCN AUC falls "from 0.76 to 0.72". Random 70/15/15 split. **Direct evidence that cutting context away from labelled nodes costs accuracy.** |
| **TwiBot-22** (Feng et al., NeurIPS 2022 D&B) | 1M users (139,943 bots), 92.9M nodes, 170.2M edges, 14 relations | All 35 baselines run on the full dataset with a "7:2:1 random split" (our measured split is not random: 7.8% / 28% / 29.4% bots). BotRGCN trains **full-batch** on all users. Only Dehghan et al. hit "out-of-memory". The 10 sub-communities (5k bot + 5k human each) serve as generalisation folds. Reported results: BotRGCN acc 79.7 / F1 57.5, GAT 79.5, GCN 78.4, RGT 76.5. |
| **FLAG** (Yang et al., KDD 2025) | Instagram/Reddit repurposed | Class ratio set to **1:10** to mimic fraud imbalance. This is label-ratio subsampling, not structural slicing (secondary-source summary, not checked against the ACM version). |
| **"What does the bot say?"** (Feng et al., ACL 2024) | TwiBot-20 test 1,183. TwiBot-22 test **340** | "We downsample the original training split into 1,000 users due to limits in computational budget". The 340-account TwiBot-22 test split comes from **Tan et al. 2023 = "BotPercent: Estimating Bot Populations in Twitter Communities" (Findings of EMNLP 2023)** [V]. I could not find how the 340 were chosen in the BotPercent text [unverified]. BotPercent itself uses the 10 communities (10k, 50/50) plus resampled 5k communities at 10-90% bot share. |
| **BotSCL** (Wu et al., ICPR 2024) | TwiBot-20/22 | Motivated by bot heterophily. Sampling details not found in the abstract [unverified]. |

**Takeaway [I].** The serious benchmarks keep the graph whole and vary the *label budget*. The LLM papers subsample *target users* because of inference cost, but that sampling is node-level and does not cut the neighbourhood away.

## 3. Industry practice: big graphs without slicing (the centrepiece)

**The pattern across companies [I, synthesised from the sources below]:** (i) store the whole graph; (ii) for each *target* node, fetch a bounded k-hop neighbourhood (sampled, importance-weighted or truncated); (iii) or precompute neighbourhood aggregates offline as features; (iv) train on labelled targets at natural or re-weighted prevalence; (v) split by time; (vi) serve on ego-subgraphs.

| Approach | Industry evidence [V] | Fits 16 GB laptop at 1M nodes / ~3-10M edges? [I] |
|---|---|---|
| **Precomputed neighbourhood features into GBDT/DNN** | **Facebook DEC** (USENIX Security 2021): over 20,000 "deep features" aggregated from direct and indirect neighbours. "Deep" means graph fan-out, not network depth. Trained in stages: lots of noisy automated labels, then a small human-labelled set. In production more than 2 years, removing a further ~27% of abusive accounts. **Uber** (2021 blog): RGCN driver/rider fraud scores added as 2 features to a 200-feature production model gave about 15% better precision; the scores ranked 4th and 39th by importance. Uber built "one giant graph" rather than city graphs and filtered very high-degree entities. **GADBench**: RF/XGB-Graph beat GNNs. | **Yes, easily.** A sparse matrix times features for 3 relations × 2 directions × 2 hops is seconds to minutes in scipy. LightGBM on 1M × a few hundred features fits in a few GB. |
| **Precompute-once propagation (SGC/SIGN/SAGN)** | SIGN on papers100M (110M nodes). Inference an order of magnitude faster than sampling methods. | **Yes.** 1M × 64 float32 = 256 MB per operator. 9-12 operators is about 2.3-3 GB (half in float16). Training is then just an MLP. |
| **Full-batch GNN** | TwiBot-22 BotRGCN official code trains full-batch on 1M users. | **Probably yes on CPU or MPS** with low-dimensional (≤128) features and 2 layers: activations about 0.5 GB per layer, CSR adjacency about 0.1 GB. Peak RSS needs measuring. |
| **Mini-batch neighbour sampling (GraphSAGE/PinSAGE, PyG `NeighborLoader`)** | Pinterest PinSAGE (3B nodes). **LinkedIn LiGNN** (KDD 2024): 2-hop sampling with PPR/forward-push and random walks, temporal neighbour sampling, 7x training speed-up from adaptive neighbour sampling and batching. AWS reference model: RGCN (DGL). | **Yes,** but slower than precompute on CPU. Python-side sampling dominates (SALIENT measured PyG single-thread sampling as the bottleneck on ogbn-products). A fallback if full-batch runs out of memory. |
| **Cluster-GCN / GraphSAINT** | Amazon2M in 2.2 GB. Used widely in academia, less visible in production blogs. | **Yes**, but needs METIS (`pymetis`) and the normalisation done correctly. More setup than SIGN, little gain for 2-layer models. |
| **Ego-subgraph (k-hop) inference** | **AWS** DGL fraud solution: "extract a subgraph induced by k-hop out-neighborhood of the target nodes". At k=2 a 1,030-transaction batch scores in 5.4 s, and the subgraph covers 54% of the training graph (100% at k=3). Training graph 217,935 nodes / 2.65M edges, time-split 80/20. **Ant AGL** (VLDB 2020): builds a self-contained k-hop neighbourhood per node via MapReduce. Billions of nodes. 2-layer GAT trained in about 14 h, inference in 1.2 h. Ant InfDetect: up to 100M nodes and billions of edges. | **Yes, and this is the key trick for our LLM steps:** sample target users, then pull each one's 1-hop neighbours from the *full* graph for the prompt. The neighbourhood is never truncated by a slice. |
| **Distributed graph engines** (AliGraph, DGL-distributed, GraphLearn-for-PyTorch, Neptune ML) | AliGraph: 492.9M vertices / 6.82B edges, caching gives 40-50% faster training. Grab uses RGCN with AWS Neptune. Twitter's TwHIN embeds a graph of over 1B nodes, with offensive-content detection as one downstream task. | **No and not needed.** These exist for 10⁸-10¹⁰ edges across many machines. At 10⁷ edges a single process is enough. Cite them only to place our setup in context. |

**Other industry habits worth copying [I unless marked]:**
- **Time-based splits.** AWS splits 80/20 by time [V]. LiGNN models temporal neighbourhoods [V]. TwiBot-22 has no clean per-user label time, so a random stratified re-split is our honest substitute.
- **Natural-prevalence evaluation.** Balance the classes for training if you must (re-weighting, focal loss, downsampling negatives), but report on a holdout at true prevalence with PR-AUC or precision@k (GADBench reports AUROC, AUPRC and Rec@K [V]).
- **Weak labels then strong labels** (DEC [V]). This matches TwiBot-22, whose Snorkel labels came partly from GCN/GAT/R-GCN models [V]. There is a circularity risk: graph models can look good by matching labels that graph models helped create.

## 4. Evaluation hygiene when subsampling

1. **Prevalence.** The paper's sub-communities are forced to 50/50. Edge-homophily chance then drops from about 0.86 to about 0.5, so raw homophily numbers are not comparable across slices [I]. Report **adjusted homophily** and **label informativeness**. Platonov et al. (NeurIPS 2023) designed adjusted homophily to correct for class imbalance and to be comparable across datasets. LI measures how much a neighbour's label tells you, and "is not a measure of homophily" [V]. Both are still *edge-sample statistics*: a degree-biased sampler changes which edges survive, so recompute them on every slice [I].
2. **Transductive vs inductive.** In transductive full-graph training, test nodes' features flow into training nodes' representations. That is standard (TwiBot-22 and GADBench do it), but say so. For the LLM step, the prompt may include neighbour *features*, never neighbour *labels* drawn from val/test [I].
3. **Leakage.** Platonov et al. (ICLR 2023) found that duplicate nodes in squirrel/chameleon leaked across train and test and "strongly affects GNN performance" [V]. Check for duplicate descriptions and tweets across our splits [I].
4. **Split skew.** The official TwiBot-22 split (7.8% / 28% / 29.4% bots) is not the "7:2:1 random" split the paper describes. Report results on both the official split and a stratified random re-split [I].
5. **Metrics.** Report macro-F1 *and* PR-AUC (plus AUROC) at natural prevalence. Report F1 at a threshold tuned on validation, not at 0.5 [I].
6. **Show the slice is representative.** For any slice, put a table next to the full graph with: number of users, bot %, isolated-node share per relation, degree distribution (KS D-statistic against full, the Leskovec & Faloutsos test [V]), edge homophily and its chance level, adjusted homophily, LI, and the in-degree-only AUROC (crawl-artefact check) [I].

## 5. Recommendation for this project (deadline Sat 7 Nov, laptop only)

| Rank | Option | Pros | Cons / bias introduced | Cost on 16 GB laptop [I] |
|---|---|---|---|---|
| **1 (do this)** | **Full 1M graph for every non-LLM model, plus a node-sampled LLM subset with full-graph ego-nets.** Features: cheap per-user metadata, minus `verified` (matching our shortcut check), plus cheap text (TF-IDF/hashing → SVD-64, or a small multilingual-E5 model on descriptions only if time allows). Models: MLP; SGC/SIGN-style precomputed propagation with separate in/out operators per relation; GBDT on [x, mean-neighbour-x per relation, degrees] (GADBench's XGB-Graph); full-batch GCN/SAGE; LINKX (built for large non-homophilous graphs). LLM steps (embeddings, DGP LoRA) on a **stratified random sample at 14% prevalence** (e.g. about 1,300 train + 3-5k test), with the prompt's neighbour summary taken from each user's 1-hop neighbourhood in the **full** graph. Also score the GNNs on exactly those test users for a paired comparison. | Matches industry practice and both benchmarks (GADBench, DGraph). No structural sampling bias. The heterophily numbers we already measured apply directly. Paired comparison on one shared test set. | The LLM and GNN arms use different feature sets (LLM text vs cheap features). Report a "GNN + LLM-embedding on the subset" row to bridge them. Transductive leakage is standard but must be stated. | Feature build and propagation: under 1 h. Each GBDT/MLP/SIGN run: minutes. Full-batch GNN: about 10-60 min per run, peak RSS to be measured. LLM embedding of about 5-6k users: same as current runs. |
| **2 (fallback)** | **Own slice: 20-50k target users, stratified at 14%, plus their 1-hop neighbours (induced on targets ∪ neighbours, neighbours unlabelled or labels hidden).** Not RW/snowball. | Fits the whole LLM pipeline end to end. Keeps each target's real neighbourhood (DGraph "background nodes" logic). | Neighbour-of-neighbour edges are missing. Hubs come in as neighbours (that is fine as long as they are not targets). Measure and report the item-6 table. | About 1 day of work plus embeddings of 20-50k users (at the edge of feasible). |
| **3 (secondary only)** | **The paper's sub-communities**, e.g. 1-2 of the 10. | Comparable to BotPercent and TwiBot-22's own generalisation tests. | 50/50 balance distorts prevalence and the homophily chance level. 290-7,900 follow edges per 10k users and as few as 2-10% of users with any follow edge, so the "graph" barely exists. Hashtag clusters select on content. | Cheap, but it answers a different question. |
| **Avoid** | Random-node induced subgraph; BFS/snowball/plain RW/forest fire from random seeds. | Simple. | Induced subgraphs go empty (about 0.0004 edge retention at 2%). Degree-biased samplers under-sample bots and strengthen the in-degree artefact. | n/a |

**Suggested schedule [I].** Wed-Thu: build cheap features, run propagation/GBDT/MLP/SIGN on 1M (official split and a stratified re-split), and draw the stratified LLM sample with ego-net summaries from the full graph. Fri: full-batch GCN/SAGE/FAGCN/LINKX on 1M (if memory allows; otherwise `NeighborLoader` with fanout [10, 5]), plus LLM/DGP runs on the sample. Sat: paired evaluation on the shared test set (macro-F1, PR-AUC, AUROC), a representativeness table for the sample against the full graph, and an optional sub-community run as a generalisation check.

**What to write in the report [I].** "Following GADBench and DGraph, we keep the full graph and subsample only labelled target users for the LLM components. Each sampled user's neighbourhood is drawn from the full graph (ego-subgraph inference, as in AGL/AWS). We avoid BFS/RW slicing because degree-biased samplers interact with TwiBot-22's crawl artefact (in-degree AUROC 0.65)."

---

## Sources

- Leskovec, J. & Faloutsos, C. *Sampling from Large Graphs.* KDD 2006. https://cs.stanford.edu/people/jure/pubs/sampling-kdd06.pdf
- Gjoka, M. et al. *Walking in Facebook: A Case Study of Unbiased Sampling of OSNs* (INFOCOM 2010); arXiv "A Walk in Facebook". https://arxiv.org/abs/0906.0060
- Hamilton, W. et al. *Inductive Representation Learning on Large Graphs (GraphSAGE).* NeurIPS 2017. https://arxiv.org/abs/1706.02216
- Ying, R. et al. *Graph Convolutional Neural Networks for Web-Scale Recommender Systems (PinSAGE).* KDD 2018. https://arxiv.org/abs/1806.01973 ; Pinterest Engineering blog: https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48
- Chiang, W.-L. et al. *Cluster-GCN.* KDD 2019. https://arxiv.org/abs/1905.07953
- Zeng, H. et al. *GraphSAINT: Graph Sampling Based Inductive Learning Method.* ICLR 2020. https://arxiv.org/abs/1907.04931
- Wu, F. et al. *Simplifying Graph Convolutional Networks (SGC).* ICML 2019. https://arxiv.org/abs/1902.07153
- Frasca, F., Rossi, E. et al. *SIGN: Scalable Inception Graph Neural Networks.* 2020. https://arxiv.org/abs/2004.11198
- Tang, J. et al. *GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection.* NeurIPS 2023 D&B. https://arxiv.org/abs/2306.12251
- Huang, X. et al. *DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection.* NeurIPS 2022 D&B. https://arxiv.org/abs/2207.03579
- Feng, S. et al. *TwiBot-22: Towards Graph-Based Twitter Bot Detection.* NeurIPS 2022 D&B. https://arxiv.org/abs/2206.04564 ; BotRGCN training code: https://github.com/LuoUndergradXJTU/TwiBot-22/blob/master/src/BotRGCN/twibot_22/train.py
- Feng, S. et al. *What Does the Bot Say? Opportunities and Risks of LLMs in Social Media Bot Detection.* ACL 2024. https://arxiv.org/abs/2402.00371
- Tan, Z., Feng, S. et al. *BotPercent: Estimating Bot Populations in Twitter Communities.* Findings of EMNLP 2023. https://arxiv.org/abs/2302.00381
- Yang, C. et al. *FLAG: Fraud Detection with LLM-enhanced Graph Neural Network.* KDD 2025 (summary via https://chatpaper.com/paper/175073 ; list at https://github.com/AI4Risk/awesome-graph-based-fraud-detection)
- Wu, Q. et al. *BotSCL: Heterophily-aware Social Bot Detection with Supervised Contrastive Learning.* ICPR 2024. https://arxiv.org/abs/2306.07478
- Platonov, O. et al. *Characterizing Graph Datasets for Node Classification: Homophily-Heterophily Dichotomy and Beyond.* NeurIPS 2023. https://arxiv.org/abs/2209.06177
- Platonov, O. et al. *A Critical Look at the Evaluation of GNNs under Heterophily: Are We Really Making Progress?* ICLR 2023. https://arxiv.org/abs/2302.11640
- Xu, T. et al. *Deep Entity Classification: Abusive Account Detection for Online Social Networks.* USENIX Security 2021. https://www.usenix.org/conference/usenixsecurity21/presentation/xu-teng
- Uber Engineering. *Fraud Detection: Using Relational Graph Learning to Detect Collusion.* 2021. https://www.uber.com/blog/fraud-detection/
- Borisyuk, F. et al. *LiGNN: Graph Neural Networks at LinkedIn.* KDD 2024. https://arxiv.org/abs/2402.11139
- Zhang, D. et al. *AGL: A Scalable System for Industrial-purpose Graph Machine Learning* (Ant Group). VLDB 2020. https://arxiv.org/abs/2003.02454
- Ant Financial. *InfDetect: a Large Scale Graph-based Fraud Detection System for E-Commerce Insurance.* https://arxiv.org/abs/2003.02833
- Zhu, R. et al. *AliGraph: A Comprehensive Graph Neural Network Platform.* VLDB 2019. https://arxiv.org/abs/1902.08730
- AWS ML Blog. *Build a GNN-based real-time fraud detection solution using DGL without using external graph storage.* https://aws.amazon.com/blogs/machine-learning/build-a-gnn-based-real-time-fraud-detection-solution-using-the-deep-graph-library-without-using-external-graph-storage ; Neptune version: https://aws.amazon.com/blogs/machine-learning/build-a-gnn-based-real-time-fraud-detection-solution-using-amazon-sagemaker-amazon-neptune-and-the-deep-graph-library
- Grab Engineering. *Graph for fraud detection.* https://engineering.grab.com/graph-for-fraud-detection
- El-Kishky, A. et al. *TwHIN: Embedding the Twitter Heterogeneous Information Network.* KDD 2022. https://arxiv.org/abs/2202.05387
- Kaler, T. et al. *SALIENT: Accelerating Training and Inference of GNNs with Fast Sampling and Pipelining.* MLSys 2022. https://arxiv.org/abs/2110.08450
