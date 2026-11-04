# How Industry Handles Big Graphs: A Reading Guide

Nov 4, 2026 · @Prajan

## How to read this guide

Big companies almost never shrink their graph to make it fit; they shrink the work done on it. This guide shows how, through four papers: Facebook's fake-account system, Uber's fraud graph, the GADBench benchmark, and Pinterest's PinSAGE.

Each paper section has three layers, so you can stop at any depth:

1. **In plain words**: the idea, explained without jargon.
2. **How it works**: the mechanism, one level deeper.
3. **What it gives us**: what we can copy for TwiBot-22.

Under each section, a **Words in this section** box defines every technical term the section uses. The last section has the full glossary, A to Z. Terms in **bold** are in the glossary.

Why this matters for us: TwiBot-22 has 1,000,000 users and 170 million edges. Our laptop can run graph models on all of it, but it cannot run an LLM on all of it. These papers show how others split that problem.

## The big idea: cut the computation, not the graph

**In plain words.** Imagine judging whether one person in a city of a million is a scammer. You don't redraw the city map with only 10,000 people on it. You keep the map, and look closely at that person and the few people around them. Industry does the same with graphs.

**How it works.** There are three common ways to do this, and real systems often combine them:

1. **Look around one user at a time** (**neighbour sampling**). Each training step takes a small batch of users plus a sample of their neighbours from the full **graph**. Memory stays small, and no user or edge is thrown away.
2. **Do the graph work once, in advance** (**precomputed features**). Average each user's neighbours' features over 1, 2 and 3 hops, once, then train a simple model like an **MLP** or a tree model on the result.
3. **Cheap first, expensive second** (**two-stage system**). A fast model scores everyone; the costly model (a **GNN** or an **LLM**) only looks at the few users the fast model flags.

Slicing the graph instead, i.e. keeping only some users and the edges between them, changes what you measure. Keep a random 1% of users and you keep only about 0.01% of their edges. That is why TwiBot-22's own 10,000-user **sub-communities** have almost no follow edges.

**What it gives us.** Run the graph models on all 1M users, and run the LLM only on a sample of users. Give the LLM each sampled user's real neighbours, taken from the full graph.

> **Words in this section**
>
> - **Graph**: dots (users) joined by lines (relations such as "follows").
> - **Neighbour**: a user joined to you by an edge; **hop** = one step along an edge.
> - **Neighbour sampling**: picking a few neighbours per user each training step, instead of all.
> - **Precomputed features**: graph information turned into ordinary numbers once, before training.
> - **Two-stage system**: a cheap filter followed by an expensive model on the flagged few.
> - **MLP**: a basic neural network that sees only a user's own features.
> - **GNN**: graph neural network, which also mixes in neighbours' features.
> - **Sub-community**: one of the 10 user groups the TwiBot-22 authors cut out (10k users each).

## Paper 1: Facebook's Deep Entity Classification (DEC)

[Xu et al., "Deep Entity Classification: Abusive Account Detection for Online Social Networks", USENIX Security 2021](https://www.usenix.org/conference/usenixsecurity21/presentation/xu-teng)

**In plain words.** Facebook had to find fake and abusive accounts among billions of users. A fake account can lie about itself, but it is hard to fake who it is connected to. So DEC judges each account partly by the accounts and other things linked to it, and what is linked to those. DEC runs on all users continuously. Facebook estimates it removed **27% more abusive accounts** than older methods were already catching.

**How it works.**

1. **Features from the neighbourhood.** For each account, DEC looks at the entities linked to it in Facebook's graph, and the entities linked to those. It summarises their properties with simple statistics such as mean, maximum and percentiles. That yields **more than 20,000 features** per account. The "deep" in the name means this **fan-out** across the graph, not a deep neural network.
2. **Two kinds of labels, used in stages.** Human reviewers give a small set of very accurate labels. Automated rules give a huge set of noisy labels. DEC first learns from the noisy labels, then refines on the accurate ones. The paper calls this **multi-stage multi-task learning (MS-MTL)**.
3. **One model, many abuse types.** The same model flags several types of abusive account at once.

**What it gives us.**

- **Neighbour statistics are the industrial workhorse.** For each TwiBot-22 user and each relation, we can compute the mean, max and count of neighbours' features. That is cheap, runs on all 1M users, and works with any model.
- **TwiBot-22's labels are DEC-style weak labels.** They came from labelling rules plus models (including GCN and GAT), checked against 1,000 expert labels. So a graph model might score well partly because graph models helped make the labels. We should say this in the report.
- **Clean labels are precious.** TwiBot-22's 1,000 expert-labelled users would be a stronger, if small, test set than the noisy labels.

> **Words in this section**
>
> - **Abusive account**: fake, spam, scam or hijacked account.
> - **Feature**: one number that describes a user, e.g. follower count.
> - **Fan-out**: stepping outward from a user to its neighbours, then theirs.
> - **Aggregation**: squashing many neighbours into one number (mean, max, count).
> - **Weak (noisy) labels**: cheap labels from rules or models; often wrong.
> - **Expert labels**: labels checked by trained humans; few but accurate.
> - **MS-MTL**: train on noisy labels first, then on clean ones; one model, several tasks.

## Paper 2: Uber's graph learning for collusion fraud

[Uber Engineering, "Fraud Detection: Using Relational Graph Learning to Detect Collusion", 21 May 2021](https://www.uber.com/blog/fraud-detection/)

**In plain words.** Some Uber fraud is teamwork: several users cooperating to cheat (Uber calls it collusion). One account alone looks normal; the group gives it away. Uber built a graph of users and the things they share, trained a graph model on it, and handed the graph model's scores to its existing fraud model as two extra inputs. Precision went up by **about 15%**, with almost no rise in false alarms. Uber notes this model was an experiment, **not used in production**.

**How it works.**

1. **One big graph, several kinds of link.** Users are nodes; edges have types (the post says distinguishing connection types "amplifies the signal"). Very high-degree entities were filtered out, so one hub couldn't link everyone.
2. **A relational GNN (RGCN).** An RGCN learns separate weights for each edge type, so each kind of link counts differently. It gives every user two scores: one as a driver, one as a rider. The label was whether the user had a **chargeback** in a time window.
3. **Graph scores become features.** The two scores joined a production model with 200 features. They ranked **4th and 39th** in feature importance.
4. **A time-based test.** Trained on 4 months of history, tested on the **6 weeks after** the split date: the model only sees the past, as in real use.

**What it gives us.**

- **Graph output as a feature, not the final answer.** We can train a GNN on TwiBot-22, then feed its score into an MLP or tree model with text and metadata. If that beats the same model without the score, the graph helped.
- **Keep edge types apart.** TwiBot-22 has following, followers, mention, retweet, reply and more. An RGCN, or one feature set per relation, matches Uber's approach.
- **Drop or cap huge hubs.** TwiBot-22's largest node has 270,344 edges. Capping degree, like Uber, stops one celebrity from linking everyone.

> **Words in this section**
>
> - **Collusion**: several accounts cheating together.
> - **Precision**: of the users flagged, the share that really are fraud.
> - **False positive (false alarm)**: an honest user flagged as fraud.
> - **RGCN**: relational graph convolutional network; a GNN with separate weights per edge type.
> - **Chargeback**: a payment reversed after a dispute; Uber's fraud label.
> - **Feature importance**: how much a model relies on each input.
> - **Temporal split**: train on older data, test on newer data.
> - **Hub**: a node with a huge number of edges.

## Paper 3: GADBench, a fair test of fraud-detection models

[Tang et al., "GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection", NeurIPS 2023 Datasets and Benchmarks](https://arxiv.org/abs/2306.12251) · [code](https://github.com/squareRoot3/GADBench)

**In plain words.** Many papers claim their new graph model catches fraud best, but each tests on different data in a different way. GADBench ran 29 models on the same 10 fraud and anomaly datasets with the same rules. The surprise: an old-fashioned **tree model**, given each node's own features plus its neighbours' averaged features, beat the specialised fraud GNNs.

**How it works.**

1. **Ten real datasets.** They range from 8,405 nodes (Weibo) to 5.8 million (T-Social). The anomaly share runs from 1.3% (DGraph-Fin) to 21.8% (Tolokers). The paper describes no cutting-down of the graphs.
2. **Two label budgets.** *Fully supervised* uses a normal split of the labelled nodes. *Semi-supervised* **trains each model on only 100 labels (20 anomalies, 80 normal), with a second 100-label set for validation**, repeated over 10 random draws. They shrink the labels, not the graph.
3. **The tree baselines.** **RF-Graph** and **XGB-Graph** join each node's own features with the mean of its 1-hop and its 2-hop neighbours' features (the default; sum and max were also tried), then a **random forest** or **XGBoost** predicts. XGB-Graph beat the best GNN by 2.0**% AUROC on average** when fully supervised; RF-Graph beat the best GNN by 2.8% semi-supervised.
4. **Three metrics.** **AUROC**, **AUPRC** and **Rec@K**, all at each dataset's natural anomaly rate.

**What it gives us.**

- **A must-run baseline.** XGB-Graph on TwiBot-22 (own features + mean neighbour features per relation) is exactly the tree baseline we planned. On TwiBot-20 it never ran: the log was empty because of a library crash, now diagnosed.
- **Our TwiBot-20 result is consistent with GADBench.** There, no GNN beat a plain MLP. GADBench has no Twitter data, so it cannot confirm that result directly. But it points the same way: plain GCN and GIN were only on par with an MLP, and plain tree models with no graph at all beat every GNN on average AUPRC.
- **Copy the evaluation.** Keep the natural bot share (14%), and report AUPRC alongside macro-F1. A 100-label setting would also show how each model copes with few labels, which is close to the 1,300 users our LLM step trains on.

> **Words in this section**
>
> - **Anomaly detection**: finding the rare odd ones out; fraud is one kind.
> - **Benchmark**: a fixed set of datasets and rules so models can be compared fairly.
> - **Tree model**: a model built from many yes/no questions ("followers > 100?").
> - **Random forest / XGBoost**: two popular ways of combining many trees.
> - **RF-Graph / XGB-Graph**: tree models given neighbour-averaged features.
> - **Supervised / semi-supervised**: trained with many labels / with very few.
> - **AUROC**: chance a random fraud case scores above a random normal one (0.5 = guessing).
> - **AUPRC (PR-AUC)**: precision averaged over recall levels; honest when fraud is rare.
> - **Rec@K**: share of all frauds found among the top K scored users.

## Paper 4: Pinterest's PinSAGE, plus the precompute shortcut

[Ying et al., "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (PinSAGE), KDD 2018](https://arxiv.org/abs/1806.01973), built on [GraphSAGE (Hamilton et al., NeurIPS 2017)](https://arxiv.org/abs/1706.02216); shortcut methods [SGC (ICML 2019)](https://arxiv.org/abs/1902.07153) and [SIGN (2020)](https://arxiv.org/abs/2004.11198)

**In plain words.** Pinterest ran a GNN on a graph of **3 billion nodes and 18 billion edges**, far too big for any one machine's memory. The trick: never load the whole graph into the model. For each item, take a small, well-chosen set of neighbours, compute on that, move on. The precompute methods go further: do the neighbour averaging once, then train a plain model.

**How it works.**

1. **GraphSAGE: sample, then combine.** For each target node, pick a fixed number of neighbours (say 10), and 10 of each of theirs. Combine their features with an **aggregator** (mean, pooling). Because it learns a rule rather than one vector per node, it works on nodes it never saw: it is **inductive**.
2. **PinSAGE: pick important neighbours.** Instead of random neighbours, it runs short **random walks** from each node and keeps the most-visited ones. That gave a **46% gain** offline over plain k-hop neighbourhoods.
3. **Industrial plumbing.** Training used **mini-batches** fed by a producer-consumer pipeline (CPUs prepare neighbourhoods while GPUs train). Final embeddings for all 3B nodes were computed with **MapReduce**.
4. **The precompute shortcut (SGC, SIGN).** Multiply the features by the adjacency matrix once for 1 hop, twice for 2 hops, and so on, then train an MLP on the stacked results. SGC reports up to 100x speed-ups; SIGN scaled to 110 million nodes and 1.5 billion edges.

**What it gives us.**

- **Our 1M users fit on the laptop.** TwiBot-22 is 3,000 times smaller than Pinterest's graph. Precomputing 1-, 2- and 3-hop features per relation takes minutes on the CPU. No slicing needed.
- **Sampling helps the LLM step.** DGP already trims each user's neighbours to the most relevant few, which is PinSAGE's idea applied to text.
- **Inductive matters for honest tests.** A model that can score unseen users can be trained on one part of the graph and tested on another.

> **Words in this section**
>
> - **Mini-batch**: a small group of examples processed in one training step.
> - **Aggregator**: the rule that merges neighbours' features (mean, max, pooling).
> - **Inductive / transductive**: works on new, unseen nodes / only on nodes seen in training.
> - **Random walk**: hopping from node to random neighbour, again and again.
> - **k-hop neighbourhood**: everything within k steps of a node.
> - **Embedding**: a list of numbers that summarises a node.
> - **Adjacency matrix**: a table saying which nodes are linked.
> - **MapReduce**: a way to split a huge computation across many machines.
> - **SGC / SIGN**: GNNs whose neighbour averaging is done once, up front.

## Putting it together for TwiBot-22

All four papers point the same way: keep the full graph for the cheap models, and spend the expensive model only where it pays.

&#91;embedded content: TwiBot-22 plan · full graph for cheap models, sample for the LLM\]

Both arms meet on one test set; the dashed arrow is Uber's idea of feeding a graph score into the text model.

**The plan, in five steps:**

1. **Full graph, cheap features (DEC, GADBench).** All 1,000,000 users. Each user's features: bio embedding plus allowed profile fields, with `verified` excluded as a shortcut. Then, per relation, the mean, max and count of neighbours' features.
2. **Full-graph models (GADBench, SIGN).** MLP, XGB-Graph, precomputed 1-3 hop propagation, and full-batch GCN, SAGE and LINKX, all on CPU.
3. **A fair sample for the LLM (PinSAGE idea).** A stratified random set of users at the natural 14% bot share, drawn from our own stratified split. Each sampled user's neighbours come from the **full** graph, not a cut-down one.
4. **Graph scores as features (Uber).** Feed the best graph model's score into the text model, and see whether it adds anything.
5. **One shared test set.** Score every model on the same test users. Report macro-F1, AUROC and AUPRC at the natural bot share.

**Questions to think about while reading.** These are the decisions you can guide:

- TwiBot-22's labels came partly from GCN and GAT models. How much should we trust a graph model that scores well on them?
- Should the 1,000 expert-labelled users become a separate, cleaner test set?
- Uber tested on later data than it trained on. TwiBot-22 tweets have dates. Is a time-based split worth adding?
- Which matters more for the report: the best score, or showing *why* the graph helps or does not?

## Glossary, A to Z

| Term | Meaning |
| --- | --- |
| Abusive account | Fake, spam, scam or hijacked account. |
| Adjacency matrix | A table saying which nodes are linked. |
| Aggregation / aggregator | Merging many neighbours into one number (mean, max, count). |
| Anomaly detection | Finding the rare odd ones out; fraud detection is one kind. |
| AUPRC (PR-AUC) | Precision averaged over recall levels; honest when fraud is rare. |
| AUROC | Chance a random fraud case scores above a random normal one; 0.5 = guessing. |
| Benchmark | A fixed set of datasets and rules so models can be compared fairly. |
| Chargeback | A payment reversed after a dispute; Uber's fraud label. |
| Collusion | Several accounts cheating together. |
| Edge / relation | A link between two nodes; its type says what kind (follows, mentions). |
| Embedding | A list of numbers that summarises a user or a text. |
| Expert labels | Labels checked by trained humans; few but accurate. |
| False positive | An honest user flagged as fraud. |
| Fan-out | Stepping outward from a user to its neighbours, then theirs. |
| Feature | One number describing a user, e.g. follower count. |
| Feature importance | How much a model relies on each input. |
| GNN | Graph neural network; mixes a user's features with its neighbours'. |
| Graph | Nodes (users) joined by edges (relations). |
| Hop / k-hop neighbourhood | One step along an edge / everything within k steps. |
| Hub | A node with a huge number of edges. |
| Inductive / transductive | Works on unseen nodes / only on nodes seen in training. |
| MapReduce | Splitting a huge computation across many machines. |
| Mini-batch | A small group of examples processed in one training step. |
| MLP | A basic neural network that sees only a user's own features. |
| MS-MTL | Facebook DEC's training: noisy labels first, clean labels second, several tasks in one model. |
| Neighbour sampling | Using a few sampled neighbours per user each step, not all. |
| Precision | Of the users flagged, the share that really are fraud. |
| Precomputed features | Graph information turned into ordinary numbers once, before training. |
| Random forest / XGBoost | Two ways of combining many decision trees. |
| Random walk | Hopping from node to random neighbour, again and again. |
| Rec@K | Share of all frauds found among the top K scored users. |
| RF-Graph / XGB-Graph | Tree models given neighbour-averaged features (GADBench). |
| RGCN | Relational GCN; a GNN with separate weights per edge type. |
| SGC / SIGN | GNNs whose neighbour averaging is done once, up front. |
| Sub-community | One of TwiBot-22's 10 author-made user groups (10k users each). |
| Supervised / semi-supervised | Trained with many labels / with very few. |
| Temporal split | Train on older data, test on newer data. |
| Tree model | A model built from many yes/no questions. |
| Two-stage system | A cheap filter, then an expensive model on the flagged few. |
| Weak (noisy) labels | Cheap labels from rules or models; often wrong. |

## Sources

- Xu et al. [Deep Entity Classification: Abusive Account Detection for Online Social Networks](https://www.usenix.org/conference/usenixsecurity21/presentation/xu-teng). USENIX Security 2021.
- Uber Engineering. [Fraud Detection: Using Relational Graph Learning to Detect Collusion](https://www.uber.com/blog/fraud-detection/). 21 May 2021.
- Tang et al. [GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection](https://arxiv.org/abs/2306.12251). NeurIPS 2023.
- Ying et al. [Graph Convolutional Neural Networks for Web-Scale Recommender Systems (PinSAGE)](https://arxiv.org/abs/1806.01973). KDD 2018.
- Hamilton et al. [Inductive Representation Learning on Large Graphs (GraphSAGE)](https://arxiv.org/abs/1706.02216). NeurIPS 2017.
- Wu et al. [Simplifying Graph Convolutional Networks (SGC)](https://arxiv.org/abs/1902.07153). ICML 2019.
- Frasca et al. [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/abs/2004.11198). 2020.
- Feng et al. [TwiBot-22: Towards Graph-Based Twitter Bot Detection](https://arxiv.org/abs/2206.04564). NeurIPS 2022.
- Our longer notes: `reports/graph_slicing_research.md` in the project repository.
