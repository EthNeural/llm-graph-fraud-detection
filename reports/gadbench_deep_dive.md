# GADBench deep dive: fact-check and implementation notes

Sources opened and read in full: the paper PDF (arXiv 2306.12251v2, 16 Nov 2023; https://arxiv.org/abs/2306.12251) and a fresh clone of the code (https://github.com/squareRoot3/GADBench: readme.md, utils.py, models/detector.py, models/gnn.py, preprocess.ipynb, benchmark.py). "Table N" and "Figure N" refer to the paper. Statements marked **[inference]** are mine, not the authors'.

## Summary table

| # | Claim | Verdict |
|---|---|---|
| 1 | 29 models, 10 datasets, identical conditions | VERIFIED for counts; "identical conditions" OVERSTATED |
| 2 | 11,944 (Amazon) to 5.8M; anomaly 1.3% to 14.5%; uncut | WRONG on the smallest dataset and on the max anomaly rate; "uncut" NOT IN PAPER |
| 3 | 100 labels (20/80), 10 draws; fully = traditional splits | VERIFIED, but incomplete (there is also a 100-label validation set) |
| 4 | AUROC, AUPRC, Rec@K | VERIFIED (K = number of anomalies in the test set) |
| 5 | Mean of neighbours, append, tree model | OVERSTATED: sum/mean/max, 1-4 layers, concatenated across hops incl. raw features; defaults mean, 2 layers |
| 6 | +2.0 AUROC (full), +2.8 AUROC (semi) vs best GNN | VERIFIED, but default hyperparameters and an average; GNNs win on several datasets |
| 7 | Trees natively handle tabular anomalies and imbalance better | OVERSTATED: the paper's reasons are decision boundaries and feature type, not imbalance |
| 8 | One-hop average captures most of the graph value | NOT SHOWN as stated; Figure 2 ablates layers 0-4 and the paper says 2 are adequate |
| 9 | No GNN beating MLP is normal; GADBench confirms | PARTLY: MLP is on par with GCN/GIN, but no Twitter/bot dataset, and GNNs do beat MLP when specialised |
| 10 | 100-label block proxies a 1,300-user LLM set | WEAK proxy; reasonable only as a label-scarcity analogy |

## Part A: claim by claim

**1. 29 models, 10 datasets.** Abstract: "a detailed comparison across 29 distinct models on ten real-world GAD datasets". Table 1 also lists 29. The README only mentions "25" and "26" models in its example commands, so the count is the paper's, not the CLI's. "Identical conditions" is too strong. Section 3.3 standardises splits, metrics and a random-search protocol, but the appendix says "we did not perform hyperparameter search for heterogeneous graph neural networks (RGCN, CARE-GNN, and H2-FDetector)". Table 4 shows SVM and XGBOD as OOT (out of time, over a day) on T-Social. The code also sets hidden size 16 for T-Social (benchmark.py, utils.py). The paper reports two regimes (default vs tuned), and the headline numbers use defaults.

**2. Sizes and rates.** Table 3: the largest is T-Social at 5,781,065 nodes, so "5.8M" is right. The smallest is **Weibo at 8,405**, then Reddit at 10,984. Amazon (11,944) is third smallest, so the "Amazon" part is WRONG. Anomaly rates: the lowest is DGraph-Fin at 1.3%, which is right. The highest is **Tolokers at 21.8%**, not YelpChi's 14.5% (14.5% is the second highest). Section 3.2 caps the ratio at 25%. "Graphs uncut" is not stated. **[inference]** Reddit and Weibo are user-item/hashtag graphs turned into user graphs (Appendix B and Table 3, "Under Same Post/Hashtag"), and the Elliptic and DGraph-Fin preprocessing keeps only the nodes flagged "mark" (`preprocess.ipynb`). Nothing is described as subsampled.

**3. Splits.** Section 3.3: "we standardize the training set across all datasets to include a total of 100 labels—20 positive labels (anomalous nodes) and 80 negative labels (normal nodes). ... ten random splits". Fully supervised: "we preserve pre-existing data splits when available", otherwise 40/20/40 (Table 3 gives per-dataset training ratios, 40% to 70%). The code adds a detail the paper omits: `preprocess.ipynb` also draws a **second 100-label block (20 anomalies, 80 normal) as validation**, and all other nodes become test. So semi-supervised uses 200 labels in total (100 train, 100 validation), with the test set being the entire rest of the graph.

**4. Metrics.** "AUROC, AUPRC calculated by average precision, and the Recall score within top-k predictions (Rec@K) ... We set k as the number of anomalies within the test set." Code: `k = labels.sum()`; `RecK = hits in top-k / total anomalies`.

**5. RF-Graph / XGB-Graph mechanism.** The paper (Sec 3.1) defines `h^(l) = Aggregate{...}` and `Score = TreeEnsemble([h^0 || h^1 || ... || h^L])`, with "Aggregate(·) can take on any aggregation function such as mean, max, or sum pooling ... the aggregation process is parameter-free". The code (`models/gnn.py`, `GIN_noparam`):

```python
def __init__(self, num_layers=2, agg='mean', init_eps=-1, **kwargs):
    self.gnn = dglnn.GINConv(None, activation=None, init_eps=init_eps, aggregator_type=agg)
def forward(self, graph):
    h = graph.ndata['feature']; h_final = h.detach().clone()
    for i in range(self.num_layers):
        h = self.gnn(graph, h)
        h_final = torch.cat([h_final, h], -1)
    return h_final
```

Verified details:
- The output is own raw features, then hop-1 aggregate, then hop-2 aggregate, and so on, all concatenated. At default L=2 the feature width is 3x the original.
- Each hop re-aggregates the previous hop's output (hop 2 is the neighbours-of-neighbours aggregate), not a separate hop-2 neighbourhood.
- `init_eps=-1` makes GIN's `(1+eps)*self` term zero, so each hop is a pure neighbour aggregate with no self-loop. **[inference from DGL's GINConv formula]**
- No normalisation, scaling or other preprocessing. Defaults (Table 9) are mean and L=2. The search space is {sum, mean, max} x L in {1,2,3,4}.
- The graph is used as given. Edges are not typed, so multiple relations are merged (see Part B).

So "average of neighbours appended to own features" is the default behaviour, but one-hop-only is not the default; it is a 2-hop concatenation.

**6. Headline gaps.** Section 4.1 (default hyperparameters, Figure 1; per-dataset numbers in Tables 11/12): "XGB-Graph surpasses BWGNN—the best GNN model in this setting—by an absolute average improvement of 2.0% on AUROC, 12.9% on AUPRC, and 9.8% on Rec@K. In the semi-supervised context, RF-Graph presents an absolute average improvement of 2.8% on AUROC, 8.0% on AUPRC, and 3.1% on Rec@K, as compared to GHRN." My check against Tables 11/12: full AUROC 86.7 vs 84.7, semi AUROC 78.9 vs 76.1. Verified.

Caveats:
- These are averages over 10 datasets with default hyperparameters. T-Social drives a large part: fully supervised AUROC 99.3 vs 92.0 (XGB-Graph vs BWGNN), AUPRC 92.4 vs 54.9.
- After tuning (Table 4, AUPRC) the paper says "the performance gap ... narrow" and trees "still prevail among 6 out of 10 datasets". My count: trees (RF/XGB-Graph) lead on Weibo, Amazon, YelpChi, T-Finance, Elliptic and Tolokers. They do **not** lead on Reddit (NA 9.70, BWGNN 8.32 vs ~5), Questions (BWGNN 18.57 vs 18.19), DGraph-Fin (NA 4.13) or T-Social (BGNN 99.09 vs 97.63).
- Default, fully supervised, AUROC: GNN (BWGNN/GHRN) beats the best tree-graph model on Reddit (65.4/66.0 vs 62.8), Questions (71.8 vs 65.8), DGraph-Fin (76.3 vs 74.6) and marginally Tolokers (80.4 vs 80.1 for XGB-Graph; RF-Graph 81.2 wins). Default AUPRC adds Tolokers (BWGNN 49.7, GHRN 49.9 vs 49.3/48.9 for RF-/XGB-Graph).
- Inductive setting (Table 6): XGB-Graph wins Elliptic by a large margin but loses on DGraph-Fin (GHRN best).

**7. "Trees handle tabular anomalies and imbalance natively."** Not claimed. The reasons actually given (Sec 4.3):
- Decision boundaries: "anomaly instances tend to form multiple dispersed clusters and are coupled with normal instances", which suits trees' "complex and disjoint decision boundaries", while GNNs end in an MLP and give "simple and continuous decision boundaries". The evidence is a t-SNE visualisation (Figure 3), which the authors call "an initial study".
- Feature types: on text-embedding datasets (Reddit, Weibo, Questions) "GNNs showcase competitive performance"; on the 7 datasets with "diverse feature types that have low correlation (for instance, gender and age information), tree ensembles ... typically exhibit superior performance."
- Imbalance is mentioned only to explain why AUPRC and Rec@K gaps exceed AUROC gaps ("due to the imbalanced issue"), not as a reason trees win. **[inference]** The code does reweight classes: `weights = np.where(train_y == 0, 1, neg/pos)` for every tree model, so tree imbalance handling here is explicit sample weighting, not "native".

**8. "One-hop captures the majority of the value."** Not stated. Figure 2 sweeps aggregation layers 0 to 4 on all 10 datasets for both models and the text says "performance on most datasets improves when the number of layers increases from 0 to 2 ... two layers are adequate". My reading of the plot (approximate, no table of values): for XGB-Graph, layer 0 to 1 gives most of the gain on Tolokers (AUPRC ~0.37 to ~0.46) and T-Finance, while T-Social goes ~0.15 to ~0.67 at one hop and ~0.9 at two. Other datasets are nearly flat. **[inference]** So "one hop gets most of it" is plausible for most datasets but false for T-Social, and the paper does not make the claim. No ablation of sum vs mean vs max is reported (only the search space and the tuned result). Table 10 tests a different thing (post-hoc score averaging, NA).

**9. MLP / GNN / TwiBot-20.** GADBench has no Twitter or bot dataset. Social datasets: Reddit, Weibo, Questions, T-Social (Sec 3.2). What it supports:
- "the performance of GCN and GIN is on par with that of MLP—a method that does not take graph structure information into account." Standard GNNs "are not suitable for the GAD task". Default fully supervised AUROC means: MLP 80.7, GCN 78.0, GIN 80.0, GraphSAGE 80.7, BWGNN 84.7. Tuned AUPRC average (Table 4, 9 datasets): MLP 44.5, GCN 36.4, GIN 42.0, GraphSAGE 49.3, BWGNN 49.6.
- But MLP does not beat all GNNs: specialised GNNs (BWGNN, GHRN) beat it clearly.
- The more important result for us: **plain RF/XGBoost with no graph at all beat every GNN on average AUPRC**. Tuned: RF 53.7, XGBoost 55.0 vs best GNN BWGNN 49.6. Default fully supervised AUPRC: RF 51.7 / XGBoost 50.3 vs BWGNN 48.2.
- So: "a feature-only model matching or beating GNNs is common in GADBench" is supported. "TwiBot-20 specifically is normal" is only an analogy. **[inference]** It also notes the paper's own caveat that GNNs stay competitive on text-embedding features (Sec 4.3), which is our feature type (bio embeddings), so the evidence cuts both ways for us.

**10. 100-label proxy for a 1,300-user LLM set.** Weak. **[inference]** (a) Sizes differ by 13x and the GADBench block is class-fixed at 20% anomalies, which is far more balanced than TwiBot-22's ~14% bot rate only if we also stratify. (b) GADBench uses all remaining nodes as test, transductively with the full graph, while an LLM set is a different regime (frozen embeddings or fine-tuning). (c) The semi-supervised ranking changes (RF-Graph over XGB-Graph; GCN/GAT improve on Weibo) so the label budget matters for the winner. Defensible wording: "GADBench's semi-supervised block is a precedent for evaluating at small labelled budgets", not "proxy".

## Part B: implementation details for XGB-Graph on TwiBot-22

**Hyperparameters.** Defaults (Table 9 and code): XGB-Graph 100 estimators, eta 0.3, lambda 1, subsample 1, booster gbtree, aggregation mean, layers 2. Max depth is not set, so XGBoost's own default (6) applies **[inference]**. `tree_method='gpu_hist'`, class weight `neg/pos` as `sample_weight`, validation set passed as `eval_set` (the `early_stopping_rounds=20` comment is commented out, so no early stopping). RF-Graph: 100 trees, gini, `max_samples=None`, `max_features='sqrt'`, same sample weighting.
Random-search spaces (`utils.py`): XGB-Graph `n_estimators` 10-200, `eta` = 0.5*10^U(-1,0) (0.05-0.5), lambda {0,1,10}, subsample {0.5,0.75,1}, booster {gbtree, dart}, layers {1-4}, agg {sum,max,mean}. RF-Graph: 10-200 trees, {gini, entropy}, max_samples {0.5,0.75,1}, max_features {sqrt, log2, None}. README: 100 random trials, best validation AUPRC kept.

**Multi-relational graphs.** Amazon/YelpChi each have three relations (Appendix B). The standard (non-hetero) runs merge all relations into one homogeneous graph, as Table 5 says for XGB-Graph ("treat all edge types equivalently"). The README's `--datasets 10,11` runs the hetero variants for RGCN/HGT/CARE-GNN/H2FD only. Result (Table 5): per-relation modelling does not help on Amazon (XGB-Graph 98.69 AUROC fully supervised, above RGCN 92.03 and CARE-GNN 90.84) but helps on Yelp for label-scarce CARE-GNN (semi AUPRC 68.7 vs 24.8 XGB-Graph). Fully supervised on Yelp, merged XGB-Graph still wins (AUPRC 87.0 vs HGT 62.6). **No per-relation aggregation was tried for the trees.** For TwiBot-22 we would have to build it ourselves (e.g. one aggregation block per relation), which is beyond what the paper tests.

**Heterophily.** Heterophily appears only as motivation ("graphs with anomalies often exhibit strong heterophily", Sec 2) and as the stated reason standard GNNs struggle ("structure camouflage or feature heterophily"). The paper reports **no homophily numbers for any dataset**. Datasets are not characterised by edge homophily.

**Runtime and memory.** The paper gives only a chart (Figure 1 bottom: wall-clock time over all datasets, peak CPU/GPU memory on DGraph-Fin) and says trees "outperform GNNs in terms of efficiency but also exhibit lower memory consumption". No numbers are in the text. Hardware: 32-core AMD EPYC, 64 GB RAM, RTX A6000 48 GB (Appendix C). The README says running all 25 models fully supervised needs more than 48 GB of GPU memory. T-Social (73M edges): XGB-Graph and RF-Graph both completed; SVM and XGBOD did not (OOT, over one day). Aggregation is a full-graph DGL op done once up front, then a standard tree fit. **[inference]** At 1M nodes, widths of 3x(768 embedding + profile fields) are the real cost, not the graph pass.

**Social-network-like datasets.** Reddit (64-d LIWC text, 3.3% anomaly, banned users), Weibo (400-d bag-of-words, 10.3%), Tolokers (10-d profile + task stats, 21.8%, crowd workers), Questions (301-d FastText of user descriptions, 3.0%), T-Social (10-d profile/login features, 3.0%, 73M edges). Of these, the text-feature ones (Reddit, Questions) are where GNNs do best, T-Social is where trees with aggregation do best (AUPRC ~93 vs 55 for BWGNN; default; and 0.15 to 0.9 over the layer sweep in Figure 2). No homophily values reported.

## Part C: recommendations

1. **Build XGB-Graph exactly as GADBench does as the baseline**: concat `[x, mean-agg hop 1, mean-agg hop 2]`, self-loops excluded, class weight `neg/pos`. Evidence: `GIN_noparam` code and Table 9 defaults.
2. **Also run a no-graph XGBoost on the same features.** GADBench shows plain XGBoost/RF beat all GNNs on average AUPRC (Table 4: 55.0 / 53.7 vs 49.6), so it is the real reference for "does the graph help".
3. **Sweep layers 0-2 (and 3) and report it as our own small ablation.** The paper's Figure 2 shows 0 to 2 improves most datasets but T-Social needs two hops, so do not assume one hop is enough.
4. **Aggregate per relation on TwiBot-22 and compare with the merged graph.** GADBench only tried merged graphs for trees, and per-relation modelling helped on Yelp (Table 5) but not Amazon, so this is a legitimate extension.
5. **Compute edge homophily per relation ourselves.** The paper reports none, so any heterophily statement in our report must rest on our own measurements.
6. **Tell the reader the result may not transfer for text embeddings.** Sec 4.3 says GNNs are competitive on text-embedding features (Reddit, Weibo, Questions); our bio-embedding features are of that type. Consider reducing embedding dimension (PCA) for the tree model, since trees see only a subset of columns per split.
7. **Report AUPRC and Rec@K alongside AUROC**, using the paper's definitions (K = number of test anomalies). GADBench's gaps are 2.0 AUROC but 12.9 AUPRC, so AUROC alone understates differences.
8. **Rephrase the TwiBot-20 and 100-label claims.** Say "consistent with GADBench, where GCN/GIN are on par with an MLP and graph-free tree models beat most GNNs (no bot dataset included)", and say the 100-label block is a "precedent for small label budgets", not a proxy for 1,300 labels. Use this paper only for what it contains (Part A items 2, 5, 7, 8 need correcting before citation).

## References
- Paper: https://arxiv.org/abs/2306.12251 (PDF v2); Table/figure numbers as above.
- Code: https://github.com/squareRoot3/GADBench (`models/detector.py`, `models/gnn.py`, `utils.py`, `preprocess.ipynb`, `readme.md`).
