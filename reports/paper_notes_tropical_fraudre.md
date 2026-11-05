# Paper notes: TROPICAL (KAIS 2025) and FRAUDRE (ICDM 2021), read 5 Nov 2026 (external)

Both PDFs are in `research documents/` (not in git). Note: `research documents/run_all.py` is a byte-for-byte copy of the TROPICAL PDF (probably saved under the wrong name); nothing was changed.

## 1. TROPICAL: Haghighi, Soltani, Shabani, Wu, Zhang, Yao, Yang, Sheng. "Beyond pairwise relationships: a transformer-based hypergraph learning approach for fraud detection." Knowledge and Information Systems 67:7987–8022 (2025), doi 10.1007/s10115-025-02476-5 (journal version of ICDM 2024)

- **Idea:** fraudsters camouflage with pairwise links to benign users, but connect to *distant* fraudsters. Build a hypergraph from n-hop neighbourhoods (hyperedge groups), add positional encodings, run a transformer encoder; classify.
- **Motivating measurement (their eq. 3, Fig. 4a):** density of fraudsters in a fraudster's n-hop neighbourhood **rises with n** on Amazon, peaking at 3–4 hops.
- **Heterophily table (their Table 2):** fraud nodes have > 80% heterophilic connections on every relation except YelpChi R-U-R.
- **Data:** YelpChi (45,954 reviews, 14.5% fraud, 3 relations), Amazon (11,944 users, 6.9% fraud, 3 relations). Metrics AUC and macro-F1 (as ours). They report GCN, GAT, GraphSAGE and FAGCN below the fraud-specific models.
- **Clue about "DEFEND":** section 6.1.1 of this paper says the datasets were used "to validate the effectiveness of DEFEND" (text carried over from another paper by the same authors), so DEFEND is a model by this group. The best candidate is still Haghighi et al., ACM TKDD 2026, doi 10.1145/3801967 ("Beyond Smoothing: A Discriminative Enhanced Aggregation GNN for Camouflaged Fraud Detection"); confirm the name in that paper.

**Transfer test on TwiBot-22 (done 5 Nov, `results/twibot22/nhop_bot_density.json`):** bot density around val bots, hops 1–4, train labels only: union graph 0.116 → 0.081 → 0.086 → 0.059 (base rate 0.140); follow 0.124 → 0.095 → 0.095 → 0.068; interaction 0.102 → 0.060 → 0.070 → 0.040. **It does not rise**; around humans it is similar (0.075 → 0.056 → 0.083 → 0.056). On TwiBot-22's star-shaped graph, longer walks end at the (mostly human) hubs. So TROPICAL's motivation does not transfer. Caveat: walk-weighted density (A^n y / A^n 1), not distinct n-hop sets, so hubs weigh heavily.
**Our heterophily vs theirs:** TwiBot-22 bots have on average 88% heterophilic train-labelled neighbours (local homophily 0.116; Q2), the same picture as their > 80%.

## 2. FRAUDRE: Zhang, Wu, Yang, Beheshti, Xue, Zhou, Sheng. "FRAUDRE: Fraud Detection Dual-Resistant to Graph Inconsistency and Imbalance." IEEE ICDM 2021, doi 10.1109/ICDM51629.2021.00098 (Macquarie)

- **Three graph inconsistencies caused by camouflage:** *feature* (fraudsters imitate normal users' features), *topology* (fraudsters link to normal users, rarely to each other), *relation* (fraudsters mix with normal users under one relation but cluster under another). Plus **class imbalance**.
- **Model, 4 modules:** (1) a *graph-agnostic* embedding of each user's own attributes, kept separate from neighbours (so aggregation does not wash out suspiciousness at the start); (2) a *fraud-aware* graph convolution that embeds the **dissimilarity** to neighbours; (3) concatenation of all intermediate layers; (4) an imbalance-oriented loss.
- **Per-relation class homophily (their Table I, β):** normal users > 0.85 on every relation, fraudsters < 0.2 (except YelpChi R-U-R). Exactly our Q1 shape: TwiBot-22 humans ~0.93, bots 0.04–0.17.
- **Data and metrics:** Amazon, YelpChi; AUC, macro-recall, macro-F1; low label rates (10–40% training). GCN, GAT, GraphSAGE near AUC 0.50–0.55 on YelpChi at 10% labels; CARE-GNN 0.70, FRAUDRE 0.72.

**What it means for us:**
- Their *relation inconsistency* is our Q3 (cross-relation inconsistency features). On TwiBot-22 we found little separation by relation: bots are heterophilous on **all** 7 relations (Q1), so there is no relation where bots cluster. Q3's model test is running.
- Their module (1) + (3) is the "keep self separate from neighbours" design that SAGE, FAGCN (eps), and our trees follow; our Q2 result fits it: plain neighbour averaging (XGB-Graph) **hurts** on heterophilous neighbourhoods (−0.016 AUROC) and helps on homophilous ones (+0.057).
- Their module (2) (dissimilarity to neighbours) is close to our Q3 own-vs-group cosine features.
- Imbalance: we use class-weighted loss (GNNs) and validation cut-offs; same intent.

## For the report (literature review order, with these two added)
1. MAGNET (adaptive homophilous / heterophilous channels) → our Q2.
2. FRAUDRE (feature / topology / relation inconsistency, self kept separate) → our Q1 and Q3; the group's earlier work and the source of the inconsistency framing.
3. TROPICAL (higher-order, n-hop hypergraph) → our n-hop density check: does not transfer to TwiBot-22.
4. DEFEND / TKDD 2026 (discriminative intra- and inter-relation learning), once its name is confirmed.
