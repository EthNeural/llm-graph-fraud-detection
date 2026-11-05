# For the submission (internal)

"Internal" = goes into the submitted COMP8851 research project report (7 pages, template: Introduction, Literature review, Methodology + timeline, Outcome, Contributions). Everything else in the repository is external working material. Items are added here as they are decided; Sunday's writing starts from this file.

**Writing rule:** as in research papers, the report never cites feedback (supervisor or external) as the reason for a choice. Heterophily is motivated by the problem itself: fraudsters and bots camouflage by linking to benign accounts (Dou et al., 2020), so methods must be evaluated under measured heterophily. Dataset changes are justified on their merits (e.g. Elliptic has no text for an LLM to read).

## Template and project brief (`uni brief /`, checked 5 Nov 23:10)

- **Sections (latest template):** Introduction (for a layperson: aim, why it matters, how, expected vs achieved) · Literature Review · Research Methodology (sub-problems, integration with figures, timeline vs plan, difficulties and fallbacks) · Research Outcome (follow the plan; deviations as P → P′) · Individual Contributions · **Future Outlook** · **Bibliography**.
- **Citation rule:** cite work from the past 3 (recommended) to 5 (at most) years, i.e. 2021 or later. Our list to check: Dou et al. 2020 (CARE-GNN) and Yang et al. AAAI 2020 are 6 years old: keep only as one-line foundational mentions, or replace (camouflage can be motivated from MAGNET 2024 / TKDD 2026 and FRAUDRE 2021). Fine: TwiBot-20 (2021), FRAUDRE (2021), TwiBot-22 (2022), Platonov, Hays, GADBench, Mao (2023), MAGNET, TROPICAL (2024–25), TKDD 2026.
- **Project brief:** "Large Language Models for Graph-based Fraud Detection": how textual (LLM) and structural (graph) information can be jointly leveraged; expected outcome a framework prototype + technical report. **Framing for the report:** the three LLM roles the brief names map onto our pipeline: LLM node representations (multilingual-E5 / MiniLM text features into trees and GNNs), LLM reasoning over graph structure by prompting (zero-shot, DGP with neighbour summaries), and LLM fine-tuning (LoRA `target`). The main result is *when* each helps under heterophily, not "profile wins".

## Methodology: timeline

Past work (dates 2026):

| When | Work |
|---|---|
| Sep – Oct | Team pipeline on Elliptic (Bitcoin transactions) |
| 27 Oct | Elliptic dropped (no text for an LLM to read); rebuilt on Instagram (GLBench text graph) |
| 30 Oct | Instagram results; settings A (natural) and B (1:10, FLAG protocol) |
| 1 – 4 Nov | TwiBot-20 as the main dataset: encoder ladder, GNNs (GCN, SAGE, FAGCN, LINKX, FLAG-lite), zero-shot LLM, LoRA fine-tuning and a DGP replication; shortcut and crawl-artefact checks; 22-item self-review |
| 4 Nov | TwiBot-22 data preparation: 1M users, 88.2M tweets, 7 relations; shortcut, heterophily and leakage checks; sealed test set and 100k study sample; all rules recorded before any model |

Planned (estimates, 5 Nov):

| When | Work | Estimate |
|---|---|---|
| Thu 5 Nov, early | Text embeddings (multilingual-E5, MiniLM) for the 100k sample + hubs | ~3.5 h GPU (measured pace for E5) |
| Thu 5 Nov | Cheap models (MLP, trees, XGB-Graph, DEC-lite, baselines), with and without profile, sample and 1M | ~2–4 h CPU |
| Thu 5 Nov | GNNs on the full 1M graph: 1 seed for all 5, then 3 seeds for the 2 best by validation macro-F1 (tables mark 1 vs 3 seeds) | ~4–7 h CPU |
| Thu 5 Nov, 02:41–06:18 | LoRA `target` (measured: 3.6 h) | done |
| Thu 5 Nov – Fri 6 Nov | Expert rotation (~4 h), then DGP at TwiBot-20 scope (1,500-user test subset; ~9.5 h, summaries ~1 s each), then second seeds of `target` and `dgp` (~7 h) | ~20 h GPU |
| Fri 6 Nov, early | Weak → expert LoRA fine-tuning, 5-fold rotation over the 986 experts (blind) | ~3.5–4 h GPU |
| Fri 6 Nov | Confidence intervals, expert rotation for trees / MLP, sealed-set final run, comparison with TwiBot-20 | ~3 h |
| Fri night – Sat 7 Nov | TwiBot-20 follow-up batch (optional), figures | ~12 h GPU + ~4 h |
| Sun 8 Nov | Writing | 1 day |

## Literature review plan (builder, 5 Nov)

Order and argument (~400 words in the report):
1. **MAGNET**: adaptive learning for homophilous and heterophilous neighbourhoods (Haghighi et al., "Robust Graph Learning Against Camouflaged Malicious Actors", WISE 2024, LNCS, doi 10.1007/978-981-96-0567-5_12; verified on Crossref).
2. **TROPICAL**: higher-order relationships and hypergraph learning (Haghighi et al., ICDM 2024, doi 10.1109/icdm59182.2024.00019; extended in Knowledge and Information Systems 2025, doi 10.1007/s10115-025-02476-5; verified).
3. **"DEFEND"**: discriminative learning through intra- and inter-relation patterns. **Name not verified**: the closest match is Haghighi et al., "Beyond Smoothing: A Discriminative Enhanced Aggregation Graph Neural Network for Camouflaged Fraud Detection", ACM TKDD 2026, doi 10.1145/3801967. Check the model's name in the paper before citing.
   **Clue:** the TROPICAL journal paper (KAIS 2025, §6.1.1) says its datasets validate "DEFEND", so DEFEND is a model by this group; confirm which paper.
4. **FRAUDRE** (Zhang, Wu, Yang, Beheshti, Xue, Zhou, Sheng, IEEE ICDM 2021, doi 10.1109/ICDM51629.2021.00098): feature / topology / relation inconsistency caused by camouflage, plus imbalance; per-relation class homophily (normal > 0.85, fraud < 0.2), the same shape as TwiBot-22 (humans ~0.93, bots 0.04–0.17). Notes: `reports/paper_notes_tropical_fraudre.md`.
   Also available: Haghighi, "From Classic GNNs to Hyper-GNNs for Detecting Camouflaged Malicious Actors", WSDM 2023, doi 10.1145/3539597.3572989.

Argument: in online social networks, malicious accounts may deliberately connect to legitimate users (camouflage; Dou et al., 2020), weakening homophily-based graph learning. Haghighi et al. address this with adaptive neighbourhood aggregation, multi-relational learning and higher-order representations. These motivate testing whether interaction patterns, local structure and cross-relation inconsistencies give complementary signals for camouflaged accounts (our Q1–Q3, checklist E8–E10). Transfer already tested for TROPICAL's motivation: on TwiBot-22 the bot density around bots falls with hop distance (0.116 → 0.059 over hops 1–4, base rate 0.14), unlike Amazon where it rises. **Caution to state:** those papers evaluate specific fraud datasets; they do not show the methods transfer to TwiBot-22, so our experiments test that transfer.

**Related-work checks** (`reports/literature_check_findings.md`, arXiv ids verified): supported by Yang et al. AAAI 2020, Hays et al. WWW 2023, GADBench (Tang et al. NeurIPS 2023), Mao et al. NeurIPS 2023; contradicted by the TwiBot-22 leaderboard (graph methods first): discuss split, `verified` and graph-model labelling functions as candidate reasons. **Confirmed in the official code** (`reports/twibot22_repo_check.md`): BotRGCN, GCN and GAT use the skewed official split, take `verified` as an input, train a fixed 200 epochs with no validation selection, and report AUC from hard predictions.

## Future Outlook: candidate points (draft list, 6 Nov ~00:45; builder picks 3–4)

Template asks: what remains beyond scope, what further work could solve, the challenges, the most promising approaches. Each point below comes from a result of ours.

1. **Labels are the bottleneck.** Remains: weak labels overstate the empty-bio cue (5.0× vs 1.6× under independent humans), and the 986 experts are not independent of the label-makers. Challenge: human labels are costly and biased too (Hays et al.). Promising: more independent annotation chosen by active learning (label the users the models disagree on), and noise-robust training on the weak labels.
2. **Camouflage-aware graph models.** Remains: message passing adds little, and neighbour averaging hurts users in heterophilous neighbourhoods (Q2). Promising: models that adapt aggregation per node or per relation (MAGNET; the discriminative intra- / inter-relation model of the TKDD 2026 paper), tested on TwiBot-22 under our protocol.
3. **LLMs with more context and more data.** Remains: the LLM read 480 tokens, trained on 1,300 users, 1.7B parameters; DGP adds neighbour summaries (result pending). Promising: longer context, more training users, LLM-derived features fed into the tree models instead of the LLM deciding alone.
4. **Distribution shift and newer bots.** Remains: every model loses ~0.11 AUROC on the official (non-random) split, and TwiBot-22 is a 2022 snapshot; newer bots may write with LLMs, weakening text cues. Promising: time-based splits and evaluation on newer data.
5. **Benchmark practice.** Our protocol (random stratified split, sealed test, no `verified`, macro-F1 / AUROC / PR-AUC instead of accuracy) as a proposed standard for comparing graph and non-graph detectors.
6. **Other fraud graphs.** The project brief covers e-commerce and finance: test whether the same pattern (profile / own features first, little from message passing under heterophily) holds on text-rich fraud graphs such as review networks.

## Methodology: other items to include (to be filled)

- Datasets and why: Instagram (secondary), TwiBot-20, TwiBot-22 (see `artifacts/04_decisions_and_reasons.md`).
- Heterophily measures: adjusted homophily, label informativeness.
- Shortcut rule and what it excluded (`verified`).
- Expert labels (986, 105 bots): used for training **and** clean evaluation by 5-fold rotation (decided 5 Nov, before any expert score); weak labels match experts for 56 of 101 dataset bots.
- Split, sealed set, sample; metrics (macro-F1, AUROC, PR-AUC, reweighted PR-AUC, bootstrap CIs, non-verified slice).

## Outcome: TwiBot-22 findings so far (5 Nov evening; working-test users, weak labels; sealed set and experts not yet scored)

Paired bootstrap 95% CIs (2,000), seed 0, each model with its own validation cut-off (`my_build/twibot22_ci.py` → `results/twibot22/ci_pairs.json`). Sample = the 5,000 sample working-test users (same as LoRA); 1M = 49,951 working-test users, bio only. Models are HistGradientBoosting trees ("XGB-Graph style"), 3 seeds unless noted; sceptic-checked 5 Nov.

1. **Profile beats every text model.** Profile only (15 fields): macro-F1 0.701, AUROC 0.837. LoRA `target` (Qwen3-1.7B, 1 seed, 1,300 training users): 0.626 / 0.710; difference +0.073 [+0.054, +0.093] F1, +0.128 [+0.106, +0.149] AUROC. With the same 1,300 training users, profile only still gets 0.677 / 0.808. Caveat: much of the profile signal is the empty-bio flag (40.6% of bots vs 7.8% of humans), bio length and follower counts, on weak labels partly made by labelling functions; the expert check (rotation) tests this.
2. **Text adds a little on top of profile.** Tree profile + text − profile: +0.011 [−0.001, +0.025] F1, +0.012 [+0.005, +0.018] AUROC (sample); +0.012 [+0.008, +0.016] / +0.014 [+0.012, +0.016] (1M).
3. **Neighbour features help without profile, barely with it.** XGB-Graph − own features, no profile: +0.005 [−0.010, +0.021] F1 but +0.053 [+0.037, +0.070] AUROC (sample), +0.015 / +0.044 (1M). With profile: −0.001 [−0.012, +0.010] F1, +0.005 [−0.000, +0.010] AUROC (sample); +0.010 [+0.007, +0.013] / +0.012 [+0.011, +0.014] (1M, where PR-AUC +0.054). Best model overall: XGB-Graph with profile on 1M, 0.731 / 0.873 / PR-AUC 0.580.
4. **Tweets add nothing over the bio for a text-only model; averaging them in hurts.** Same 70k training / 5,000 test users: bio + 20 tweets − bio only: −0.052 [−0.074, −0.030] F1, −0.049 [−0.074, −0.026] AUROC. Mechanism checked by slices: on users with a bio the two are equal (AUROC 0.682 vs 0.681); on the 602 users without a bio (46% bots) tweets help (0.613 vs 0.500), but averaging blurs the no-bio group's contrast with everyone else (mean score 0.25 vs 0.12, against 0.46 vs 0.10 for bio only). With profile (which holds the empty-bio flag) the two are equal (0.713 vs 0.712). **Tested directly:** adding only an empty-bio flag to the bio + tweets embedding closes the gap (0.672 vs 0.673 F1; AUROC 0.775 vs 0.761).
5. **Fine-tuned LLM vs cheap text models** (5,000 users): with the same 1,300 training users LoRA beats TF-IDF+LR by +0.050 F1 [CI excludes 0]; TF-IDF+LR on 70k training users ties LoRA (+0.002 [−0.015, +0.019]). Zero-shot Qwen: 0.505 / 0.461 (answers "bot" for nearly everyone; readout checked).
6. Crawl-structure baseline (links only, neighbour counts): 0.563 / 0.672: the graph's shape alone carries some signal.
7. **Heterophily by relation and class (Q1, E8).** On all 7 relations bots' neighbours are 4–17% bots (base rate 14.0%), humans' ~93% humans; adjusted homophily 0.010–0.024 (train + val users only). The FRAUDRE shape (normal > 0.85, fraud < 0.2) holds.
8. **Local heterogeneity (Q2, E9; MAGNET's question).** Neighbour features (XGB-Graph − own-feature tree, both with profile, 1M) help users with local homophily ≥ 0.75 (n 27,002, 1% bots): +0.057 [+0.045, +0.071] AUROC, and hurt users < 0.25 (n 4,197, 77% bots): −0.016 [−0.022, −0.009]. RGCN − MLP hurts in the homophilous bin (−0.052 [−0.074, −0.030]); FAGCN − MLP gains in every bin (+0.017 to +0.039). Predicted by Mao et al. (NeurIPS 2023).
9. **Higher-order density (TROPICAL's motivation) does not transfer:** bot share among bots' n-hop neighbours 0.116 → 0.081 → 0.086 → 0.059 (hops 1–4); on Amazon it rises.
10. **Cross-relation inconsistency (Q3, E10; FRAUDRE's question) adds nothing:** 8 label-free features (follow vs interaction neighbourhoods); XGB-Graph + profile with − without, 1M seed 0: F1 −0.002 [−0.004, −0.000], AUROC +0.000 (`q3_ci_full.json`); same on the sample.
11. **GNNs, 3 seeds (1M):** FAGCN 0.690 ± 0.001 / 0.829, RGCN 0.689 ± 0.000 / 0.804; MLP control (seed 0) 0.679 / 0.799.
12. **Leaderboard reconciliation (E11, official protocol; partial, 5 Nov 22:45).** Official split, our sealed users removed (test 94,889, 29.5% bots, all-human accuracy 0.705): profile tree 0.763 acc@0.5 / 0.700 macro-F1 / 0.765 AUROC with `verified`, 0.763 / 0.695 / 0.764 without; XGB-Graph 0.769 / 0.689 / 0.758 (both); MLP 0.627 / 0.683 / 0.745 (with), FAGCN 0.603 / 0.667 / 0.731 (with), 0.625 / 0.678 / 0.745 (without); RGCN 0.645 / 0.666 / 0.724 (with). Reading: the split lowers AUROC by ~0.11; `verified` changes ≤ 0.01; graph models do not beat the tree without neighbours under the official protocol either. GNN accuracy is below all-human because our GNNs use class weights (theirs do not): compare GNNs on macro-F1 / AUROC.
13. **The profile model is not just the empty-bio cue** (slice already computed in `cheap_*.json`, "with_bio"; 3 seeds). Two tests: (i) *more clues, same users*: empty-bio flag alone AUROC 0.658 → profile (15 fields) 0.837 (sample); (ii) *same model, users where empty bio cannot help* (users with a bio: 4,398 sample / 43,694 1M, 9.7% / 9.5% bots): flag 0.500, profile 0.790 / 0.802, profile + text 0.808 / 0.826, XGB-Graph + profile 0.813 / 0.841 AUROC. Macro-F1 falls on this slice (profile 0.604 / 0.606) partly because the bot share is lower and the cut-off was chosen on all validation users, so AUROC is the fair measure here. **Limit:** this shows the other profile clues carry signal *against the weak labels*; whether they reflect real bot behaviour or label-maker habits is D8 / the expert check.

14. **Expert labels (986 experts, 105 bots; 5-fold rotation; read early 5 Nov 23:50 by the builder's decision, recorded before reading; `expert_read_early.json`).** AUROC weak-only → after the expert stage [CI of change]: LoRA `target` 0.822 → 0.724 (−0.099 [−0.164, −0.038]); profile tree 0.861 → 0.867 (+0.006); profile + text 0.879 → 0.899 (+0.020 [+0.000, +0.040]); XGB-Graph + profile (sample) 0.883 → 0.910 (+0.027 [+0.009, +0.044]), macro-F1 0.751, PR-AUC 0.599, the best; XGB-Graph (1M) 0.875 → 0.887; MLP 0.819 → 0.845 (+0.025 [+0.006, +0.046]); RGCN 0.799 → 0.866 (+0.067 [+0.034, +0.104]). The dataset's weak label as a prediction: macro-F1 0.745; its 90.5% agreement compares with 89.4% for all-human. **Reading:** the model ranking holds on expert labels; the GNN gains most from clean labels but stays below the trees; the LLM's one-epoch expert stage hurts its ranking. **Caveat:** 800 of the 1,000 expert labels trained the label-making models (TwiBot-22 paper §3.2), so weak-only models are not tested on fully independent labels.
15. **D8: empty-bio lift (bot rate empty ÷ bot rate with bio; bootstrap 2,000).** 986 experts: weak 8.7× [6.4, 12.3] vs expert 7.2× [5.3, 10.3] (difference +1.5 [−1.5, +4.6]); not independent (above). Annotation study (TwiBot-22 repo; 6 researchers, 3 per user; median ≥ 3 bot, ≤ 1 human, 2 dropped; 458 users after removing 21 sealed non-expert users; 44 bots): weak 5.0× [3.3, 7.7] vs annotators 1.6× [0.7, 3.1], difference +3.4 [+1.2, +6.1]. **Reading:** under independent human labels the empty-bio cue is much weaker; the weak labels overstate it. Wording: "consistent with Hays et al. (WWW 2023), extending it to TwiBot-22" (they studied TwiBot-20).
16. **Leaderboard reconciliation complete except one cell** (E11; seed 0; table in `reports/meeting_brief_2026-11-06.md` §7): the split lowers AUROC by ~0.11 for every model; `verified` moves trees ≤ 0.01 and GNNs in both directions; under the official protocol the own-feature tree is best on macro-F1 / AUROC and no GNN beats the MLP without edges.
