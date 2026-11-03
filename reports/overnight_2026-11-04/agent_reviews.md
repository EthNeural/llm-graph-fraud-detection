# Independent agent reviews (every 2 h)

Each entry: a fresh read-only reviewer, then my triage in overnight_notes.md.

## Review 1 (3 Nov, 15:17)
1. "Heterophily" framing unsupported: adjusted homophily ≥ 0 except friend edges; 51.6% of a bot's neighbours are bots (TEAM_GUIDE.md:29,109,154 and README.md:4 say mostly humans). Reframe as label-uninformative edges. (90%)
2. LoRA checkpoint chosen by val LOSS on 200-user sample (step6_dgp.py:376-382) vs MLP/GNN by full-val macro-F1; val AUROC still rising at epoch 2 in both A runs. Score state.pt adapter, choose by full-val F1. ~40 min GPU per A variant. (80%)
3. Test "no edge signal" directly: degree-preserving shuffle; semi-synthetic heterophily sweep (−0.4…+0.4 adjusted homophily). ~1.5 h CPU. (80%)
4. Setting B: no CIs, 54 test bots, one subsample; SAGE > MLP contradicted by AUROC/PR-AUC; 29% isolated vs 7.8% in A. (85%)
5. Connected vs isolated table: GCN loss is all on connected users. (85%)
6. LoRA prompt caps own text at 480 tokens: 81% of users cut short. Disclose. (75%)
7. Text/number mismatches (FAGCN ranges, "findings 2 and 3", E5 "best" on Instagram, "encoder matters most", TEAM_GUIDE edge counts/§7 outdated). (95%)
8. No figures; REPORT is a notebook; add limitations + matched-comparison table. (70%)

## Review 2 (3 Nov, 17:30)
1. Four sentences contradicted by their own numbers: LoRA "between 650 and 1,300" (actually ≈ LR at 325); real-edge GCN gain 0.048 "far less" than 0.059 (it isn't); "SAGE does not care" (B AUROC reverses); "nothing in B" (E5 GCN −0.029); "B curve flat" (LR +0.05). Stale header/status. (95%)
2. Setting B removes 99.3% of bot–bot edges (8,162 → 54 rows); bots' neighbours 7.8% bots < 9.1% base; isolation tied to label (41.8% bots vs 27.6% humans). B's "no edge signal" is the sampling. Fix: state it; better B′ = full graph, 1:10 only on labelled nodes. (85%)
3. Shuffle control = one rewiring; ± is model-seed only. Add SHUFFLE_SEED, 5 rewirings. (80%)
4. step6_rescore.py hard-codes "epoch 1/2" labels; only 2 of 4 checks saved; training stopped at 2 so later epochs uncovered. (85%)
5. Step 6 B will lack matched-data control and bootstrap pairs: queue control_train_size + bootstrap for B after the chain. (80%)
6. Real-vs-shuffled GCN gain may be text similarity along edges, not labels: cosine along real vs shuffled vs random, by label pair. (70%)
7. 1,300-user control is one training draw: add 5-10 draws. (70%)
8. Isolated column unstable (GCN 0.777 real vs 0.724 shuffled with identical inputs): unbold, caveat. (80%)

## Review 3 (3 Nov, 19:20)
1. Stale numbers/contradictions: finding 3 "0.048" (now 0.037), isolated "0.724" (one rewiring), signposting example MLP 0.747 (no MLP row), Instagram FLAG ⏳ vs scope, plain vs adjusted homophily vocab; README Instagram bolding and findings outdated. (90%)
2. FAGCN beats GCN as much on shuffled edges → gain is the eps residual, not signed weights; retitle finding 2. (75%)
3. B dgp prompt "none" lines reveal isolation, tied to label in B: pre-register a link-flag check + report connected-only dgp − target. (65%)
4. Instagram: label is commercial vs personal, not fraud; cite GraphAdapter as source; GCN ≥ MLP in A for all encoders: use as contrast. (80%)
5. Step 5 sample noisy (±0.04, shifts models differently); Instagram B zero-shot ≈ LR; add full-test notes; soften finding 5. (80%)
6. Findings lack a one-line answer and a setting B verdict; finding 1 vs 3 wording; finding 6 rests on one pair; finding 7 title overreaches; M-sweep reason wrong (friend-of-friend binds, 5.1). (80%)
7. Best GPU use Wed-Fri: 2 more LoRA seeds × dgp/target on A (~2.2 h per run, one night) after adding LORA_SEED + seed in TAG/CKPT; optional LoRA target on 2,600 users (~4.5 h) to test "mostly data". (70%)
8. Missing-text flags in neighbour ranking: true for step 5 (E5 + flags); NOT for step 6 (flags dropped, step6_dgp.py:441) — reviewer partly wrong. (60%)

## Review 4 (3 Nov, 21:20)
1. Link-type cue is mostly a crawl/popularity artefact: each user stores ~10 followers/followings; out-degree capped at 10, reciprocity 1-4%; all signal in incoming links; in-friend tracks followers_count (ρ 0.59). Reword. (85%)
2. Edges not label-free in direction × type: in-follow − in-friend untrained AUROC 0.713; our undirected pipelines discard it. (90%)
3. Number mismatches: finding 7 zero-shot on sample only; FAGCN 0.003–0.030; GCN 0.036–0.064; LR 0.756 vs 0.757; B +0.023; cue table shares are all users, not test. (95%)
4. Test dgp gain vs cue: within-flag AUROC, target + flags stack vs dgp. (75%)
5. Overclaims: "no GNN beats" (B E5 GCN +0.025); Qwen alike only in A; Instagram contrast causal. (85%)
6. CI for the +0.013 flag lift. (70%)
7. isolation_cue.py: no leakage; transductive edges legit; B flags partly label-built. (90%)
8. Own lists recorded depend on label (95.8% vs 81.4%). (65%)

## Review 5: planning (3 Nov, 23:15)
New evidence: B dgp highest PR-AUC, lowest AUROC; B dgp val AUROC collapsed 0.810 → 0.624 at last check (state.pt = collapsed adapter); bootstrap_ci/link_direction read only best-loss scores; link_direction A-only; bootstrap macro-F1 only; RQ2 rationale (explainability) absent from REPORT.
Plan Wed-Sat: (Wed AM CPU) adapter choice + AUROC/PR-AUC CIs in scripts; settle checkpoint rule; step6_dgp.py: LORA_SEED, seed in TAG/CKPT, adapter per check, select by full-val F1, probe. (GPU Wed ~12-22) 2 more LoRA seeds × dgp/target on A (~9 h). (Thu opt) LoRA target on 2,600 A users (4.5 h). No B seeds. Fri evening: GPU off. (Thu-Fri CPU) seed-mean bootstrap, figures. (Sat) 3-user qualitative table for RQ2, final tables. STOP: reviewer cycles after Thu, Instagram work, new encoders, B′, R-GCN, FLAG baseline, M sweep, GADBench, changelog upkeep.
Figures: 1 learning curve + LoRA points; 2 CI forest plot; 3 edge signal (plain vs adjusted homophily; real vs shuffled); 4 link direction (in/out × type, bots vs humans); 5 (opt) encoder × model dot plot.
Outline: O1 text does the work; O2 near-zero edge label signal, averaging hurts; O3 LLM roles (training-size control, target+flags); O4 crawl artefact (own heading); O5 rare class B; O6 Instagram contrast; Limitations. Appendix: full ladder, Instagram, step 5 table, connected/isolated, support users, bridges, learning-curve numbers, decision log. Drop changelog/status.
Risks: (1) rescore picks epoch 2 and A test ≥0.73 → LoRA ≈/≥ MLP(1,300), switch main row (rule fixed before test), keep old; (2) B target ≪ dgp → report next to B target+flags; lead with "B cannot separate"; (3) PR-AUC framing flips B headline → report all three metrics, claim only what holds on all.

## Review 6: QA of recent changes (4 Nov, 01:10)
No code bugs; tonight's B runs safe. Text: (1) "target + flags matches dgp" only holds in A (B: −0.054 vs dgp); (2) B MLP quoted seed-0 0.633 but CIs use 3-seed 0.613; (3) "loss and AUROC agreed in B" only for B dgp; B target like A; B dgp last adapter = collapsed epoch 3, not "epoch 2"; (4) finding 8 / limitations outdated now B has AUROC/PR-AUC CIs (several exclude 0); B flag lift CI [+0.002, +0.049]; (5) bootstrap averages per-seed AUROC (consistent with F1, fine); B renumbering correct; (6) minor rounding/status items.

## Review 7: checkpoint-rule decision (4 Nov, 03:00)
Defensible: rule written in step6_rescore.py in fc45a63 (3 Nov 15:24) before last-adapter test scores (4 Nov 01:51 / 02:25); same rule as MLP/GNNs; it keeps the lowest-loss adapter in B dgp, so it doesn't always favour later checkpoints. Risks: REPORT's "decide once the rescoring is in" reads as results-dependent (reworded); only 2 adapters compared; B target margin tiny (0.6435 vs 0.6378); must switch all runs, not only where test went up. If switched: A dgp 0.727/0.815, target 0.728/0.812; B target 0.644/0.739; dgp − target −0.001 (A), −0.016 (B); LoRA ≈/above MLP(1,300). Rerun CIs and target+flags with the chosen adapters (done via LORA_ADAPTER=chosen). Phrase: rule fixed in advance; both adapters in one table; don't call the switch "improving" LoRA.
