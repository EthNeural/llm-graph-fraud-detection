# Independent study: plan (separate from the submitted report; never overwrites it)

Mission (`uni brief /Projects – Dr. Venus Haghighi ... .pptx`): investigate LLM-based approaches for graph fraud detection, and how textual and structural information can be jointly leveraged; deliver a framework prototype + a technical report. Authorised by the builder (Fri 6 Nov ~03:30) to run independently after the main report is finished and the GPU is free.

## Data
- **BotSim-24** (Qiao et al., AAAI 2025, arXiv 2412.13420; github.com/QQQQQQBY/BotSim): 2,907 Reddit accounts (1,907 real humans, 1,000 LLM-driven bots), profiles, posts, first- and second-order comments; reply links from comment `parent_id`. **Labels exact by construction** (bots were created by the authors), so no weak-label problem. Downloaded 6 Nov 02:06 to `data/botsim24-raw/` (git-ignored; no licence stated: research use only, never committed).
- TwiBot-22 (existing) as the weak-label counterpart.

## Problems from the main study it can address
1. Weak labels (TwiBot-22) → exact labels here: does the "profile beats text, graph adds little" pattern survive clean labels?
2. LLM-written bots (Future Outlook 4): can text models still detect bots that write with LLMs?
3. Camouflage: bots reply to humans by design; measure heterophily on the reply graph before modelling.
4. LLM + graph fusion (RQ1, RQ4): encoders vs LoRA vs DGP-style neighbour summaries, with matched training sizes.
5. Shortcuts: simulated bots may differ trivially (counts generated "in proportion", unified creation time removed); run the shortcut rule first.

## Rules (as in the main study)
Rules before scores (decision log in this folder), random stratified split + sealed test, shortcut rule, macro-F1 / AUROC / PR-AUC with paired bootstrap CIs, 3 seeds, edge-off controls. CPU work never during LoRA training; GPU only after the main queue.
