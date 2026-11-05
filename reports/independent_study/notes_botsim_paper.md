# BotSim-24 paper notes (Qiao et al., AAAI 2025, 39(13), 14377–14385; arXiv 2412.13420; PDF in `research documents/independent_study/`)

- **Data:** 2,907 users (1,907 real Reddit humans from 6 news subreddits, 1,000 GPT-4o-mini agent bots); bots post and comment on three debated topics (Ukraine–Russia, Israel–Palestine, US politics). 46,518 edges, **3 edge types**: first-level commenter ↔ post author, second-level commenter ↔ post author, first-level ↔ second-level commenter; comment counts as edge weights. Their split 7:2:1 random (2,304 / 582 / 291). **Labels exact by construction** ("simulation labeling strategy").
- **Disguise by design:** metadata disguise (bot counts generated in proportion to human statistics; `created_utc`, karma removed), text disguise (LLM writes human-like text), interaction disguise.
- **Their results (Table 3, 5 seeds, accuracy / F1 for the bot class):** metadata AB 77.5 / 74.8, RF 75.7 / 72.4; text Wei et al. 50.8 / 50.4 (≈ random); RoBERTa+NN 67.6 / 30.5; GCN 72.7 / 50.5; GAT 80.3 / 73.1; **BotRGCN 89.9 / 86.7 (best)**; RGT 82.3 / 76.4; S-HGN 87.7 / 83.1. (TwiBot-22 under weak labels in the same table: best F1 57.6.)
- **Their own explanation of the GNN win (§Experimental Analysis):** human data is replayed in timeline order, so humans never initiate interactions with bots: there are human↔human, bot→human, bot↔bot edges but **no human→bot edges**. "This structural incompleteness enables GNNs to identify the intrinsic differences." Edge-perturbation (reversing a share of edge directions) changes GNN performance accordingly.

## What this means for our independent study
1. **A construction artefact in the graph, like TwiBot-20's crawl artefact.** Test: the same models with direction removed, with direction-only features ("links only": in/out counts by type), and with the missing-direction cue neutralised. If the graph gain collapses, it was the artefact.
2. **Text:** their text baselines are weak (RoBERTa+NN F1 30.5), but they did not try modern sentence encoders (multilingual-E5, MiniLM) with simple classifiers, or a fine-tuned LLM. Does LLM-written text remain detectable by an LLM? (Future Outlook 4.)
3. **Metadata:** counts generated "in proportion" to humans; run our shortcut rule first.
4. Exact labels: does "own features first, graph adds little" (TwiBot-22) hold when labels are clean but bots are LLM-driven?
