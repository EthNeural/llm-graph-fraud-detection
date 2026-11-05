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

## Corrections after the sceptic review (6 Nov ~02:40; checked against the paper and the data)
- **Split:** the paper's 2,304 / 582 / 291 sums to 3,177, not 2,907, and App. A.1 says "same data splits provided in the benchmarks": internally inconsistent. We do not reuse it (ours: 60/20/10/10, seed 42).
- **Labels:** bot labels are exact by construction; human labels rest on an unspecified annotation step (App. C.1, "verified as human through annotations"). Say "bot labels exact, human labels annotated", not "no weak-label problem".
- **F1:** the paper never says its F1 is for the bot class (remove "bot-class").
- **Edges are directed** (commenter → target): there are **no human→bot edges but many bot→human edges** (data: 17,424 L1 pairs, 1,189 L2 pairs). The paper's Discussion sentence "does not include interactions between humans and bots" is loose.
- **Edge counts:** 46,518 is the paper's figure; the released data gives ~47.6–48.8k depending on dedup (ours 47,598).
- **Topics:** bots spread disinformation on three topics but also join broader news discussion (App. C.4).
- **Metadata disguise:** the paper removes Cake Day / Post Karma / Comment Karma; post-level `created_utc` remains. **Users.csv `submission_num` equals the real post count for 1,589 / 1,907 humans but only 34 / 1,000 bots**: never derive "reported vs actual count" features.
- **Text cleaning (App. C.1):** links, formatting strings and media were removed from human text to resemble GPT text; text-length and style differences are therefore partly preprocessing.
- **The paper already tests LLM text detection (Table 5):** GPT-4-turbo 5-shot Acc 70.76 / F1 63.58; zero-shot Acc 39.18 / F1 22.03 (sample size not stated).
- **Ethics statement:** research use only.
- Hyperparameters (Table 6): 2 layers, dropout 0.3, hidden 128, 100 epochs, best-validation selection, seeds 0–4, relations {C1-P, C2-P, C1-C2}; text = RoBERTa embeddings averaged over posts and comments; 10 metadata features incl. Description (Table 10) — so their metadata baselines may use the description leak.

## From the related-work review (`notes_related_work.md`)
- Later papers report F1 ≈ 0.97–0.99 on BotSim-24 (TRACE-Bot 0.9755, CSC text-only 0.9942, BotEvo 96.46%), far above the original paper; TRACE-Bot's **personal-information-only** variant gets F1 0.9900, above its full model. None discusses the missing human→bot edges or the zero-engagement leak. High scores are consistent with leak-driven profile features.
- **fox8-23 is a weak second dataset:** bots were selected by links to three websites (fox8.news, cryptnomics.org, globaleconomics.news), and the 1,140 humans come from four older datasets (source and time confound). Use only as a secondary check, with links removed from the text.
