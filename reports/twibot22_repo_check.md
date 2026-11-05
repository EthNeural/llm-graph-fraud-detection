# TwiBot-22 official code check (github.com/LuoUndergradXJTU/TwiBot-22), 5 Nov 2026 (external)

Cloned read-only into the scratchpad; nothing executed. What the leaderboard's top graph models (BotRGCN #1, GAT #2, GCN #3) actually do, from their code:

| Item | Their code | Ours |
|---|---|---|
| Split | the released `split.csv` (`BotRGCN/twibot_22/preprocess_1.py:18`): train 7.8% bots, val 28.0%, test 29.4% | random stratified 70/20/10 (14% each) + sealed half |
| `verified` | **an input**: categorical tensor = protected, **verified**, default_profile_image (`preprocess_1.py:193`); GCN and GAT load the same tensors (`GCN_GAT/dataset.py:9-15`) | removed (0 of 95,398 verified users are bots; a labelling rule) |
| Other inputs | RoBERTa bio vector, RoBERTa tweet vector (all 1M users), numerical counts | profile 15 fields + multilingual-E5 bio PCA 32 (GNNs); bio + 20 tweets only on the 100k sample |
| Relations | 2: followers, following (`preprocess_1.py:55-64`) | 7 (following, followers, mention, retweet, reply, quote, like) |
| Training / selection | fixed 200 epochs, test at the last epoch, no validation-based selection (`train.py:74-77`) | early stopping and cut-off on validation |
| Threshold and metrics | argmax (0.5); accuracy and bot-class F1; **AUC computed from hard 0/1 predictions** (`train.py:53-61`) | validation cut-off; macro-F1, AUROC and PR-AUC from scores |

The labelling voters (paper §3) were MLP, GAT, GCN, R-GCN "following Feng et al. 2021b" = this same input encoding (with `verified`), trained on 800 of the 1,000 expert-labelled users. The 8:2 expert split is **not** released (no train/test marker in `src/annotation/`), so which 800 is unknown.

## A second, independent human-label set
`src/annotation/` also holds an **annotation study**: 500 random TwiBot-22 users, each rated by 3 experts on 0–4 (0 definitely human … 4 definitely bot), 1,500 ratings. Only 1 of these users is among the 986 experts, and they were not used to train the voters, so they are independent of how the weak labels were made. Our roles: 371 train, 73 val, 34 test_work, 22 sealed. Fine for comparing *labels with labels* (D8); too few test users to score models. Copied to `data/twibot-22-raw/annotation_study/` (not in git; licence).
