# BotSim-24: data inspection (Fri 6 Nov ~02:20, before any model)

Counts: 2,907 users (rows 0–1,906 human, 1,907–2,906 bot, per Readme), ids match between `Users.csv` and `user_post_comment.json`. Posts: humans 34,346, bots 18,297. Comments: humans 44,710 first-level + 3,415 second-level; bots 29,607 + 1,405.

## Leaks (class-revealing artefacts of the simulation) — excluded from every model
| Field | Humans | Bots | Why |
|---|---|---|---|
| post `score` = 0 | 6.6% | **100%** | simulated posts were never voted on |
| post `upvote_ratio` = 0 | **0%** | **100%** | same |
| post `num_comments` | median 23 | **always 0** | same |
| comment `comment_score` = 0 | 1.8% | **100%** | same |
| post id starts with a digit | **100%** | 28.6% | bot ids are random strings; human ids are Reddit base-36 |
| comment id starts with a digit | 0% | 27.7% | same |
| `description` empty | **85.4%** | **0%** | every bot got a generated bio (Readme) |
| `character_setting` | never | always | Readme: cannot be used |
Not leaking at a glance: post times (same range, 2023-06-20 to 2024-06-19; seconds = 0 on 1.7% vs 1.8%); `submission_num` (median 3 vs 3); `comment_num` (median 9 vs 14, max 836 vs 112: a weak, allowed signal). Text length differs (post median 75 vs 87 chars; comment 126 vs 217): allowed, but reported.

**Consequence.** The paper's metadata baselines used "10 standardized numerical" profile features (their Appendix C.1); scores, votes and ids are post-level, so their reported numbers may not use these leaks, but any model built on the raw JSON would. Our rule: use only text, profile counts (submission_num, comment_num, comment_num_1, comment_num_2), subreddit membership, timing, and the reply graph; never scores, vote ratios, num_comments, ids, description presence, or character_setting. Description *text* is also excluded (presence is a perfect cue).

## Graph (built by `my_build/independent/botsim_preprocess.py`, 02:30)
- 47,598 directed typed pairs from 73,762 comments (1,150 comments' parents not in the dataset). Types: L1 comment → post author 39,888; L2 comment → post author 3,481; L2 reply → L1 commenter 4,229.
- **Human → bot edges: 0 of 47,598** (h→h 27,653, b→h 19,752, b→b 193). Adjusted homophily −0.26 / −0.20 / −0.16 by type: strongly heterophilous (bots reply to humans).
- **In-edges are a leak:** 78.4% of humans receive a reply vs 8.4% of bots; minus in-degree alone, untrained, gives **AUROC 0.871**. Out-degree is behaviour (bots 99.9% vs humans 84.2% have one; AUROC 0.633).
- Clean single features: profile counts 0.50–0.60, subreddits 0.50–0.61, timing 0.58–0.67, mean post length 0.795, mean comment length 0.722 (AUROC, max of both directions).
- Split (seed 42, stratified): train 1,744 / val 581 / test_work 291 / test_sealed 291, 34.4% bots each.
