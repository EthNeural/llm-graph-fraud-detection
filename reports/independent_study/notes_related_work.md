# Related-work notes: LLM-driven social bot detection (BotSim-24, fox8-23)

Scope: five local PDFs plus the BotEvo abstract page. Numbers are quoted as printed. "UNVERIFIED" = could not be confirmed from the source text. Page numbers are given only where the text extraction made them clear; otherwise section/table numbers are used. Tables in the PDFs were read from `pdftotext` output, so column alignment was checked by reading values against the surrounding prose.

Reference datasets in our study: BotSim-24 = 1,907 humans / 1,000 bots / 46,518 edges (confirmed in the local BotSim paper, arXiv 2412.13420: "1,907 human accounts and 1,000 LLM-driven agent bot accounts", total 2,907, 46,518 edges). fox8-23 = 1,140 bots + 1,140 humans.

---

## 1. TRACE-Bot (arXiv 2604.02147, Wang, Lin, Wang, Wang; 2 Apr 2026)

**(1) Task and datasets.** Binary user-level detection of LLM-driven bots (Abstract; Sec. III-A). Datasets: Fox8-23 and BotSim-24 only (Sec. III-A, IV-A2). Input = three modalities per user: "personal information data", "interaction behavior data" (zlib-compressed sequences of Original/Retweet/Reply symbols), "tweet data" (scores from AIGC detectors Fast-DetectGPT and GLTR) (Sec. III-B). Model = dual channel: GPT-2 text encoder + MLP behaviour encoder, concatenated, MLP head.

**(2) Dataset statistics (Table I, "Dataset Statistics").**
| Dataset | Category | Users | Tweets |
|---|---|---|---|
| Fox8-23 | Genuine Users | 1,140 | 218,245 |
| Fox8-23 | Social Bots | 1,140 | 149,783 |
| BotSim-24 | Genuine Users | 1,000 | 129,722 |
| BotSim-24 | Social Bots | 1,906 | 129,728 |

- DISAGREEMENT with our figures: Table I gives BotSim-24 as 1,000 genuine / 1,906 bots (total 2,906), i.e. the class counts are swapped relative to the BotSim paper (1,907 humans / 1,000 bots) and the bot count is off by one. The text also says they "apply random undersampling to the bot class in the training and validation subsets to match human counts", consistent with their Table I (bots = majority) but not with BotSim's real class balance. Which side of the mix-up is an error is UNVERIFIED, but the BotSim paper supports 1,907 humans / 1,000 bots.
- Sec. III-A describes Fox8-23 as "over 9,000 accounts and 12,000 tweets", which does not match Table I (2,280 users). This matches the raw search numbers in the fox8 paper (12,226 tweets / 9,112 accounts), not the benchmark.
- No edge count is reported for BotSim-24.

**(3) Split.** Random 60/20/20 train/val/test (Sec. IV-A2). BotSim-24: undersample bots in train and val to match human counts; test set keeps original distribution. Max 10 epochs, batch 256, early stopping on val F1 (IV-A4). Label-efficiency study uses 5-fold CV with random splits (IV-E). Random, user-level split with no stated stratification; no temporal split.

**(4) Results (Table II, p. 7).**
- TRACE-Bot: Fox8-23 ACC 0.9846 / PRE 0.9825 / REC 0.9868 / F1 0.9847. BotSim-24 ACC 0.9750 / PRE 0.9567 / REC 0.9950 / F1 0.9755.
- Strongest baseline on BotSim-24: CACL (GNN) ACC 0.9700 / PRE 0.9518 / REC 0.9954 / F1 0.9731. On Fox8-23 CACL: ACC 0.9803 / F1 0.9806.
- Other BotSim-24 baselines (ACC/F1): Arin et al. (LSTM) 0.8150/0.8122; Mou et al. 0.6300/0.7050; BotRGCN1 0.5675/0.1722; BotRGCN2 0.6550/0.6497; BotRGCN12 0.6915/0.7643; BotMoE 0.6625/0.5603; UnDBot 0.5000/0.6664; SEBot 0.6000/0.5000; BotDGT 0.8417/0.8407; LMBot 0.4850/0.4278; Pasricha et al. (Digital DNA) 0.5300/0.4891; BotRuler 0.5000/0.6664 (the last two assigned by reading the extracted column order; modest risk of misalignment).
- Modality ablation (Table IV, p. 8; rows labelled "- X" are read as "keep only the stated remaining modality" by checking against the prose):
  - Personal-information only, BotSim-24: ACC 0.9900 / PRE 0.9900 / REC 0.9900 / F1 0.9900, which is higher than the full model (F1 0.9755). Fox8-23 personal-information only: ACC 0.9846 / F1 0.9847, equal to the full model. The prose confirms: "On Fox8-23, personal information data alone matches the full model, indicating strong discriminative signals in metadata".
  - Tweet-only: Fox8-23 F1 0.8350 (ACC 0.8224); BotSim-24 F1 0.7950 (ACC 0.7950).
  - Interaction-behaviour-only: Fox8-23 F1 0.8738 (ACC 0.8816); BotSim-24 F1 0.4472 (ACC 0.5500). The prose reports these BotSim figures ("only 0.4472 and 0.7950").
  - Personal info + interaction behaviour: Fox8-23 F1 0.9804; BotSim-24 F1 0.9848 (prose says 0.9852; small inconsistency).
  - Module ablation, Fox8-23 only (Table III): remove textual channel F1 0.9565; remove behavioural channel F1 0.9240; remove both F1 0.9481.
- Internal inconsistency: robustness text (IV-F) gives BotSim-24 F1 "0.9877 (under the 6:2:2 split)" and ACC 0.9875, vs Table II F1 0.9755 / ACC 0.9750.
- Strongest simple baselines: metadata-only (above) is the strongest single modality; text-only (GPT-2 on text + AIGC scores) is weak on BotSim-24 (F1 0.7950).

**(5) Roles.** LLM: GPT-2 is a text encoder (not a generator) for name/bio/text; Fast-DetectGPT and GLTR give AIGC-likelihood features. Best PLM is GPT-2 on Fox8-23 (Table V); prompting GPT-2 degrades F1 by about 1.7 points (Table VI). Graph: none in TRACE-Bot. The graph-based competitors (BotRGCN, BotMoE, CACL, BotDGT, SEBot, UnDBot) are run as baselines; how their graphs were built for BotSim-24/Fox8-23 is not described (UNVERIFIED). The text claims "social topology" is leveraged (IV-B) but no graph module exists in the method.

**(6) Artefacts/shortcuts.** Not discussed. No mention of missing human-to-bot edges, replayed human activity, or the self-revealing-phrase collection process. The limitations section (V) only notes English-only/Twitter-only scope and that traditional bots are not tested. Note that the BotSim-24 "personal information" fields are described in Twitter terms (Sec. III-B1: followers, followees, list count, verification etc.), while BotSim-24 is Reddit-based; which fields actually exist for BotSim-24 is UNVERIFIED.

**(7) Bearing on the claim.** SUPPORTS the first half. Metadata alone gives F1 0.9900 on BotSim-24 (above the full model) and equals the full model on Fox8-23; graph methods are mostly worse than this (BotRGCN variants F1 0.17 to 0.76), with CACL (0.9731) the exception. It does NOT test "once artefacts are removed" and has no graph-ablation of its own. Caveat: their GNN baselines may be poorly adapted, so their weakness is not clean evidence about message passing.

---

## 2. CSC: Calibrated Simplicity for Conflict-Aware Social Bot Detection in the LLM Era (arXiv 2609.23320v2, CIKM '26; Qian, Zhao, Niu)

**(1) Task and datasets.** Binary bot detection with calibrated late fusion of four experts (graph, profile, text, structured-feature) plus an "inconsistency expert" (Sec. 4). Main benchmarks: TwiBot-22 (primary), TwiBot-20, MGStBot-large (Sec. 5.1). BotSim-24 appears only as a "reduced BotSim-24 sanity check" (Sec. 5.1 and 5.6); fox8-23 is not used (UNVERIFIED by absence: grep for fox8 found nothing).

**(2) BotSim-24 statistics.** None reported (no counts of humans, bots or edges). Disagreement with our numbers therefore cannot be assessed. They say BotSim-24 "is Reddit-based and not schema-compatible with the TwiBot-style pipeline" and describe their input as "profile/activity text, public behavior statistics, and a lightweight comment-interaction graph" with "user identifiers, row order, names, and bot character settings" excluded (Sec. 5.6).

**(3) Split.** BotSim check: "three stratified splits"; ratios not stated (UNVERIFIED). Main benchmarks: shared train/val/test per benchmark; calibration, simplex weights, and thresholds fitted on val only (Sec. 5.2, Alg. 1). TwiBot-20 uses an MGTAB-style protocol with 3000-resample bootstrap.

**(4) Results.**
- BotSim-24 sanity check (Sec. 5.6): text branch F1 0.9942 ± 0.0057; behaviour/graph feature branch F1 0.9868 ± 0.0075; conflict-aware fusion F1 0.9983 ± 0.0029. Authors' reading: "Since text alone is already very strong on this Reddit-based benchmark, we interpret this result as auxiliary LLM-era sanity evidence". No metadata-only or graph-only split of the combined "behavior/graph" branch, so graph contribution cannot be isolated.
- TwiBot-22 (Table 1, p. 8; columns ACC / F1 / AUC): LightGBM 0.6334 / 0.5665 / 0.7781; RoBERTa-direct 0.6993 / 0.5477 / 0.7465; BotRGCN 0.6692 / 0.5771 / 0.7758; CSC 0.7248 / 0.5956 / 0.7809. A non-graph LightGBM has AUC about equal to BotRGCN.
- TwiBot-20: LightGBM 0.8115 / 0.8504 / 0.8298; CSC 0.8673 / 0.8828 / 0.9334. MGStBot-large: LightGBM 0.9098 / 0.8497 / 0.9681; CSC 0.9157 / 0.8532 / 0.9621; the text says "LightGBM retains the highest AUC" there.
- Graph-branch ablation (Table 3), TwiBot-22 F1: text-only GraphSAGE 0.5359; adding structured features 0.5780; later graph-control variants 0.5755 to 0.5839 (full graph model 0.5802). The authors conclude extra graph complexity "does not necessarily translate into better TwiBot-22 performance" (5.5.1). TwiBot-20 F1 for the same rows: 0.7669 then 0.8751 to 0.8831.
- Semantic-camouflage stress test (Sec. 5.6, on a TwiBot-style set, not BotSim): after matched-human-text replacement the text expert F1 drops from 0.9217 to 0.6780 and recall from 1.0000 to 0.6000, while graph and fused systems "remain stable on the balanced challenge set".

**(5) Roles.** LLM: only as a text encoder (RoBERTa); the "LLM era" is the motivation (bots write fluent text), not part of the model. Graph: a prototype-guided GraphSAGE-style expert; the paper argues structural signal is a stable anchor when text is camouflaged, and that simpler graph control is enough.

**(6) Artefacts.** No artefact discussion for BotSim-24. They do remove identifiers, row order, names and bot character settings before the sanity check, which is an implicit leakage precaution (Sec. 5.6). Limitations (Sec. 7) note the stress test is "a controlled proxy" and "larger platform-specific evaluations are still needed".

**(7) Bearing on the claim.** MIXED. Supports: on BotSim-24, text alone F1 0.9942 and a behaviour/graph branch 0.9868, fusion adds about 0.004 (ceiling effect); on TwiBot-22, LightGBM matches BotRGCN's AUC; graph complexity adds little (Table 3). Contradicts or complicates: the paper's thesis is that structure is an anchor when text is camouflaged (stress test), and that a graph expert beats text-only GraphSAGE (F1 0.5359 to about 0.58). But the Table 3 jump comes from adding structured features, not message passing, so it also fits "own-account features carry the signal". It never runs a graph-removed ablation under artefact control.

---

## 3. Adversarial Creation and Detection of AI-Generated Social Bot Content (arXiv 2606.07219, Trokhymovych, Baeza-Yates, Flammini, Saez-Trumper, Menczer; 5 Jun 2026)

**(1) Task and datasets.** Text-only: detect AI-generated messages, then aggregate to user level to detect "AI-powered social bots" (Sec. 5.1, 5.2). They build a multilingual paired human/AI dataset from Reddit (20 subreddits) and Telegram (16 chats), 17 languages, by imitating real users with persona + retrieved context prompts (openai/gpt-oss-20b persona; Sec. 3). Fox8-23 is the independent out-of-distribution benchmark. BotSim-24 not used (grep found no mention).

**(2) Dataset statistics.** None reported for fox8-23 (no 1,140 / 1,140 figures in the text, UNVERIFIED beyond grep). It says Fox8-23 "includes user identifiers, making it possible to evaluate our models on this user classification task" (5.2).

**(3) Split.** Own data: split at user level to avoid leakage; for each channel randomly 25% of users (min 50 for small channels) held out for test; 5% of training data as val (Sec. 3.4). Fox8-23 used only as an unseen test set. Message scores averaged per user.

**(4) Results (AUC, 95% CI).**
- Message level (Table 1; own data / Fox8-23): TC(RoBERTa) 0.969 / 0.714; TC(Gemma3) 0.964 / 0.722; TC(mBERT) 0.961 / 0.720; LFC 0.858 / 0.602; OSM-Det 0.603 / 0.620; MS-based TC 0.700 / 0.582; FastDetect 0.681 / 0.543; Binocular 0.665 / 0.550; GEC score 0.646 / 0.593.
- User level (Table 3; own data / Fox8-23): TC(mBERT) 0.991 / 0.989; TC(RoBERTa) 0.993 / 0.972; TC(Gemma3) 0.993 / 0.941; LFC 0.960 / 0.907; OSM-Det 0.626 / 0.861; MS-based TC 0.766 / 0.744; GEC 0.726 / 0.811; Binocular 0.754 / 0.688; FastDetect 0.797 / 0.672.
- "With just 20 messages per user, our TC (mBERT) model achieves an AUC of approximately 0.97" on Fox8-23 (Fig. 4, Sec. 5.2). Text claim: "The best model achieves near-perfect accuracy on the Fox8-23 benchmark."
- Strongest simple baseline: text-only classifiers are the whole study; the "simple" LFC (linguistic-feature gradient boosting) gets 0.907 user-level on Fox8-23. No metadata-only baseline.

**(5) Roles.** LLM: generator (gpt-oss-20b and others) to create adversarial training data; fine-tuned encoders (mBERT, XLM-R/"RoBERTa", Gemma 3 1B) as detectors. Graph: none. Authors say "Integrating multi-modal content and activity metadata would likely yield significant improvements" (Sec. 6.1) and note detectors could be "integrated into larger detection systems that include account metadata and behavioral" features.

**(6) Artefacts/shortcuts.** Directly relevant on fox8-23 (Sec. 6.1 Limitations): "This dataset has two important limitations. First, it was collected before AI content generation had reached its current sophistication. Second, it was built by identifying bots that openly revealed themselves, meaning the bots it contains were relatively easy to spot. Together, these factors suggest that detection models performing well on this benchmark may struggle significantly against more advanced bots found on today's platforms." They also note possible automated activity inside their own human baseline (Telegram), whereas "older Reddit data remains unaffected" by recent LLMs. A text-only AUC of 0.97 to 0.99 on fox8-23 is consistent with text carrying an easy signal there (this inference is ours).

**(7) Bearing on the claim.** Neutral to supportive on the "easy dataset" side: strong text-only user-level AUC on fox8-23 and an explicit statement that fox8-23 is easy. Says nothing about graph message passing. Does not test own-account features beyond text.

---

## 4. Anatomy of an AI-powered malicious social botnet (arXiv 2307.16336, Yang and Menczer; 30 Jul 2023): the fox8 paper

**(1) Task and datasets.** Case study of a ChatGPT-using Twitter botnet, behavioural characterisation, and tests of Botometer and LLM-text detectors (Abstract, Sec. 3 to 5). Introduces fox8-23.

**(2) Dataset statistics.**
- 1,140 bots + 1,140 humans = fox8-23 (Sec. 3, end). Humans: 285 randomly selected from each of four pre-existing annotated datasets: botometer-feedback, gilani-17, midterm-2018, varol-icwsm. Bot tweets: up to 200 recent tweets per bot, plus friend and follower lists, via the Twitter V1.1 API. Human tweets were also up to 200 per account, "collected years before the release of LLMs like ChatGPT".
- Bot profile averages (Sec. 4.1): followers 74.0 (SD 36.7), friends 140.4 (SD 236.6), tweets 149.6 (SD 178.8).
- Bot follow network (Sec. 4.2): in-degree 13.7 (SD 5.2), out-degree 13.4 (SD 5.8); reply network avg in-degree 3.4 (SD 2.3), out-degree 3.1 (SD 1.9); largest weakly connected component 1,036 accounts (reply), 1,058 (retweet).
- Search origin: the phrase "as an ai language model", 1 Oct 2022 to 23 Apr 2023, gave 12,226 tweets by 9,112 unique accounts. Not our study's numbers (no BotSim data here).

**(3) Split.** None. This is a characterisation paper; the only supervised-style test is thresholding the OpenAI detector's average score on all of fox8-23 (Sec. 5.3), with the threshold tuned on the same data (so the F1 is an in-sample figure).

**(4) Results.**
- Botometer (overall and sub-scores): recall near zero at threshold 2.5; overall R = 0.04, fake follower 0.03, financial 0.00, self declared 0.01, spammer 0.03, echo chamber 0.00, other 0.04 (Fig. 6, Sec. 5.1).
- GPTZero: all probabilities near zero on fox8 bot tweets (5.2).
- OpenAI AI-text classifier, account-level mean score: bots 57.7 vs humans 48.6 (t = 30.6, p < 0.001; SD 2.6 vs 9.7); best F1 = 0.84 at threshold 52.7 (5.3). In the field (1,986 random accounts), 815 are labelled bots at that threshold, with many false positives, so the authors call it "premature to rely solely on this method".
- No metadata-only or graph-only supervised baselines.

**(5) Roles.** LLM: ChatGPT is the bots' content generator (they infer this from self-revealing tweets, 1,205 of which are catalogued in Table 1: harmful content 980 (81.3%), beyond capability 148 (12.3%), other forbidden 49 (4.1%), positive 23 (2.0%), others 5). Graph: used descriptively; the follow network is dense and engineered, and the authors say the bots "can be detected through their coordination patterns" (Abstract). No GNN.

**(6) Artefacts/shortcuts (important nuance).**
- The 1,140 bots were NOT identified directly by the phrase. The phrase search produced a pool in which a 100-account manual sample found "76% are likely humans posting or retweeting ChatGPT outputs". The 1,140 were then extracted as "all 1,140 accounts linking to any of these websites": fox8.news, cryptnomics.org, globaleconomics.news (Sec. 3). Labels therefore rest on shared-URL co-membership plus manual validation, and self-revealing tweets (1,205) exist in only part of the botnet. About 3% of bot tweets link to the three sites (Sec. 4.5).
- Collection asymmetry: bots are recent accounts (tweets from 2022 to 2023); humans come from four older annotated datasets with tweets collected years earlier (acknowledged as an anti-contamination step, but it is also a source and time confound).
- The bot follow/reply/retweet network is dense because the bots follow each other. The paper computed network statistics only among the 1,140 bots and a convenience "baseline" group of 7,972 accounts that also posted the phrase (reply within-group probability 0.2% for fox8 vs 0.016% baseline). Whether human nodes in fox8-23 have comparable graph data is UNVERIFIED.
- Authors' own caveats (Sec. 6): one botnet only, "fox8 is likely the tip of the iceberg: the operators of other LLM-powered bots may not be as careless"; not all content is guaranteed LLM-generated; future bots will "likely cease posting self-revealing tweets".

**(7) Bearing on the claim.** Mixed and indirect. Supports that own-account profile/behaviour features are unusual for bots (e.g. near-identical in/out degrees around 13.5, confined tweet-type mix) and that Botometer (own-account features trained pre-ChatGPT) fails, which cuts against "own-account features carry most of the signal" for a classifier trained on older bots. The strongest structural signal (dense bot-to-bot follow cluster) is a collective/coordination signal, which supports graph value, but it is also a selection artefact of how the botnet was found (shared domains, mutual following).

---

## 5. LMBot: Distilling Graph Knowledge into Language Model for Graph-less Deployment in Twitter Bot Detection (arXiv 2306.17408v3, WSDM '24; Cai et al.)

**(1) Task and datasets.** Twitter bot detection with a fine-tuned language model distilled from a GNN teacher so inference needs no graph. Datasets: Cresci-2015, Cresci-2017, Midterm-2018, TwiBot-20 (Sec. 4.1). Not LLM-driven bots; it predates BotSim-24 and does not use fox8-23 (grep found no mention).

**(2) Statistics for BotSim-24/fox8-23.** None. Their own: Cresci-2015 5,301 users (graph); Cresci-2017 14,368 users (no graph); Midterm-2018 50,538 users (no graph); TwiBot-20 229,580 users in the graph but only 11,826 annotated used (Sec. 4.1).

**(3) Split.** "following [42], we randomly split users into 1:1:8 ratio for train, valid, and test respectively", five runs, mean and std reported (Sec. 4.1). Note the unusually small training share.

**(4) Results (Table 1, p. 7; accuracy / F1, TwiBot-20).**
- LMBot-LM 85.63 / 87.61; LMBot-GNN/MLP 85.25 / 87.38.
- Graph methods: RGT 84.70 / 87.19; SimpleHGN 84.65 / 87.13; BotRGCN 84.42 / 86.90.
- Text, no graph: RoBERTa-finetune 84.05 / 86.13 (about 1 F1 point below RGT and about 1.5 below LMBot-LM).
- Metadata/feature methods: SGBot 79.58 / 83.51; BotHunter 73.43 / 78.01; LOBO 76.17 / 80.64; Kudugunta et al. 58.08 / 48.07.
- Cresci-2015: LMBot-GNN/MLP 99.06 / 99.26; SGBot 97.76 / 98.23; BotRGCN 98.60 / 98.90; RoBERTa-finetune 98.13 / 97.43. Cresci-2017 (no graph): LMBot 98.34 / 98.91; SGBot 95.49 / 97.04. Midterm-2018: LMBot 99.21 / 99.53; SGBot 98.61 / 99.18.
- GLNN (GNN distilled into MLP): GLNN-RGT 83.02 / 85.60; GLNN-BotRGCN 82.56 / 85.17 on TwiBot-20.
- Consistency check: LM and GNN/MLP predictions agree on 99.59% (Cresci-2017) and 95.77% (TwiBot-20) of points (Fig. 6).
- TRACE-Bot's Table II reports LMBot as a baseline: BotSim-24 ACC 0.4850 / F1 0.4278; Fox8-23 ACC 0.8969 / F1 0.8962. That is TRACE-Bot's run of LMBot, not LMBot's own paper.

**(5) Roles.** LLM: a pre-trained masked/encoder LM (RoBERTa, T5, BART, DeBERTa), fine-tuned on a textual serialisation of each user; not a generative LLM and not about LLM-written bots. Graph: GNN is a teacher (BotRGCN, RGT, SimpleHGN, GCN, GAT, RGCN combinations) whose knowledge is distilled back into the LM; motivated by the cost of multi-hop neighbour fetching and "graph sampling bias" (Sec. 1, Fig. 1).

**(6) Artefacts.** No dataset-artefact discussion beyond noting API sampling bias in neighbour fetching (twenty random neighbours per query, Sec. 1). Argues text methods "can be easily deceived when bots steal tweets" and feature methods by tampering metadata.

**(7) Bearing on the claim.** SUPPORTS "graph adds little" in a narrow sense: on TwiBot-20 the graph-free fine-tuned RoBERTa is within about 1 F1 point of the best graph-only methods and LMBot-LM (graph-free at inference) is the best, and the gain over graph methods is 0.2 to 0.5 F1 points. But the graph still helps in training (teacher), and metadata-only SGBot on TwiBot-20 is about 4 F1 points below RGT, so it does not support "own-account metadata carries most of the signal" on that dataset; text carries it there. It is not an LLM-bot or artefact-controlled study.

---

## 6. BotEvo (J. King Saud Univ. Computer and Information Sciences, vol. 38, art. 460, 28 May 2026; Springer DOI 10.1007/s44443-026-00814-3)

Source: Springer page, fetched (WebFetch returned a summary, not the verbatim abstract; reading of the full text was truncated at 100,000 of about 154,500 characters, so everything below is from the abstract/summary and the early part of the page). Treat details marked as such with caution.

- Title: "BotEvo: LLM-Driven social bot detection via behavioral evolution modeling and cross-modal fusion". Authors listed include Yuxin Zhang, Kai Qiao, Shuhao Shi, Jiaxin Liu, Zihao Liu, Jian Chen, Lei Li, Bin Yan (list truncated on page).
- Method: dynamic weighting of individual posts, implicit gating to fuse text, behavioural and sentiment features, and two-stage threshold optimisation for risk-stratified classification. Described as graph-agnostic: it uses user post sequences only; GCN, GAT, BotRGCN appear as baselines.
- Datasets: BotSim-24 stated as "2,907 accounts (1,907 human, 1,000 bot)", agreeing with our figures; and a Twibot-20 + SDAR mixed set (11,826 labelled Twibot-20 samples + 3,000 LLM-generated SDAR samples; 3-way classification). Fox8-23 not mentioned.
- Split: 70% train / 20% val / 10% test, stratified.
- Results: BotSim accuracy 96.81%, F1 96.46%, MCC 92.95%; Twibot-20/SDAR accuracy 88.93%, F1 89.30%, MCC 0.8227; "outperforms 14 state-of-the-art baselines" (feature, text, graph, LLM-driven).
- Artefacts: none found in the fetched portion (UNVERIFIED for the full text).
- Bearing: a graph-free model (text + behaviour + sentiment) reaches F1 96.46% on BotSim-24, which is below TRACE-Bot's metadata-driven 0.9755/0.9900 and CSC's text-only 0.9942 sanity figure, but those use different splits and feature sets, so no direct comparison is warranted.

---

## Cross-paper summary relevant to the claim

"Own-account features carry most of the signal; graph message passing adds little once artefacts are removed."

| Paper | Evidence on BotSim-24/fox8-23 | Direction |
|---|---|---|
| TRACE-Bot | Personal-information-only F1 0.9900 (BotSim-24), equal to full model on Fox8-23 (0.9847); most GNN baselines F1 0.17 to 0.84, CACL 0.9731 | Supports first half; no artefact control; GNN baselines may be poorly adapted (UNVERIFIED) |
| CSC | BotSim-24 text-only F1 0.9942; behaviour/graph branch 0.9868; fusion 0.9983 | Ceiling effect; graph not isolated; its main thesis favours structure under text camouflage |
| Trokhymovych et al. | Text-only user-level AUC 0.941 to 0.989 on fox8-23; dataset "relatively easy to spot" | Supports "dataset is easy"; silent on graph |
| Yang and Menczer (fox8) | Botometer recall 0.04; dense bot follow cluster (in/out degree about 13.5); bots found via 3 shared domains | Shows structural signal exists but is tied to how the botnet was assembled |
| LMBot | Graph-free RoBERTa within about 1 F1 of best GNN on TwiBot-20 | Supports "graph adds little" on a non-LLM-bot dataset |
| BotEvo | Graph-agnostic model, BotSim F1 96.46% | Consistent with graph not being necessary |

No paper in this set discusses the BotSim-24 construction artefact (no human-to-bot edges because human activity is replayed). The only artefact-type remarks are: fox8-23 bots were self-revealing and found via shared links (fox8 paper Sec. 3; Trokhymovych et al. Sec. 6.1), and humans in fox8-23 come from older datasets (fox8 paper Sec. 3). The BotSim-24 artefact claim in our study rests on our own reading of the BotSim paper, not on these five papers.

Flags for the reader:
1. TRACE-Bot Table I swaps/changes the BotSim-24 class counts (1,000 genuine / 1,906 bots) versus 1,907 humans / 1,000 bots in the BotSim paper and BotEvo.
2. TRACE-Bot Table II Pasricha/BotRuler BotSim-24 rows were assigned from extracted column order (low risk of misalignment).
3. The user's premise "fox8 bots found via the phrase 'as an AI language model'" is only partly right: the phrase gave the candidate pool; the 1,140 final bots were selected by links to fox8.news, cryptnomics.org, globaleconomics.news.
4. BotEvo details come from a summarising fetch tool and a partial page read.
