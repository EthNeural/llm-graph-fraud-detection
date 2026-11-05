> **Snapshot** of the whole project as of **Fri 6 Nov 2026, ~00:45**. Read in this order:
> **Part 0**: TwiBot-22 explained from the ground up, in plain words (start here).
> **Part A**: the full working report (`REPORT.md`): Instagram and TwiBot-20.
> **Part B**: TwiBot-22: data, checks, decisions, **results so far**, the graph questions Q1–Q3, the leaderboard conflict, the expert stage, what is running, what is next.
> **Part C**: external feedback and the TwiBot-20 to-do.
> **Part D**: every decision with its time and reason (all three datasets).
> **Appendices**: the TwiBot-22 checklist, the TwiBot-20 to-do, the notes for the submitted report (internal), and the full run log of 5 Nov with every exit code.
> Live status: `reports/STATUS.md`. Nothing here is a final result until the sealed run (Fri 6 Nov).

# Part 0: TwiBot-22 explained (plain words first)

## TwiBot-22, explained from the ground up

Written Thu 5 Nov 2026, ~00:45; updated ~19:00 with the first results (sections 6, 7, 10, 11). This explains what we are doing with TwiBot-22 and why. Each section starts in plain words and then goes one level deeper. A short glossary follows each section. Exact numbers and rules live in `reports/twibot22_data_checklist.md`; the reasons for each decision are in `04_decisions_and_reasons.md`.

---

### 1. The whole thing in three sentences

We have a million Twitter accounts, and about one in seven is a bot. We want to know whether looking at **who an account is connected to** helps catch bots, beyond reading **what the account writes**. On our first Twitter dataset (TwiBot-20) the answer was "no, the writing does all the work". TwiBot-22 is the second, bigger test, where the writing is weaker.

**Glossary**
- **Bot**: an account run by software, not by a person.
- **Connections**: follows, mentions, retweets, replies, quotes and likes between accounts.

---

### 2. What TwiBot-22 is

**Plain version.** TwiBot-22 is a public research dataset collected in early 2022. It has profiles for 1,000,000 accounts and 88 million of their tweets, plus who follows, mentions and retweets whom. Every account is marked "bot" or "human".

**One level deeper.** The labels are not hand-checked one by one. With a million accounts, that is impossible. The dataset's authors combined many automatic rules and models into a single vote. These are called **weak labels**: they are often right, but not always. For 986 accounts, human experts also checked the label. When we compare the two, the dataset's "bot" label matches the experts for only 56 of 101 accounts. So the expert labels are our most trustworthy check, and we protect them (section 6).

**Deeper still.** Some of the rules used to make the labels are things a model could see, such as the blue "verified" tick. If a rule made the label, a model that finds the rule is not detecting bots; it is copying the label-maker. That is why we hunt for **shortcuts** (section 5).

**Glossary**
- **Weak labels**: labels produced automatically, cheaply, with some errors.
- **Expert labels**: labels checked by people; few, but more reliable.
- **Labelling function**: one automatic rule used to vote on a label.

---

### 3. The question: do connections help?

**Plain version.** On many social networks, birds of a feather flock together: friends tend to resemble each other. If bots mostly connected to bots, a bot's friends would give it away. But fraudsters and bots often connect to ordinary users to blend in. Then a bot's friends look normal and tell you little.

**One level deeper.** The first pattern (similar accounts connect) is **homophily**. The second (different accounts connect) is **heterophily**. Which one a dataset has must be measured, not assumed. We measured it for each of the 7 kinds of connection.

**Deeper still.** Plain "share of links between same-label accounts" is misleading when one class is rare: with 86% humans, most links join two humans just by chance. So we use **adjusted homophily**, which subtracts that chance level: 0 means "no better than random", 1 means "perfectly same-label". On TwiBot-22 every relation scores between **0.011 and 0.038**, essentially no label signal. A bot's neighbours are 11.6% bots, *below* the 14% you would expect by chance. That is heterophily.

**Glossary**
- **Homophily / heterophily**: links mostly between similar / different accounts.
- **Adjusted homophily**: homophily corrected for chance; around 0 means links carry no label signal.
- **Neighbour**: an account linked to this one.

---

### 4. Too big to do everything: cut the work, not the graph

**Plain version.** Reading 20 tweets for each of a million accounts with a modern language model would take many hours on a laptop. We could shrink the dataset, but cutting the network apart removes links and changes the problem.

**One level deeper.** We copy what large companies do: **keep the whole network, and do the expensive work on a sample**. Facebook's abuse system (called DEC) computes simple features for every account, but trains and tests on samples; it measures how many abusive accounts it has by hand-labelling 20,000 random users a day.
- **Every one of the 1M accounts:** cheap features (profile numbers, a short summary of the bio, summaries of each account's neighbours).
- **A random 100,000 accounts** (the same 14% bot share): the expensive reading of bio + 20 tweets, and the language-model step.
- Their neighbours still come from the full million, so no link is lost.

**Deeper still.** TwiBot-22's network is **star-shaped**: only 10,039 accounts (1%) have any outgoing links; everyone else only receives them. So every account's neighbour features come from those ~10k "hub" accounts. We therefore read the tweets of the 100k sample **and** of the hubs (110,057 accounts in total).

**Glossary**
- **Sample**: a random subset that keeps the same mix as the whole.
- **Stratified**: drawn so each group (bots/humans, train/test) keeps its share.
- **Hub**: one of the ~10k accounts with outgoing links.

---

### 5. Making it fair, part 1: shortcuts and the guard

**Plain version.** A model can look clever by spotting a giveaway that has nothing to do with bot behaviour. A classic example: an image classifier for "wolf vs husky" that learned "snow means wolf" because the wolf photos had snow. On TwiBot-20, the "verified" tick alone predicted the label almost perfectly, because the dataset labelled verified accounts as human.

**One level deeper.** We tested every input for shortcuts with one fixed rule: if a simple split on one field puts 200 or more training accounts into a group that is 99% one class, it is a shortcut. On TwiBot-22:
- **Only "verified" is a shortcut**: 0 of 95,398 verified accounts are bots. It is removed.
- **All 814 model features pass**, as do 43,469 words and every posting app.
- Some strong but fair clues remain and are reported:
  - an **empty bio**: 40.6% of bots vs 7.8% of humans;
  - **no tweets**: 17.6% vs 4.8%;
  - a **default profile picture**: 13.0% vs 1.7%.

**Deeper still.** Removing a column is not enough if a later script reads it by mistake. So "verified" and the labels now live in a separate evaluation-only folder, and a **guard** in the code stops any model that is handed a column called verified, label, y or split. We tested that it catches them.

**Glossary**
- **Shortcut**: an input that predicts the label for reasons unrelated to the task.
- **Guard**: code that refuses to run if a forbidden input appears.

---

### 6. Making it fair, part 2: rules first, and a sealed test

**Plain version.** If you try many ideas and keep whichever scores best on the test, the test score flatters you. Medical trials avoid this by writing their analysis plan before they see results. Kaggle competitions keep a hidden test set that only counts at the end.

**One level deeper.** We do both.
- **Every rule is written down and committed before any score:** the split, the sample, the text recipe, the metrics, how a model is chosen.
- **The test accounts are split in two:**
  - a **working half** (49,951 accounts), used while building;
  - a **sealed half** (50,937 accounts, including all 986 expert accounts), scored **once**, at the very end.
- A second guard refuses to score sealed accounts before that final run.

**Deeper still.** Choices such as "which model version is best" are made on a separate **validation** set (200k accounts), never on test. On TwiBot-20 some of those choices were made after test scores were seen, a mistake we listed and are not repeating. The expert labels get a second job, borrowed from Facebook's DEC (noisy labels first, clean labels second). Sealing all 986 for one final score would waste them, because the weak labels disagree with the experts too often to learn the expert view. So they are **rotated**: split into 5 parts; train on 4, score the 5th with a model that never saw it; repeat. Every expert is used for training and is still scored cleanly once. This runs **blind**: the scores are saved but nobody looks until the final run.

**Glossary**
- **Train / validation / test**: learn from / choose with / judge on.
- **Sealed set**: test accounts kept untouched until the final run.
- **Pre-registration**: fixing the rules before seeing results.

---

### 7. What each account "looks like" to a model

**Plain version.** A model cannot read words directly. Each account is turned into a list of numbers that describe it. There are three kinds of description.

1. **Its writing.** For each account, the bio plus its **20 most recent tweets**.
   - Retweets are kept, and links and @mentions are left as written, as on TwiBot-20.
   - A text model (**multilingual-E5**, which handles many languages) reads the bio and each tweet separately and turns each into 768 numbers. The account's description is the average.
2. **Its profile.** 15 fields, such as follower count, account age, empty bio and default picture ("verified" excluded).
3. **Its neighbours.** For each of the 7 kinds of link and each direction:
   - how many neighbours it has;
   - the average of their descriptions;
   - plus averages over neighbours-of-neighbours.

   In total 814 numbers per account.

**One level deeper.** The neighbour averages are exactly the idea behind Facebook's DEC and the GADBench benchmark's strongest simple method: describe an account by summaries of its neighbourhood, then let a simple model use them.

**Deeper still.** The text reading is the slowest step: 2.06 million pieces of text for the 110,057 sampled and hub accounts, 153.5 minutes on the Mac's GPU (finished 5 Nov, early morning). A smaller text model (MiniLM) followed as a cross-check (37.7 minutes).

**Glossary**
- **Embedding**: the list of numbers a text model produces; similar texts get similar numbers.
- **multilingual-E5 / MiniLM**: the two text models we use (main / check).
- **Feature**: one number describing an account.

---

### 8. The models we will compare

**Plain version.** From simplest to most complex:
- **Baselines** that show what simple clues alone give:
  - profile only;
  - "links only" (just the counts of connections), which measures how much the way the data was collected gives away.
- **Text-only model** (an MLP): reads only the writing.
- **Trees with neighbour summaries** (XGB-Graph, DEC-lite): the industry-style approach.
- **Graph neural networks** (GCN, SAGE, FAGCN, LINKX, RGCN): models that pass information along the links.
- **A language model fine-tuned for the task** (Qwen + LoRA, including our replication of the DGP method), on the 100k sample.

**One level deeper.** Each cheap model is run twice: **with** the profile fields and **without** them. Without profile, the comparison is clean: does adding the network help the writing? With profile, we see the best detector we can build. The expensive models (graph networks, language model) run with profile only, to save time.

**Deeper still.** The headline comparison is on the 100k sample, where every model reads the same text. A full-million run with bio only checks that the answer holds at full scale. The language model must read the same text as the encoders it is compared with; on TwiBot-20 it read less (480 tokens vs up to 2,000 characters), which external feedback rightly flagged.

**Glossary**
- **MLP**: a small neural network; here the "text only" reference.
- **GNN** (graph neural network): a model that mixes each account's description with its neighbours'.
- **Fine-tuning / LoRA**: adapting a large language model to our task by training a small add-on.

---

### 9. How we keep score

**Plain version.** "Accuracy" is misleading here: a model that calls everyone human is right 86% of the time and catches no bots. So we use measures that reward finding the bots.

**One level deeper.**
- **Macro-F1:** the average of how well we do on bots and on humans, each counted equally.
- **AUROC:** how well the model ranks bots above humans (0.5 = coin flip, 1 = perfect).
- **PR-AUC:** how precise the model stays as it catches more bots. This is the most telling measure when bots are rare. Facebook's DEC paper makes the same point.

**Deeper still.**
- We also report PR-AUC as if bots were only **5% or 1%** of accounts. This is done by reweighting the test accounts, not by changing the data, because real fraud is often rarer than 14%; Facebook reports 3.8–5.2% abusive accounts.
- Every number comes with a **confidence interval** (by resampling the test accounts 2,000 times).
- Every number is also shown on **non-verified accounts only**, because on TwiBot-20 the verified accounts (all human) inflated every score by 0.06–0.09.

**Glossary**
- **Confidence interval**: the range the true score probably lies in; if two models' ranges overlap a lot, their difference may be luck.
- **Reweighting**: counting some test accounts more, to mimic a different bot share.

---

### 10. Where we are now (Fri 6 Nov, ~00:45)

- **Done:** all data checks; the sample feature table with bio + tweets (830 numbers per account, no shortcuts); the fine-tuned language model reading only the account's text (LoRA `target`); text-only baselines; every cheap model, with and without profile, on the sample and on all 1M; the five graph neural networks on the full 1M graph (the 2 best with 3 seeds); confidence intervals; a sceptic review; a literature check; three graph questions (Q1–Q3, below); a check of the official TwiBot-22 code; the expert stage for the language model, the trees, the MLP and RGCN, read early on the 986 experts (the builder's decision, Thu 23:50); the empty-bio test (D8); the leaderboard reconciliation (all but one run).
- **Running:** the language model with neighbour summaries (DGP) on the GPU, scored early Friday, then a second seed of both language-model versions until about 13:00; the expert stage for FAGCN; the last reconciliation run.
- **Next:** the final run on the sealed set and the experts (Friday afternoon), the comparison with TwiBot-20, figures Saturday, the report Sunday.

---

### 11. What we found so far (working test users, weak labels)

**Plain version.** On TwiBot-22, an account's **profile** (follower counts, empty bio, account age, ...) gives bots away far better than anything it writes. Reading its tweets adds almost nothing. Its connections add a little, mostly when the profile is left out.

**One level deeper** (macro-F1 / AUROC, higher is better; 0.5 AUROC = coin flip):
- **Profile only** (15 fields, a simple tree model): **0.701 / 0.837**.
- **Fine-tuned language model on the text:** 0.626 / 0.710. A word-count model (TF-IDF) trained on 70,000 accounts ties it.
- **Profile + text:** 0.713 / 0.848, only a little above profile alone.
- **+ neighbours** (the industry-style tree with neighbour summaries): 0.712 / 0.852 on the sample, **0.731 / 0.873 on all 1M** (the best so far).
- **Graph neural networks** (all 1M): 0.678–0.689 / 0.800–0.828, within about 0.01 of the same network with the links switched off. Message passing along heterophilous links adds little, as on TwiBot-20.

**Deeper still.**
- **Why tweets do not help:** 40.6% of bots have an empty bio, against 7.8% of humans. A text model on the bio alone spots that easily. Averaging 20 tweets into the description blurs it, so bio + tweets scores *lower* than bio alone (−0.05 macro-F1, same users). Once the profile (which has an "empty bio" field) is added, the two are equal.
- **The big caveat:** the weak labels were made partly by automatic rules, some of them profile-based. A model that wins with profile fields may partly be copying the label-maker. The expert rotation and the sealed final run test exactly this.

**The three graph questions (Q1–Q3), plain version.**
- **Q1: who links to whom?** Bots' neighbours are only 4–17% bots, against 14% of all users; humans' neighbours are ~93% humans. Bots hide among humans (heterophily), on every one of the 7 relations.
- **Q2: does it differ from user to user?** Yes. Neighbour features help users whose neighbours look like them (mostly humans: +0.057 AUROC) and slightly *hurt* users whose neighbours do not (mostly bots: −0.016). This is the pattern Mao et al. (NeurIPS 2023) predict, and the question MAGNET asks.
- **Q3: do the relations disagree?** We built 8 "inconsistency" numbers (e.g. how different an account's follow-neighbours are from its interaction-neighbours). They add nothing: 0.729 vs 0.731 macro-F1 on all 1M.
- **TROPICAL's pattern does not carry over:** on Amazon, bots get denser further out in the graph; on TwiBot-22 bot density *falls* with distance (0.116 → 0.059 over 1–4 hops).

**The leaderboard conflict, plain version.** The TwiBot-22 paper ranks graph models first. Their official code shows why the comparison is not like ours: they use a skewed split (test 29.4% bots), give the models `verified` (no verified user is a bot), train a fixed 200 epochs, and rank by accuracy. We are re-running our own models their way (results so far: the split matters, `verified` hardly does, and graph models still do not beat the tree without neighbours).

**Checking against human labels, plain version.** The dataset's labels were made by machines. 986 users were also labelled by experts, and 458 more by an independent group of 3 researchers per user.
- **The ranking holds with expert labels:** profile-based trees with neighbour features are still best (AUROC 0.910); the graph network gains most when retrained on expert labels (+0.067) but stays below them.
- **Empty bio is a weaker clue than the dataset says.** By the dataset's labels, empty-bio users are 5× as likely to be bots; by the independent humans, on the same users, only 1.6×. The 986 experts cannot settle this, because their labels were used to train the label-making machines.

**Glossary**
- **TF-IDF:** counting which words an account uses, weighted by how rare they are.
- **Message passing:** a graph network updating each account with its neighbours' descriptions.

---

# Part A: working report (Instagram, TwiBot-20)

## Progress Report

A living report, updated as experiments finish. The newest entries are at the top of each
section. For how everything works, see [TEAM_GUIDE.md](TEAM_GUIDE.md).

**Last updated:** 3 November 2026, 20:15

**Scope.** TwiBot-20 is the main dataset: steps 1–6, settings A and B. Instagram is the secondary dataset, steps 1–5 only: it checks whether the step 1–5 findings hold on a second, multilingual graph. Step 6 is not planned for Instagram. One step 6 run takes about 3–6 h on our Mac, so that time goes to the main dataset.

### Status

| Experiment | TwiBot-20 A | TwiBot-20 B | Instagram A | Instagram B |
|---|---|---|---|---|
| Step 1: homophily | ✅ | ✅ | ✅ | ✅ |
| Step 2: baselines | ✅ | ✅ | ✅ | ✅ |
| Step 3: encoders | ✅ 7 encoders | ✅ 7 encoders | ✅ 3 encoders | ✅ 3 encoders |
| Step 4: MLP / GCN / SAGE / FAGCN | ✅ + bridge graph | ✅ | ✅ | ✅ |
| Step 5: zero-shot reasoner | ✅ | ✅ | ✅ | ✅ |
| Step 6: DGP (LoRA), dgp | ✅ 1 seed | ✅ 1 seed | not in scope | not in scope |
| Step 6: DGP (LoRA), target | ✅ 1 seed | ✅ 1 seed | not in scope | not in scope |
| FLAG-lite and LINKX baselines (step 4) | ✅ | ✅ | ✅ | ✅ |

✅ done · 🔄 running · ⏳ to do

**Current run (GPU, from 4 Nov 04:26):** step 6 LoRA seeds 1 and 2 for `dgp` and `target` on TwiBot-20 A (`LORA_SEED`), each rescored right after; about 11 h in total. CPU alongside: the semi-synthetic heterophily sweep.
- Done today: step 6 `target` on TwiBot-20 A (13:30–15:42); Qwen3-1.7B embeddings (15:42–19:14, 3.5 h).
- On CPU after the chain: the setting B training-size control and bootstrap.

### Data preparation

#### Sources and labels
- **TwiBot-20** (Feng et al., CIKM 2021), the TwiBot-22-format release (`TwiBot-20-Format22.zip`), read straight from the zip by `my_build/prep_twibot20.py`. Class 1 = bot. We keep the **11,826 labelled users** and drop the 217,754 unlabelled support users (see "dropped support users").
- **Instagram**, the GLBench version (Li et al., 2024) that FLAG also uses, already prepared as `x_text`, `y` and `edge_index` files; we did not rebuild it. Class 1 = commercial account (business, brand or media), a stand-in for a minority class, not fraud.

#### Text per user
- **TwiBot-20:** the bio plus the user's first 20 tweets, one per line (whitespace trimmed, line breaks inside a tweet removed), cut at 2,000 characters. **61.6% of users reach that cap**, so the models see the bio and as many tweets as fit. 12.9% of users have no bio, 1.3% have no tweets, 37 have no text at all.
- **Instagram:** the bio (median 119 characters, maximum 523); 99 users have none.
- **Profile metadata is excluded.** TwiBot-20's "verified" flag alone gives macro-F1 0.803: a collection shortcut (Hays et al., WWW 2023), not text or graph signal.

#### Edges
- **TwiBot-20:** friend and follow links between labelled users, stored both ways (undirected), merged into one edge type for steps 4–5; step 6 keeps the two types. 16,349 undirected edges; no self-loops or duplicates.
- **Instagram:** 72,005 undirected follow edges; no self-loops or duplicates.

#### Splits and the two settings
| | Users | Class 1 | Edges | Train / val / test |
|---|---|---|---|---|
| TwiBot-20 A (natural) | 11,826 | 6,589 bots (55.7%) | 16,349 | 8,278 / 2,365 / 1,183 (official split) |
| TwiBot-20 B (1:10) | 5,761 | 524 bots | 5,274 | 4,000 / 1,164 / 597 |
| Instagram A (natural) | 11,339 | 4,115 commercial (36.3%) | 72,005 | 6,803 / 2,268 / 2,268 (stratified 60/20/20, seed 42) |
| Instagram B (1:10) | 7,946 | 722 commercial | 38,193 | 4,767 / 1,589 / 1,590 |

**Setting B** keeps every class-0 user and a random sample of class-1 users (seed 42) so that class 1 is 10% of class 0, following FLAG's protocol (§4.1.1), with only the edges whose two ends are kept. Splits are restricted to the kept users. The subsampling removes almost all bot–bot edges on TwiBot-20 (54 rows vs 8,162 in A).

#### Encoding (step 3)
TwiBot-20 text is longer than an encoder reads, so each piece (the bio and each tweet) is encoded separately, up to 128 tokens, and the L2-normalised piece vectors are averaged per user. Decoder LLMs (Qwen2.5-1.5B, Qwen3-1.7B) and Twitter-RoBERTa are mean-pooled over tokens. A user with no text gets a zero vector, and two 0/1 flags ("no bio", "no tweets") are added for TwiBot-20, so nobody is dropped. Features are standardised with training-set statistics only.

#### Profile metadata: which fields are shortcuts? (TwiBot-20 A, full test set)

We use every profile field unless it is a shortcut, under a rule fixed before looking at results (`my_build/metadata_check.py`): a field is excluded if (a) it is a documented collection or labelling artefact, or (b) one split on it labels a large group almost perfectly (a depth-1 tree leaf with at least 10% of training users that is at least 99% one class).

| Field | Single-field depth-1 tree: macro-F1 / AUROC | Purest large leaf (share of users, purity) | Spearman with in-friend count | Verdict |
|---|---|---|---|---|
| verified | 0.803 / 0.800 | – | – | **shortcut**: documented (Hays et al., WWW 2023, §4.1) |
| followers count | 0.731 / 0.731 | 31.7%, 77.0% | +0.59 | allowed |
| following count | 0.456 / 0.534 | 16.9%, 68.8% | −0.03 | allowed |
| tweet count | 0.594 / 0.597 | 49.9%, 65.8% | +0.38 | allowed |
| listed count | 0.714 / 0.716 | 27.0%, 81.6% | +0.59 | allowed |
| account age | 0.620 / 0.621 | 52.6%, 66.5% | +0.33 | allowed |

| Qwen2.5-1.5B embeddings + | LR macro-F1 (AUROC) | MLP macro-F1 (AUROC), 3 seeds | Gain over text only (paired bootstrap) |
|---|---|---|---|
| nothing (text only) | 0.756 (0.822) | 0.766 (0.840) | – |
| **the 5 allowed fields** | **0.804 (0.874)** | **0.806 (0.885)** | LR +0.048 [+0.027, +0.067]; MLP +0.040 [+0.024, +0.056] |
| all 6 fields incl. verified (reference only) | 0.861 (0.926) | 0.844 (0.921) | +0.10 |
| the 5 allowed fields alone (no text) | 0.748 (0.769) | 0.742 (0.796) | – |

The allowed fields add a real +0.04–0.05 to text. Two caveats: followers and listed count correlate ρ 0.59 with the crawl artefact (in-friend count), so part of their gain is the same collection effect; and every comparison elsewhere in this report (encoders, GNNs, LLMs) is text-only, so these rows are a separate "text + profile" track, not a replacement. The DGP numerical-summary variant (`VARIANT=dgp_num`) uses only the 5 allowed fields.

#### Checks on the prepared data
| Check | Result |
|---|---|
| Graph: self-loops, duplicate edges, symmetry | none, none, symmetric (both datasets) |
| Duplicate text across splits (`duplicate_text.py`) | 1 of 1,183 TwiBot-20 test users and 12 of 2,268 Instagram test users share a training user's normalised text; dropping them changes Qwen2.5 + LR by ≤ 0.001 |
| Raw vs cleaned text (`clean_text_check.py`, TF-IDF + LR) | URLs → `URL`, @mentions → `@USER`, retweet prefixes removed: −0.007 to +0.002 macro-F1; deleting URLs and mentions: −0.007 to −0.026. Every CI includes 0, so cleaning would not have helped, and removing URLs and mentions slightly hurts (they carry signal: 93.8% of TwiBot-20 users post URLs, 33.4% of tweets are retweets) |
| Language | 2.8% of TwiBot-20 and 5.6% of Instagram users write mostly non-Latin text; kept, and the multilingual encoder covers it |

### Latest results (macro-F1, test set)

**Which test set?** Each heading says so. Most tables use the **full test set**: TwiBot-20 1,183 users (A) and 597 (B); Instagram 2,268 (A) and 1,590 (B). One table, "LLM reasoner vs trained models", uses the smaller **step 5 test sample**, because LLM scoring is slow. Its trained rows are also a single seed, so the same model can show a slightly different number there (for example, Qwen2.5 + GCN is 0.707 in the ladder and 0.698 there).

#### TwiBot-20, encoder ladder (full test set, mean of 3 seeds)

| Encoder | A: MLP | A: GCN | A: SAGE | A: FAGCN | B: MLP | B: GCN | B: SAGE | B: FAGCN |
|---|---|---|---|---|---|---|---|---|
| MiniLM | 0.726 | 0.686 | 0.721 | 0.722 | 0.588 | 0.550 | 0.565 | 0.584 |
| E5-base (English) | 0.760 | 0.700 | 0.740 | 0.733 | 0.564 | 0.589 | 0.567 | 0.542 |
| Twitter-RoBERTa | 0.749 | 0.692 | 0.743 | 0.722 | 0.612 | 0.579 | 0.609 | 0.571 |
| multilingual-E5 | 0.747 | 0.699 | 0.741 | 0.726 | 0.606 | 0.585 | 0.589 | 0.567 |
| Qwen3-Embedding | 0.727 | 0.691 | 0.707 | 0.710 | 0.599 | 0.574 | 0.558 | 0.579 |
| Qwen2.5-1.5B | 0.766 | 0.707 | 0.750 | 0.744 | 0.613 | 0.577 | **0.624** | 0.554 |
| Qwen3-1.7B (the step 6 LLM) | **0.769** | 0.706 | 0.750 | 0.740 | 0.593 | 0.581 | 0.579 | 0.570 |
| Graph only | – | 0.577 | 0.568 | 0.562 | – | 0.497 | 0.497 | 0.492 |

FAGCN (Bo et al., AAAI 2021; PyG `FAConv`, eps 0.3) is built for heterophily: each edge gets a learned weight in [−1, 1], so a neighbour can be subtracted as well as added. It has two propagation layers between an input and an output linear layer, the same hidden size (128) and about the same parameter count as GCN (197k). It also uses the same learning rate, weight decay, early stopping and seeds. Its own settings (eps 0.3, dropout 0.5) are fixed, not tuned, like every GNN here. In setting B (1:10), FAGCN is below MLP for every encoder (by 0.004–0.058), and its seeds vary more (std up to 0.02).

References for TwiBot-20 A: TF-IDF 0.702; metadata without "verified" 0.755 (not a valid model).

#### Heterophily and fraud baselines: FLAG-lite and LINKX (full test set, mean of 3 seeds)

Two more graph baselines, run with the same training, seeds and validation cut-off rule as step 4 (`my_build/step4_gnn.py`):
- **FLAG-lite**: FLAG (Yang et al., KDD 2025) without its LLM. Its skip-GNN, `GCN(X, A) + Linear(X)` (eq. 6), on a graph pruned by its semantic-similarity sampling: each user keeps up to its 10 most text-similar neighbours with cosine > 0 (`GRAPH=similar10`, 1-hop). FLAG's own ablation (Table 5) credits these two parts with most of its gain. The pruning drops about 25% of TwiBot-20's edges and 48% of Instagram's.
- **LINKX** (Lim et al., NeurIPS 2021; PyG `LINKX`): a non-homophilous baseline that never averages neighbours. One MLP reads each user's adjacency row, one its features, and a third combines them.

| Encoder + model | TwiBot-20 A | TwiBot-20 B | Instagram A | Instagram B |
|---|---|---|---|---|
| Qwen2.5 + MLP (no edges) | **0.766** | 0.613 | 0.607 | **0.575** |
| Qwen2.5 + GCN | 0.707 | 0.577 | **0.609** | 0.549 |
| Qwen2.5 + skip-GCN | 0.752 | 0.612 | 0.604 | 0.558 |
| Qwen2.5 + GCN, similar-10 graph | 0.719 | 0.585 | 0.607 | 0.562 |
| Qwen2.5 + FLAG-lite (skip-GCN, similar-10) | 0.754 | **0.615** | 0.602 | 0.568 |
| Qwen2.5 + LINKX | 0.752 | 0.607 | 0.598 | 0.562 |
| multilingual-E5 + MLP | **0.747** | 0.606 | 0.605 | 0.561 |
| multilingual-E5 + FLAG-lite | 0.731 | **0.635** | **0.613** | **0.565** |
| multilingual-E5 + LINKX | 0.730 | 0.615 | 0.602 | 0.557 |
| Graph only: GCN / skip-GCN / LINKX | 0.577 / 0.578 / **0.595** | 0.497 / 0.497 / 0.488 | 0.548 / 0.547 / 0.532 | 0.500 / 0.498 / 0.521 |

- **The skip connection does most of the work.** It lifts GCN from 0.707 to 0.752 on TwiBot-20 A, the same "keep your own features" effect as SAGE and FAGCN. Similarity pruning adds about 0.01 to GCN.
- **On TwiBot-20 A neither FLAG-lite (0.754) nor LINKX (0.752) beats MLP (0.766)**, matching every other GNN here. Setting B's best (multilingual-E5 + FLAG-lite 0.635 ± 0.018 vs MLP 0.606) is inside B's ±0.05 noise.
- **LINKX graph-only (0.595) beats GCN graph-only (0.577) in A**: reading who links to whom helps more than averaging. It is still far below the untrained in/out link-direction score (AUROC 0.713 vs LINKX's 0.633), because our adjacency is undirected.

#### Instagram (secondary dataset, steps 1–5): full test set; LR one fit, MLP and GNNs mean of 3 seeds

On Instagram, class 1 is **commercial accounts** (business, brand or media) against personal ones: a stand-in for a minority class, not fraud. FLAG uses the same Instagram graph (from GLBench, Li et al. 2024) as a fraud proxy at 1:10, our setting B. Its reported macro-F1 there is 0.473–0.483 for text-embedding GNNs and at most 0.507 for FLAG itself (Table 4), near the 0.476 an all-majority predictor scores; our Qwen2.5 + LR reaches 0.578 in B. Splits, subsample and encoder differ (FLAG uses Sentence-BERT), so this is not a head-to-head comparison. Text alone is much weaker here than on TwiBot-20 (AUROC about 0.63 against 0.84).

| Encoder | A: LR | A: MLP | A: GCN | A: SAGE | A: FAGCN | B: LR | B: MLP | B: GCN | B: SAGE | B: FAGCN |
|---|---|---|---|---|---|---|---|---|---|---|
| MiniLM | 0.588 | 0.585 | 0.607 | 0.588 | 0.596 | 0.546 | 0.536 | 0.552 | 0.543 | 0.566 |
| multilingual-E5 | **0.612** | 0.605 | 0.606 | 0.602 | **0.613** | 0.563 | 0.561 | 0.555 | 0.572 | 0.560 |
| Qwen2.5-1.5B | 0.608 | 0.607 | 0.609 | 0.604 | 0.606 | **0.578** | 0.575 | 0.549 | 0.562 | 0.534 |
| Graph only | – | – | 0.548 | 0.544 | 0.522 | – | – | 0.500 | 0.498 | 0.494 |

References: TF-IDF (balanced) 0.578 (A), 0.552 (B). Every model sits within about 0.03 of LR. Unlike TwiBot-20, GCN is level with or above MLP for all three encoders in setting A (macro-F1 +0.001 to +0.022; AUROC 0.639–0.642 vs 0.609–0.635), and graph-only reaches 0.548. In B no graph model is consistently better or worse.

#### LLM reasoner vs trained models (step 5 test sample, 1 seed)

**Sample sizes:** 400 test users for TwiBot-20 A, 300 for Instagram A and 600 for Instagram B, stratified, the same users for every row in a column. For TwiBot-20 B the "sample" is the whole test set (597). The trained rows here are one run (seed 0), not the mean of 3 seeds used elsewhere.

Two different LLMs appear here. **Step 5 uses Qwen2.5-1.5B-Instruct** (zero-shot only). **Step 6 uses Qwen3-1.7B**, the same family as the DGP paper's Qwen3-8B, both zero-shot and fine-tuned with LoRA. The trained baselines use Qwen2.5-1.5B *embeddings* (step 3) with LR or GCN on top. Every row is scored on the same step 5 test sample.

| Step | Model | TwiBot-20 A | TwiBot-20 B | Instagram A | Instagram B |
|---|---|---|---|---|---|
| 3–4 | Qwen2.5-1.5B embeddings + LR (trained) | **0.749** | **0.633** | 0.607 | 0.546 |
| 3–4 | Qwen2.5-1.5B embeddings + GCN (trained) | 0.698 | 0.573 | **0.623** | **0.562** |
| 5 | Qwen2.5-1.5B-Instruct zero-shot: target only | 0.584 | 0.541 | 0.553 | 0.522 |
| 5 | Qwen2.5-1.5B-Instruct zero-shot: raw neighbours | 0.591 | 0.547 | 0.552 | 0.499 |
| 5 | Qwen2.5-1.5B-Instruct zero-shot: DGP summaries | 0.550 | 0.541 | 0.537 | 0.538 |
| 6 | Qwen3-1.7B zero-shot: target only | 0.590 | 0.503 | not in scope | not in scope |
| 6 | Qwen3-1.7B zero-shot: DGP summaries | 0.562 | 0.527 | not in scope | not in scope |
| 6 | Qwen3-1.7B LoRA: target only | 0.694 | 0.618 | not in scope | not in scope |
| 6 | Qwen3-1.7B LoRA: DGP summaries | 0.726 | 0.628 | not in scope | not in scope |

**TwiBot-20 B, step 6 `dgp`** (597 test users, 54 bots; the step 5 "sample" is the whole test set; 1 seed): LoRA + DGP scores macro-F1 0.628, AUROC 0.723, PR-AUC 0.296, against Qwen2.5 + LR 0.634 / 0.799 / 0.282 and MLP 0.613 / 0.788 / 0.269 (mean of 3 seeds; the seed-0 run in the step 5 table is 0.633). Paired bootstrap against that 3-seed MLP on all 4,000 training users (`B/bootstrap_ci.json`): macro-F1 +0.015 [−0.051, +0.081], AUROC −0.064 [−0.124, −0.008], PR-AUC +0.027 [−0.080, +0.143]. So in B LoRA is level on macro-F1 and PR-AUC (CIs about ±0.07–0.11 wide) and significantly lower on AUROC. With 54 test bots, B's apparent PR-AUC lead is noise.

**TwiBot-20 B, step 6 `target`**: LoRA on the user's own text scores 0.618 / AUROC 0.683 / PR-AUC 0.188 (zero-shot 0.503), so `dgp` − `target` is +0.010 macro-F1, +0.040 AUROC, +0.108 PR-AUC (paired bootstrap queued). Unlike A, the link-type cue does not account for it: target LoRA + 3 link flags (LR fitted on validation) scores only 0.574 (AUROC 0.687), −0.054 [−0.102, −0.006] below `dgp`, and `dgp` ranks better than `target` within most flag groups (no follow link: AUROC 0.702 vs 0.611; no friend link: 0.718 vs 0.681; has follow link is the exception, 0.770 vs 0.814). So in B the summaries seem to add something beyond "which links exist", but B's edges are themselves label-dependent and its CIs are wide. Validation loss and AUROC agreed in B (best check at epoch 2.0: loss 0.335, AUROC 0.810), unlike A.

**Sample noise.** On 400 users, macro-F1 moves by about ±0.04 from test-set noise alone, and the sample shifts models in different directions against the full test set: LoRA + DGP 0.711 → 0.726, LoRA target 0.704 → 0.694, LR 0.756 → 0.749. So on the sample dgp − target is +0.032; on the full test set it is +0.007. Use the full-test numbers (training-size and bootstrap sections) for any LoRA claim.

**Validation samples.** LLM scoring is slow, so the zero-shot rows (steps 5 and 6) pick their cut-off on a stratified validation *sample*: 200 users for TwiBot-20 A, 400 for B, 150 for Instagram A and 300 for Instagram B. The trained models pick theirs on the full validation set. The step 6 LoRA runs use that same sample for early stopping, then pick their cut-off on the full validation set (2,365 users for TwiBot-20 A).

The two "zero-shot: DGP summaries" rows (0.550 and 0.562) differ in the LLM and in how the summaries were made. Step 5 used its own summaries; step 6 used the paper's diffusion trimming and 10-token summaries. So they are not a like-for-like comparison of the two LLMs.

LoRA dgp is one seed (42). On the full TwiBot-20 A test set (1,183 users) it scores 0.711 macro-F1, AUROC 0.794. Every model picks its own cut-off by the same rule: the one that maximises macro-F1 on validation, applied unchanged to test. On the 400-user step 5 sample its AUROC (0.828) is the highest, but that does not hold on the full test set: AUROC 0.794, against 0.822 for Qwen2.5 + LR and 0.840 for MLP. So it is behind on ranking too, not just on the cut-off. Even the best cut-off chosen on test itself (not a valid model) gives only 0.726.

#### LoRA checkpoint rule: both adapters (full test set, 1 seed)

`my_build/step6_rescore.py` scores each run's last-check adapter and applies the text-only models' rule, written into the script before any last-adapter test score existed (commit `fc45a63`, 3 Nov 15:24): keep the adapter with the higher full-validation macro-F1. Only two adapters per run were saved and compared.

| Run | Lowest val loss: epoch, test macro-F1 / AUROC | Last check: epoch, test macro-F1 / AUROC | Full-val macro-F1 (lowest loss / last) | Kept by the rule |
|---|---|---|---|---|
| A `dgp` | 1: 0.711 / 0.794 | 2: **0.727** / 0.815 | 0.718 / 0.743 | last |
| A `target` | 1: 0.704 / 0.782 | 2: **0.728** / 0.812 | 0.718 / 0.737 | last |
| B `dgp` | 2: **0.628** / 0.723 | 3: 0.502 / 0.546 (collapsed) | 0.648 / 0.538 | lowest loss |
| B `target` | 1: 0.618 / 0.683 | 2: **0.644** / 0.739 | 0.638 / 0.644 | last |

**Under this rule the DGP summaries add nothing:** `dgp` − `target` is −0.001 in A (0.727 vs 0.728) and −0.016 in B (0.628 vs 0.644). LoRA on 1,300 users is then level with text-only models on the same 1,300 users (MLP 0.726 on the same draw). **The tables and findings below still use the lowest-validation-loss adapters** until the team decides which rule is primary; CIs for the rule's adapters (`LORA_ADAPTER=chosen` in `bootstrap_ci.py` and `link_direction.py`) are running.

#### Where GCN loses: connected vs isolated test users (TwiBot-20 A, full test set, mean of 3 seeds)

| Qwen2.5-1.5B + | Connected (1,096 users) | Isolated (87 users) |
|---|---|---|
| MLP | **0.767** | 0.743 |
| GCN | 0.701 | 0.777 |
| SAGE | 0.752 | 0.734 |
| FAGCN | 0.742 | 0.770 |

GCN's whole loss is on connected users, where neighbours are averaged in. The isolated column is only 87 users (47 bots) and is not reliable: isolated users have no neighbours in either graph, yet GCN scores 0.777 on them when trained on the real edges and 0.720–0.758 (mean 0.738) when trained on five shuffled versions. The difference comes only from weights learned on a different graph.

#### How much label signal do the edges carry? (all graphs)

Plain edge homophily is misleading when classes are unbalanced, so this table also gives two measures from Platonov et al. ([arXiv 2209.06177](https://arxiv.org/abs/2209.06177)) that correct for chance. **Adjusted homophily** is 0 when edges join same-label users no more often than chance. **Label informativeness** (0 to 1) says how much a neighbour's label tells you about yours. Both use all labels, test included: they describe the data and are never a model input. From `my_build/graph_stats.py` (`results/graph_stats.json`).

| Graph | Edges | Edge homophily | Chance level | Adjusted homophily | Label informativeness |
|---|---|---|---|---|---|
| TwiBot-20 A, all edges | 16,349 | 0.532 | 0.501 | 0.062 | 0.003 |
| TwiBot-20 A, friend only | 11,819 | 0.508 | 0.515 | −0.015 | 0.000 |
| TwiBot-20 A, follow only | 4,922 | 0.591 | 0.554 | 0.081 | 0.005 |
| TwiBot-20 A, + bridge edges | 46,362 | 0.588 | 0.501 | 0.175 | 0.022 |
| TwiBot-20 B, all edges | 5,274 | 0.879 | 0.878 | 0.014 | 0.000 |
| Instagram A, all edges | 72,005 | 0.586 | 0.551 | 0.078 | 0.005 |
| Instagram B, all edges | 38,193 | 0.861 | 0.856 | 0.033 | 0.002 |

Every graph sits near zero once chance is removed. Setting B's high edge homophily (0.86–0.88) only reflects the 1:10 ratio: almost every edge touches the large class. On TwiBot-20, B's near-zero value is also caused by how B is built: it keeps 8% of bots and only edges whose two ends are both kept, so bot–bot edge rows fall from 8,162 (A) to 54 (B). A bot's neighbours are then 7.8% bots, below B's 9.1% base rate, and isolation becomes tied to the label (41.8% of B's bots are isolated, against 27.6% of its humans; 7.3% vs 8.5% in A). Friend links on TwiBot-20 join same-label users no more often than chance (adjusted −0.015); the label signal in TwiBot-20's links sits elsewhere, in who lists whom (see "who lists whom"). The bridge edges raise adjusted homophily to 0.175, yet GCN still gets worse on them (robustness check below).

#### Per-class homophily: is there "camouflage"? (FLAG's Figure 5 test)

FLAG (KDD 2025, Fig. 5) argues its 1:10 fraud datasets show *neighbourhood camouflage*: minority nodes have low per-node homophily, majority nodes high. We ran the same histogram on our four graphs, adding the chance level (each class's share of all edge ends, which is what random edges would give). `my_build/class_homophily.py`, figure `my_build/figures/class_homophily.png`.

| Graph | Minority: mean homophily | Minority: chance | Majority: mean | Majority: chance |
|---|---|---|---|---|
| TwiBot-20 A | 0.543 | 0.484 | 0.581 | 0.516 |
| TwiBot-20 B (1:10) | 0.091 | 0.065 | 0.935 | 0.935 |
| Instagram A | 0.389 | 0.341 | 0.692 | 0.659 |
| Instagram B (1:10) | 0.099 | 0.078 | 0.921 | 0.922 |

Every class sits a few points above chance. The low-minority / high-majority pattern FLAG reads as camouflage appears in both 1:10 settings, but it is almost exactly what chance predicts once one class is 10 times the other; it comes from the imbalance, not from the minority hiding among the majority. On TwiBot-20 A there is no camouflage at all (bots 0.54 vs humans 0.58).

![Per-class homophily](my_build/figures/class_homophily.png)

#### TwiBot-20 graph: dropped support users

Our TwiBot-20 graph keeps only the 11,826 labelled users. The benchmark also ships 217,754 unlabelled support users, and its own GCN/GAT code keeps them as neighbours. `my_build/playground_bridges.py` counts what we lost (`results/twibot20/support_bridges.json`).

- 8,082 support users (3.7%) are **bridges**: linked to 2 or more labelled users. 171,896 (79%) link to exactly one, so they connect nothing. The largest hub has only 25 labelled neighbours.
- 786 of our 923 isolated users sit next to a bridge (75 of the 87 isolated test users).

| Graph | Edges | Edge homophily |
|---|---|---|
| Random-edge baseline | – | 0.507 |
| Ours (labelled only) | 16,349 | 0.532 |
| Bridge edges only (labelled–support–labelled, collapsed) | 30,240 | 0.619 |
| Ours + bridge edges | 46,362 | 0.588 |

Homophily here uses all labels, test included: it describes the data and is never a model input. The published TwiBot-20 GCN/GAT scores (F1 0.809 / 0.853) are binary F1 with profile metadata, so they are not comparable with our text-only macro-F1.

#### Control: training-set size (TwiBot-20 A)

Step 6 fine-tunes on **1,300** training users (the DGP paper's size); the text-only models above train on all **8,278**. `my_build/control_train_size.py` retrains Qwen2.5-1.5B + LR / MLP on the exact 1,300 users step 6 used. Validation, cut-off and test sets are unchanged. MLP is the mean of 3 seeds.

| Model | Training users | Full test set: macro-F1 | Full test set: AUROC | Step 5 sample: macro-F1 |
|---|---|---|---|---|
| Qwen2.5-1.5B + MLP | 8,278 | 0.766 | 0.840 | 0.754 |
| Qwen2.5-1.5B + LR | 8,278 | 0.756 | 0.822 | 0.749 |
| Qwen2.5-1.5B + MLP | 1,300 | 0.726 | 0.796 | 0.723 |
| Qwen2.5-1.5B + LR | 1,300 | 0.725 | 0.786 | 0.712 |
| Qwen3-1.7B + MLP (same model as LoRA, used as an encoder) | 8,278 | **0.769** | **0.840** | 0.759 |
| Qwen3-1.7B + LR | 8,278 | 0.765 | 0.826 | 0.759 |
| Qwen3-1.7B + MLP | 1,300 | 0.724 | 0.792 | 0.729 |
| Qwen3-1.7B + LR | 1,300 | 0.717 | 0.795 | 0.713 |
| Qwen3-1.7B LoRA + DGP summaries (1 seed) | 1,300 | 0.711 | 0.794 | 0.726 |
| Qwen3-1.7B LoRA, target text only (1 seed) | 1,300 | 0.704 | 0.782 | 0.694 |

**Same model, two roles.** Qwen3-1.7B used as a frozen encoder with an MLP scores 0.724 on 1,300 users and 0.769 on all users; fine-tuned with LoRA on the same 1,300 users it scores 0.711 (DGP summaries) and 0.704 (target text). So the model version does not explain the gap: in setting A, Qwen2.5 and Qwen3 behave alike as encoders (in B, Qwen3 is lower: LR 0.584 vs 0.634). With the same 1,300 users, the text-only models lose about 0.03–0.04, and the gap to LoRA + DGP shrinks from 0.055 to 0.015 on the full test set. On the step 5 sample LoRA is level with them. The full-train rows reproduce steps 3 and 4 exactly, which checks the setup.

**Which 1,300 users?** The table uses the same draw as step 6. Over 6 random stratified 1,300-user draws (`control_train_size.py`, `DRAWS=6`), LR scores 0.730 ± 0.005 and MLP 0.733 ± 0.006 on the full test set (MLP: 3 seeds per draw). The step 6 draw is the lowest of the six (LR 0.725, MLP 0.726), so the typical gap between LoRA + DGP (0.711) and a text-only model on 1,300 users is about 0.02, not 0.014. LoRA itself was trained on one draw only.

**Learning curve** (Qwen2.5-1.5B embeddings, full test set, MLP mean of 3 seeds; `my_build/bootstrap_ci.py`):

| Training users | 325 | 650 | 1,300 | 2,600 | 5,200 | 8,278 |
|---|---|---|---|---|---|---|
| LR | 0.714 | 0.720 | 0.725 | 0.743 | 0.754 | 0.756 |
| MLP | 0.697 | 0.713 | 0.726 | 0.751 | 0.754 | 0.766 |

LoRA + DGP at 1,300 users (0.711) is below every text-only point from 650 users up, and level with LR trained on 325 users (0.714), a quarter of its data. LoRA target only (0.704) is below them all. Equal training data closes most of the gap, but not all of it.

#### Are the gaps real? Paired bootstrap 95% CIs (TwiBot-20 A, full test set)

Each pair is scored on the same 2,000 resamples of the 1,183 test users; seeded models count as the mean of their 3 seeds. A CI that excludes 0 means the gap is larger than test-set noise. LoRA is one seed, so its CI covers test-set noise only, not seed noise.

| Comparison (Qwen2.5-1.5B embeddings unless LoRA) | Difference | 95% CI | Excludes 0? |
|---|---|---|---|
| GCN − MLP (all train) | −0.059 | [−0.082, −0.037] | yes |
| FAGCN − MLP (all train) | −0.021 | [−0.039, −0.004] | yes |
| SAGE − MLP (all train) | −0.015 | [−0.031, +0.000] | no (borderline) |
| FAGCN − GCN (all train) | +0.038 | [+0.022, +0.055] | yes |
| MLP: 1,300 − all train | −0.040 | [−0.055, −0.023] | yes |
| LoRA + DGP − MLP (both 1,300) | −0.014 | [−0.041, +0.013] | no |
| LoRA + DGP − LR (both 1,300) | −0.014 | [−0.042, +0.016] | no |
| LoRA + DGP (1,300) − MLP (all train) | −0.054 | [−0.080, −0.027] | yes |
| LoRA target only − MLP (both 1,300) | −0.022 | [−0.047, +0.003] | no |
| LoRA + DGP − LoRA target only | +0.007 | [−0.015, +0.030] | no |

Same test on **Instagram A** (Qwen2.5-1.5B embeddings, 2,268 test users; `results/A/bootstrap_ci.json`): GCN − MLP +0.002 [−0.016, +0.020], SAGE − MLP −0.003 [−0.018, +0.012], FAGCN − MLP −0.002 [−0.017, +0.013]; MLP on 1,300 users − all users −0.035 [−0.051, −0.020]. So on Instagram the graph models are indistinguishable from MLP, while on TwiBot-20 A GCN is clearly below it (−0.059).

#### Control: the same GNNs on shuffled edges (full test set, mean of 3 seeds)

`GRAPH=shuffled` keeps every user's degree but rewires who links to whom at random (`my_build/step4_gnn.py`, `step4_gnn_shuffled.json`). If a GNN scores the same on shuffled edges, the real edges give it nothing.

| Model | TwiBot-20 A: real edges | A: shuffled (5 rewirings) | TwiBot-20 B: real edges | B: shuffled (5 rewirings) |
|---|---|---|---|---|
| Qwen2.5 + MLP (no edges) | 0.766 | 0.766 | 0.613 | 0.613 |
| Qwen2.5 + GCN | **0.707** | 0.670 ± 0.009 | 0.577 | 0.588 ± 0.022 |
| Qwen2.5 + SAGE | 0.750 | 0.751 ± 0.006 | 0.624 | 0.615 ± 0.014 |
| Qwen2.5 + FAGCN | **0.744** | 0.725 ± 0.002 | 0.554 | 0.575 ± 0.008 |
| E5-base + GCN | **0.700** | 0.661 ± 0.007 | **0.589** | 0.559 ± 0.004 |
| E5-base + SAGE | 0.740 | 0.731 ± 0.004 | 0.567 | 0.577 ± 0.017 |
| E5-base + FAGCN | 0.733 | 0.738 ± 0.007 | 0.542 | 0.568 ± 0.013 |
| Graph only (GCN) | **0.577** | 0.552 ± 0.012 | 0.497 | 0.497 ± 0.000 |

Each shuffled value is the mean ± std over 5 random rewirings (each itself the mean of 3 model seeds). Bold: real edges more than 2 rewiring std above shuffled.

- **In A, real edges beat random ones for GCN** by 0.037–0.039 (both encoders), and for graph-only by 0.025. FAGCN gains 0.019 with Qwen2.5 and nothing with E5-base; SAGE gains 0–0.009. GCN's gain is a large part of its gap to MLP (−0.059). Label signal cannot explain it (label informativeness 0.003); similar text along real edges can (next table).
- **In B, no consistent direction**: real edges help E5-base + GCN (+0.030), shuffled edges score higher for FAGCN (+0.021–0.026) and Qwen2.5 + GCN. B has almost no bot–bot edges (see the homophily note).

**Why real edges help GCN without carrying labels: linked users write alike.** Mean cosine similarity of centred Qwen2.5-1.5B embeddings (`my_build/edge_similarity.py`; centring removes the common direction that pushes every raw cosine of these vectors to about 0.9):

| TwiBot-20 A | All | Bot–bot | Bot–human | Human–human |
|---|---|---|---|---|
| Real edges | **0.263** | 0.274 | 0.235 | 0.299 |
| Shuffled edges (mean of 5 rewirings) | 0.053 | 0.066 | 0.044 | 0.058 |
| Random pairs | 0.034 | 0.062 | 0.021 | 0.022 |

Real edges join users with similar text whatever their labels, even across bot–human pairs (0.235). So GCN loses less on real edges because averaging similar users blurs a user's own vector less, not because the edges reveal labels. Setting B shows the same pattern (0.336 real vs 0.071 shuffled).

#### Who lists whom: a crawl artefact that carries label signal (TwiBot-20 A)

TwiBot-20 stored, for each labelled user, a sample of about 10 accounts they follow (`friend`) and 10 accounts following them (`follow`). So a link between two labelled users means one appears in the other's 10-sample; out-degree inside the labelled graph is capped at 10, and only 1.2% (friend) and 3.8% (follow) of links are mutual. Our pipelines symmetrise these links, and steps 4–5 also merge the two types. `my_build/link_direction.py` (`results/twibot20/A/link_direction.json`) looks at direction:

| Link, as stored | Share with ≥1: bots | humans | Test AUROC of the count |
|---|---|---|---|
| Outgoing friend (user lists someone) | 0.387 | 0.358 | 0.517 |
| Outgoing follow | 0.118 | 0.088 | 0.509 |
| **Incoming friend** (user is in someone's following sample) | 0.388 | **0.634** | 0.306 (0.694 reversed) |
| **Incoming follow** (user is in someone's follower sample) | **0.499** | 0.238 | 0.640 |

- **All the signal is in incoming links**, and it is strong: the untrained score in-follow minus in-friend reaches test AUROC 0.713, above graph-only GCN (0.61). Outgoing links carry almost none.
- **It tracks follower count**, a metadata field we exclude as a shortcut: Spearman ρ between in-friend count and `followers_count` is 0.59 (0.69 among humans), and −0.44 for in-follow.
- **Whether a user's own lists were recorded also depends on the label**: 95.8% of bots have a recorded follow list against 81.4% of humans (friend lists 95.0% vs 88.0%).

We read this as a property of how the dataset was collected (who appeared in whose sample, and how popular they are), not as relational signal a deployed detector would have. It fits Hays et al. (WWW 2023, §4.1): TwiBot-20 was collected "starting from well-known (verified) seed users" by breadth-first traversal, and its verified accounts were labelled human automatically, so a user's position in the crawl is tied to its label. It explains why graph-only models still score 0.577 although neighbour labels are uninformative.

**How much do these cues add to text?** LR on Qwen2.5-1.5B embeddings, full test set, paired bootstrap 95% CIs:

| Features | Macro-F1 | AUROC | Gain over text |
|---|---|---|---|
| Text | 0.756 | 0.822 | – |
| Text + 3 undirected link flags (has friend / follow / friend-of-friend) | 0.770 | 0.833 | +0.013 [−0.003, +0.029] |
| Text + 4 log in/out degrees | 0.776 | 0.845 | +0.019 [+0.001, +0.038] |
| 3 link flags only | 0.633 | 0.655 | |
| 4 log in/out degrees only | 0.660 | 0.717 | |

In setting B the three flags lift LR from 0.634 to 0.656 (+0.023 [+0.002, +0.049], `B/link_direction.json`), but B's edges themselves depend on every user's label through its subsampling, so that number is partly built from the labels.

**Does step 6 `dgp` gain only through this cue? In A, apparently yes.** The `dgp` prompt writes "none" for each missing link type, so it exposes the three undirected flags; `target` does not. Combining the target-only LoRA score with the three flags (an LR fitted on the validation set) scores 0.731 macro-F1 (AUROC 0.803), against 0.711 (0.794) for `dgp`: +0.020 [−0.002, +0.040]. Within groups of users with the same flag, `dgp` is better in some (no friend link: AUROC 0.803 vs 0.775; has follow link: 0.786 vs 0.770) and worse in others (has friend: 0.768 vs 0.777). So in A the summaries' own contribution is not established: a 4-feature LR on top of target LoRA matches `dgp`. In setting B the same check goes the other way (target + flags 0.574 vs `dgp` 0.628, see the setting B step 6 paragraph).

#### Robustness check: step 4 on the bridge graph (TwiBot-20 A, full test set)

Step 4 rerun with the bridge edges added (`GRAPH=bridges`, `results/twibot20/A/step4_gnn_bridges.json`): 92,724 edge rows instead of 32,698, and 137 isolated users instead of 923. **This is a robustness check only. Steps 5 and 6 and all tables above stay on the labelled-only graph.** "Isolated" still means isolated in the labelled-only graph, so the groups are the same users in both runs. Macro-F1, mean of 3 seeds:

| Encoder | MLP | GCN: original → bridges | SAGE: original → bridges | FAGCN: original → bridges |
|---|---|---|---|---|
| MiniLM | 0.726 | 0.686 → 0.651 (−0.035) | 0.721 → 0.725 (+0.005) | 0.722 → 0.716 (−0.007) |
| E5-base (English) | 0.760 | 0.700 → 0.671 (−0.029) | 0.740 → 0.746 (+0.006) | 0.733 → 0.744 (+0.010) |
| Twitter-RoBERTa | 0.749 | 0.692 → 0.670 (−0.021) | 0.743 → 0.741 (−0.002) | 0.722 → 0.730 (+0.007) |
| multilingual-E5 | 0.747 | 0.699 → 0.676 (−0.023) | 0.741 → 0.741 (0.000) | 0.726 → 0.735 (+0.008) |
| Qwen3-Embedding | 0.727 | 0.691 → 0.665 (−0.025) | 0.707 → 0.719 (+0.012) | 0.710 → 0.716 (+0.006) |
| Qwen2.5-1.5B | **0.766** | 0.707 → 0.677 (−0.029) | 0.750 → **0.761** (+0.010) | 0.744 → 0.743 (−0.001) |
| Graph only | – | 0.577 → 0.538 | 0.568 → 0.532 | 0.562 → 0.519 |

- MLP is identical in both runs (it ignores edges), which confirms only the graph changed.
- GCN drops on every encoder (std 0.002–0.010), most on users who were isolated before (Qwen2.5: 0.777 → 0.666; the 0.777 starting point is itself unstable, see the connected/isolated note). Average degree rises from about 2.8 to about 7.8, so GCN's averaging gives each user's own text less weight. This happens even though the bridge graph is the only one with clear label signal (adjusted homophily 0.175, label informativeness 0.022).
- SAGE, which keeps its own features separate, gains slightly: Qwen2.5 + SAGE comes within 0.005 of MLP.
- FAGCN holds steady or gains slightly (−0.007 to +0.010). The extra edges do not hurt it the way they hurt GCN.
- Graph-only drops by about 0.04. One hypothesis, not tested: in the labelled-only graph, degree partly reflects a user's position in the dataset's BFS crawl, and the bridge edges blur that.

### Limitations (so far)

- **One seed for step 6 so far, and that run was unseeded on the LoRA side.** The first `dgp` and `target` runs fixed the training users (SEED=42) and batch order, but not the LoRA start weights or dropout (no `torch.manual_seed`), so they cannot be reproduced exactly. Seeded runs (`LORA_SEED=1, 2`, same 1,300 users and summaries) are running on TwiBot-20 A; the unseeded run will be the third. Qwen3-1.7B instead of the paper's Qwen3-8B; the paper uses 5 seeds.
- **LoRA reads less text than the encoders.** Every model works from the same prepared text (bio + tweets, cut at 2,000 characters). The encoders read all of it; the LoRA prompt caps the user's own text at 480 tokens of whole lines, which cuts 81% of TwiBot-20 users short (13.2 of 16.6 lines kept on average, 800-user sample). This biases LoRA vs MLP against LoRA; dgp vs target is unaffected (same cap).
- **LoRA and the other models choose their checkpoint differently.** LoRA keeps the lowest validation loss on a validation sample (200 users in A, 400 in B); MLP and GNNs keep the best full-validation macro-F1. In both TwiBot-20 A runs and in B `target` the loss was lowest at epoch 1 while validation AUROC kept rising (A dgp 0.748 → 0.772, A target 0.765 → 0.781, B target 0.699 → 0.773). In B `dgp` loss and AUROC agreed (best at epoch 2.0), and the run ended on a collapsed check (epoch 3.0: AUROC 0.624). The last-check adapters (epoch 2 in A and B `target`, epoch 3 in B `dgp`) are being scored with the full-validation rule (`my_build/step6_rescore.py`).
- **GNN hyperparameters are fixed and shared, not tuned.**
- **The `dgp` prompt reveals which link types a user has** (its "none" lines), a label cue worth +0.013 to LR in A (CI includes 0) and +0.023 [+0.002, +0.049] in B. In A, target LoRA plus these flags matches `dgp`; in B it does not (see "who lists whom" and the setting B step 6 paragraph). `target` does not see it.
- **TwiBot-20's links reflect how it was collected.** Each user stores a sample of about 10 followers and 10 followings; incoming links track follower count (excluded metadata), and whether a user's own lists were recorded depends on the label (95.8% of bots vs 81.4% of humans for follow lists).
- **Step 5's neighbour ranking uses multilingual-E5 vectors with the two 0/1 "no bio" / "no tweets" flags still attached** (`step5_llm_reason.py:180`), so a shared flag can outweigh text similarity when a user has more than 8 neighbours. Step 6's diffusion drops the flags (`step6_dgp.py:441`).
- **Setting B is small and changes the graph.** Its test set has 54 bots; one random bot subsample (seed 42); 29% of users are isolated in B against 7.8% in A (37% of B's test bots). Its paired bootstrap CIs are about ±0.05: GCN − MLP −0.036 [−0.087, +0.012], SAGE − MLP +0.012 [−0.034, +0.058], FAGCN − MLP −0.058 [−0.112, −0.002]. Only the FAGCN gap is outside noise. B's MLP learning curve is flat and noisy (0.622 at 325 users, 0.613 at 4,000); LR rises from 0.583 to 0.634. B cannot separate most models. B also has almost no bot–bot edges (54 rows), so its "no edge signal" result reflects the subsampling, not rare fraud as such. `results/twibot20/B/bootstrap_ci.json`.
- **No duplicate-text leakage.** 1 of 1,183 TwiBot-20 A test users (12 of 2,268 on Instagram A) has exactly the same normalised text as a training user; dropping them changes Qwen2.5 + LR by at most 0.001 (`my_build/duplicate_text.py`).
- **TwiBot-20 graph = labelled users only** (see "dropped support users"; bridge check below).
- **Edge signal near zero**, so these results speak to near-random edges, not to strongly heterophilous graphs (see the adjusted-homophily table).

### Findings so far

**In one line:** on TwiBot-20 a strong text encoder does the work. Neighbour labels carry almost no signal, so no GNN and no neighbour-summary prompt is significantly better than a text-only model; the label signal that the links do carry comes from how the dataset was collected (who appears in whose follower sample) and tracks follower count. A fine-tuned LLM roughly matches text-only models trained on the same 1,300 users but stays below them when those use all 8,278.

1. **A good text encoder is where the gains are.** On TwiBot-20 A, Qwen3-1.7B + MLP (0.769) and Qwen2.5-1.5B + MLP (0.766) beat TF-IDF (0.702) by about 0.065. The neural encoders span 0.726–0.769, and no graph model improves on its encoder's MLP.
2. **Neighbour averaging (GCN) hurts; models that keep a user's own features lose less; none beats MLP on TwiBot-20.** GCN is 0.036–0.064 below MLP for every encoder in A (Qwen2.5: −0.059, 95% CI [−0.082, −0.037]). SAGE (own features kept separate) is within about 0.02 (−0.015, borderline). FAGCN is 0.003–0.030 below MLP and beats GCN by 0.02–0.04, but it beats GCN by as much on shuffled edges (Qwen2.5 0.725 vs 0.670), so its gain comes from mixing its input back in at every layer (eps), not from signed edge weights. All GNNs use the same fixed, untuned hyperparameters. Two more baselines built for this problem also stay below MLP on TwiBot-20 A: FLAG-lite (FLAG's skip-GNN on its similarity-pruned graph, 0.754) and LINKX (0.752). Even on the bridge graph, the only one with clear label signal (adjusted homophily 0.175), GCN gets worse by 0.02–0.04. On Instagram A, where text alone is weak (AUROC about 0.63), GCN is level with MLP (+0.002, CI [−0.016, +0.020]). This is consistent with averaging hurting when a user's own text is strong and being neutral when it is weak, though the two datasets also differ in language, label meaning and density.
3. **Neighbour labels carry almost no signal; neighbours write alike; who lists whom is a crawl artefact.** Adjusted homophily is −0.015 to 0.08 on every original graph and label informativeness at most 0.005. Real edges join users with similar text whatever their labels (centred cosine 0.26 vs 0.05 for shuffled edges), so on TwiBot-20 A real edges are less harmful than random ones (GCN +0.037 over 5 rewirings) but never better than no edges. The links do carry label signal in their direction: being in other users' follower samples vs their following samples gives an untrained AUROC of 0.713, and it tracks follower count. We read it as an artefact of how TwiBot-20 was collected. Our undirected pipelines discard it.
4. **The graph alone is weak:** 0.577 (TwiBot-20 A) and 0.497 (B). On the bridge graph it drops to 0.538.
5. **The zero-shot LLM reasoner (Qwen2.5-1.5B-Instruct, step 5) loses to trained classifiers on TwiBot-20 A and B and Instagram A.** On Instagram B it roughly ties the weak trained baselines on its 600-user sample (zero-shot DGP summaries 0.538 vs LR 0.546; zero-shot target AUROC 0.567 vs LR 0.524). Adding neighbours (raw or summaries) does not reliably help.
6. **Encoder type matters more than size.** The two decoder LLMs lead (Qwen3-1.7B 0.769, Qwen2.5-1.5B 0.766), then English E5-base (110M, 0.760), ahead of multilingual-E5 (278M, 0.747) and Qwen3-Embedding (0.6B, 0.727). On Instagram A the encoders are tied.
7. **Fine-tuning helps a lot; at equal data the fine-tuned LLM roughly matches text-only models.** Qwen3-1.7B goes from 0.562 zero-shot to 0.726 with LoRA on the step 5 sample (0.711 on the full test set; zero-shot was scored on the sample only). The same model as a frozen encoder + MLP scores 0.724 on the same 1,300 users and 0.769 on all 8,278. Against Qwen2.5 + MLP on 1,300 users the gap is −0.014 (95% CI [−0.041, +0.013]) on the same draw and about −0.02 over 6 draws. LoRA was never trained on more than 1,300 users, so whether more data would close the full-data gap is untested. One seed. **The DGP neighbour summaries add +0.007** over target-only LoRA (0.711 vs 0.704; CI [−0.015, +0.030]), and in A target LoRA plus three link flags matches `dgp` (0.731 vs 0.711, +0.020 [−0.002, +0.040]), so there the summaries' own contribution is not established. In setting B, `dgp` beats `target` on AUROC (0.723 vs 0.683) and target + flags does not catch up (0.574 macro-F1), but B has 54 test bots and label-dependent edges. Zero-shot they hurt (0.562 vs 0.590, step 5 sample).
8. **Setting B separates few models on macro-F1.** Its test set has 54 bots and its macro-F1 CIs are about ±0.05; only FAGCN − MLP excludes 0. On AUROC more gaps are clear: GCN − MLP −0.072 [−0.129, −0.018], FAGCN − MLP −0.141 [−0.196, −0.090], FAGCN − GCN −0.069, and LoRA `dgp` − MLP −0.064. Its subsampling also removes almost every bot–bot edge (54 rows, against 8,162 in A).

### Decision log

| Date | Decision | Why |
|---|---|---|
| 3 Nov | Measure the link-type cue before reading the setting B step 6 result | The `dgp` prompt shows which link types a user has; in B isolation is tied to the label |
| 3 Nov | Report adjusted homophily and label informativeness next to edge homophily | Plain homophily is inflated by class imbalance (0.88 in setting B, 0.014 adjusted); the standard fix from the heterophily literature |
| 3 Nov | Training-set size control: LR / MLP retrained on step 6's 1,300 users | LoRA trains on 1,300 users and the text-only models on 8,278, which confounds "encoder vs predictor" with data size |
| 3 Nov | Qwen3-1.7B added as a step 3 encoder (same checkpoint as step 6, mean-pooled like Qwen2.5) | Removes the model-version confound between the encoder rows (Qwen2.5) and the LoRA rows (Qwen3) |
| 3 Nov | TwiBot-20 = main dataset (steps 1–6); Instagram = secondary (steps 1–5) | TwiBot-20 has the benchmark split, typed friend/follow relations and real bot labels. Step 6 costs 3–6 h per run, so it goes to the main dataset |
| 3 Nov | Added FAGCN (PyG `FAConv`, eps 0.3) to step 4 | A GNN built for heterophily: signed edge weights can subtract a neighbour. Tests whether finding 2 is about GCN or about graphs |
| 3 Nov | Bridge graph used only as a step 4 robustness check; steps 5 and 6 stay on the labelled-only graph | Keeps all main tables on one graph; the step 4 check already answers whether the graph construction drove finding 2 |
| 3 Nov | Measured the dropped support users (bridges, homophily) before changing any model | Findings 2, 3 and 6 depend on the graph; the benchmark keeps support users as neighbours |
| 2 Nov | Step 6 summaries generated in batches of 4 on TwiBot-20 | At 16, the GPU memory cache grew to 16 GB and the Mac swapped |
| 2 Nov | Step 6 relations treated as undirected | Kept directed, only 38% of users had a friend link; undirected, 93% have a neighbour (matches steps 4–5) |
| 2 Nov | Step 6 LLM = Qwen3-1.7B, not step 5's Qwen2.5-1.5B-Instruct | Same family as the paper's Qwen3-8B; every LLM row names its model |
| 2 Nov | Step 6 = DGP replication with Qwen3-1.7B, 1,300 training users, B=10, K=2, M=8, r=16, lr 1e-4, one seed, no numeric summaries | Closest to the paper that fits a 16 GB Mac; the official code is not public yet |
| 2 Nov | Diffusion features X = multilingual-E5 embeddings | The paper says "raw node features"; our raw features are text |
| 1 Nov | TwiBot-20 graph = labelled users only; the 217,754 support users dropped | No labels and no text needed for them, so prep stays small. The effect on the graph was not checked then (see 3 Nov) |
| 1 Nov | Added TwiBot-20 (official split, bio + 20 tweets) | A second, larger heterophilous dataset with real text |
| 1 Nov | Excluded TwiBot-20 metadata | "Verified" alone gives macro-F1 0.803: a shortcut, not text or graph signal. Hays et al. (WWW 2023, §4.1, Table 2) find the same: a depth-1 decision tree on "verified" reaches accuracy 0.82 / F1 0.86 on TwiBot-20, 0.05 / 0.03 below the best published model, because collection started from verified seed users and "verified accounts were automatically labeled human" |
| 1 Nov | Encoder ladder: six encoders, everything else fixed | To separate encoder size and type from the graph's effect |
| 30 Oct | Threshold chosen on validation for every model | Scores are not calibrated; 0.5 is badly placed when class 1 is rare |
| 30 Oct | Settings A (natural) and B (1:10, FLAG-style) | Tests both the data as it is and rare fraud. FLAG (Yang et al., KDD 2025, §4.1.1) builds its text-rich fraud datasets the same way: the minority class "randomly selected so that the final ratio … is about 1:10" |
| 27 Oct | Dropped Elliptic, rebuilt on Instagram | Elliptic has no natural text for an LLM to read |

### Next steps

1. Running tonight (GPU): Qwen3-1.7B as an encoder (step 3, then LR / MLP / GNNs and the 1,300-user control), step 6 `dgp` and `target` on TwiBot-20 B, then scoring the last-check LoRA adapters.
2. Decide which LoRA checkpoint rule is primary. The full-validation macro-F1 rule was fixed in `step6_rescore.py` (commit `fc45a63`) before the last-check adapters were scored; both are reported (see "LoRA checkpoint rule").
3. FLAG (Yang et al., KDD 2025; no public code): running a FLAG-lite baseline from its two non-LLM components (semantic-similarity top-10 neighbour sampling, skip-GNN `GNN(X) + Linear(X)`), which its own ablation (Table 5) credits with most of the gain. FLAG also selects models by best validation macro-F1 (§4.1.2), the rule `step6_rescore.py` applies.
4. Optional, by Saturday night: more LoRA seeds on TwiBot-20 A (about 2.2 h per run, so 2 seeds × dgp and target fit one night; needs a separate `LORA_SEED` and the seed in the checkpoint name first), a semi-synthetic heterophily sweep (CPU), figures.
5. Low priority: the M ∈ {4, 8, 16} sweep. The limit of 8 binds only for friend-of-friend (mean 5.1 kept in A); 1-hop metapaths keep 0.8–1.8.

### Changelog

- **4 Nov 2026 (05:40):** Metadata shortcut check: only "verified" is a shortcut under the pre-set rule; text + the 5 allowed fields gains +0.04–0.05 macro-F1 (significant). `VARIANT=dgp_num` (DGP's numerical summaries) added and queued after the LoRA seeds.
- **4 Nov 2026 (05:00):** New Data preparation section (sources, text, edges, splits and settings, encoding, checks). Raw vs cleaned text check: cleaning changes TF-IDF + LR by −0.026 to +0.002, all CIs include 0. Noted the 2,000-character text cap (61.6% of TwiBot-20 users).
- **4 Nov 2026 (04:40):** `LORA_SEED` added to step 6; seeds 1 and 2 started on TwiBot-20 A. Duplicate-text check: 1 TwiBot-20 test user and 12 Instagram test users share a training user's text; removing them changes Qwen2.5 + LR by at most 0.001. Heterophily sweep started.
- **4 Nov 2026 (05:30):** FLAG-lite (similar-10 graph + skip-GCN) and LINKX on all four settings: neither beats MLP on TwiBot-20 A; the skip connection recovers most of GCN's loss.
- **4 Nov 2026 (04:40):** Cited Hays et al. (WWW 2023) for the metadata exclusion and the crawl artefact.
- **4 Nov 2026 (04:15):** FLAG (KDD 2025) read: setting B's 1:10 protocol and the validation-F1 model selection both match it; Instagram comparison with its Table 4; FLAG-lite baseline and per-class homophily histograms (FLAG Fig. 5) started.
- **4 Nov 2026 (03:30):** Rescoring done: the full-validation rule keeps the last-check adapter in A (both) and B `target`, and the lowest-loss one in B `dgp`. Under it `dgp` − `target` is −0.001 (A) and −0.016 (B). New section with both adapters; main tables unchanged pending a decision.
- **4 Nov 2026 (01:40):** After review 6 (QA): B MLP baseline quoted as the 3-seed mean the CIs use; checkpoint-rule wording covers B (B target like A; B dgp ended on a collapsed check); "target + flags matches dgp" limited to A; finding 8 adds B's AUROC CIs.
- **4 Nov 2026 (01:20):** Step 6 `target` on TwiBot-20 B: 0.618 / AUROC 0.683 / PR-AUC 0.188 (`dgp` 0.628 / 0.723 / 0.296). In B, target LoRA + link flags does not reach `dgp` (0.574), unlike A.
- **3 Nov 2026 (23:50):** `bootstrap_ci.py` now also gives AUROC and PR-AUC CIs and can use either LoRA adapter (`LORA_ADAPTER`); setting B: LoRA + DGP vs MLP AUROC −0.064 (significant), PR-AUC +0.027 (not).
- **3 Nov 2026 (23:20):** Step 6 `dgp` on TwiBot-20 B: LoRA 0.628 macro-F1 (LR 0.634), AUROC 0.723 (LR 0.799), PR-AUC 0.296 (LR 0.282); zero-shot 0.527.
- **3 Nov 2026 (22:00):** After review 4: TwiBot-20's link signal is in link direction (incoming links), tracks follower count and reflects how the dataset was collected (`link_direction.py`); the cue section, finding 3 and the one-line answer reworded. Target LoRA + 3 link flags matches `dgp` (0.731 vs 0.711). Number fixes (GCN/FAGCN ranges, LR 0.756, B +0.023, zero-shot on the sample only); softened two overclaims.
- **3 Nov 2026 (20:30):** Instagram A paired bootstrap: every GNN − MLP gap is within ±0.02 and includes 0.
- **3 Nov 2026 (20:15):** After review 3: Findings rewritten (one-line answer first; finding 2 credits FAGCN's gain to its input residual, shown by shuffled edges; Instagram contrast; finding 6 by encoder type; finding 7 with the same-model comparison; new finding 8 on setting B). Qwen3-1.7B encoder rows (ladder, training-size control). Link-type cue table (`isolation_cue.py`). Instagram label described; step 5 sample noise noted; stale numbers fixed.
- **3 Nov 2026 (18:30):** Shuffled-edge control over 5 rewirings (A: GCN +0.037–0.039 on real edges; B: no consistent direction). 6 training draws of 1,300 users: MLP 0.733 ± 0.006; the step 6 draw is the lowest, so LoRA's typical gap at equal data is about 0.02.
- **3 Nov 2026 (18:00):** After review 2: corrected the learning-curve, shuffled-edge and setting B sentences against their numbers; isolated column marked unreliable; setting B's bot–bot edge loss documented; new edge text-similarity table (real edges join similar-text users regardless of label). Fixed epoch labels in `step6_rescore.py`. Queued 4 more rewirings, 5 more training draws, and the setting B control and bootstrap.
- **3 Nov 2026 (16:05):** Shuffled-edge control (A: real edges slightly better than random for GCN/FAGCN; B: no difference; SAGE indifferent). Setting B paired bootstrap: CIs about ±0.05, only FAGCN − MLP excludes 0.
- **3 Nov 2026 (16:00):** After an independent review: connected/isolated table; Limitations section; fixed ranges in findings 2 and the B FAGCN note; setting B SAGE claim checked against AUROC/PR-AUC; finding 1 reworded. Queued step 6 rescoring of the epoch-2 adapters and a shuffled-edge control.
- **3 Nov 2026 (15:45):** Step 6 `target` on TwiBot-20 A: LoRA 0.704 (full test), 0.694 (step 5 sample); zero-shot 0.590. DGP summaries add +0.007 on the full test set.
- **3 Nov 2026 (15:00):** Adjusted homophily and label informativeness for every graph (near 0 everywhere). Learning curve and paired bootstrap CIs on TwiBot-20 A: the LoRA vs MLP gap at equal training size is not significant.
- **3 Nov 2026 (14:30):** FAGCN on TwiBot-20 B; Instagram graph-only GCN/SAGE rows. Rerunning Instagram GCN/SAGE reproduced every earlier number exactly.
- **3 Nov 2026 (14:15):** FAGCN results on TwiBot-20 A (both graphs) and Instagram A and B. Instagram step 4 now has MLP rows and the full MLP / GCN / SAGE / FAGCN table. Corrected the claim that DGP ranks best: on the full test set its AUROC is 0.794, against 0.822 for LR and 0.840 for MLP. Added a note on validation samples.
- **3 Nov 2026 (14:00):** TwiBot-20 made the main dataset, Instagram the secondary one (steps 1–5); every LLM row now names its model. FAGCN added to step 4.
- **3 Nov 2026 (afternoon):** Step 4 robustness check on the bridge graph: GCN drops 0.02–0.04 on every encoder, SAGE gains up to 0.012. Steps 5 and 6 stay on the labelled-only graph. Started step 6 `VARIANT=target` on TwiBot-20 A.
- **3 Nov 2026:** TwiBot-20 A step 6 `dgp` finished (LoRA 0.726 on the step 5 sample, 0.711 on the full test set). Counted the dropped support users: 8,082 bridges; bridge edges are more homophilous (0.619) than our graph (0.532).
- **2 Nov 2026:** TwiBot-20 step 5 finished (A and B). Step 6 (DGP replication) written and started on TwiBot-20 A. Repository created.

---

# Part B: TwiBot-22 (status Fri 6 Nov 2026, ~00:45)

Data preparation, checks and every rule were fixed **before** any model was scored (the lesson from TwiBot-20). Models were then run on the **working** test users only; the sealed set and the 986 experts are untouched until the final run.

## B1. Data

| Item | Value |
|---|---|
| Users | 1,000,000, all labelled; 140k bots (14.0%) |
| Labels | Weak (Snorkel over labelling functions incl. `verified`, spam keywords and GCN/GAT/R-GCN models). Experts: 986 users; dataset "bot" labels match the experts for 56 of 101 |
| Relations used (7) | following 2.61M, followers 1.11M, mention 1.46M, retweet 0.65M, reply 0.36M, quote 0.17M, like 0.34M undirected edges. Co-list (61M pairs) reported only |
| Tweets | 9 files, 104.4 GB; 88,217,457 tweets (= the `post` edges in `edge.csv`), 0 decode errors, dated 21 Mar 2006 – 15 Mar 2022 |
| Text per user | bio + the 20 most recent tweets (17.8M tweets for 933,872 users); retweets kept; links and @mentions raw; no total length cap; each piece ≤ 128 tokens (text rules fixed 4 Nov ~22:45, as TwiBot-20 except the cap) |
| Licence | CC BY-NC-ND 4.0: no data in the repository |

## B2. Splits, sealed set and sample

| Part | Users | Bots |
|---|---|---|
| train | 699,309 | 14.0% |
| val | 199,803 | 14.0% |
| test_work (used while building) | 49,951 | 14.0% |
| test_sealed (scored once, at the end; includes all 986 experts) | 50,937 | 13.9% |

- **Study sample** (headline with tweets): 100,000 users drawn stratified by role and label, plus the 986 experts = 100,986 users (70k train, 20k val, 5k test_work, 5,986 sealed).
- **Hub users** (any outgoing edge): 10,039. Their tweets are embedded as neighbour providers only.
- **Users to embed with tweets:** 110,057.
- **Official split:** reported only for comparison; it is not random (train 7.8% bots, test 29.4%; all verified users in train).

## B3. Checks (all before any model)

| Check | Result |
|---|---|
| Heterophily | Adjusted homophily 0.011–0.038 on every relation; label informativeness ≤ 0.0034. A val bot's train neighbours are 11.6% bots (chance 14.0%) |
| Graph shape | Star-shaped: only 10,039 users (1%) have any outgoing edge. No out-degree cap (max 907); reciprocity 0.3–1% |
| Crawl artefact | Humans receive edges 1.8–4× more often (in-degree AUROC 0.36–0.49 for "bot"); users with no link are 20.2% bots vs 12.1% with a link. Covered by a links-only model and with/without-link-count runs |
| Shortcuts (fixed rule: ≥200 train users at ≥99% one class) | Only `verified` (0 of 95,398 verified users are bots). 0 of the 814 feature columns fail (rechecked on the multilingual-E5 table). Tiny pure-human tails (≤1.2% of users, mostly famous accounts) reported |
| Strong allowed cues | empty bio 40.6% of bots vs 7.8% of humans; no tweets 17.6% vs 4.8%; default picture 13.0% vs 1.7%; follower count alone AUROC 0.733 |
| Tweet text | No bot-pure word among 43,469; 8 tiny human-pure words (topic communities) and 3 newsroom apps (SocialFlow, SocialNewsDesk, Echobox), reported. 0.56% of val users share an exact tweet with a train user |
| Row order | Every per-user file in `label.csv` order; the 124,231 zero embedding rows equal the 124,231 empty bios exactly |
| Guard | `verified` and the labels are outside every model input (`eval_only/`); scripts stop if they appear; sealed users refused before the final run |

## B4. Decisions (all fixed before the scores they govern; reasons in Part D)

1. **Headline (option B):** multilingual-E5 on bio + 20 tweets, on the 100k sample, neighbours from the full graph. Full-graph 1M models use bio only.
2. **Profile fields kept** (all except `verified`). Cheap models run with and without profile; GNNs and the LLM with profile.
3. **All 7 relations.**
4. **No artificial 1:10 setting.** PR-AUC also reported with the test set reweighted to 5% and 1% bots.
5. **Metrics and selection** (`twibot22_metrics.py`): macro-F1, AUROC, PR-AUC, paired bootstrap CIs; cut-off and model choice on validation only; 3 seeds; slices: all, non-verified, with bio, with tweets.
6. **LLM (H7):** Qwen3-1.7B, 1,300 training users, TwiBot-20's LoRA settings; adapter chosen by validation macro-F1 on 1,000 validation users (2,000 measured at 23.6 min per check, over the 10-min limit).
7. **Experts (5 Nov 13:22):** used for training **and** clean evaluation by 5-fold rotation (LoRA `target`: 1 more epoch per fold; trees: 50 more boosting rounds per fold). Blind until the final run. Replaces the 4 Nov "light recalibration".
8. **DGP scope (5 Nov 13:47):** TwiBot-20 scope: 1,300 train, the same 1,000 validation users, a 1,500-user stratified test subset of the working test users (seed 42, committed before scoring). dgp vs target compared on that subset. Reason: summaries take ~1 s each (5.6 h instead of 8.6 h).
9. **GNNs:** all 5 with 1 seed, then the 2 best by validation macro-F1 with 3 seeds; tables mark seed counts. Run only while no LoRA trains (they pause automatically otherwise).
10. **GPU never idle overnight:** queues are planned to run until ~13:00 the next day.

## B5. Results so far (working test users, weak labels; sceptic-checked)

All numbers macro-F1 / AUROC / PR-AUC, mean of 3 seeds unless noted. Sample = the 5,000 sample working-test users (bio + 20 tweets, the same users as the LLM); 1M = 49,951 working-test users (bio only). Trees are HistGradientBoosting.

| Model | Sample | 1M |
|---|---|---|
| Zero-shot Qwen3-1.7B (no training; says "bot" for nearly everyone) | 0.505 / 0.461 / 0.139 | – |
| LoRA `target` (fine-tuned on text, 1,300 train, 1 seed) | 0.626 / 0.710 / 0.312 | – |
| TF-IDF + LR, 70k train / 1,300 train | 0.628 / 0.727 / 0.307 · 0.576 / 0.640 / 0.243 | – |
| MiniLM + MLP, 70k train | 0.607 / 0.684 / 0.291 | – |
| Links only (neighbour counts) | 0.563 / 0.672 / 0.214 | 0.558 / 0.675 / 0.214 |
| Empty-bio flag alone | 0.668 / 0.658 / 0.264 | 0.670 / 0.662 / 0.266 |
| Text only, tree (own text) | 0.618 / 0.712 / 0.320 | 0.680 / 0.772 / 0.362 |
| **Profile only** (15 fields) | **0.701 / 0.837 / 0.497** | 0.707 / 0.846 / 0.508 |
| Profile + text, tree | 0.713 / 0.848 / 0.518 | 0.720 / 0.860 / 0.525 |
| XGB-Graph, no profile | 0.631 / 0.763 / 0.361 | 0.694 / 0.816 / 0.447 |
| **XGB-Graph + profile** | 0.712 / 0.852 / 0.533 | **0.731 / 0.873 / 0.580** |
| GNNs (full 1M, profile + bio, 7 relations; FAGCN, RGCN 3 seeds, others seed 0) | – | FAGCN 0.690 / 0.829, RGCN 0.689 / 0.804, SAGE 0.684 / 0.810, LINKX 0.681 / 0.811, GCN 0.678 / 0.808; MLP control 0.679 / 0.799 |

**Findings with paired 95% CIs** (`ci_pairs.json`):
1. Profile beats every text model: profile − LoRA +0.073 [+0.054, +0.093] F1, +0.128 AUROC. With LoRA's same 1,300 training users, profile still gets 0.677 / 0.808.
2. Text adds a little on top of profile: +0.011 [−0.001, +0.025] F1 (sample), +0.012 [+0.008, +0.016] (1M).
3. Neighbours help without profile (+0.053 AUROC, sample; +0.044, 1M), barely with it (sample −0.001 F1; 1M +0.010 F1, +0.054 PR-AUC).
4. Tweets do not help a text-only model: bio + tweets − bio only, same users: −0.052 [−0.074, −0.030] F1. Mechanism checked by slices: equal on users with a bio; averaging blurs the empty-bio cue (40.6% of bots vs 7.8% of humans).
5. GNNs: all within ~0.01 F1 of the MLP control and below the trees on the same own features: message passing along heterophilous links adds little (as on TwiBot-20).
6. LoRA vs cheap text models: with the same 1,300 users LoRA beats TF-IDF by +0.05 F1; with 70k users TF-IDF ties it.

**Main caveat:** much of the profile signal is one cue (empty bio), on weak labels partly made by labelling rules. The expert rotation and D7 (gain on weak vs expert labels) test whether "profile wins" means "copies the labeller".

## B6. Graph questions Q1–Q3 and the TROPICAL check (checklist E8–E10; designed 19:20, before any number)

| Question | Result |
|---|---|
| **Q1. Who links to whom** (per relation, direction, class; train + val users) | Bots' neighbours are 4–17% bots against a 14.0% base rate; humans' ~93% humans. Adjusted homophily 0.010–0.024 on all 7 relations. Out-edges, P(neighbour bot \| bot): following 0.062, followers 0.165, mention 0.050, retweet 0.049, reply 0.055, quote 0.043, like 0.057. Same shape as FRAUDRE's fraud data (normal > 0.85, fraud < 0.2) |
| **Q2. Local heterogeneity** (each user's share of train-labelled neighbours with the same label; 1M, working test) | Val users: 53% of bots and 68% of humans have a train neighbour; mean local homophily bots 0.116, humans 0.925. **XGB-Graph − own tree (profile):** +0.057 [+0.045, +0.071] AUROC for users ≥ 0.75 (n 27,002, 1% bots); −0.016 [−0.022, −0.009] for users < 0.25 (n 4,197, 77% bots). **RGCN − MLP:** −0.052 [−0.074, −0.030] in the ≥ 0.75 bin. **FAGCN − MLP:** +0.017 to +0.039 in every bin. Matches Mao et al. (NeurIPS 2023) |
| **TROPICAL's motivation** (bot density around bots by hop) | 0.116 → 0.081 → 0.086 → 0.059 over hops 1–4: falls with distance (on Amazon it rises). Does not transfer |
| **Q3. Cross-relation inconsistency** (8 label-free features: follow vs interaction neighbourhoods; no shortcuts, single-feature val AUROC 0.53–0.62) | Add nothing. 1M, XGB-Graph + profile with − without: F1 −0.002 [−0.004, −0.000], AUROC +0.000 (0.729 / 0.872 / 0.579 vs 0.731 / 0.873 / 0.580). Sample: ~0.708–0.718 / 0.851–0.853, the same as without |

**GNN seeds:** FAGCN 0.690 ± 0.001 / 0.829 ± 0.001, RGCN 0.689 ± 0.000 / 0.804 ± 0.003 (3 seeds); the other GNNs and the MLP control seed 0.

**All metrics** (`all_metrics_table.txt`, 1M working test, all-human accuracy 0.860): XGB-Graph + profile accuracy 0.869, bot-F1 0.538, macro-F1 0.731, AUROC 0.872, PR-AUC 0.579 (0.366 reweighted to 5% bots, 0.146 to 1%). Accuracy is misleading here: calling everyone human scores 0.860.

## B7. The leaderboard conflict (checklist E11)

**The conflict.** The TwiBot-22 paper ranks graph models (BotRGCN, RGT, ...) first; we find graph ≈ no graph.

**What their official code does** (`reports/twibot22_repo_check.md`): BotRGCN, GCN and GAT take `verified` as an input (`preprocess_1.py:193`; no verified user is a bot), use the official split (3 contiguous blocks: train 7.8% / val 28.0% / test 29.4% bots, although the paper says "7:2:1 random"), use 2 relations, train a fixed 200 epochs with no validation selection, and compute AUC from 0/1 predictions. The leaderboard ranks by accuracy; on bot-F1 a random forest (Kantepe, 0.587) beats BotRGCN (0.575); leaderboard recall is below 50%. The weak labels were partly made by GCN / GAT / R-GCN.

**Reconciliation run** (our models under the official protocol; `REPLICATION=1`; our sealed users removed from every part):


Official split, our sealed users removed: test 94,889 users, **29.5% bots → calling everyone human scores 0.705 accuracy**. Our split: test 49,951, 14.0% bots → all-human 0.860.

| Model | `verified` in / out | Accuracy @0.5 | Macro-F1 | AUROC |
|---|---|---|---|---|
| **Official split** | | | | |
| Own-feature tree (profile + bio, no neighbours) | in / out | 0.763 / 0.763 | 0.700 / 0.695 | 0.765 / 0.764 |
| XGB-Graph + profile | in / out | 0.769 / 0.769 | 0.689 / 0.691 | 0.758 / 0.758 |
| MLP (no edges) | in / out | 0.627 / 0.644 | 0.683 / 0.674 | 0.745 / 0.734 |
| FAGCN | in / out | 0.603 / 0.625 | 0.667 / 0.678 | 0.731 / 0.745 |
| RGCN | in / out | 0.645 / 0.640 | 0.666 / 0.653 | 0.724 / 0.726 |
| **Our random split** | | | | |
| Own-feature tree (profile + bio, no neighbours) | in / out | 0.878 / – | 0.721 / 0.720 | 0.863 / 0.860 |
| XGB-Graph + profile | in / out | 0.885 / 0.869 | 0.730 / 0.731 | 0.874 / 0.873 |
| MLP (no edges) | in / out | 0.770 / 0.843 | 0.686 / 0.679 | 0.824 / 0.799 |
| FAGCN | in / out | 0.816 / 0.850 | 0.674 / 0.689 | 0.799 / 0.828 |
| RGCN | in / out | *running* / 0.836 | – / 0.689 | – / 0.800 |

Seed 0 each. All models use the 1M bio-only features (15 profile fields + multilingual-E5 bio). "Out" on our split = the main results.

**What it says**
1. **The split matters far more than `verified`.** AUROC drops by ~0.11 from our random split to the official one, for every model. `verified` moves the trees by ≤ 0.01; for the GNNs it goes both ways (MLP +0.025 AUROC on our split, FAGCN −0.03), within single-seed noise of a few hundredths.
2. **Graph models do not win under their protocol either.** On the official split the tree without neighbours is the best on macro-F1 and AUROC; XGB-Graph is 0.006 higher only on accuracy; FAGCN and RGCN do not beat the MLP without edges (RGCN 0.02 below it without `verified`).
3. **Caveat:** our GNNs use class weights (theirs do not), which trades accuracy for catching bots, so their accuracy at 0.5 is below all-human. Compare GNNs on macro-F1 / AUROC.

**Likely explanation of the leaderboard:** their ranking rests on accuracy on a skewed split, graph models given `verified`, fixed-epoch training without validation selection, and labels partly made by the same graph-model families. Under one common protocol, we do not reproduce a graph advantage.


## B8. Expert stage (checklist G7; weak → expert, 5-fold rotation, blind)

| Family | Status |
|---|---|
| LoRA `target` | ✅ done 17:14 (1 epoch per fold from the best adapter) |
| Trees: profile only, profile + text, XGB-Graph + profile (sample), XGB-Graph + profile (1M) | ✅ done 19:41 (+50 rounds per fold, warm start) |
| GNNs: MLP control, RGCN, FAGCN (1M) | 🔄 MLP ✅ 13 min, RGCN ✅ 112 min, FAGCN running (~01:00; pauses while a LoRA trains). 30 epochs at lr 1e-3, class-weighted, on the ~789 experts of the other 4 folds |
| DGP | ✗ not enough GPU time |

Read early on Thu 23:50 (B9), except FAGCN (still running): see B9. D7 (graph gain on weak vs expert labels on the sealed non-expert users) stays for the final run.

## B9. Expert results read early and D8 (Thu 23:50, builder's decision; recorded in Part D before reading)

Each expert is scored by a model that never saw that expert (5-fold rotation). 95% bootstrap CIs. FAGCN still running.

| Model | AUROC weak-only → after expert stage | Change [95% CI] | Macro-F1 after |
|---|---|---|---|
| LoRA `target` (text) | 0.822 → 0.724 | **−0.099** [−0.164, −0.038] | 0.735 |
| Profile only (tree) | 0.861 → 0.867 | +0.006 [−0.017, +0.030] | 0.736 |
| Profile + text (tree) | 0.879 → 0.899 | +0.020 [+0.000, +0.040] | 0.752 |
| **XGB-Graph + profile** (sample) | 0.883 → **0.910** | +0.027 [+0.009, +0.044] | **0.751** |
| XGB-Graph + profile (1M) | 0.875 → 0.887 | +0.012 [−0.002, +0.028] | 0.714 |
| MLP, no edges (1M) | 0.819 → 0.845 | +0.025 [+0.006, +0.046] | 0.728 |
| RGCN (1M) | 0.799 → 0.866 | **+0.067** [+0.034, +0.104] | 0.725 |
| *The dataset's own weak label as a prediction* | – | – | 0.745 |

- **The ranking holds on expert labels:** profile-based trees with neighbour features best; the GNN gains most from expert labels but stays below the trees; LoRA's ranking worsens (1 epoch on ~789 users disturbs the adapter; its F1 gain comes from the new cut-off).
- **90.5% agreement is not much:** on these 986 users, calling everyone human agrees 89.4%. On bots, the weak label and the experts agree for 56 of the 105 expert bots.

**D8: is "empty bio → bot" a real bot habit or a label-maker habit?** Lift = bot rate among empty-bio users ÷ bot rate among users with a bio.

| Users | Weak labels | Human labels | Weak − human |
|---|---|---|---|
| 986 experts (103 empty bio) | 8.7× [6.4, 12.3] | experts 7.2× [5.3, 10.3] | +1.5 [−1.5, +4.6] |
| 458 annotation-study users (44 bots, 62 empty bio) | 5.0× [3.3, 7.7] | annotators **1.6×** [0.7, 3.1] | **+3.4** [+1.2, +6.1] |

- **The 986 experts cannot settle it:** 800 of their labels trained the label-making models (TwiBot-22 paper §3.2), so agreement is built in. The annotation study (6 researchers, 3 per user) never trained them.
- **Reading:** under independent human labels the empty-bio cue is much weaker; the weak labels overstate it (difference excludes 0). Mostly a label-maker habit, possibly a small real effect (1.6× CI includes 1, sample small). Consistent with Hays et al. (WWW 2023), who warn that benchmark labels are artefacts of collection and labelling (studied TwiBot-20, not TwiBot-22); extending it to TwiBot-22 is, as far as we found, new.

## B10. Running and next (Fri 6 Nov, ~00:45)

- **GPU (unattended, pushes itself):** DGP path summaries → LoRA training + scoring (~04:00–05:00) → dgp vs target on the 1,500 subset → LoRA `target` seed 1 → `dgp` seed 1 → dgp vs target with 2 seeds (~13:00).
- **CPU:** last reconciliation run (RGCN, our split, `verified` in); FAGCN expert stage (pauses during LoRA).
- **Fri 9:00:** tutor meeting (`reports/meeting_brief_2026-11-06.md`).
- **Fri afternoon, final run** (`FINAL_RUN=1`, the sealed non-expert users, read once): headline table; D7; FAGCN expert result; dgp vs target.
- **Then:** comparison with TwiBot-20; figures Saturday; report Sunday 8 Nov.

---

# Part C: external feedback (5 Nov) and the TwiBot-20 to-do

External feedback (`artifacts/google_feedback.txt`) raised three TwiBot-20 points. Each was checked against our own files:

1. **Checkpoint confound.**
   - It is real: LoRA picked its checkpoint on a 200-user validation loss, while the MLP and GNNs picked on full-validation macro-F1.
   - The matching rule is "choose by full-validation macro-F1", not "always use the last adapter". It picks the last adapter in setting A, but the lowest-loss one for setting B's `dgp`.
   - Under it, dgp − target is about 0 (0.727 vs 0.728). Seed noise (0.02–0.03) is larger than any dgp − target gap.
2. **Prompt cap.**
   - It is real: LoRA reads 480 tokens, and 81% of users are cut; the encoders read up to 2,000 characters.
   - It weakens the "frozen encoder vs fine-tuning" claim but does not make it invalid. The training-size confound (1,300 vs 8,278) is controlled separately.
   - Fix planned: a matched-text control.
3. **Setting B noise.**
   - It is real: 54 test bots, macro-F1 CIs about ±0.05.
   - But noise cannot create "no difference". On AUROC the gaps are clear and favour text models (GCN − MLP −0.072, FAGCN − MLP −0.141, LoRA dgp − MLP −0.064).
   - Fix: lead setting B with AUROC and PR-AUC.

TwiBot-22 takes the lesson as checklist row G8: the LLM reads the same text as the encoders. Every TwiBot-20 item is in the appendix below.

---

# Part D: decisions and reasons (copy of `artifacts/04_decisions_and_reasons.md`)

## Decisions and reasons: Instagram, TwiBot-20, TwiBot-22

Written Wed 4 Nov 2026, evening, so that anyone picking up this project (a teammate, the supervisor, a new Claude session) can see **what was decided, when, and why**, without reading every chat. It covers all three datasets in the order they were used. Detailed numbers live in `REPORT.md` (Instagram, TwiBot-20) and `reports/twibot22_data_checklist.md` (TwiBot-22); this file is the reasoning.

**The project in one paragraph.** COMP8851 (Macquarie). Question: do LLM-based and graph-based methods help detect bots or fraud-like accounts when the graph is **heterophilous**, that is, when bad accounts mostly connect to good ones, so a user's neighbours do not share its label? The supervisor asked for heterophily to be measured, not assumed. We compare text-only models, graph models (GNNs and neighbour-feature trees) and LLMs (zero-shot reasoning and LoRA fine-tuning, including a replication of the DGP method) on the same users.

---

### 1. Instagram (27 Oct – 3 Nov): the first text graph

| Decision | Why |
|---|---|
| **Dropped Elliptic** (27 Oct), the team's original Bitcoin dataset | Elliptic has no natural text, so an LLM has nothing to read. The project is about LLMs on graphs |
| **Instagram from GLBench** (Li et al., 2024), used as prepared | A real text graph (bios) with follow edges, and the same version FLAG (KDD 2025) uses, so results can be set next to a published fraud-detection paper |
| **Class 1 = commercial account**, treated as "a stand-in for a minority class, not fraud" | Instagram has no fraud labels; saying "fraud" would overclaim |
| **Two settings: A (natural, 36% commercial) and B (1:10, FLAG-style)** | A tests the data as it is; B copies FLAG's protocol for a rare class (§4.1.1) |
| **Cut-off chosen on validation for every model** (30 Oct) | Scores are not calibrated, and 0.5 is a bad cut-off when the class is rare |
| **Became the secondary dataset** (3 Nov), steps 1–5 only | TwiBot-20 has real bot labels, a benchmark split and typed relations. The LLM fine-tuning step costs 3–6 GPU hours per run, so it went to the main dataset |

**What Instagram showed.** Text is weak there (AUROC about 0.63), and every model sits within about 0.03 macro-F1 of a logistic regression on the text embeddings. GCN is level with the text-only MLP (+0.002, CI includes 0). So on Instagram the graph neither helps nor hurts.

---

### 2. TwiBot-20 (1 – 4 Nov): the main dataset

#### Data decisions

| Date | Decision | Why |
|---|---|---|
| 1 Nov | **Added TwiBot-20** (official split; text = bio + 20 tweets, capped at 2,000 characters) | A real Twitter bot dataset with text and a benchmark split |
| 1 Nov | **Kept only the 11,826 labelled users**; dropped 217,754 unlabelled "support" users | They have no labels and no text in our use; keeps preparation small. Checked later (3 Nov): they bridge 8,082 users, so a robustness run on the bridge graph was added |
| 1 Nov | **Excluded all profile metadata** | `verified` alone gave macro-F1 0.803. Hays et al. (WWW 2023) explain why: the crawl started from verified accounts and "verified accounts were automatically labeled human". The other fields were excluded with it, without a check (our own mistake #1, see `reports/twibot20_lessons_for_twibot22.md`) |
| 4 Nov | **Shortcut rule written down and run**: only `verified` fails; the 5 other fields add +0.04–0.05 | Fixed the 1 Nov over-caution. Profile fields became a separate "text + profile" track because every other comparison was already text-only |
| 1 Nov | **Encoder ladder**: six text encoders, everything else fixed | Separates the effect of the encoder from the effect of the graph |
| 2 Nov | **DGP replication** with Qwen3-1.7B, 1,300 training users, LoRA r = 16 | The closest to the paper that fits a 16 GB Mac; the paper's code is not public |
| 3 Nov | **TwiBot-20 = main dataset, Instagram = secondary** | See section 1 |
| 3 Nov | **Report adjusted homophily and label informativeness**, not only edge homophily | Plain homophily is inflated by class imbalance (0.88 vs 0.014 adjusted in setting B) |
| 3 Nov | **Training-size control**: text models retrained on the LLM's 1,300 users | The LLM trained on 1,300 users and text models on 8,278; that confounded "LLM vs encoder" with data size |
| 3 Nov | **Added FAGCN, FLAG-lite, LINKX** | GNNs designed for heterophily; tests whether "graphs do not help" is about GCN or about the data |

#### What TwiBot-20 showed

1. **A good text encoder does the work**: Qwen2.5 + MLP 0.766, Qwen3 + MLP 0.769 macro-F1, against TF-IDF 0.702.
2. **No graph model beats the text-only MLP.** GCN is 0.04–0.06 below it; FAGCN, FLAG-lite and LINKX stay below too.
3. **Edges carry almost no label signal** (adjusted homophily −0.015 to 0.08). The signal links do carry is a **crawl artefact**: who appears in whose follower sample, which tracks follower count.
4. **The fine-tuned LLM roughly matches text models at equal data** (1,300 users), and stays below them when those use all 8,278.
5. **Verified users inflate every score**: all 326 verified test users are human. Without them, target LoRA drops from 0.704 to 0.641 macro-F1 (confirmed from raw data on 4 Nov, `check_verified_subset.py`).
6. **Trees do not beat the MLP** (GADBench's "trees win" does not hold here), and the crawl-artefact gain for trees (+0.025) disappears on non-verified users (−0.009). Tree baseline rerun 4 Nov after fixing a silent crash (torch and sklearn clashing over OpenMP).

#### Mistakes we made, and why TwiBot-22 is done differently

The 22-item review (`reports/twibot20_lessons_for_twibot22.md`) found, among others:
- rules (checkpoint choice) set **after** some test scores were seen;
- the first LLM runs had **no seed**;
- **verified users never removed from the test set**;
- the tree baseline **crashed silently**;
- results files that could be **overwritten**.

None of these changed the main conclusion, but they shifted numbers by 0.02–0.05. TwiBot-22's whole approach (checklist first, rules fixed before any score, sealed test set) comes from this list.

---

### 3. TwiBot-22 (4 Nov onwards): the second, larger test

**Why TwiBot-22.**
- It has 1M users and 14% bots, which is fraud-like and within the 1.3–21.8% range of the GADBench fraud benchmarks.
- Its text is weak: bio word counts give macro-F1 0.561 on users with a bio.
- So it tests whether the graph helps when the text cannot do the work.
- Facebook reports 3.8–5.2% abusive accounts (DEC paper), so 14% is not "real-world fraud". The report calls it a **fraud-like rare class**.

#### Fixed before any model score (times are git commit times, 4 Nov)

| Time | Decision | Why |
|---|---|---|
| 14:22 | **Our own split**: stratified 70/20/10, seed 42, all 986 expert users in test | The official split is not random: train has 7.8% bots, test 29.4%, and all 95k verified users sit in train |
| 14:25 → ~21:30 | **Relations: all 7** (following, followers, mention, retweet, reply, quote, like); co-list reported only | First 5 by coverage; changed to 7 under "don't miss information". Quote and like add links for only 8,856 users, at small cost. Co-list (61M pairs) is too dense for the laptop |
| afternoon | **Keep the whole 1M graph for cheap models; use a sample for expensive work** | Industry and benchmarks cut the computation, not the graph (`reports/graph_slicing_research.md`). Cutting a subgraph is what broke TwiBot-20's setting B: it removed 8,108 of 8,162 bot–bot edges |
| evening | **No artificial setting B** | The natural 14% is already rare. For rarer rates we instead report PR-AUC with the test set reweighted to 5% and 1% bots (evaluation only) |
| 16:29 | **Stopped the all-1M Qwen bio embedding** (26% done, ~9 h left) | It would have used the GPU for a weak, interim input (bio only), when the final text is bio + tweets |
| 20:38 → 20:58 | **Option B is the headline**: multilingual-E5 on bio + 20 most recent tweets, on a **100k stratified sample**, neighbours from the full graph. Full-graph 1M models use bio only. Option A (tweets for all 1M) later, as an extra, never the headline | Follows Facebook DEC (cheap features on everyone, expensive work on samples) and TwiBot-20 (same text for every compared model). Avoids a 10–15 GPU-hour risk before the TwiBot-20 queue on Thursday night. The 100k sample has ~10× TwiBot-20's test size, so it is less noisy. Report limit: "with tweets" results are for a random sample |
| evening | **Profile fields kept** (all except `verified`). Cheap models: with and without profile. Expensive models (GNNs, LLM): with profile only | TwiBot-20's text-only main track was partly an accident of order. Profile fields are real signal (follower count alone: AUROC 0.733). Running the cheap models both ways still shows what the graph adds without profile. If time allows, rerun TwiBot-20's GNNs and LLM with profile so the two datasets match |
| ~21:40 | **`verified` and the labels moved out of the profile file** into `eval_only/`; every model script stops if it sees them (`twibot22_guard.py`) | Prevents an accidental shortcut: on TwiBot-20, `verified` alone scored 0.803. The profile file also held the label columns, which was a worse risk |
| ~22:05 | **Sealed test set: 50% of test plus all 986 experts**, scored once at the end | Like Kaggle's private leaderboard, it stops choices drifting toward the test set. The experts are the cleanest labels we have (dataset "bot" labels match them only 56 of 101 times), so they stay untouched until the end |
| ~22:45 | **Text rules**: retweets kept; no total length cap (all 20 tweets, each piece ≤ 128 tokens, averaged per user); links and @mentions kept raw | All three follow TwiBot-20, except the cap: there it was a file-size convenience and here it would cut 70.7% of users to ~15 tweets. TwiBot-20's cleaning check found placeholders gave no gain and deleting links slightly hurt. One difference to report: TwiBot-22 uses the 20 most recent tweets, TwiBot-20 the first 20 in file order |
| ~22:20 | ~~**Weak → expert recalibration, light version**~~ (replaced 5 Nov ~13:20, next row), at the very end: trees and MLP keep their weights; a small logistic regression on their score + a few profile fields is fitted on the experts with 5-fold rotation | Facebook DEC trains on noisy labels then clean ones (30M, then 240k). We have only 986 clean labels (105 bots): too few for full fine-tuning, enough for recalibration. Rotation keeps every expert's score clean, so the seal holds. A check, not a headline |
| 5 Nov ~13:20 | **Weak → expert fine-tuning with 5-fold rotation**, for LoRA `target` (1 epoch from its best adapter, ~789 experts per fold) and for trees / MLP (a real second training stage). Every model is also scored weak-only on the same 986 experts | The builder judged that sealing all 986 expert labels for one final score wastes them: the weak labels disagree with the experts too often (56 of 101) for weak-label training to learn the expert view. Training on all experts would leave no clean test, so they are rotated: each expert trains 4 folds and is scored once by a model that never saw it, and all 105 expert bots stay in the evaluation. Measured timings (5 Nov) put the LLM rotation at ~3.5–4 h, so the 4 Nov "too costly" reason no longer holds. The rotation runs blind; scores are read once, in the final run. Fixed before any expert score |
| 5 Nov ~13:45 | **DGP at TwiBot-20 scope**: 1,300 training users (same settings), the same 1,000 validation-check users as `target`, and a **1,500-user stratified test subset** of the working test users (seed 42, `dgp_test_subset_1500.npy`, committed before any scoring). **dgp vs target is compared on that same subset**; `target` keeps its 5,000-user score for the other comparisons | Time: LLM summaries run at ~1 s each (0.94 s measured). Full scope (5,000 test users) needs 26,674 node + 13,126 path summaries, ~8.6 h; the 1,500 subset needs 18,234 + 6,815, ~5.6 h (most summaries belong to the 2,300 train and validation users' neighbours). TwiBot-20 scored DGP on a test of similar size. The subset scores `target` at 0.632 / 0.718 vs 0.626 / 0.710 on all 5,000, so it is representative |
| 5 Nov 17:15 | **GNNs run on the CPU during DGP's summary phase only**; paused automatically while a LoRA trains | Both fit in the time left; LoRA training needs the whole machine (no GNNs on the CPU while it trains) |
| 5 Nov 19:22 | **Three graph questions (Q1–Q3)**: homophily by relation and class, local heterogeneity, cross-relation inconsistency; fixed before any number | The camouflage literature (MAGNET, TROPICAL, FRAUDRE) asks exactly these; testing them shows whether its ideas transfer to TwiBot-22 rather than assuming it |
| 5 Nov 19:44 | **Leaderboard reconciliation (E11)**: our models under the official split, with and without `verified`; replication only, sealed users removed | The TwiBot-22 paper ranks graph models first, we find graph ≈ no graph; this tests whether the split or the `verified` shortcut explains the gap |
| 5 Nov ~20:15 | **D8, empty-bio test (builder's design)**: lift of the empty-bio cue under weak vs expert labels, on the 986 experts and the separate 500-user annotation study; final run only | Tells whether empty bio is a real bot habit or a habit of the label-makers |
| 5 Nov 20:38 | **Expert stage for the GNNs** (MLP control, RGCN, FAGCN; 30 epochs at lr 1e-3 per fold), blind | So every model family gets the same weak → expert treatment; if graph models gain more from expert labels, the weak labels hid what the graph can do |
| 5 Nov ~23:50 | **Expert rotation scores read early, Thu 5 Nov ~23:50, by the builder's choice; no model or rule changed after this.** Read: the 986 expert users only (LoRA `target`, trees, MLP, RGCN; FAGCN still running, left out) and D8. The sealed non-expert users stay unread until the final run | Builder's decision |
| ~22:05 | **100k sample**: stratified by role and label, seed 42, plus the experts (100,986 users); 10,039 hub users embedded as neighbour providers only | The hubs are the only users with outgoing links, so everyone's neighbour features come from them. Adding them as study users would make them ~10% of the sample instead of ~1% |

#### What the checks found (no model run yet)

- **Shortcuts:** only `verified` (0 of 95,398 verified users are bots), across 17 profile fields, 28 graph cues and all 814 feature columns.
- **Graph:** near-zero label signal on all 7 relations (adjusted homophily 0.011–0.038). A bot's neighbours are 11.6% bots, below the 14% chance level, which is heterophily. Only 10,039 users (1%) have any outgoing link, so the graph is star-shaped.
- **Strong cues, allowed but reported:** empty bio (40.6% of bots vs 7.8% of humans); no tweets (17.6% vs 4.8%); default profile picture (13.0% vs 1.7%).
- **Data integrity:**
  - 88,217,457 tweets read, exactly the "posted" links in `edge.csv`;
  - 0 decode errors;
  - no tweets after the collection end;
  - every file in `label.csv` row order.

---

### 4. Rules that now apply to everything

1. **Question every claim.** A low-cost sceptic agent checks new claims; several of our own statements were corrected this way and the corrections are recorded.
2. **Decide before scoring.** Split, sample, metrics, model choice and cut-offs are written down and committed before any test number.
3. **Honest metrics.** Macro-F1, AUROC and PR-AUC with bootstrap confidence intervals, plus a non-verified slice. Accuracy alone hides an "always human" model.
4. **No shortcuts in the inputs**, checked by a fixed rule, and guarded in code.
5. **Before any job longer than 30 minutes**, say in one line what it gives and why the report needs it; ask if the value is weak.
6. **No dataset files in the repository** (TwiBot-22 is CC BY-NC-ND).

### 5. Still open (4 Nov, late evening)

- Text rules for the tweets: keep retweets? length cap? (checklist C5)
- Tweet checks: copy-paste tweets across train and test, spam keywords, posting app (C6, C7, D4).
- Feature rebuild with multilingual-E5 bios, then re-check shortcuts (H2).
- Shared scoring module (G3, G5), then the freeze and the first model runs.
- The builder's go after a 10k-user timing test, before embedding tweets.

---

# Appendix 1: TwiBot-22 data checklist (copy of `reports/twibot22_data_checklist.md`)

## TwiBot-22 data-processing checklist

Written Wed 4 Nov 2026, 19:50, before any TwiBot-22 model score. Built from `twibot20_lessons_for_twibot22.md` (the TwiBot-20 mistakes) and `twibot22_plan.md`. Rules recorded here are kept; any later change is written down with its reason and time.

Status: ✅ done and checked · 🟡 partly done · ⬜ to do · ➖ not needed (with the reason)

### A. Raw data

| # | Item | Status | Note |
|---|---|---|---|
| A1 | `user.json`, `label.csv`, `split.csv`, `edge.csv` complete | ✅ | `edge.csv` 6,642,968,601 bytes; `user.json` matched byte for byte against a second download |
| A2 | 9 tweet files, each complete (ends with `]`) | ✅ | All 9 in `tweets/` (10.0–12.5 GB, 104.4 GB total), each ends with `}]`. Decode errors are counted during extraction (C1) |
| A3 | Tweet count per file and in total, against the paper's ~88M | ✅ | 88,217,457 tweets = exactly the `post` edges in `edge.csv` (`tweets_top20_summary.json`) |
| A4 | No dataset file in git (CC BY-NC-ND 4.0) | ✅ | `data/` is git-ignored |

### B. Labels and split

| # | Item | Status | Note |
|---|---|---|---|
| B1 | Our split: stratified 70/20/10, seed 42, 14.0% bots in each part | ✅ | `split_ours.npy` |
| B2 | All 986 expert-labelled users forced into test | ✅ | |
| B3 | Official split kept for comparison only (non-random: all 95k verified users in train) | ✅ | |
| B4 | **Sealed test set** (decided by the builder 4 Nov ~22:05: 50% of test + all experts) | ✅ | `twibot22_sealed_sample.py` → `roles.npy`: test_work 49,951 (14.0% bots), test_sealed 50,937 (49,951 + 986 experts). Drawn before the sample |
| B5 | **100k study sample** | ✅ | `sample_idx.npy`: 100,000 drawn stratified by role and label (70k train, 20k val, 5k test_work, 5k sealed; 14.0% bots each) + the 986 experts = 100,986 users |
| B6 | Weak labels documented: dataset "bot" matches experts for 56 of 101 | ✅ | |
| B7 | **Only train-split labels are ever used for training**, including DEC's stage 1 ("1M weak labels" in the plan means the train part only) | ⬜ | Write into every training script |
| B8 | **Expert users are never in train / val / working test.** All 986 sit in test_sealed. **Changed by the builder 5 Nov ~13:20:** they are no longer used only for a final score; they are also used for training, by 5-fold rotation (G7), so each expert is scored once by a model that never saw it | ✅ | All 986 in test_sealed; rotation in G7 |

### C. Tweets

| # | Item | Status | Note |
|---|---|---|---|
| C1 | Extraction counts: tweets read, kept, regex misses, JSON decode errors, per file (lesson T2) | ✅ | `twibot22_tweets_all.py`: 0 author misses, 0 date misses, 0 decode errors in all 9 files |
| C2 | **Most recent by date, not first met in file order** (T3) | ✅ | Two passes: index every tweet's author and date, select the 20 latest per user across all 9 files (deterministic ties), then decode. 17,838,598 tweets for 933,872 users |
| C3 | Wanted users = all 1M labelled users (top 20 each by date); the sample and hubs are subsets of that file | ✅ | Changed 4 Nov 20:58 (F6). All 88.2M tweet authors are labelled users |
| C4 | Users with 0 tweets: count by class; text = bio only, with a flag | ✅ | On the files: 17.6% of bots vs 4.8% of humans have 0 tweets. A strong cue like empty bio: allowed, reported (D6) |
| C5 | **Text rules** (decided by the builder 4 Nov ~22:45, before any score) | ✅ | (1) **Retweets kept**, as TwiBot-20 (`prep_twibot20.py` has no filter); train share 33.6% of bots' vs 36.4% of humans' tweets. (2) **No total length cap**: all 20 tweets, each piece (bio, each tweet) ≤ 128 tokens, pieces averaged per user as TwiBot-20's `step3_embed.py`; TwiBot-20's 2,000-character cap was a file-size convenience and would cut 70.7% of TwiBot-22 users. (3) **Links and @mentions kept raw**, as TwiBot-20 (its cleaning check: no gain; deleting slightly hurt); tweets with a link 46.0% bots vs 44.1% humans. Difference to report: TwiBot-20 took the first 20 tweets in file order, TwiBot-22 the 20 most recent |
| C6 | Language mix by class | ✅ | `twibot22_tweet_checks.py`: main language English for 78.4% of bots and 74.4% of humans, then Spanish (5.5% / 7.0%). Descriptive |
| C7 | Duplicate tweets across train and val | ✅ | 0.56% of val users have an exact original tweet (≥30 chars) also posted by a train user. For those 157 val bots the copy writers are 31% bots (chance 14%): some bot-to-bot copying, touching ~0.5% of users. Reported, not excluded |
| C8 | Tweet dates within the collection window; no tweets after 15 Mar 2022 | ✅ | 2006-03-21 to 2022-03-15 01:16 UTC; 0 after the collection end; all timestamps +00:00 |

### D. Profile fields

| # | Item | Status | Note |
|---|---|---|---|
| D1 | Shortcut rule run; only `verified` fails, so it is excluded | ✅ | |
| D2 | Shortcut rule re-run on **our** split's train (it was run on the official split) | ✅ | `twibot22_shortcuts.py`: only `verified` fails. 17 profile fields + 28 graph cues (has-edge, degree per relation) |
| D3 | Leaf-size rule uses a count (≥200 users), not 10% (P2) | ✅ | Plus a stricter every-cut-off tail scan: 5 tiny pure-human tails (≤1.2% of users: very high followers, listed, tweets, following; oldest accounts). The followers and listed tails are 80–90% verified and not pure without them; the tweet-count, following and age tails stay ~99% human but cover ≤0.3%. Reported, not excluded |
| D4 | **Labelling-function leakage**: spam keywords in bios and tweets, by class | ✅ | 43,469 words used by ≥200 train users checked: **no bot-pure word**. 8 human-pure words, each 200–244 users (topic communities, e.g. French politics, `#ucustrike`). Apps: 3 newsroom tools (SocialFlow, SocialNewsDesk, Echobox) ~99% human, 400–715 users each. Tiny; reported, not excluded (same treatment as the D3 tails) |
| D5 | Account age parsed from ISO dates, not zero for everyone (X3) | ✅ | `twibot22_profile_check.py` uses `pd.to_datetime`. The TwiBot-20 parsers in `metadata_check.py` and `step6_dgp.py` are still wrong for TwiBot-22 |
| D7 | **Labels were partly made by graph models** (GCN, GAT, R-GCN were labelling functions). Graph models may re-learn the labeller. Check: compare each graph model's gain on weak-label test users vs on expert-labelled users | ⬜ | Planned as DEC question 3; applies to every graph model |
| D8 | **Empty-bio cue: real bot habit (A) or label-maker habit (B)?** Planned by the builder 5 Nov ~20:15, before any expert number; run in the final run only. On the same 986 expert users: bot rate among empty-bio users and among users with a bio, and the **lift** (empty ÷ with bio), under the weak labels and under the expert labels; bootstrap 95% CIs (2,000) on each lift and on their difference. A predicts a similar lift under both sources; B predicts the lift shrinks towards 1 under expert labels. Caveats to report: (1) the expert labels are only partly independent: the paper's 7 voting models were trained on 800 of the 1,000 expert-labelled users, and which 800 is not released; (2) experts are also people looking at the profile, so a lift that survives supports A but does not prove it; (3) only ~100–150 empty-bio experts, so a null result means 'not enough data'. **Second source, added 5 Nov ~20:40 (before any number):** the official repo's annotation study, 500 TwiBot-22 users × 3 experts on 0–4 (`data/twibot-22-raw/annotation_study/`); only 1 overlaps the 986 and none trained the voters, so caveat (1) does not apply. Fixed rule: bot if the median rating ≥ 3, human if ≤ 1, dropped if 2 (not sure); same lift comparison against the weak labels on those users (labels vs labels, no model, so their train / val roles do not matter); read in the final run. Reason: the profile's lead over text rests largely on this cue (findings 1, 4) | ✅ | Run early 5 Nov 23:50 (builder's decision): experts weak 8.7× vs expert 7.2× (not independent); annotation study weak 5.0× vs annotators 1.6×, difference +3.4 [+1.2, +6.1]; `expert_read_early.json`. Final run (`FINAL_RUN=1`) |
| D6 | "Empty bio" and "no tweets" flags: strong label cues (empty bio 40.6% of bots vs 7.8% of humans; no tweets 17.6% vs 4.8%). Report a flag-only baseline and with-bio / with-tweets slices | ⬜ | |

### E. Graph

| # | Item | Status | Note |
|---|---|---|---|
| E1 | Relations fixed: all 7 (following, followers, mention, retweet, reply, quote, like) | ✅ | 5 fixed at 14:25; changed to 7 by the builder at ~21:30, before any score (H3) |
| E2 | Homophily, adjusted homophily, label informativeness for all 8 relations | ✅ | All near zero (0.011–0.038) |
| E3 | Star-shape check: only 10,039 users have outgoing edges | ✅ | |
| E4 | Degree AUROC on held-out users (R3); out-degree caps and reciprocity (R5) | ✅ | `twibot22_graph_checks.py`, on **val** users (no test label read): in-degree AUROC 0.36–0.49, out-degree 0.50; no cap (max out-degree 907); reciprocity 0.3–1% |
| E5 | Train-to-held-out neighbour leakage vs chance | ✅ | On val users: a bot's train neighbours are 11.6% bots (chance 14.0%), a human's 92.5% human (chance 86.0%). Edges do not give away bot labels |
| E6 | co-list counts use int8, so they can overflow above 127 shared lists (R2) | ➖ | co-list is reported only and never used as model edges. Note it in the report |
| E7 | Relation names in `edge.csv` all present and non-empty (R4) | ✅ | Every relation in the heterophily table has edges |
| E8 | **Q1. Homophily by relation, direction and class** (builder, 5 Nov ~19:20, fixed before any number): for each of the 7 relations and each direction, P(neighbour is a bot \| node is a bot) and P(neighbour is a bot \| node is a human) against the 14% base rate, plus adjusted homophily per direction. Labels of train + val users only (no test, no sealed) | ✅ | Done 5 Nov 19:25: bots' neighbours 4–17% bots (base 14.0%), adjusted homophily 0.010–0.024 on all 7 relations; `relation_homophily_by_class.json`. `twibot22_relation_checks.py` |
| E9 | **Q2. Local heterogeneity** (fixed ~19:20): each user's local homophily = share of the user's TRAIN-labelled neighbours with the user's own label, on the union graph and per relation. (a) Distribution on val users (deciles, share above 0.5, share with no train neighbour). (b) Where graph models gain: on the working test users, FAGCN, RGCN vs the MLP control (same network, no edges) and XGB-Graph vs own-feature tree (1M, with profile), per local-homophily bin (no train neighbour, [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1]); AUROC and macro-F1 per bin. A test user's own label is used only to slice scores (as in heterophily papers), never as an input | ✅ | Done 5 Nov 19:25: neighbour features help homophilous users (+0.057 AUROC), hurt heterophilous ones (−0.016); RGCN hurts the homophilous bin (−0.052); `local_homophily.json`. `twibot22_local_homophily.py` |
| E10 | **Q3. Cross-relation inconsistency** (fixed ~19:20): label-free features per user from two relation groups, follow (following + followers) and interaction (mention, retweet, reply, quote, like), both directions: cosine distance between the groups' mean neighbour bio embedding (PCA 32), distance between their mean neighbour profiles, own-vs-group cosine for each group and its difference, log degree ratio, flags for an empty group. Test: (a) each feature alone, val AUROC; (b) XGB-Graph + these features vs XGB-Graph, sample (with and without profile, 3 seeds) and 1M (with profile, seed 0), paired bootstrap CI. Shortcut rule run on them first | ✅ | Done 5 Nov 21:21: 8 features, no shortcuts; add nothing (1M: F1 −0.002 [−0.004, −0.000], AUROC +0.000; sample the same); `q3_ci_full.json`. `twibot22_relation_checks.py`, `EXTRA=inconsistency` |
| E11 | **Leaderboard reconciliation** (builder, 5 Nov ~19:50, fixed before any number). Why: the TwiBot-22 paper ranks graph models first (accuracy on the official split); we find graph ≈ no graph. 2 × 2 design: split {ours (random stratified), official `split.csv` (train 7.8% / val 28.0% / test 29.4% bots)} × `verified` {out, in as an input column}. Ours / out = the existing results. Models: own-feature tree and XGB-Graph (with profile, 1M bio-only table), MLP control, FAGCN, RGCN (same settings as the main GNNs, seed 0). Metrics: accuracy and bot-class F1 at a 0.5 probability threshold (the paper's convention) and at our validation cut-off, plus macro-F1 and AUROC. **Our sealed users (incl. experts) are removed from every official part** (train, val and test), so the seal holds; `verified` enters only these replication models, never the main ones (explicit override, `REPLICATION=1`). GNNs first (they must finish before DGP's LoRA starts) | 🟡 | All cells done except RGCN on our split with `verified` (running). 

### F. Features and embeddings

| # | Item | Status | Note |
|---|---|---|---|
| F1 | Standardisation and PCA fitted on train only | ✅ | `twibot22_features.py` |
| F2 | Rows kept in `label.csv` order everywhere; any join by id, never by position (X2) | ✅ | `twibot22_graph_checks.py`: all artefacts 1M rows; the 124,231 zero embedding rows equal the 124,231 empty bios exactly. Keep this assertion in every new script. `prep_twibot22.py` (superseded) sorts by id and would need a reindex if reused |
| F3 | Output folders per design (sample, text recipe), so no stale cache is reused (X1, lesson 20) | ✅ | `twibot22_embed_text.py` writes `data/twibot-22/sample110k_<enc>_bio20tw_raw/{emb,ids,user_ids}.npy`, asserts the row order after saving, refuses to overwrite. Timing test (10k users): 213 pieces/s, 2.7 h estimated for 110,057 users. Full run started 4 Nov 22:56 with the builder's go (E5, then MiniLM) |
| F4 | Bio-only and bio + tweets embeddings saved as separate feature sets (E3) | ✅ | Bio only: `emb_bio_{minilm,e5}_all.npy` (1M). Bio + 20 tweets: `data/twibot-22/sample110k_{e5,minilm}_bio20tw_raw/` (110,057 users; E5 153.5 min, MiniLM 37.7 min, both exit 0; row order asserted; 3,344 users without text = zero rows) |
| F5 | Neighbour features use neighbours' features only, never their labels | ✅ | |
| F6 | **Headline text model and comparison, fixed by the builder 4 Nov 20:58, before any score (replaces the 20:38 rule).** Option B: headline text model = multilingual-E5. **Headline comparison = the 100k stratified sample** (natural 14% bots), text = bio + 20 most recent tweets, neighbour features from the full 1M graph (hub users' tweets embedded as neighbour providers). **Full-graph models on all 1M users use bio only** (reported as the full-scale check). Option A (bio + tweets for all 1M) is run afterwards as an extra check if time allows; it never replaces the headline. MiniLM = check; Qwen2.5-1.5B on the sample only. Before any tweet embedding: a 10k-user timing test, the estimate, and the builder's go; not while a TwiBot-20 LoRA seed run is on the GPU. Report limit: "with tweets" results are for a 100k random sample, not all 1M users | ✅ decided | Tweet extraction covers all 1M users (top 20 each), so it serves B and A |
| F7 | Users with no neighbours in a direction get a mean of 0 (= the train mean after standardisation); the separate log-count column (0) tells models it is missing | ✅ | `twibot22_features.py` `mean_agg`. GNNs need the same flag |

### G. Run hygiene

| # | Item | Status | Note |
|---|---|---|---|
| G1 | `OMP_NUM_THREADS=1` and no torch import with sklearn boosting (the TwiBot-20 silent segfault) | ⬜ | In the CPU model scripts |
| G2 | Every launcher logs each job's exit code | ⬜ | |
| G7 | **Weak → expert fine-tuning with 5-fold rotation** (builder, 5 Nov ~13:20, replacing the 4 Nov light version; reason: dataset "bot" matches experts for only 56 of 101, so weak-label models learn the labelling rules; using all experts for training would leave no clean test). Folds: StratifiedKFold(5, shuffle, seed 42) on the expert label (~789 train / ~197 held out, 84 / 21 bots). **LoRA target:** start from the best target adapter (validation), 1 epoch on the 4 training folds, step 6 settings, no selection; macro-F1 cut-off chosen on the fold model's own training folds (`twibot22_expert_rotation_llm.py`, queued after DGP, ~3.5-4 h GPU, **blind**: writes scores, prints no metric). **GNNs (builder, 5 Nov ~20:35):** FAGCN, RGCN and the MLP control: weak model as in `twibot22_gnn.py`, then per fold 30 epochs at lr 1e-3 on the experts, no early stopping, cut-off on the fold's training experts (`twibot22_expert_rotation_gnn.py`, blind, pauses while LoRA trains). **Trees / MLP:** the same folds, a real second training stage (continue / refit on experts), CPU. Every model also scored on the same 986 experts weak-only, for the paired comparison. Read once, in the final run. Fixed before any expert score. Also answers D7 | ⬜ | Replaces DEC stage 4 Read early 5 Nov 23:50 (builder's decision): LoRA, trees, MLP, RGCN; FAGCN pending; `expert_read_early.json` |
| G8 | **The LLM reads the same text as the encoders it is compared with** (same pieces, same token limit), so "frozen encoder vs fine-tuned LLM" is like for like. Lesson from TwiBot-20 (LoRA 480 tokens vs encoders 2,000 characters; external feedback #2, 5 Nov) | ⬜ | Set when the LLM step is configured |
| G6 | **Label and `verified` out of every model input** (step 3, builder 4 Nov ~21:40) | ✅ | `twibot22_guard.py`: `users.parquet` now has profile fields only (no verified, no label/y/split); verified in `eval_only/verified.parquet` (95,398), read only for the non-verified slice and checks; `check_columns()` stops any model input named verified/label/y/split (tested). Shortcut results identical after the move. `withheld` is 59 users, not 0 (comment fixed); near-constant, left out |
| G3 | Selection rule, seeds (3), metrics (macro-F1, AUROC, AUPRC, bootstrap CIs) written into the scripts | ✅ | `twibot22_metrics.py`: val-only cut-off and selection, seeds 0–2, paired bootstrap CIs (2,000), self-tested |
| G4 | Every result also scored on non-verified test users and with-bio users | ✅ | Slices in `twibot22_metrics.slices()`: all, non-verified, with bio, with tweets; optionally restricted to the 100k sample |
| G5 | **No artificial setting B on TwiBot-22.** The natural 14% bot share is already within graph-fraud benchmark rates (1.3–21.8% in GADBench), and TwiBot-20's 1:10 subsample removed 8,108 of 8,162 bot–bot edges. For rarer rates, report **PR-AUC with test users reweighted to 5% and 1% bots** (bots weighted so they make up 5% or 1% of the test weight; graph, training and cut-off unchanged). Wording in the report: "fraud-like rare class", not "fraud" | ✅ | In `twibot22_metrics.py`; self-test: a random score gives 0.14 / 0.05 / 0.01 as expected |

### H. Added after the step-list review and its sceptic pass (4 Nov, 20:28)

| # | Item | Status | Note |
|---|---|---|---|
| H1 | Hub users (10,039 with an outgoing edge) get bio + tweet embeddings as neighbour-feature providers; flagged, not added as study users | ✅ | `hub_idx.npy`; 968 are also in the random sample. Users to embed (sample ∪ hubs): 110,057 |
| H2 | Feature table with the headline bio encoder | ✅ | `features_full_e5.npz` (multilingual-E5 bio PCA, 7 relations, 814 columns; split reused from file; `bio_encoder` recorded). Shortcut rule rerun on it: 0 of 814 fail. The MiniLM table `features_full.npz` is kept as the check |
| H3 | **Model-input contract, step 1 fixed by the builder 4 Nov ~21:25, before any score.** Profile fields (all except `verified`) are kept as inputs. **Cheap models** (MLP, trees, XGB-Graph, DEC-lite, links-only and profile-only baselines): run **both** versions, with profile and without profile (text + graph only). **Expensive models** (GNNs on 1M users, LLM + LoRA): run **with profile** only, so no information is left out (LLM: profile as numbers in the prompt, as TwiBot-20's `dgp_num`). The "what does the graph add without profile" question is answered by the cheap models. If TwiBot-22 finishes early: rerun TwiBot-20's GNN and LLM with profile too, so the two datasets match. **Step 2 (4 Nov ~21:30): all 7 relations** (following, followers, mention, retweet, reply, quote, like) as model edges and neighbour features; co-list reported only | ✅ | Every new tweet feature passes the shortcut rule before use |
| H4 | Tweet script checks each file's completeness itself (ends with `]`, size stable) before marking it done (lesson T1) | ✅ | `complete()` asserts the closing `]` before indexing |
| H5 | Parse tweet dates with `pd.to_datetime(utc=True)` before "most recent" (X6), checked on a sample | ✅ | Selection uses parsed seconds; checked against a full JSON parse of a 600 MB slice (532,984 tweets, all authors and dates match) |
| H6 | Results naming that cannot overwrite: one folder per design and seed; seeds fixed per model (lessons 18–20) | ✅ | `save_scores()` writes `runs/<design>/<model>_s<seed>.npz` and refuses to overwrite |
| H7 | **LLM budget, decided by the builder 5 Nov ~00:50:** Qwen3-1.7B, **1,300 training users**, same LoRA settings as TwiBot-20 (B = 10, K = 2, M = 8, r = 16, lr 1e-4). Order tonight: smoke test of step 6 on 20 users (5 min); if it passes, after E5 finishes: summaries → LoRA `target` → LoRA `dgp`, then the MiniLM embedding (cheap models only). Each sub-step checkpointed and logged. No GNNs on CPU while LoRA trains. If the smoke test fails: nothing is launched; the failure is written to the overnight notes | ✅ decided | **Details decided 5 Nov ~02:15:** (1) adapter saved at every check, chosen by macro-F1 on a fixed 2,000-user stratified validation subset; if one check takes > 10 min in the smoke test, use 1,000 users or check less often. (2) An MLP on exactly LoRA's 480-token text is added (G8). (3) Profile in the LLM: decided later; `target` and `dgp` run text-only as on TwiBot-20. (4) DGP relation types: 2 groups, follow = following + followers, interaction = mention + retweet + reply + quote + like (6 path types, as TwiBot-20's 2 relations). (5) MiniLM done; the LLM runs tonight. (6) GNNs (changed by the builder 5 Nov ~02:35, replaces "all 5 × 3 seeds"): all 5 GNNs with 1 seed each, then the 2 best by validation macro-F1 with 3 seeds each; never all 5 × 3. Results tables mark each GNN as 1 seed or 3 seeds. Thursday daytime on CPU, not during LoRA. LoRA training users: 1,300 drawn from the sample's train part, stratified, seed 42; tested on the sample's working-test users only |
| H7b | **DGP scope (builder, 5 Nov ~13:45):** 1,300 train, the same 1,000 validation users as `target`, a 1,500-user stratified test subset of test_work (seed 42, saved and committed before scoring: `results/twibot22/A/dgp_test_subset_1500.npy`). dgp vs target on that subset (`twibot22_dgp_vs_target.py`, paired bootstrap). Reason: summaries ~1 s each; 5.6 h instead of 8.6 h. GPU queue: zero-shot rescore → rotation smoke test → expert rotation → DGP → target seed 1 → dgp seed 1 | ✅ decided | `step6_dgp.py` `DGP_TEST` |
| H8 | Sealed users: ids saved to disk before the sample draw; every scoring script refuses them before the final run | ✅ | `twibot22_guard.score_mask()` returns test_work unless `FINAL_RUN=1`; `assert_not_sealed()` stops otherwise (tested) |
| H10 | **Sample neighbour features from bio + tweet embeddings** (headline, F6). Checked 5 Nov 00:50: 98.8% of sample users have all their 1-hop neighbours embedded (59% of links, because a few sample hubs link out widely), but only 18% of 2-hop neighbours are. So: **1-hop** per-relation and union text means from the multilingual-E5 bio + tweet embeddings, averaged over embedded neighbours, plus a "share of neighbours embedded" column; **2-hop** stays on the bio-only embedding (all 1M users have it); profile columns unchanged. Own file next to the embeddings; shortcut rule rerun on it | ✅ | `twibot22_features_sample.py` → `features_sample_e5tw.npz`: 100,986 × 830 (profile 255, text 544, graph 31 columns; text PCA explained 0.51; 106,713 of 110,057 embedded users have text). Shortcut rule rerun: **none** (strongest single column: own log followers, val AUROC 0.740) |
| H9 | Extraction time | ✅ | 7 min in total (index 2.5 min at ~900 MB/s, select 1 min, decode 4 min) |

Simplified (descriptive only, not gates): C6 language mix, C8 date range (folded into the C1 counts), C4 zero-tweet counts. G5 reweighted PR-AUC is computed from saved scores after the runs; it changes no model.

### Freeze (dropped)

No separate freeze commit (builder, 5 Nov). Decisions are recorded with their times in this file and the git history; models start once the rows they need are done.

### Order for tonight and tomorrow morning

1. `tweet_1` lands → A2.
2. Quick checks, ~30 min CPU: D2, D3, E4, E5. Then B4 (sealed set) before B5 (sample).
3. Fix the tweet script (C1–C3, C8), draw the sample (B5), extract tweets (~30–40 min).
4. Tweet checks: C4, C6, C7, D4. Decide the text rules (C5).
5. F3, then a 5-minute embedding timing test, then the full embedding.
6. G1–G3, freeze, first model runs.

---

# Appendix 2: TwiBot-20 to-do (copy of `reports/twibot20_todo.md`)

## TwiBot-20: everything still to do, in one place

Written Thu 5 Nov 2026, ~00:30. It collects every open TwiBot-20 item from `twibot20_lessons_for_twibot22.md` (the 22 lessons), `twibot22_plan.md` (step 8), the TwiBot-22 checklist (H3: profile track), and the external feedback (`artifacts/google_feedback.txt`, reviewed 5 Nov). Items are grouped so they can run **in one go**: one GPU queue, one CPU batch, one round of report edits. ✅ = done, ⬜ = to do.

**When (decided by the builder 5 Nov ~00:40):** TwiBot-22 is finished first (its checklist, models, sealed set), then compared with TwiBot-20's current results, and only **then** is this whole batch run. Target: TwiBot-22 done by Fri 6 Nov evening, this batch Fri night to Sat. If time runs short, GPU jobs run in this order: G1 + G2 (seeds, adapter rule), G4 (matched text), G3 (`dgp_num`). **If time runs out, TwiBot-20's current results are used as they are** (builder, 5 Nov) and compared with TwiBot-22; the report then states the open caveats from this list (adapter rule, text cap, setting B on macro-F1, verified slice for non-LLM models).

### Already done (4 Nov)

- ✅ Tree baseline rerun after the silent crash (`run_tree_baseline.sh`, A and B, exit codes logged), with PR-AUC and a non-verified slice. Lesson 17.
- ✅ Verified-subset check from raw data: target LoRA 0.704 → 0.641 on non-verified test users (`check_verified_subset.py`). Lesson 2, LoRA part.

### Batch 1: GPU queue (~12 h, after TwiBot-22)

Run in one queue, each job logs its exit code and writes to its own named file (lessons 19-20).

| # | Job | Why | Source |
|---|---|---|---|
| G1 | **LoRA seed 2** for `target` and `dgp` (A), seeds 1 and 2 so no headline rests on the unseeded run | Seed noise (0.02-0.03) is larger than every dgp − target gap | lessons 7, 8; plan step 8 |
| G2 | In those runs: **save the adapter at every check** and select by **full-validation macro-F1** (not the 200-user validation loss) | Same selection rule as the MLP; Google feedback #1; lessons 8, 11 | feedback #1 |
| G3 | **`dgp_num`** (DGP with numeric profile summaries, the 5 allowed fields) | The profile track for the LLM, matching TwiBot-22's "expensive models with profile" (H3) | plan step 8; H3 |
| G4 | **Matched-text control**: re-embed TwiBot-20 with the encoders reading exactly LoRA's 480-token strings (Qwen tokenizer), then MLP / LR on them | LoRA reads 480 tokens, encoders up to 2,000 characters; 81% of users are cut for LoRA. Equal text answers "frozen encoder vs fine-tuning" fairly | lesson 10; feedback #2 |

Check before queueing: the step 6 launcher accepts the selection rule and saves every check (`step6_dgp.py:416-418` saved only 2 adapters).

### Batch 2: CPU (can run while the GPU queue runs)

| # | Job | Why | Source |
|---|---|---|---|
| C1 | **Save test scores** for LR, MLP and the GNNs (they are retrained inside `bootstrap_ci.py` and never saved), then score them on **non-verified test users** | Every TwiBot-20 headline must also be shown without the 326 verified (all human) test users | lesson 2; top-10 #3 |
| C2 | **GNNs with profile** (the 5 allowed fields added to the input): GCN, SAGE, FAGCN, LINKX | Match TwiBot-22's design (H3), so the two datasets compare like for like | H3 |
| C3 | `bootstrap_ci.py` and `link_direction.py`: read the **seeded** LoRA score files (seeds 1, 2) and average over seeds | They hard-code the unseeded file | lesson 18 |
| C4 | **Bootstrap CIs on the tree baseline's crawl-artefact result** (+0.025 on all test users, −0.009 on non-verified) plus a sceptic check | New finding from 4 Nov; must be tested before it goes in the report | 4 Nov tree rerun |
| C5 | Step 5 neighbour ranking: use unit-length vectors without the 2 flags | A shared "no text" flag can outrank text similarity | lesson 9 |

### Batch 3: report edits (`REPORT.md`, after batches 1-2)

| # | Edit | Source |
|---|---|---|
| R1 | Main LoRA tables use the **full-validation adapter rule**, with both adapters shown and a note that the rule was fixed after some test scores were known | feedback #1; lesson 8 |
| R2 | Setting B finding: **lead with AUROC and PR-AUC** (macro-F1 cannot separate models with 54 test bots). On AUROC the gaps are clear and favour text models (GCN − MLP −0.072, FAGCN − MLP −0.141, LoRA dgp − MLP −0.064). Word it as "B cannot separate models on macro-F1", not "no difference" | feedback #3; finding 8 |
| R3 | Every headline also on non-verified test users (from C1) | lesson 2 |
| R4 | Text-cap and training-size caveats next to every "encoder vs LoRA" claim, with the G4 result | lessons 10, 12 |
| R5 | Note: TwiBot-20 used each user's first 20 tweets in file order; TwiBot-22 uses the 20 most recent | TwiBot-22 checklist C5 |
| R6 | Note: many comparisons on one test set and no sealed set on TwiBot-20 (cannot be fixed now; TwiBot-22 has one) | lesson 22 |

### Not doing (and why)

- Raising LoRA's token cap to 2,000 characters: too much GPU memory for a 16 GB Mac; G4 equalises from the other side instead.
- A sealed test set for TwiBot-20 after the fact: every test number has already been seen.

---

# Appendix 3: notes for the submitted report (internal; copy of `reports/report_for_submission.md`)

## For the submission (internal)

"Internal" = goes into the submitted COMP8851 research project report (7 pages, template: Introduction, Literature review, Methodology + timeline, Outcome, Contributions). Everything else in the repository is external working material. Items are added here as they are decided; Sunday's writing starts from this file.

**Writing rule:** as in research papers, the report never cites feedback (supervisor or external) as the reason for a choice. Heterophily is motivated by the problem itself: fraudsters and bots camouflage by linking to benign accounts (Dou et al., 2020), so methods must be evaluated under measured heterophily. Dataset changes are justified on their merits (e.g. Elliptic has no text for an LLM to read).

### Template and project brief (`uni brief /`, checked 5 Nov 23:10)

- **Sections (latest template):** Introduction (for a layperson: aim, why it matters, how, expected vs achieved) · Literature Review · Research Methodology (sub-problems, integration with figures, timeline vs plan, difficulties and fallbacks) · Research Outcome (follow the plan; deviations as P → P′) · Individual Contributions · **Future Outlook** · **Bibliography**.
- **Citation rule:** cite work from the past 3 (recommended) to 5 (at most) years, i.e. 2021 or later. Our list to check: Dou et al. 2020 (CARE-GNN) and Yang et al. AAAI 2020 are 6 years old: keep only as one-line foundational mentions, or replace (camouflage can be motivated from MAGNET 2024 / TKDD 2026 and FRAUDRE 2021). Fine: TwiBot-20 (2021), FRAUDRE (2021), TwiBot-22 (2022), Platonov, Hays, GADBench, Mao (2023), MAGNET, TROPICAL (2024–25), TKDD 2026.
- **Project brief:** "Large Language Models for Graph-based Fraud Detection": how textual (LLM) and structural (graph) information can be jointly leveraged; expected outcome a framework prototype + technical report. **Framing for the report:** the three LLM roles the brief names map onto our pipeline: LLM node representations (multilingual-E5 / MiniLM text features into trees and GNNs), LLM reasoning over graph structure by prompting (zero-shot, DGP with neighbour summaries), and LLM fine-tuning (LoRA `target`). The main result is *when* each helps under heterophily, not "profile wins".

### Methodology: timeline

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

### Literature review plan (builder, 5 Nov)

Order and argument (~400 words in the report):
1. **MAGNET**: adaptive learning for homophilous and heterophilous neighbourhoods (Haghighi et al., "Robust Graph Learning Against Camouflaged Malicious Actors", WISE 2024, LNCS, doi 10.1007/978-981-96-0567-5_12; verified on Crossref).
2. **TROPICAL**: higher-order relationships and hypergraph learning (Haghighi et al., ICDM 2024, doi 10.1109/icdm59182.2024.00019; extended in Knowledge and Information Systems 2025, doi 10.1007/s10115-025-02476-5; verified).
3. **"DEFEND"**: discriminative learning through intra- and inter-relation patterns. **Name not verified**: the closest match is Haghighi et al., "Beyond Smoothing: A Discriminative Enhanced Aggregation Graph Neural Network for Camouflaged Fraud Detection", ACM TKDD 2026, doi 10.1145/3801967. Check the model's name in the paper before citing.
   **Clue:** the TROPICAL journal paper (KAIS 2025, §6.1.1) says its datasets validate "DEFEND", so DEFEND is a model by this group; confirm which paper.
4. **FRAUDRE** (Zhang, Wu, Yang, Beheshti, Xue, Zhou, Sheng, IEEE ICDM 2021, doi 10.1109/ICDM51629.2021.00098): feature / topology / relation inconsistency caused by camouflage, plus imbalance; per-relation class homophily (normal > 0.85, fraud < 0.2), the same shape as TwiBot-22 (humans ~0.93, bots 0.04–0.17). Notes: `reports/paper_notes_tropical_fraudre.md`.
   Also available: Haghighi, "From Classic GNNs to Hyper-GNNs for Detecting Camouflaged Malicious Actors", WSDM 2023, doi 10.1145/3539597.3572989.

Argument: in online social networks, malicious accounts may deliberately connect to legitimate users (camouflage; Dou et al., 2020), weakening homophily-based graph learning. Haghighi et al. address this with adaptive neighbourhood aggregation, multi-relational learning and higher-order representations. These motivate testing whether interaction patterns, local structure and cross-relation inconsistencies give complementary signals for camouflaged accounts (our Q1–Q3, checklist E8–E10). Transfer already tested for TROPICAL's motivation: on TwiBot-22 the bot density around bots falls with hop distance (0.116 → 0.059 over hops 1–4, base rate 0.14), unlike Amazon where it rises. **Caution to state:** those papers evaluate specific fraud datasets; they do not show the methods transfer to TwiBot-22, so our experiments test that transfer.

**Related-work checks** (`reports/literature_check_findings.md`, arXiv ids verified): supported by Yang et al. AAAI 2020, Hays et al. WWW 2023, GADBench (Tang et al. NeurIPS 2023), Mao et al. NeurIPS 2023; contradicted by the TwiBot-22 leaderboard (graph methods first): discuss split, `verified` and graph-model labelling functions as candidate reasons. **Confirmed in the official code** (`reports/twibot22_repo_check.md`): BotRGCN, GCN and GAT use the skewed official split, take `verified` as an input, train a fixed 200 epochs with no validation selection, and report AUC from hard predictions.

### Methodology: other items to include (to be filled)

- Datasets and why: Instagram (secondary), TwiBot-20, TwiBot-22 (see `artifacts/04_decisions_and_reasons.md`).
- Heterophily measures: adjusted homophily, label informativeness.
- Shortcut rule and what it excluded (`verified`).
- Expert labels (986, 105 bots): used for training **and** clean evaluation by 5-fold rotation (decided 5 Nov, before any expert score); weak labels match experts for 56 of 101 dataset bots.
- Split, sealed set, sample; metrics (macro-F1, AUROC, PR-AUC, reweighted PR-AUC, bootstrap CIs, non-verified slice).

### Outcome: TwiBot-22 findings so far (5 Nov evening; working-test users, weak labels; sealed set and experts not yet scored)

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

---

# Appendix 4: run log, Thu 5 Nov (copy of `reports/overnight_notes_2026-11-05.md`)

## Overnight notes, Thu 5 Nov 2026 (external)

- 02:20 Step 6 smoke test started (DATASET=twibot22, VARIANT=dgp, SMOKE=1: 20 train, 50 val, 20 test users).
- 02:24 Finding: LLM node summaries run at ~1 s each (316 s per chunk of 320 prompts). The full `dgp` run would need summaries for tens of thousands of neighbours (8,300 targets × 3 paths × up to 8) plus ~25k path summaries: an estimated 20-30 h. Not possible tonight.
- Plan agreed for tonight: if the smoke test passes, run LoRA `target` only (no summaries needed). Validation checks on 2,000 users unless the smoke test shows one check would take > 10 min, then 1,000. `dgp` waits for the builder's decision on a smaller scope. No GNNs during LoRA.
- 02:41 Smoke test PASSED. Timings:
```
SMOKE TEST PASSED (end to end). Timings:
  node_summaries    20.28 min for 1,300 items (0.94 s each)
  path_summaries     1.69 min for 219 items (0.46 s each)
  fine_tune          2.05 min for 20 items (6.16 s each)
  test_scoring       0.25 min for 20 items (0.75 s each)
  val check         0.59 min for 50 users -> 23.6 min for 2,000 (limit 10 min: OVER, use 1,000 users or fewer checks)
```
- 02:41 Starting LoRA target (VAL_CHECK=1000).
- 02:42 Note on the validation-check rule: a 2,000-user check was measured at 23.6 min (> 10 min), so the builder's remedy was applied: 1,000 users. Extrapolated, a 1,000-user check takes ~11.8 min, just above 10 min; kept at 1,000 because that is the remedy the builder named. Checks run twice per epoch. Smoke numbers for planning the rest: training ~9.5 s per step of 4 users (~51 min per epoch of 1,300 users), scoring ~0.75 s per user (5,000 test users ~1 h), node summaries 0.94 s each, path summaries 0.46 s each. Estimated LoRA target total: ~4-6 h, done around 07:00-09:00 (estimate).
- 06:18 LoRA target finished, exit code 0. Last lines:
```
  {'check': 2, 'epoch': 1.4984615384615385, 'val_loss': 0.732, 'val_macro_f1': 0.6217, 'val_auroc': np.float64(0.704), 'check_minutes': 9.14}
  {'check': 3, 'epoch': 2.0, 'val_loss': 0.5291, 'val_macro_f1': 0.6162, 'val_auroc': np.float64(0.6669), 'check_minutes': 9.33}
  {'check': 4, 'epoch': 2.4984615384615383, 'val_loss': 0.749, 'val_macro_f1': 0.6196, 'val_auroc': np.float64(0.6773), 'check_minutes': 9.28}

                                                      

                                                              
Qwen3-1.7B LoRA target (full test) acc 0.795  macroF1 0.626  F1_1 0.374  AUROC 0.710  PR-AUC 0.312  | conn 0.625  iso 0.620

Same test sample as step 5:
Qwen3-1.7B LoRA target acc 0.797  macroF1 0.621  F1_1 0.362  AUROC 0.743  PR-AUC 0.341  | conn 0.617  iso 0.617
Qwen3-1.7B zero-shot target acc 0.810  macroF1 0.484  F1_1 0.073  AUROC 0.392  PR-AUC 0.118  | conn 0.492  iso 0.450

Saved -> /Users/pj/Desktop/Macquarie - Sem 4/PROJECT/Graph-based-fraud-detection-system/my_build/results/twibot22/A/step6_lora.json
=== exit code 0
```
- 13:14 Step 1 text baselines (TF-IDF+LR, MiniLM+MLP) exit 0.
```
LoRA target (rescored here, cut-off on its 1,000 val users): {'n': 5000, 'bots': 700, 'macro_f1': 0.6258268349978193, 'auroc': 0.7097636212624584, 'pr_auc': 0.3117658616534659, 'pr_auc_at_5pct_bots': 0.1355174285272691, 'pr_auc_at_1pct_bots': 0.03046781241349008}
tfidf_lr_full70k {'n': 5000, 'bots': 700, 'macro_f1': 0.6277243150773428, 'auroc': 0.7270598006644519, 'pr_auc': 0.3073773846262937, 'pr_auc_at_5pct_bots': 0.12980747527796996, 'pr_auc_at_1pct_bots': 0.02842308125742991, 'minus_lora_target': {'macro_f1': [0.0019872451382913375, -0.015143776910713867
minilm_mlp_full70k seed 0 {'n': 5000, 'bots': 700, 'macro_f1': 0.6031720237640975, 'auroc': 0.6809059800664452, 'pr_auc': 0.2934477378951583, 'pr_auc_at_5pct_bots': 0.12741901585936968, 'pr_auc_at_1pct_bots': 0.02970204458157622, 'minus_lora_target': {'macro_f1': [-0.022378653266130984, -0.041911960
minilm_mlp_full70k seed 1 {'n': 5000, 'bots': 700, 'macro_f1': 0.6109970612124651, 'auroc': 0.6813441860465117, 'pr_auc': 0.2829647125808818, 'pr_auc_at_5pct_bots': 0.11877874499942107, 'pr_auc_at_1pct_bots': 0.02698765283912572, 'minus_lora_target': {'macro_f1': [-0.01487286365178505, -0.0342863075
minilm_mlp_full70k seed 2 {'n': 5000, 'bots': 700, 'macro_f1': 0.6053591516146144, 'auroc': 0.6888003322259136, 'pr_auc': 0.2960759059142913, 'pr_auc_at_5pct_bots': 0.12879045136999392, 'pr_auc_at_1pct_bots': 0.031110635826162577, 'minus_lora_target': {'macro_f1': [-0.02030092577907114, -0.040413601
tfidf_lr_matched1300 {'n': 5000, 'bots': 700, 'macro_f1': 0.575765747047815, 'auroc': 0.6401501661129569, 'pr_auc': 0.24264754711067998, 'pr_auc_at_5pct_bots': 0.09704941491671179, 'pr_auc_at_1pct_bots': 0.02062665496190425, 'minus_lora_target': {'macro_f1': [-0.04998096775492642, -0.066778336624091
minilm_mlp_matched1300 seed 0 {'n': 5000, 'bots': 700, 'macro_f1': 0.5765818013683321, 'auroc': 0.6097279069767442, 'pr_auc': 0.2345723383104145, 'pr_auc_at_5pct_bots': 0.09487997780513902, 'pr_auc_at_1pct_bots': 0.020384570398131886, 'minus_lora_target': {'macro_f1': [-0.0494326581914654, -0.069605
minilm_mlp_matched1300 seed 1 {'n': 5000, 'bots': 700, 'macro_f1': 0.580203269995581, 'auroc': 0.6146441860465116, 'pr_auc': 0.2337058576688688, 'pr_auc_at_5pct_bots': 0.09374388210797924, 'pr_auc_at_1pct_bots': 0.020003088637714796, 'minus_lora_target': {'macro_f1': [-0.045652491024263726, -0.06441
minilm_mlp_matched1300 seed 2 {'n': 5000, 'bots': 700, 'macro_f1': 0.5653162281911631, 'auroc': 0.5840106312292359, 'pr_auc': 0.21650790163396474, 'pr_auc_at_5pct_bots': 0.08713150412001633, 'pr_auc_at_1pct_bots': 0.021664731098819928, 'minus_lora_target': {'macro_f1': [-0.06043881552551307, -0.0790
Saved /Users/pj/Desktop/Macquarie - Sem 4/PROJECT/Graph-based-fraud-detection-system/my_build/results/twibot22/text_baselines.json
=== exit code 0
```
- 13:22 Builder decision: experts used for training by 5-fold rotation (G7, B8). LoRA rotation queued after DGP (blind).
- 13:47 Builder decision: DGP at TwiBot-20 scope (1,300 train, 1,000 val, 1,500 stratified test subset, seed 42, committed before scoring); dgp vs target on that subset. Reason: summaries ~1 s each (5.6 h vs 8.6 h). Queue: zero-shot -> rotation smoke -> rotation -> DGP -> target s1 -> dgp s1 (until ~13:00 Fri).
- 14:00 Zero-shot on 5,000 test users: done (readout checked: not inverted).
```
zero-shot target, 5,000 working-test users: {'n': 5000, 'bots': 700, 'macro_f1': 0.504552682080655, 'auroc': 0.4612513289036545, 'pr_auc': 0.13935416785982876, 'pr_auc_at_5pct_bots': 0.04986199351296926, 'pr_auc_at_1pct_bots': 0.009982694280251808}
same with 1 - p (sign check only, not a result): {'n': 5000, 'bots': 700, 'macro_f1': 0.5531060914221793, 'auroc': 0.5387486710963454, 'pr_auc': 0.19786860249547744, 'pr_auc_at_5pct_bots': 0.07743622133900813, 'pr_auc_at_1pct_bots': 0.016326891613152955}
Saved /Users/pj/Desktop/Macquarie - Sem 4/PROJECT/Graph-based-fraud-detection-system/my_build/results/twibot22/A/step6_Qwen3-1.7B_target_zeroshot_full.npz
=== exit status (from log) 0
```
- 14:01 Rotation smoke test: PASSED.
```
                                                                 

                                                            
fold 0 saved (no metric printed: blind)
ROTATION SMOKE TEST PASSED (1.0 min for 20 train + 10 held out)
=== exit code 0
```
- 14:01 Expert rotation (LoRA target, 5 folds, blind) started.
- 17:14 Expert rotation finished, exit 0 (blind: scores saved, no metric looked at).
```
                                                              
fold 4 saved (no metric printed: blind)
Saved /Users/pj/Desktop/Macquarie - Sem 4/PROJECT/Graph-based-fraud-detection-system/my_build/results/twibot22/A/expert_rotation_target.npz (blind: read in the final run)
=== exit code 0
```
- 17:14 DGP started (TwiBot-20 scope: 1,300 train, 1,000 val, 1,500 test subset). Est. ~9.5 h (5.6 h summaries).
- 17:15 GNNs started (DGP summary phase: no LoRA training).
- 18:17 GNN seed 0 done; 2 best by validation macro-F1: FAGCN,RGCN -> seeds 1, 2.
- 18:21 Check: bio-only tree on the SAME sample users (70k train / 5,000 test): text-only F1 0.673 AUROC 0.761 vs bio + tweets 0.618 / 0.712; with profile 0.712 / 0.846 vs 0.713 / 0.848. Averaging tweets in hides the empty-bio cue; with profile they are equal. To be checked by the sceptic before the report.
- 18:28 CIs for key pairs done; sceptic check done (verdicts: profile-vs-LoRA needed a matched 1,300 run -> profile 0.677 / 0.808, still ahead; empty-bio mechanism confirmed by slices). Findings in report_for_submission.md.
- 19:57 GNNs all done.
- 20:38 GNN expert rotation (MLP, RGCN, FAGCN; blind) started.
- 19:45–22:45 E11 leaderboard reconciliation (REPLICATION=1): official split with/without `verified` done for trees and MLP / FAGCN / RGCN (RGCN without `verified` running); our split with `verified` running. Exit 0 for each finished cell.
- 21:21 Q3 full 1M (XGB-Graph + profile + inconsistency features, seed 0) done, exit 0: 0.729 / 0.872 / PR-AUC 0.579; paired CI vs without: F1 −0.002 [−0.004, −0.000], AUROC +0.000 (`q3_ci_full.json`). Q3 adds nothing.
- 22:43 GNN expert rotation: MLP done (13 min), RGCN done (112 min), saved blind; FAGCN running.
- 22:45 DGP node summaries 49 / 57 (~7 min per chunk, CPU shared with 3 jobs).
- 23:50 Builder's decision: expert rotation scores read early (decision line committed first, 851a08b). Read: 986 experts only, plus the TwiBot-22 annotation study without its 21 sealed non-expert users. Script `twibot22_expert_read.py`, output `expert_read_early.json` / `.log` (84d8da2). FAGCN left out (still running).
  - Expert stage vs weak-only (AUROC): trees +0.006 to +0.027, MLP +0.025, RGCN +0.067 [+0.034, +0.104], LoRA −0.099 [−0.164, −0.038] (its macro-F1 +0.040). Best after the expert stage: XGB-Graph + profile (sample) 0.751 / 0.910 / 0.599. The dataset's weak label itself: macro-F1 0.745 on the experts.
  - D8: lift on the experts, weak 8.74 [6.39, 12.33] vs expert 7.22 [5.31, 10.30] (difference +1.52 [−1.50, +4.64]); on the annotation study (458 users, 44 bots), weak 5.01 [3.30, 7.67] vs annotation 1.64 [0.70, 3.06] (difference +3.36 [+1.20, +6.14]).
  - Paper (arXiv 2206.04564v6, `research documents/`): the 1,000 experts are 17 researchers of the dataset's own group, and their labels (800 of them) trained the label-making classifiers, so expert and weak labels are not independent. The annotation study (6 researchers, most of whom have published on bot detection; 3 per user; whether they are from the same group is not stated) never trained the label-makers: the independent check.

