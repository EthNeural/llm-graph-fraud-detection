# TwiBot-22, explained from the ground up

Written Thu 5 Nov 2026, ~00:45. This explains what we are doing with TwiBot-22 and why. Each section starts in plain words and then goes one level deeper. A short glossary follows each section. Exact numbers and rules live in `reports/twibot22_data_checklist.md`; the reasons for each decision are in `04_decisions_and_reasons.md`.

---

## 1. The whole thing in three sentences

We have a million Twitter accounts, and about one in seven is a bot. We want to know whether looking at **who an account is connected to** helps catch bots, beyond reading **what the account writes**. On our first Twitter dataset (TwiBot-20) the answer was "no, the writing does all the work". TwiBot-22 is the second, bigger test, where the writing is weaker.

**Glossary**
- **Bot**: an account run by software, not by a person.
- **Connections**: follows, mentions, retweets, replies, quotes and likes between accounts.

---

## 2. What TwiBot-22 is

**Plain version.** TwiBot-22 is a public research dataset collected in early 2022. It has profiles for 1,000,000 accounts and 88 million of their tweets, plus who follows, mentions and retweets whom. Every account is marked "bot" or "human".

**One level deeper.** The labels are not hand-checked one by one. With a million accounts, that is impossible. The dataset's authors combined many automatic rules and models into a single vote. These are called **weak labels**: they are often right, but not always. For 986 accounts, human experts also checked the label. When we compare the two, the dataset's "bot" label matches the experts for only 56 of 101 accounts. So the expert labels are our most trustworthy check, and we protect them (section 6).

**Deeper still.** Some of the rules used to make the labels are things a model could see, such as the blue "verified" tick. If a rule made the label, a model that finds the rule is not detecting bots; it is copying the label-maker. That is why we hunt for **shortcuts** (section 5).

**Glossary**
- **Weak labels**: labels produced automatically, cheaply, with some errors.
- **Expert labels**: labels checked by people; few, but more reliable.
- **Labelling function**: one automatic rule used to vote on a label.

---

## 3. The question: do connections help?

**Plain version.** On many social networks, birds of a feather flock together: friends tend to resemble each other. If bots mostly connected to bots, a bot's friends would give it away. But fraudsters and bots often connect to ordinary users to blend in. Then a bot's friends look normal and tell you little.

**One level deeper.** The first pattern (similar accounts connect) is **homophily**. The second (different accounts connect) is **heterophily**. Our supervisor asked us to measure which one TwiBot-22 has, not assume it. We measured it for each of the 7 kinds of connection.

**Deeper still.** Plain "share of links between same-label accounts" is misleading when one class is rare: with 86% humans, most links join two humans just by chance. So we use **adjusted homophily**, which subtracts that chance level: 0 means "no better than random", 1 means "perfectly same-label". On TwiBot-22 every relation scores between **0.011 and 0.038**, essentially no label signal. A bot's neighbours are 11.6% bots, *below* the 14% you would expect by chance. That is heterophily.

**Glossary**
- **Homophily / heterophily**: links mostly between similar / different accounts.
- **Adjusted homophily**: homophily corrected for chance; around 0 means links carry no label signal.
- **Neighbour**: an account linked to this one.

---

## 4. Too big to do everything: cut the work, not the graph

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

## 5. Making it fair, part 1: shortcuts and the guard

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

## 6. Making it fair, part 2: rules first, and a sealed test

**Plain version.** If you try many ideas and keep whichever scores best on the test, the test score flatters you. Medical trials avoid this by writing their analysis plan before they see results. Kaggle competitions keep a hidden test set that only counts at the end.

**One level deeper.** We do both.
- **Every rule is written down and committed before any score:** the split, the sample, the text recipe, the metrics, how a model is chosen.
- **The test accounts are split in two:**
  - a **working half** (49,951 accounts), used while building;
  - a **sealed half** (50,937 accounts, including all 986 expert accounts), scored **once**, at the very end.
- A second guard refuses to score sealed accounts before that final run.

**Deeper still.** Choices such as "which model version is best" are made on a separate **validation** set (200k accounts), never on test. On TwiBot-20 some of those choices were made after test scores were seen, a mistake we listed and are not repeating. At the very end there is one more step, borrowed from Facebook: use the clean expert labels to **recalibrate** the models (a light correction, rotated so each expert is scored by a model that never saw it).

**Glossary**
- **Train / validation / test**: learn from / choose with / judge on.
- **Sealed set**: test accounts kept untouched until the final run.
- **Pre-registration**: fixing the rules before seeing results.

---

## 7. What each account "looks like" to a model

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

**Deeper still.** The text reading is the slowest step: 2.06 million pieces of text for the 110,057 sampled and hub accounts, about 2.7–3.5 hours on the Mac's GPU. It is running now. A smaller text model (MiniLM) runs afterwards as a cross-check.

**Glossary**
- **Embedding**: the list of numbers a text model produces; similar texts get similar numbers.
- **multilingual-E5 / MiniLM**: the two text models we use (main / check).
- **Feature**: one number describing an account.

---

## 8. The models we will compare

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

## 9. How we keep score

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

## 10. Where we are now (5 Nov, ~00:45)

- **Done:**
  - all data downloaded and checked (88,217,457 tweets, 0 errors);
  - shortcuts and heterophily measured;
  - the split, the sealed set and the 100k sample fixed;
  - the guards in place;
  - the scoring code written and tested;
  - every decision written down.
- **Running:** the text reading (multilingual-E5, then MiniLM), done around 02:30–03:30.
- **Next:**
  1. a final check of the outputs;
    3. the first models on Thursday;
  3. the language model on Thursday night;
  4. the sealed test and the comparison with TwiBot-20 on Friday;
  5. TwiBot-20's open items afterwards, if time allows;
  6. the report on Sunday.
- **No model has been scored yet.** That is on purpose: every rule had to be fixed first.
