# Team Guide (plain-language version)

This guide explains the project in everyday words. You don't need to know the code or the maths.
Every technical word is explained the first time it appears, and there is a glossary at the end.
For the latest numbers, see [REPORT.md](REPORT.md).

---

## 1. What is this project about?

Imagine you work at Twitter or Instagram and you need to find **suspicious accounts**: bots
pretending to be people, or businesses pretending to be ordinary users. You have two clues:

1. **What the account writes**: its bio and its posts.
2. **Who it is connected to**: its friends and followers. Together these connections form a
   **graph**, a network where each account is a dot and each friendship is a line.

**AI language models** (LLMs, the technology behind ChatGPT) are very good at reading text.
**Graph models** (GNNs, graph neural networks) are good at using connections. Our project asks:

> **What's the best way to combine an AI that reads text with the network of who-knows-whom,
> to catch suspicious accounts?**

## 2. The twist: suspicious accounts hide among normal friends

A common assumption is "birds of a feather flock together": you look like your friends. Many
graph models rely on it. If your friends are bots, you're probably a bot.

**Fraudsters break this rule on purpose.** A bot follows lots of real people so it looks
normal. This is called **heterophily**: connected accounts are often *different*, not similar.
So everything here is tested with this in mind.

**What we measured:** on Instagram, a business account's friends are business accounts only
39% of the time. Business accounts are 36% of all users, so a business account's friends are
barely more "businessy" than a random crowd. **The network tells us almost nothing.** That makes
this a hard, realistic test.

## 3. The two datasets

| | Instagram | TwiBot-20 (Twitter) |
|---|---|---|
| Accounts | 11,339 | 11,826 |
| The "suspicious" group | Business, brand and media accounts (36%) | Bots (56%) |
| What we read | The bio | The bio + up to 20 tweets |
| Connections | 72,005 friendships | 11,890 "friend" + 5,018 "follow" links |

**Why not the Bitcoin dataset (Elliptic) we started with?** It has no real text, only numbers,
so a text-reading AI had nothing to read.

**One rule we chose on purpose.** Twitter also gives profile facts such as "verified ✓" and
follower counts. "Verified" alone catches bots very well, because almost no bot is verified. But
that is a **shortcut**: it tells us nothing about reading text or using the network, which is
what we study. So our models **only use text and connections**. We report the shortcut
separately, so readers can see how strong it is.

## 4. Two versions of each test

- **Setting A, "as it is":** the data with its natural mix of normal and suspicious accounts.
- **Setting B, "rare fraud":** we keep only 1 suspicious account for every 10 normal ones. That
  is closer to real life, where fraud is rare, and it is much harder.

## 5. How we make the comparison fair (the exam analogy)

Every model is treated like a student sitting the same exam:

- **Practice questions (training set, 60%):** the model learns from these.
- **Mock exam (validation set, 20%):** used to make choices, such as how strict the model should be.
- **Final exam (test set, 20%):** used **once**, at the very end, for the reported score. Nobody may peek.

All models get **the same three sets** of accounts, so no model gets an easier exam.

**How strict to be.** Each model gives every account a "suspicion score" from 0 to 1. Someone
must decide the cut-off: above 0.5? Above 0.3? We pick the cut-off that works best on the
**mock exam**, then use it unchanged on the final exam.

**Why we don't use plain accuracy.** In Setting B, a lazy model that says "everyone is normal"
is right **91%** of the time and catches **zero** fraud. So our main score is **macro-F1**, which
gives equal weight to "how well do you catch suspicious accounts" and "how well do you clear
normal ones". The lazy model scores badly on it, as it should. (0.5 is roughly guessing; 1.0 is perfect.)

We also report separately how models do on accounts **with** connections and accounts **with
no connections at all**, because the network can only help the first group.

## 6. The six steps, as a story

### Step 1: Look at the data
We count accounts and connections and measure how much friends resemble each other (section 2).
The result: barely more than random.

### Step 2: The simplest possible guess
Before any AI, we try a basic **word-counting** model: which words appear more in business or
bot accounts ("official", "shop", "news"…)? This is **TF-IDF + logistic regression**. It sets
the bar: a fancy AI that can't beat word-counting isn't worth its cost.

### Step 3: Let an AI read the text
An AI model reads each bio (and tweets) and turns it into a list of numbers that captures its
meaning, a kind of **"meaning fingerprint"** (called an **embedding**). Similar texts get
similar fingerprints. A simple classifier then learns which fingerprints look suspicious.

We tried **six AI readers**, from small (22 million parameters, the "knobs" inside the model)
to large (1.5 billion). These include a model trained on tweets and one that reads 100
languages (**multilingual-E5**; please always use that full name). Everything else stayed the
same, so any difference comes from the reader.

### Step 4: Add the network
Now we give the classifier the connections too, using two kinds of graph model:

- **GCN** = *blending your opinion with your friends' average.* Each account's fingerprint is
  mixed with its friends'. That's bad news when a bot's friends are mostly real people: the bot
  gets blended into looking normal.
- **GraphSAGE (SAGE)** = *listening to yourself and to your friends separately.* The model can
  learn "trust the account's own text more than its friends".

We also run two "control" versions: the **same model with no connections** (MLP) and a model
with **only connections and no text**. These show exactly what the network adds.

### Step 5: Ask a chatbot directly
Instead of fingerprints, we give a chatbot-style AI (Qwen) a written description and ask:
*"Is this account a bot? Answer Yes or No."* We test three versions of the description:

1. Only the account's own text.
2. The account's text + its 8 most similar friends' full texts.
3. The account's text + **one-line summaries** of those 8 friends. This idea comes from a
   paper called **DGP**: show the main person in full detail and the friends as quick summaries,
   so the friends don't drown out the main person.

The chatbot has **never seen our labels** (this is called **zero-shot**). It only uses general
knowledge. We read how strongly it leans towards "Yes" as its suspicion score.

### Step 6: Teach the chatbot with examples (in progress)
Step 5's chatbot was guessing from general knowledge. In step 6 we **train it on 1,300 labelled
examples**, as the DGP paper does, and repeat the test.

- **LoRA**: rather than retraining the whole AI (1.7 billion knobs, far too big for a laptop),
  we attach a small set of **extra adjustable knobs** (about 0.4% extra) and train only those.
  Think of sticky notes added to a textbook instead of rewriting the book.
- **Smarter choice of friends ("diffusion distance")**: we keep the friends who **move in the
  same circles** as the account (similar text, spread along the network), up to 8 per
  connection type.
- **Two-level summaries**: each friend is summarised in about 10 words, then each group of
  friends ("friends", "followers", "friends of friends") is summarised again in about 10 words.
- **The key test**: we train the same AI twice, **with** and **without** the friend summaries.
  If the "with" version wins, the network helps once the AI is trained. If not, heterophily
  means the friends are just noise.

Compared with the paper, we use a smaller AI (Qwen3-1.7B instead of 8B) because a 16 GB
laptop can't train the big one.

## 7. What we've found so far (in plain words)

1. **A better AI reader is the biggest win.** Swapping word-counting for a good AI reader
   improves results more than anything the network adds. (Twitter: 0.70 → 0.77 macro-F1.)
2. **Blending with friends (GCN) makes things worse.** It happens on Twitter for every AI
   reader, because bots' friends are mostly real people. Listening separately (SAGE) is much safer.
3. **The network alone is weak:** 0.58 on Twitter, and close to guessing when fraud is rare.
4. **The untrained chatbot loses clearly** to a simple classifier trained on our data
   (Twitter: 0.58 vs 0.75). Showing it the friends' summaries doesn't help.
5. **Bigger isn't always better.** A small model trained on tweets beat a model five times its size.
6. **Still open (step 6):** does *training* the chatbot close the gap, and do friend summaries
   help once it is trained?

## 8. Running it yourself

You need Python with the packages in `my_build/requirements.txt`, and the data in `data/` (not
in this repo, see section 3). Then, from the project folder:

```bash
pip install -r my_build/requirements.txt

python my_build/step1_explore.py                    # step 1
SETTING=A python my_build/step2_baseline.py         # steps 2-5 work the same way;
SETTING=A python my_build/step3_embed.py            # use SETTING=B for the "rare fraud" version
SETTING=A python my_build/step4_gnn.py
SETTING=A python my_build/step5_llm_reason.py
python my_build/compare.py                          # prints every result in one table
```

For Twitter, put `DATASET=twibot20` in front, e.g. `DATASET=twibot20 SETTING=A python my_build/step4_gnn.py`.

**Some steps take hours.** The slow parts are saved as they go, so if the computer stops, just
run the same command again and it carries on. For long runs on a Mac:
- start the command with `caffeinate -ims` (stops the Mac sleeping);
- keep the charger in and the **lid open** (locking the screen is fine);
- close Chrome;
- make sure no shutdown timer is set.

## 9. Where things are

| Path | What's there |
|---|---|
| `README.md` | One-page summary of the project |
| `REPORT.md` | Latest results, status and decisions (updated as we go) |
| `my_build/step1_explore.py` … `step6_dgp.py` | The six steps above |
| `my_build/common.py` | Shared rules: same exam for everyone, same scoring |
| `my_build/compare.py` | Prints all results side by side |
| `my_build/results/` | Every result, as small `.json` files |
| `my_build/study_notes.html` | Detailed study notes (open in a web browser) |
| `reports/` | Assessment 2 draft and the codebase study guide |

## 10. Working together

- Please **don't upload** the big saved "fingerprint" files (`*.npy`) or training checkpoints.
  The repo already blocks them.
- New experiments should use the shared rules in `common.py` (same splits, same scoring), so
  their numbers can be compared fairly with ours.
- If you change how the data is split or scored, every earlier number changes. Let's discuss first.
- Write new results and decisions in [REPORT.md](REPORT.md).

## 11. Glossary

| Word | Plain meaning |
|---|---|
| Graph | A network: accounts are dots, connections are lines |
| Heterophily | Connected accounts tend to be *different* (bots befriending real people) |
| Homophily | Connected accounts tend to be *similar* ("birds of a feather") |
| LLM | Large language model, a ChatGPT-style AI that reads and writes text |
| GNN | Graph neural network, an AI that learns from a network |
| Embedding | A "meaning fingerprint": a list of numbers that represents a text |
| GCN | A graph model that blends each account with its friends |
| GraphSAGE / SAGE | A graph model that keeps the account and its friends separate |
| MLP | The same kind of model with no network, used as a comparison |
| TF-IDF | Simple word counting that highlights unusual words |
| Zero-shot | Asking an AI to do a task it was never trained on |
| LoRA | Training a small add-on instead of the whole AI ("sticky notes") |
| DGP | A method from a 2026 paper: main account in full, friends as short summaries |
| Training / validation / test set | Practice questions / mock exam / final exam |
| Threshold | The cut-off score above which we call an account suspicious |
| Macro-F1 | Our main score; fair even when fraud is rare (1.0 = perfect) |
| AUROC | How well the scores rank suspicious above normal accounts (0.5 = random) |
| Parameters | The internal "knobs" an AI learns; more knobs = bigger model |
