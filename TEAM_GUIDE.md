# Team Guide

This guide explains the whole `my_build` pipeline for the team: what each part does, why it was
set up that way, how to run it, and how to read the results. You don't need to have read the
code first. For the latest numbers and status, see [REPORT.md](REPORT.md).

## 1. The question

The topic brief is "LLM-based approaches for graph fraud detection". GNNs mostly ignore node
text, while LLMs read text well but don't see the graph. The supervisor added one requirement:
**evaluate under heterophily**. Fraudsters camouflage themselves by connecting to normal users,
so "neighbours share your label" is often false. That assumption is exactly what GCN-style
models rely on.

So the question is: **when the graph is heterophilous, how much does an LLM help, and in which role?**

| LLM role | What happens | Where |
|---|---|---|
| Encoder | The LLM turns text into a vector; a classifier or GNN decides | Steps 3, 4 |
| Zero-shot reasoner | The LLM reads a prompt about the user and their neighbours and answers Yes/No | Step 5 |
| Fine-tuned reasoner | The same prompts, with the LLM trained on our labels with LoRA | Step 6 |

## 2. Why not Elliptic

The team's first pipeline used the Elliptic Bitcoin graph. It has **no natural text**: the
"text" had to be generated from numeric features, so an LLM has nothing real to read. The
rebuild uses two text-attributed graphs instead.

## 3. Datasets

**Instagram.** 11,339 users with bios and 72,005 undirected connections.
- Class 1 (36.3%) are commercial accounts (business, brand, media). This is inferred from their
  most telling words ("we", "official", "products", "news"); the dataset does not name its classes.
- Commercial accounts play the role of the fraud-like class.

**TwiBot-20.** 11,826 labelled Twitter users, 55.7% of them bots.
- Text is the bio plus up to 20 tweets (capped at 2,000 characters).
- Two edge types between labelled users: friend (11,890) and follow (5,018).
- We use the official split.

**No metadata.** TwiBot-20 also has profile numbers (verified, follower counts and so on).
"Verified" alone scores macro-F1 0.803, because only 1 of 6,589 bots is verified. That is a
shortcut that says nothing about text or graph, so all models use **text + graph only**. The
shortcut is reported as a reference line (`twibot_shortcut.py`).

**Getting the data.** The data is not in the repo. Put Instagram in `data/instagram/`
(`x_text.npy`, `edge_index.npy`, `y.npy`). For TwiBot-20, request the dataset from its authors,
then run `python my_build/prep_twibot20.py path/to/TwiBot-20-Format22.zip`.

## 4. Two settings

| | Setting A | Setting B |
|---|---|---|
| Class ratio | Natural | 1:10. All class 0 kept, plus a random 10% as many class 1 users |
| Graph | All edges | Induced subgraph (only edges between kept users) |
| Why | The data as it is | Rare fraud, as in real systems (taken from the FLAG paper) |

Choose them with environment variables: `SETTING=A|B` and `DATASET=instagram|twibot20`.

## 5. Heterophily in one paragraph

Edge homophily (the share of edges joining same-label users) is **misleading** under class
imbalance. On Instagram it is 0.586, but class 1's neighbours are class 1 only 39.2% of the
time, against a 36.3% chance level. The honest measure is **class-insensitive homophily**
(Lim et al. 2021), which subtracts chance: 0.078 on Instagram, almost random. In Setting B,
edge homophily jumps to 0.86 while the fraud class gets *more* heterophilous. Step 1 computes
all four measures.

## 6. Evaluation rules (every step follows them)

These live in `common.py`, so every model is compared on equal terms.

1. **Fixed splits.** They are made once and saved to `results/.../splits.npz`: stratified 60/20/20 for Instagram, the official split for TwiBot-20.
2. **Train on train only.** Vectorisers and scalers are fitted on training data; there is no test leakage.
3. **Validation chooses, test reports.** Hyperparameters, the epoch and the **decision threshold** are all picked on validation. `best_threshold()` takes the cut-off with the best validation macro-F1. Test is scored once.
4. **One `evaluate()`** reports:
   - macro-F1 (the main metric; accuracy is misleading when "always predict normal" scores 64–91%);
   - class-1 F1;
   - AUROC;
   - PR-AUC (a random model scores the class-1 share);
   - macro-F1 on **connected** and **isolated** test users separately.
5. **GNNs use 3 seeds** and report mean ± std. LLM steps use one seed because of cost; their output says so.

## 7. The steps

| Step | File | What it does | Main lesson |
|---|---|---|---|
| 1 | `step1_explore.py` | Statistics, four homophily measures, node-homophily plot | The graph knows almost nothing about class 1 |
| 2 | `step2_baseline.py` | Majority class; TF-IDF + logistic regression | The bar to beat (Instagram ~0.58, TwiBot-20 ~0.70) |
| 3 | `step3_embed.py` | Six LLM encoders + logistic regression | Better encoders give the biggest gains |
| 4 | `step4_gnn.py` | MLP (no edges), GCN, GraphSAGE on each encoder, plus graph-only | GCN hurts under heterophily; SAGE holds up better |
| 5 | `step5_llm_reason.py` | Zero-shot Qwen2.5-1.5B-Instruct: target / raw neighbours / summaries | Zero-shot loses to trained models; summaries don't help |
| 6 | `step6_dgp.py` | DGP replication: LoRA fine-tuning of Qwen3-1.7B | In progress |
| – | `compare.py` | Prints every result for both settings | |

**Step 3, the encoders**, in order of size: MiniLM (22M), E5-base English (110M),
Twitter-RoBERTa (125M), multilingual-E5 (278M, of which 192M is vocabulary), Qwen3-Embedding
(0.6B) and Qwen2.5-1.5B. Please write **"multilingual-E5"**, not plain "E5", so it isn't
confused with the English model. Decoder models are mean-pooled. On TwiBot-20, each tweet is
encoded separately and the vectors are averaged per user, so the whole text is used.

**Step 4, GCN vs SAGE.** GCN averages a node *together with* its neighbours. SAGE keeps a
separate weight for the node itself. Under heterophily, GCN blurs fraud users into their normal
neighbours. The **MLP** (same network, no edges) and **graph-only** (degree features, no text)
controls make the comparison clean.

**Step 5, DGP-style prompting.**
- The prompt holds the target's full text plus its 8 most similar neighbours (by multilingual-E5 cosine similarity), either raw or summarised in about 25 words.
- The score is P(Yes) from the Yes/No logits.
- Run on a stratified test sample (400 users for TwiBot-20 A), with trained baselines re-scored on the same sample.

**Step 6, the DGP replication** (Li et al., AAAI 2026, [arXiv:2507.21653](https://arxiv.org/abs/2507.21653)).
1. **Metapaths.** TwiBot-20 uses friend, follow and friend-of-friend; Instagram uses 1-hop and 2-hop. Each relation is treated as undirected.
2. **Diffusion-based trimming.** h = Z·X, with Z = 1/K Σₖ Tᵏ and T = D⁻¹A. The top 8 neighbours by distance are kept. X = multilingual-E5 embeddings.
3. **Two levels of summaries.** Each neighbour is summarised in 10 tokens; each metapath gets a 10-token summary of those summaries. The prompts are task-agnostic.
4. **Fine-tuning.** Plain cross-entropy on the Yes/No token, LoRA (r=16) on the attention layers, 1,300 training users (the paper's size), and early stopping on validation loss.
5. **Two variants.** `VARIANT=target` (no neighbours) vs `VARIANT=dgp`, to test whether neighbours help once the model is trained.
6. **Differences from the paper:**
   - Qwen3-1.7B instead of 8B (16 GB Mac);
   - no numeric summaries (metadata excluded);
   - fixed hyperparameters instead of a grid;
   - one seed.

## 8. Running it

```bash
pip install -r my_build/requirements.txt

python my_build/step1_explore.py
for S in A B; do
  SETTING=$S python my_build/step2_baseline.py
  SETTING=$S python my_build/step3_embed.py       # slow the first time; embeddings cached
  SETTING=$S python my_build/step4_gnn.py
  SETTING=$S python my_build/step5_llm_reason.py  # slow; summaries cached
done
python my_build/compare.py

# TwiBot-20: add DATASET=twibot20 to each command, e.g.
DATASET=twibot20 SETTING=A python my_build/step4_gnn.py

# Step 6 (hours; resumes automatically after an interruption)
caffeinate -ims env DATASET=twibot20 SETTING=A VARIANT=dgp python -u my_build/step6_dgp.py
DATASET=twibot20 SETTING=A VARIANT=target PROBE=1 python my_build/step6_dgp.py   # 20-step speed test
```

**Long runs on a laptop:**
- Run under `caffeinate` and keep the charger in.
- **Keep the lid open.** Closing it sleeps the Mac; locking the screen is fine.
- Close Chrome (and ideally VS Code). Swap slowed one overnight run 5–10×.
- Make sure no shutdown timer is set: `pgrep -l shutdown` should print nothing.

## 9. Where things are

```
my_build/
  common.py, step1..6, compare.py      the pipeline
  prep_twibot20.py, twibot_shortcut.py TwiBot-20 preparation and the metadata reference line
  study_notes.html                     study notes for every concept (open in a browser)
  results/
    A/, B/                             Instagram results per setting (*.json), splits.npz
    twibot20/A/, twibot20/B/           TwiBot-20 results per setting
    summaries*.json                    cached LLM neighbour summaries
    emb_*.npy                          cached embeddings (NOT in git, ~400 MB)
reports/                               Assessment 2 draft, codebase study guide
REPORT.md                              living progress report
```

Each result file is a list of rows with the same keys (`model`, `macro_f1`, `f1_class1`,
`auroc`, `pr_auc`, `macro_f1_connected`, `macro_f1_isolated`, plus `_std` for GNNs).
`compare.py` prints them all.

**Reading `compare.py`:** step 5 and step 6 rows are scored on a test **sample**. Compare them
with the "(same sample)" rows next to them, not with the full-test rows above.

## 10. Working together

- Don't commit `*.npy` embeddings or step 6 checkpoints; `.gitignore` already blocks them.
- New experiments should use `load()`, `get_splits()`, `best_threshold()` and `evaluate()` from `common.py`, and write a `results/<setting>/<name>.json` file. Add it to `FILES` in `compare.py`.
- If you change a split or the threshold rule, every earlier number changes too. Discuss it first.
- Log decisions and results in [REPORT.md](REPORT.md).

## 11. Glossary

- **Heterophily:** connected nodes tend to have different labels.
- **Macro-F1:** F1 for each class, averaged equally.
- **AUROC:** the probability that a random positive scores above a random negative.
- **PR-AUC:** area under the precision–recall curve.
- **Mean pooling:** averaging the token vectors into one text vector.
- **Zero-shot:** no training on our labels.
- **LoRA:** training small low-rank add-on matrices instead of the whole LLM.
- **Metapath:** a sequence of edge types, e.g. friend-of-friend.
- **Diffusion distance:** how differently two nodes look after features are spread along random walks.
- **Induced subgraph:** kept nodes plus only the edges between them.
