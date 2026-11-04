"""
Step 5: LLM role (b) "reasoner". The LLM reads a prompt built from the graph and decides.

Run:  python my_build/step5_llm_reason.py     (needs step 3 embeddings; ~20-30 min first run)

Pipeline (Dual Granularity Prompting idea):
  1. Node-level summarisation: condense each neighbour's bio into <= B_NODE tokens with a
     TASK-AGNOSTIC prompt (no mention of the label), cached to results/summaries.json.
  2. Prompt the LLM zero-shot in three ways, to isolate what helps:
       target : target bio only                                (no graph)
       raw    : target bio + K neighbours' FULL bios           (fine + fine)
       dgp    : target bio + K neighbours' SUMMARIES           (fine + coarse)  <- DGP
  3. Score = P("Yes") vs P("No") from one forward pass. Threshold picked on a VAL sample.

Neighbour choice: the K neighbours most similar to the target (cosine on E5 embeddings).
On a heterophilous graph this filters out the most dissimilar (likely other-class) neighbours.

Label meaning is INFERRED from the data (class-1 words: we, official, products, magazine,
news, click...), so the prompt asks "business/brand account vs personal account".
TwiBot-20 (DATASET=twibot20): the question becomes "bot or human?", the text is bio + tweets,
and in the raw variant each neighbour's text is cut to RAW_CHARS so the prompt stays short
(a full neighbour is up to 2,000 characters; 8 of them would not fit comfortably).
LLMs are slow, so we evaluate on a stratified sample of the test set, and compare against
the step 3/4 models on exactly the same sample.
"""
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split

from common import (load, load_embeddings, original_ids, get_splits, isolated_mask, evaluate,
                    best_threshold, RESULTS_DIR, SHARED_DIR, SEED, SETTING, DATASET, TWITTER)
from step3_embed import make_clf, tune_C, DEVICE

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# LLM evaluation samples (stratified). Setting B needs more, or class 1 would be ~27 users.
N_TEST, N_VAL = (300, 150) if SETTING == "A" else (600, 300)
if DATASET == "twibot20":
    # B's whole official test set is 597 users (54 bots), so B uses all of it.
    N_TEST, N_VAL = (400, 200) if SETTING == "A" else (597, 400)
if DATASET == "twibot22":
    # H7 (5 Nov): validation checks on a fixed 2,000-user stratified subset (VAL_CHECK to change it)
    N_TEST, N_VAL = 400, int(os.environ.get("VAL_CHECK", 2000))
K_NEIGH = 8                  # neighbours shown in the prompt
B_NODE = 32                  # token budget per neighbour summary
SUMMARY_FILE = SHARED_DIR / "summaries.json"     # keyed by ORIGINAL user id, shared by A and B

SUMMARY_PROMPT = "Summarise the following text in at most 25 words.\n\nText: {text}"
QUESTION = ("Is this Instagram account a business, brand or media account "
            "(rather than a personal account)? Answer Yes or No.")
TARGET_LABEL, NBR_LABEL, EMPTY, RAW_CHARS = "Target account bio", "Bios", "(empty bio)", None
if TWITTER:
    QUESTION = ("Is this Twitter account a bot (an automated account) rather than a human? "
                "Answer Yes or No.")
    TARGET_LABEL, NBR_LABEL, EMPTY, RAW_CHARS = ("Target account (bio, then recent tweets)",
                                                 "Texts", "(no text)", 400)
SUMMARY_CHARS = 1000 if TWITTER else None


def sample(idx, y, n):
    # Stratified subsample so class balance matches the full split.
    if n >= len(idx):
        return np.sort(idx)
    keep, _ = train_test_split(idx, train_size=n, stratify=y[idx], random_state=SEED)
    return np.sort(keep)


def pick_neighbours(nodes, edge_index, emb):
    # For each node: its K most similar neighbours by cosine similarity (emb is unit-length).
    src, dst = edge_index
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    starts = np.searchsorted(src, np.arange(len(emb) + 1))
    chosen = {}
    for v in nodes:
        nbrs = dst[starts[v]:starts[v + 1]]
        sims = emb[nbrs] @ emb[v]
        chosen[int(v)] = nbrs[np.argsort(-sims)[:K_NEIGH]].tolist()
    return chosen


def load_llm():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")  # left pad for generation
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(DEVICE).eval()
    return tok, model


def chat(tok, user_msg):
    return tok.apply_chat_template([{"role": "user", "content": user_msg}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def summarise(ids, x_text, tok, model, batch_size=16 if DATASET == "instagram" else 4):
    # Node-level summarisation with a fixed token budget, cached across runs.
    # Returns {node id in this setting: summary}; the cache file uses original ids.
    orig = original_ids()
    key = lambda i: str(orig[i])
    cache = json.load(open(SUMMARY_FILE)) if SUMMARY_FILE.exists() else {}
    todo = [i for i in ids if key(i) not in cache]
    for i in todo:
        if not x_text[i].strip():
            cache[key(i)] = EMPTY
    todo = [i for i in todo if key(i) not in cache]

    for b in tqdm(range(0, len(todo), batch_size), desc="summarising"):
        chunk = todo[b:b + batch_size]
        # TwiBot-20: summarise the first SUMMARY_CHARS characters (bio + first tweets), to fit memory
        prompts = [chat(tok, SUMMARY_PROMPT.format(text=x_text[i][:SUMMARY_CHARS])) for i in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
        out = model.generate(**enc, max_new_tokens=B_NODE, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, s in zip(chunk, texts):
            cache[key(i)] = s.strip().replace("\n", " ")
        if b % (batch_size * 20) == 0:                     # save progress now and then
            json.dump(cache, open(SUMMARY_FILE, "w"))
    json.dump(cache, open(SUMMARY_FILE, "w"))
    return {i: cache[key(i)] for i in ids}


def build_prompt(tok, v, variant, x_text, neighbours, summaries):
    bio = x_text[v].strip() or EMPTY
    msg = f"{TARGET_LABEL}:\n{bio}\n"
    if variant != "target":
        nbrs = neighbours[v]
        if not nbrs:
            msg += "\nThis account has no connections.\n"
        else:
            msg += f"\n{NBR_LABEL} of {len(nbrs)} connected accounts"
            msg += " (summarised):\n" if variant == "dgp" else ":\n"
            for j in nbrs:
                t = summaries[j] if variant == "dgp" else (x_text[j].strip()[:RAW_CHARS] or EMPTY)
                msg += "- " + t.replace("\n", " ") + "\n"
    return chat(tok, msg + "\n" + QUESTION)


@torch.no_grad()
def score(prompts, tok, model, batch_size=8 if DATASET == "instagram" else 4):
    # P(Yes) from the logits of the first answer token. One forward pass, no generation.
    yes, no = tok.encode("Yes")[0], tok.encode("No")[0]
    probs = []
    for b in tqdm(range(0, len(prompts), batch_size), desc="scoring", leave=False):
        enc = tok(prompts[b:b + batch_size], return_tensors="pt", padding=True).to(DEVICE)
        logits = model(**enc).logits[:, -1, [yes, no]].float()
        probs.append(logits.softmax(-1)[:, 0].cpu().numpy())
    return np.concatenate(probs)


def same_sample_baselines(y, train, val_s, test_s, edge_index, iso_test):
    # Qwen + LR and Qwen + GCN from steps 3/4, scored on the SAME test sample (fair comparison).
    from step4_gnn import train_one
    from torch_geometric.nn import GCNConv
    emb = load_embeddings("qwen").astype(np.float32)
    out = []

    val = get_splits(y)[1]
    clf = make_clf(tune_C(emb, y, train, val)).fit(emb[train], y[train])
    t = best_threshold(y[val], clf.predict_proba(emb[val])[:, 1])
    p = clf.predict_proba(emb[test_s])[:, 1]
    out.append(evaluate("qwen+LR (same sample)", y[test_s], (p > t).astype(int), p, iso_test))

    mu, sd = emb[train].mean(0), emb[train].std(0) + 1e-6
    x = torch.as_tensor((emb - mu) / sd)
    prob = train_one(GCNConv, x, torch.as_tensor(edge_index), torch.as_tensor(y),
                     torch.as_tensor(train), torch.as_tensor(val), seed=0)
    t = best_threshold(y[val], prob[val])
    out.append(evaluate("qwen+GCN (same sample)", y[test_s], (prob[test_s] > t).astype(int),
                        prob[test_s], iso_test))
    return out


def main():
    x_text, edge_index, y = load()
    train, val, test = get_splits(y)
    val_s, test_s = sample(val, y, N_VAL), sample(test, y, N_TEST)
    iso_test = isolated_mask(edge_index, len(y))[test_s]
    print(f"Setting {SETTING}. LLM samples: val {len(val_s)}, test {len(test_s)} (isolated in test: {iso_test.sum()})")

    neighbours = pick_neighbours(np.concatenate([val_s, test_s]), edge_index,
                                 load_embeddings("e5"))
    to_summarise = sorted({j for nbrs in neighbours.values() for j in nbrs})
    print(f"Neighbours to summarise: {len(to_summarise)}")

    tok, model = load_llm()
    summaries = summarise(to_summarise, x_text, tok, model)

    # Show one example so you can SEE what the LLM is given.
    ex = next(v for v in test_s if neighbours[v])
    print("\n--- example DGP prompt ---\n" + build_prompt(tok, ex, "dgp", x_text, neighbours, summaries))

    results, probs = [], {}
    for variant in ["target", "raw", "dgp"]:
        p_val = score([build_prompt(tok, v, variant, x_text, neighbours, summaries) for v in val_s],
                      tok, model)
        p_test = score([build_prompt(tok, v, variant, x_text, neighbours, summaries) for v in test_s],
                       tok, model)
        t = best_threshold(y[val_s], p_val)
        results.append(evaluate(f"LLM {variant}", y[test_s], (p_test > t).astype(int),
                                p_test, iso_test))
        probs[variant] = p_test

    del model
    print("\nSame test sample, trained models:")
    results += same_sample_baselines(y, train, val_s, test_s, edge_index, iso_test)

    json.dump(results, open(RESULTS_DIR / "step5_llm.json", "w"), indent=2)
    np.savez(RESULTS_DIR / "step5_probs.npz", test_nodes=test_s, **probs)


if __name__ == "__main__":
    main()
