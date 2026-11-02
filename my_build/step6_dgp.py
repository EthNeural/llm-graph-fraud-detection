"""
Step 6: DGP replication. The LLM itself is the classifier, fine-tuned with LoRA on our labels.

Paper: Li, Hu, Hooi, He, Chen. "DGP: A Dual-Granularity Prompting Framework for Fraud Detection
with Graph-Enhanced LLMs", AAAI 2026 (arXiv 2507.21653). Their code is not public yet, so this
follows the equations in the paper.

Run (one script for every run; each phase is cached, so re-running resumes where it stopped):
  caffeinate -ims env DATASET=twibot20 SETTING=A VARIANT=dgp python -u my_build/step6_dgp.py
  ... VARIANT=target ...                      the same LLM fine-tuned on the target's text alone
  ... PROBE=1 ...                             20 training steps only: prints speed and memory, saves nothing

Pipeline (paper section in brackets):
  1. Metapaths [3.2]  TwiBot-20 has two edge types, friend and follow, so the metapaths are
     friend, follow and friend-of-friend. Instagram has one edge type: 1-hop and 2-hop.
  2. Diffusion-based trimming [4.2, eq. 6-9]  T = D^-1 A_P,  Z = 1/K sum_{k=0..K} T^k,  h = Z X,
     keep the M neighbours in N_P(v) with the smallest ||h_u - h_v||.
     X = multilingual-E5 embeddings (the paper says "raw node features"; ours are text).
  3. Node summaries [eq. 5]  task-agnostic, "Summarize the text within 10 tokens", frozen LLM.
  4. Metapath summaries [eq. 10]  the M node summaries joined, then summarised again (10 tokens).
  5. Prompt [eq. 12]  target's full text + one summary line per metapath + the step 5 question.
     No numerical summaries: the only numbers we have are the TwiBot-20 metadata we exclude
     (report 4.6), so this is the paper's "w/o NumSumm" variant.
  6. Fine-tune [eq. 13]  cross-entropy over the full vocabulary on the first answer token
     ("Yes"/"No"), no class weights, LoRA on all attention projections, AdamW, batch size 4,
     up to 10 epochs, early stopping on validation loss.
  7. Score [eq. 14]  P(Yes) = softmax(logit_Yes, logit_No), exactly as in step 5.

Laptop-sized choices (the paper used Qwen3-8B on 4x A100):
  Qwen3-1.7B (same family), 1,300 training users (the paper's train size, stratified sample of
  our train split), fixed hyperparameters instead of the paper's grid, one seed.

Fair-evaluation rules, as in every step: saved splits; checkpoint and threshold chosen on
validation (best_threshold); test scored once; reported through evaluate() on the full test
set AND on the step 5 test sample, next to zero-shot and Qwen2.5 + LR / MLP / GCN on that sample.
"""
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from common import (load, load_embeddings, original_ids, get_splits, isolated_mask, evaluate,
                    best_threshold, RESULTS_DIR, SHARED_DIR, DATA_DIR, SEED, SETTING, DATASET)
from step3_embed import DEVICE
from step5_llm_reason import (sample, same_sample_baselines, QUESTION, TARGET_LABEL, EMPTY,
                              N_TEST, N_VAL, SUMMARY_CHARS)

MODEL_ID = os.environ.get("MODEL", "Qwen/Qwen3-1.7B")
VARIANT = os.environ.get("VARIANT", "dgp")
assert VARIANT in ("target", "dgp"), "VARIANT must be target or dgp"
PROBE = os.environ.get("PROBE") == "1"
FORCE = os.environ.get("FORCE") == "1"           # run even if the estimate is over MAX_HOURS
FP32 = os.environ.get("FP32") == "1"             # fallback if fp16 training goes NaN

B = 10                       # B_node = B_meta, the paper's best summary budget
K_HOPS = 2                   # diffusion steps K
M = 8                        # neighbours kept per metapath
N_TRAIN = 1300               # labelled training users, as in the paper (Table 1)
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
LR = 1e-4
BATCH = 4
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 10))
PATIENCE = 2                 # evaluations (= 1 epoch) without a lower validation loss
EXPECTED_EPOCHS = 3          # used only for the time estimate; early stopping decides
SAVE_EVERY = 50              # optimizer steps between checkpoints
MAX_HOURS = 10
TARGET_TOKENS = 480 if DATASET == "twibot20" else 256

SHORT = MODEL_ID.split("/")[-1]
NAME = f"{SHORT} LoRA {VARIANT}"
TAG = f"{SHORT}_{VARIANT}"
CKPT = RESULTS_DIR / f"step6_{TAG}"
SUMMARY_PROMPT = "Summarize the text within {b} tokens.\n\nText: {text}"
NODE_SUMMARY_FILE = SHARED_DIR / f"summaries_b{B}_{SHORT}.json"      # keyed by original id, A and B
PATH_SUMMARY_FILE = RESULTS_DIR / f"metapath_summaries_b{B}_{SHORT}.json"   # per setting

if DATASET == "twibot20":
    METAPATHS = {"friend": ["friend"], "follow": ["follow"], "friend-of-friend": ["friend", "friend"]}
    PATH_TEXT = {"friend": "Friend links", "follow": "Follow links",
                 "friend-of-friend": "Friends of friends"}
else:
    METAPATHS = {"neighbour": ["link"], "2-hop": ["link", "link"]}
    PATH_TEXT = {"neighbour": "Connected accounts", "2-hop": "Accounts two steps away"}


# ---------------- 1-2. metapaths and diffusion-based trimming ----------------
def build_typed_edges(out_file, zip_path=Path.home() / "Downloads" / "TwiBot-20-Format22.zip"):
    # One-off: prep_twibot20.py merged friend and follow into one edge type. Re-read them typed,
    # directed as stored (source -> target), between labelled users only.
    ids = np.load(DATA_DIR / "ids.npy")
    index = {u: i for i, u in enumerate(ids)}
    import pandas as pd
    edges = {"friend": [], "follow": []}
    for ch in pd.read_csv(zipfile.ZipFile(zip_path).open("Twibot-20/edge.csv"), chunksize=5_000_000):
        ch = ch[ch.relation.isin(list(edges)) & ch.source_id.isin(index) & ch.target_id.isin(index)]
        for r, a, b in zip(ch.relation, ch.source_id, ch.target_id):
            if a != b:
                edges[r].append((index[a], index[b]))
    np.savez(out_file, **{r: np.array(sorted(set(e)), dtype=np.int64).reshape(-1, 2) for r, e in edges.items()})


def typed_relations():
    # Relation name -> (2, E) edges in the current setting's node numbering. Each relation is
    # stored both ways (undirected), as in steps 4-5 and like the paper's symmetric relations:
    # kept directed, only 38% of users have an outgoing friend link (93% have a neighbour when
    # undirected).
    if DATASET == "instagram":
        return {"link": load()[1]}
    f = DATA_DIR / "edge_typed.npz"
    if not f.exists():
        print("Reading friend/follow edge types from the TwiBot-20 zip (one-off, ~1 min)...")
        build_typed_edges(f)
    keep = original_ids()
    new_id = np.full(len(np.load(DATA_DIR / "y.npy")), -1)
    new_id[keep] = np.arange(len(keep))
    out = {}
    for r, e in np.load(f).items():
        e = new_id[e.T]
        e = e[:, (e >= 0).all(0)]
        out[r] = np.hstack([e, e[::-1]])
    return out


def metapath_adjacency(rel_edges, path, n):
    # A_P = A_r1 A_r2 ... (eq. 3), binarised, without the node itself (a 2-hop path returns to v).
    mats = [sp.csr_matrix((np.ones(rel_edges[r].shape[1]), rel_edges[r]), shape=(n, n)) for r in path]
    A = mats[0]
    for nxt in mats[1:]:
        A = A @ nxt
    A = (A > 0).astype(np.float32).tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def diffusion_embeddings(A, X):
    # h = Z(K) X with Z(K) = 1/K sum_{k=0..K} T^k and T = D^-1 A (eq. 6-7). K sparse products,
    # never forming T^k itself. Rows with no neighbours stay at X.
    deg = np.asarray(A.sum(1)).ravel()
    T = sp.diags(1.0 / np.maximum(deg, 1)) @ A
    h, term = X.copy(), X
    for _ in range(K_HOPS):
        term = T @ term
        h += term
    return h / K_HOPS


def trim_neighbours(targets, X, rel_edges, n):
    # For each metapath and target: the top-M neighbours by diffusion distance (eq. 8-9).
    chosen = {}
    for name, path in METAPATHS.items():
        A = metapath_adjacency(rel_edges, path, n)
        h = diffusion_embeddings(A, X)
        for v in targets:
            nbrs = A.indices[A.indptr[v]:A.indptr[v + 1]]
            d = np.linalg.norm(h[nbrs] - h[v], axis=1)
            chosen[(int(v), name)] = nbrs[np.argsort(d, kind="stable")[:M]].tolist()
        sizes = [len(chosen[(int(v), name)]) for v in targets]
        print(f"  metapath {name:18s} edges {A.nnz:>10,}   targets with neighbours "
              f"{np.mean(np.array(sizes) > 0):.1%}   mean kept {np.mean(sizes):.1f}")
    return chosen


# ---------------- 3-4. node and metapath summaries ----------------
def load_llm(dtype):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(DEVICE)
    return tok, model


def chat(tok, user_msg):
    # enable_thinking=False: Qwen3 answers straight away, so the FIRST token is the answer.
    return tok.apply_chat_template([{"role": "user", "content": user_msg}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def generate(tok, model, prompts, batch_size=4 if DATASET == "twibot20" else 16):
    # Greedy, at most B new tokens. Sorted by length so batches carry little padding.
    # Batch 4 on TwiBot-20 (as in step 5): at 16, the MPS memory cache grew to 16 GB and swapped.
    model.eval()
    order = np.argsort([len(p) for p in prompts])
    out = [None] * len(prompts)
    for b in range(0, len(prompts), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(DEVICE)
        gen = model.generate(**enc, max_new_tokens=B, do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            out[i] = t.strip().replace("\n", " ") or EMPTY
        if DEVICE == "mps":
            torch.mps.empty_cache()                            # every batch has a new shape
    return out


def cached_summaries(cache_file, items, make_prompt, tok, model, desc, chunk=320):
    # items: {cache key: source text or None}. None / empty text -> no LLM call.
    cache = json.load(open(cache_file)) if cache_file.exists() else {}
    for k, text in items.items():
        if k not in cache and not (text or "").strip():
            cache[k] = EMPTY if text is not None else "none"
    todo = [k for k in items if k not in cache]
    print(f"  {desc}: {len(items) - len(todo):,} cached, {len(todo):,} to generate")
    for b in tqdm(range(0, len(todo), chunk), desc=desc):
        keys = todo[b:b + chunk]
        for k, s in zip(keys, generate(tok, model, [make_prompt(items[k]) for k in keys])):
            cache[k] = s
        json.dump(cache, open(cache_file, "w"))                # save progress every chunk
    json.dump(cache, open(cache_file, "w"))
    return cache


# ---------------- 5. prompts ----------------
def fit_target(tok, text):
    # Keep the bio, then as many whole tweets (lines) as fit in TARGET_TOKENS.
    lines, kept, used = text.strip().split("\n"), [], 0
    for line in lines:
        n = len(tok(line, add_special_tokens=False)["input_ids"]) + 1
        if used + n > TARGET_TOKENS:
            if not kept:                                       # one very long first line: cut it
                ids = tok(line, add_special_tokens=False)["input_ids"][:TARGET_TOKENS]
                kept.append(tok.decode(ids))
            break
        kept.append(line)
        used += n
    return "\n".join(kept).strip() or EMPTY


def build_prompt(tok, v, x_text, path_sum, orig):
    msg = f"{TARGET_LABEL}:\n{fit_target(tok, x_text[v])}\n"
    if VARIANT == "dgp":
        msg += "\nSummaries of linked accounts:\n"
        for name in METAPATHS:
            msg += f"- {PATH_TEXT[name]}: {path_sum.get(f'{orig[v]}|{name}', 'none')}\n"
    return chat(tok, msg + "\n" + QUESTION)


# ---------------- 6-7. training and scoring ----------------
def pad_batch(seqs, pad_id):
    # Left padding, so the last position is every prompt's real last token. Qwen uses RoPE
    # (relative positions), so the shift from left padding does not change attention.
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s)
        mask[i, L - len(s):] = 1
    return ids.to(DEVICE), mask.to(DEVICE)


def last_logits(model, seqs, pad_id):
    ids, mask = pad_batch(seqs, pad_id)
    # logits_to_keep=1: only the last position's vocabulary logits (saves ~0.5 GB per batch)
    return model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1].float()


@torch.no_grad()
def score(model, seqs, pad_id, yes, no, labels=None, batch_size=4, desc="scoring"):
    # P(Yes) for every prompt (eq. 14); with labels, also the mean training loss (eq. 13).
    model.eval()
    order = np.argsort([len(s) for s in seqs])
    p, loss = np.empty(len(seqs)), 0.0
    for b in tqdm(range(0, len(seqs), batch_size), desc=desc, leave=False):
        idx = order[b:b + batch_size]
        logits = last_logits(model, [seqs[i] for i in idx], pad_id)
        p[idx] = logits[:, [yes, no]].softmax(-1)[:, 0].cpu().numpy()
        if labels is not None:
            tgt = torch.tensor([yes if labels[i] else no for i in idx], device=DEVICE)
            loss += F.cross_entropy(logits, tgt, reduction="sum").item()
    return (p, loss / len(seqs)) if labels is not None else p


def make_batches(lengths, epoch):
    # Shuffle each epoch (seeded by the epoch, so a resumed run continues the same order), sort
    # within chunks of 100 by length to cut padding, then shuffle the batch order.
    rng = np.random.default_rng(SEED + epoch)
    perm = rng.permutation(len(lengths))
    batches = []
    for c in range(0, len(perm), 100):
        chunk = sorted(perm[c:c + 100], key=lambda i: lengths[i])
        batches += [chunk[b:b + BATCH] for b in range(0, len(chunk), BATCH)]
    return [batches[i] for i in rng.permutation(len(batches))]


def adapter_state(model):
    from peft import get_peft_model_state_dict
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def fine_tune(model, seqs, y_tr, val_seqs, y_val, pad_id, yes, no, probe=False):
    from peft import set_peft_model_state_dict
    CKPT.mkdir(parents=True, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    st = dict(epoch=0, batch=0, step=0, best_loss=float("inf"), wait=0, history=[], done=False)
    if not probe and (CKPT / "state.pt").exists():
        saved = torch.load(CKPT / "state.pt", weights_only=False)
        st = saved["state"]
        set_peft_model_state_dict(model, saved["adapter"])
        opt.load_state_dict(saved["opt"])
        print(f"Resumed: epoch {st['epoch'] + 1}, batch {st['batch']}, step {st['step']}, "
              f"best val loss {st['best_loss']:.4f}")
    if st["done"]:
        return

    lengths = [len(s) for s in seqs]
    steps_per_epoch = math.ceil(len(seqs) / BATCH)
    evals = {steps_per_epoch // 2, steps_per_epoch}            # twice per epoch
    t0, timed = time.time(), 0

    def save():
        torch.save({"state": st, "adapter": adapter_state(model), "opt": opt.state_dict()},
                   CKPT / "state.tmp")
        os.replace(CKPT / "state.tmp", CKPT / "state.pt")      # never leave a half-written file

    while st["epoch"] < MAX_EPOCHS:
        batches = make_batches(lengths, st["epoch"])
        bar = tqdm(total=steps_per_epoch, initial=st["batch"], desc=f"epoch {st['epoch'] + 1}")
        while st["batch"] < steps_per_epoch:
            idx = batches[st["batch"]]
            model.train()
            logits = last_logits(model, [seqs[i] for i in idx], pad_id)
            tgt = torch.tensor([yes if y_tr[i] else no for i in idx], device=DEVICE)
            loss = F.cross_entropy(logits, tgt)                # eq. 13: full vocabulary, no weights
            if not torch.isfinite(loss):
                sys.exit("Loss is NaN/inf. Re-run with FP32=1 (fp32 base model, ~2x slower).")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            st["batch"] += 1
            st["step"] += 1
            timed += 1
            bar.update(1)
            bar.set_postfix(loss=f"{loss.item():.3f}")

            if timed == 20:
                per_step = (time.time() - t0) / 20
                epoch_h = per_step * steps_per_epoch / 3600
                # eval: forward only (~1/3 of a training step per example)
                eval_h = (2 * len(val_seqs) / BATCH) * per_step / 3 / 3600
                mem = torch.mps.driver_allocated_memory() / 2**30 if DEVICE == "mps" else 0
                expected = EXPECTED_EPOCHS * (epoch_h + eval_h)
                print(f"\n  {per_step:.1f} s/step -> {epoch_h:.1f} h per epoch (+{eval_h:.2f} h val checks). "
                      f"Expected ~{expected:.1f} h for {EXPECTED_EPOCHS} epochs; worst case "
                      f"{MAX_EPOCHS * (epoch_h + eval_h):.1f} h for {MAX_EPOCHS}. GPU memory {mem:.1f} GB")
                if probe:
                    return per_step
                if expected > MAX_HOURS and not FORCE:
                    save()
                    sys.exit(f"Estimated {expected:.1f} h > {MAX_HOURS} h. Checkpoint saved; stopping "
                             "to ask. Re-run with FORCE=1 to continue (it resumes from here).")
            if st["step"] % SAVE_EVERY == 0:
                save()
            if st["batch"] in evals:
                p, vloss = score(model, val_seqs, pad_id, yes, no, labels=y_val, desc="val check")
                from sklearn.metrics import roc_auc_score
                st["history"].append(dict(epoch=st["epoch"] + st["batch"] / steps_per_epoch,
                                          val_loss=round(vloss, 4),
                                          val_auroc=round(roc_auc_score(y_val, p), 4)))
                print(f"\n  val check {st['history'][-1]}")
                if vloss < st["best_loss"]:
                    st["best_loss"], st["wait"] = vloss, 0
                    torch.save(adapter_state(model), CKPT / "best_adapter.pt")
                else:
                    st["wait"] += 1
                save()
                if st["wait"] >= PATIENCE:
                    print(f"  early stopping: no lower validation loss for {PATIENCE} checks")
                    st["done"] = True
                    save()
                    return
        bar.close()
        st["epoch"] += 1
        st["batch"] = 0
        save()
    st["done"] = True
    save()


# ---------------- comparison rows ----------------
def mlp_same_sample(y, train, val, test_s, edge_index, iso_test):
    # Qwen2.5 + MLP (step 4's text-only control), scored on the step 5 test sample.
    from step4_gnn import train_one, NoGraph
    emb = load_embeddings("qwen").astype(np.float32)
    mu, sd = emb[train].mean(0), emb[train].std(0) + 1e-6
    prob = train_one(NoGraph, torch.as_tensor((emb - mu) / sd), torch.as_tensor(edge_index),
                     torch.as_tensor(y), torch.as_tensor(train), torch.as_tensor(val), seed=0)
    t = best_threshold(y[val], prob[val])
    return evaluate("qwen+MLP (same sample)", y[test_s], (prob[test_s] > t).astype(int),
                    prob[test_s], iso_test)


def save_rows(rows):
    # Merge into step6_lora.json, replacing rows of the same name (target and dgp runs add up).
    f = RESULTS_DIR / "step6_lora.json"
    old = json.load(open(f)) if f.exists() else []
    names = {r["model"] for r in rows}
    json.dump([r for r in old if r["model"] not in names] + rows, open(f, "w"), indent=2)
    print(f"\nSaved -> {f}")


def main():
    print(f"Step 6 | {DATASET} setting {SETTING} | {NAME} | 1 seed (seed {SEED})")
    print("Before a long run: caffeinate, charger in, Chrome and VS Code closed, and no shutdown "
          "timer (pgrep -l shutdown prints nothing).\n")
    x_text, edge_index, y = load()
    n = len(y)
    train, val, test = get_splits(y)
    train_s = sample(train, y, N_TRAIN)
    val_s, test_s = sample(val, y, N_VAL), sample(test, y, N_TEST)    # the step 5 samples
    iso = isolated_mask(edge_index, n)
    orig = original_ids()
    print(f"train {len(train_s)} (of {len(train)}), val {len(val)} (check sample {len(val_s)}), "
          f"test {len(test)} (step 5 sample {len(test_s)})")

    dtype = torch.float32 if FP32 else torch.float16
    tok, model = load_llm(dtype)
    yes = tok.encode("Yes", add_special_tokens=False)[0]
    no = tok.encode("No", add_special_tokens=False)[0]

    path_sum = {}
    if VARIANT == "dgp" and not PROBE:
        targets = np.concatenate([train_s, val, test])
        X = load_embeddings("e5").astype(np.float32)
        if DATASET == "twibot20":
            X = X[:, :-2]                                      # drop the 0/1 "no text" flags
        print("Diffusion-based metapath trimming:")
        chosen = trim_neighbours(targets, X, typed_relations(), n)
        need = sorted({u for nb in chosen.values() for u in nb})
        cut = lambda t: t[:SUMMARY_CHARS] if SUMMARY_CHARS else t
        node_prompt = lambda t: chat(tok, SUMMARY_PROMPT.format(b=B, text=cut(t)))
        node_sum = cached_summaries(NODE_SUMMARY_FILE, {str(orig[u]): x_text[u] for u in need},
                                    node_prompt, tok, model, "node summaries")
        items = {f"{orig[v]}|{name}": ("\n".join(node_sum[str(orig[u])] for u in nb) if nb else None)
                 for (v, name), nb in chosen.items()}
        path_sum = cached_summaries(PATH_SUMMARY_FILE, items, node_prompt, tok, model,
                                    "metapath summaries")
        torch.mps.empty_cache() if DEVICE == "mps" else None

    enc = lambda nodes: [tok(build_prompt(tok, v, x_text, path_sum, orig), add_special_tokens=False)
                         ["input_ids"] for v in nodes]
    tr_seqs, valcheck_seqs = enc(train_s), enc(val_s)
    print(f"prompt tokens: median {int(np.median([len(s) for s in tr_seqs]))}, "
          f"max {max(len(s) for s in tr_seqs)}")
    print("\n--- example prompt ---\n" + tok.decode(tr_seqs[0]) + "\n----------------------")

    # Zero-shot rows (same model, no training), on the step 5 samples. Cached.
    zs_file = RESULTS_DIR / f"step6_{TAG}_zeroshot.npz"
    if not PROBE and not zs_file.exists():
        p_val = score(model, valcheck_seqs, tok.pad_token_id, yes, no, desc="zero-shot val")
        p_test = score(model, enc(test_s), tok.pad_token_id, yes, no, desc="zero-shot test")
        np.savez(zs_file, val=p_val, test=p_test)

    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    if PROBE:
        fine_tune(model, tr_seqs, y[train_s], valcheck_seqs, y[val_s], tok.pad_token_id, yes, no, probe=True)
        return
    fine_tune(model, tr_seqs, y[train_s], valcheck_seqs, y[val_s], tok.pad_token_id, yes, no)
    hist = torch.load(CKPT / "state.pt", weights_only=False)["state"]["history"]
    print("validation checks:", *hist, sep="\n  ")

    # Best adapter -> full validation (threshold) -> full test, once. Scores cached.
    sc_file = RESULTS_DIR / f"step6_{TAG}_scores.npz"
    if not sc_file.exists():
        set_peft_model_state_dict(model, torch.load(CKPT / "best_adapter.pt"))
        p_val = score(model, enc(val), tok.pad_token_id, yes, no, desc="full val")
        p_test = score(model, enc(test), tok.pad_token_id, yes, no, desc="full test")
        np.savez(sc_file, val=p_val, test=p_test, test_nodes=test)
    sc, zs = np.load(sc_file), np.load(zs_file)
    del model
    torch.mps.empty_cache() if DEVICE == "mps" else None

    rows = []
    t = best_threshold(y[val], sc["val"])
    rows.append(evaluate(f"{NAME} (full test)", y[test], (sc["test"] > t).astype(int),
                         sc["test"], iso[test]))
    pos = {v: i for i, v in enumerate(test)}
    p_s = sc["test"][[pos[v] for v in test_s]]
    print("\nSame test sample as step 5:")
    rows.append(evaluate(NAME, y[test_s], (p_s > t).astype(int), p_s, iso[test_s]))
    tz = best_threshold(y[val_s], zs["val"])                   # zero-shot: as in step 5
    rows.append(evaluate(f"{SHORT} zero-shot {VARIANT}", y[test_s], (zs["test"] > tz).astype(int),
                         zs["test"], iso[test_s]))
    base = same_sample_baselines(y, train, val_s, test_s, edge_index, iso[test_s])
    base.append(mlp_same_sample(y, train, val, test_s, edge_index, iso[test_s]))
    save_rows(rows + base)


if __name__ == "__main__":
    main()
