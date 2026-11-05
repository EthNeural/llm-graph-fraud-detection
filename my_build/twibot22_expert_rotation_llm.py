"""
TwiBot-22 weak -> expert fine-tuning of LoRA target, with 5-fold rotation over the 986 expert users
(decided by the builder 5 Nov ~14:00, replacing "experts only for the final score"; checklist B8, G7).

Why: the weak labels disagree with the experts often (dataset "bot" matches experts for 56 of 101), so
models trained on weak labels learn the labelling rules. Using all experts for training would leave no
clean test, so they are rotated: each expert is used for training in 4 folds and scored once, by a model
that never saw it. All 105 expert bots stay in the evaluation.

Rules, fixed before any expert score is seen:
  - start: the best LoRA target adapter (chosen on validation, step 6); the base model is not retrained
  - folds: StratifiedKFold(5, shuffle, random_state 42) on the expert label
  - training: 1 epoch on the ~789 experts of the other 4 folds, expert labels, step 6's settings
    (lr 1e-4, batch 4, r 16, the same prompt and 480-token text); no early stopping, no selection
  - cut-off for macro-F1: chosen on the fold model's scores of its own 4 training folds (never on the
    held-out fold); the weak-only model keeps its validation cut-off
  - also scores all 986 experts with the weak-only adapter, so both are compared on the same users

BLIND: the experts are sealed users, so this script prints no metric. It only writes scores; they are
read once, in the final run (FINAL_RUN=1). Resumable: each finished fold is saved.
Run: DATASET=twibot22 VARIANT=target VAL_CHECK=1000 HF_HUB_OFFLINE=1 python -u my_build/twibot22_expert_rotation_llm.py
GPU, ~3.5-4 h (5 x ~31 min training + scoring).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

assert os.environ.get("DATASET") == "twibot22" and os.environ.get("VARIANT") == "target"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step6_dgp as s6
from step6_dgp import torch, F

ROOT = Path(__file__).resolve().parent
ROT_SMOKE = os.environ.get("ROT_SMOKE") == "1"   # 1 fold, 20 train / 10 held-out experts, separate folder: checks it runs
_sm = "_smoke" if ROT_SMOKE else ""
OUT = s6.RESULTS_DIR / f"expert_rotation_target{_sm}.npz"
FOLD_DIR = s6.RESULTS_DIR / f"expert_rotation_target_folds{_sm}"
EXPERTS = ROOT.parent / "data" / "twibot-22-raw" / "expert_labels" / "label_manual.csv"
N_FOLDS = 5


def train_one_epoch(model, seqs, y_tr, pad_id, yes, no):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=s6.LR)
    batches = s6.make_batches([len(s) for s in seqs], 0)
    t0 = time.time()
    for b, idx in enumerate(batches):
        model.train()
        logits = s6.last_logits(model, [seqs[i] for i in idx], pad_id)
        tgt = torch.tensor([yes if y_tr[i] else no for i in idx], device=s6.DEVICE)
        loss = F.cross_entropy(logits, tgt)
        if not torch.isfinite(loss):
            sys.exit("Loss is NaN/inf")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if b % 50 == 0:
            print(f"    step {b}/{len(batches)}  {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from sklearn.model_selection import StratifiedKFold
    from twibot22_metrics import val_cutoff

    x_text, _, y_weak = s6.load()
    ids = np.load(s6.DATA_DIR / "ids.npy")
    split = np.load(s6.DATA_DIR / "split.npy")
    exp = pd.read_csv(EXPERTS)
    pos = {u: i for i, u in enumerate(ids)}
    nodes = np.array([pos[u] for u in exp.id])
    y_exp = (exp.label.values == "bot").astype(int)
    assert len(nodes) == 986 and (split[nodes] == "sealed").all()
    assert not np.isin(nodes, np.where(np.isin(split, ["train", "val", "test"]))[0]).any()
    folds = np.zeros(len(nodes), int)
    for k, (_, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=42).split(nodes, y_exp)):
        folds[te] = k
    FOLD_DIR.mkdir(parents=True, exist_ok=True)

    tok, base = s6.load_llm(torch.float16)
    yes = tok.encode("Yes", add_special_tokens=False)[0]
    no = tok.encode("No", add_special_tokens=False)[0]
    orig = s6.original_ids()
    if ROT_SMOKE:                                          # tiny: fold 0 only, 20 train + 10 held out
        keep = np.concatenate([np.where(folds != 0)[0][:20], np.where(folds == 0)[0][:10]])
        nodes, y_exp, folds = nodes[keep], y_exp[keep], folds[keep]
    seqs = [tok(s6.build_prompt(tok, v, x_text, {}, orig), add_special_tokens=False)["input_ids"] for v in nodes]
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **s6.LORA))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    start = torch.load(s6.RESULTS_DIR / "step6_Qwen3-1.7B_target" / "best_adapter.pt")

    weak_file = FOLD_DIR / "weak_only.npy"
    if not weak_file.exists():
        set_peft_model_state_dict(model, start)
        np.save(weak_file, s6.score(model, seqs, tok.pad_token_id, yes, no, desc="weak-only adapter, experts"))
    sc = np.load(s6.RESULTS_DIR / "step6_Qwen3-1.7B_target_scores.npz")
    weak_cut = val_cutoff(y_weak[sc["val_nodes"]], sc["val"])

    t_all = time.time()
    for k in range(1 if ROT_SMOKE else N_FOLDS):
        f = FOLD_DIR / f"fold{k}.npz"
        if f.exists():
            continue
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        print(f"fold {k}: train {len(tr)} experts ({y_exp[tr].sum()} bots), held out {len(te)}", flush=True)
        set_peft_model_state_dict(model, start)                  # every fold starts from the weak adapter
        train_one_epoch(model, [seqs[i] for i in tr], y_exp[tr], tok.pad_token_id, yes, no)
        p_tr = s6.score(model, [seqs[i] for i in tr], tok.pad_token_id, yes, no, desc="own training folds")
        p_te = s6.score(model, [seqs[i] for i in te], tok.pad_token_id, yes, no, desc="held-out fold")
        np.savez(f, held_out=te, p=p_te, cut=val_cutoff(y_exp[tr], p_tr))
        print(f"fold {k} saved (no metric printed: blind)", flush=True)

    if ROT_SMOKE:
        print(f"ROTATION SMOKE TEST PASSED ({(time.time() - t_all) / 60:.1f} min for 20 train + 10 held out)")
        return
    p_rot, cut_rot = np.empty(len(nodes)), np.empty(len(nodes))
    for k in range(N_FOLDS):
        z = np.load(FOLD_DIR / f"fold{k}.npz")
        p_rot[z["held_out"]], cut_rot[z["held_out"]] = z["p"], float(z["cut"])
    np.savez(OUT, nodes=nodes, rows=np.load(s6.DATA_DIR / "rows.npy")[nodes], y_expert=y_exp,
             y_weak=y_weak[nodes], fold=folds, p_weak_only=np.load(weak_file), cut_weak_only=weak_cut,
             p_rotated=p_rot, cut_rotated=cut_rot)
    print("Saved", OUT, "(blind: read in the final run)")


if __name__ == "__main__":
    main()
