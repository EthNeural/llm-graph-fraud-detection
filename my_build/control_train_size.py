"""
Control: is "encoder beats LoRA predictor" just "more training data"?

Step 6 fine-tunes the LLM on 1,300 training users (the DGP paper's train size); the step 3/4
text-only models (LR, MLP) were trained on the full training split (8,278 on TwiBot-20 A).
Here the same LR and MLP are trained on BOTH: the full split and the exact 1,300 users step 6
used, so the training data matches LoRA's. Everything else is unchanged: full validation set
for C, early stopping and the cut-off; scored on the full test set AND the step 5 test sample.

Run:  DATASET=twibot20 SETTING=A python my_build/control_train_size.py
      ENC=qwen,qwen3 (default: every listed encoder whose embeddings exist)
CPU only; minutes. Results: results/<dataset>/<setting>/control_train_size.json
"""
import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from common import load, load_embeddings, get_splits, isolated_mask, evaluate, best_threshold, RESULTS_DIR
from step3_embed import make_clf, tune_C, emb_file
from step4_gnn import train_one, NoGraph, SEEDS
from step5_llm_reason import sample, N_TEST
from step6_dgp import N_TRAIN

ENC = os.environ.get("ENC", "qwen,qwen3").split(",")
DRAWS = int(os.environ.get("DRAWS", 1))   # DRAWS=k adds k-1 more random 1,300-user training draws


def mean_rows(label, runs):
    out = {"model": label}
    for k in runs[0]:
        if k != "model":
            vals = [r[k] for r in runs]
            out[k] = round(float(np.mean(vals)), 4)
            out[k + "_std"] = round(float(np.std(vals)), 4)
    return out


def main():
    _, edge_index, y = load()
    train, val, test = get_splits(y)
    train_s, test_s = sample(train, y, N_TRAIN), sample(test, y, N_TEST)   # step 6 / step 5 samples
    iso = isolated_mask(edge_index, len(y))
    print(f"train full {len(train):,}, train sample {len(train_s):,}, val {len(val):,}, "
          f"test {len(test):,}, test sample {len(test_s):,}")

    rows = []
    for name in ENC:
        if not emb_file(name).exists():
            print(f"(skipping {name}: no embeddings)")
            continue
        emb = load_embeddings(name).astype(np.float32)
        extra = [(f"{len(train_s)} train draw {d}",
                  np.sort(train_test_split(train, train_size=len(train_s), stratify=y[train], random_state=d)[0]))
                 for d in range(1, DRAWS)]
        for tr_name, tr in [("full train", train), (f"{len(train_s)} train", train_s)] + extra:
            print(f"\n=== {name}, {tr_name} ===")
            # LR: deterministic, one fit.
            clf = make_clf(tune_C(emb, y, tr, val)).fit(emb[tr], y[tr])
            t = best_threshold(y[val], clf.predict_proba(emb[val])[:, 1])
            for ev_name, ev in [("full test", test), ("step 5 sample", test_s)]:
                p = clf.predict_proba(emb[ev])[:, 1]
                rows.append(evaluate(f"{name} + LR | {tr_name} | {ev_name}", y[ev], (p > t).astype(int),
                                     p, iso[ev]))
            # MLP: step 4's text-only control, 3 seeds, standardised with the TRAINING users' stats.
            mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-6
            x = torch.as_tensor((emb - mu) / sd)
            runs = {"full test": [], "step 5 sample": []}
            for seed in SEEDS:
                prob = train_one(NoGraph, x, torch.as_tensor(edge_index), torch.as_tensor(y),
                                 torch.as_tensor(tr), torch.as_tensor(val), seed)
                t = best_threshold(y[val], prob[val])
                for ev_name, ev in [("full test", test), ("step 5 sample", test_s)]:
                    runs[ev_name].append(evaluate(f"  seed {seed} {ev_name}", y[ev],
                                                  (prob[ev] > t).astype(int), prob[ev], iso[ev]))
            for ev_name, r in runs.items():
                rows.append(mean_rows(f"{name} + MLP | {tr_name} | {ev_name}", r))

    out = RESULTS_DIR / "control_train_size.json"
    new = {r["model"] for r in rows}
    old = [r for r in json.load(open(out)) if r["model"] not in new] if out.exists() else []
    json.dump(old + rows, open(out, "w"), indent=2)
    print(f"\n{'model':48s} {'macroF1':>8s} {'AUROC':>6s}")
    for r in old + rows:
        print(f"{r['model']:48s} {r['macro_f1']:8.3f} {r['auroc']:6.3f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
