"""
Two checks on the headline comparisons (TwiBot-20, Qwen2.5-1.5B embeddings, full test set).

1. Learning curve: LR and MLP trained on 325 ... all training users (stratified samples,
   validation and test unchanged), so the LoRA point (1,300 users) can be read against it.
2. Paired bootstrap 95% CIs for macro-F1 differences: resample the test users 2,000 times,
   score BOTH models on the same resample, report the mean difference and its 2.5-97.5%
   range. A range that crosses 0 means the gap is not distinguishable from test-set noise.
   Seeded models (MLP, GCN, ...) are scored as the mean over their 3 seeds in each resample.

Run:  DATASET=twibot20 SETTING=A python my_build/bootstrap_ci.py     (CPU, ~15 min)
"""
import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_geometric.nn import GCNConv, SAGEConv, FAConv

from common import load, load_embeddings, get_splits, best_threshold, RESULTS_DIR
from step3_embed import make_clf, tune_C
from step4_gnn import train_one, NoGraph, SEEDS
from step5_llm_reason import sample

N_BOOT = 2000
SIZES = [325, 650, 1300, 2600, 5200]                 # sizes >= the training split are skipped


def predictions(conv, x, edge_index, y, tr, val, test):
    # (n_seeds, n_test) 0/1 predictions, each seed with its own validation cut-off.
    out = []
    for seed in SEEDS:
        prob = train_one(conv, x, edge_index, torch.as_tensor(y), torch.as_tensor(tr), torch.as_tensor(val), seed)
        out.append(prob[test] > best_threshold(y[val], prob[val]))
    return np.array(out)


def lr_predictions(emb, y, tr, val, test):
    clf = make_clf(tune_C(emb, y, tr, val)).fit(emb[tr], y[tr])
    t = best_threshold(y[val], clf.predict_proba(emb[val])[:, 1])
    return (clf.predict_proba(emb[test])[:, 1] > t)[None]


def macro_f1(y, preds, idx):
    return np.mean([f1_score(y[idx], p[idx], average="macro") for p in preds])


def main():
    _, e, y = load()
    train, val, test = get_splits(y)
    y_test = y[test]
    emb = load_embeddings("qwen").astype(np.float32)
    edge_index = torch.as_tensor(e)

    def features(tr):
        mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-6
        return torch.as_tensor((emb - mu) / sd)

    # 1. Learning curve
    curve = []
    for n in [m for m in SIZES if m < len(train)] + [len(train)]:
        tr = sample(train, y, n)
        lr = lr_predictions(emb, y, tr, val, test)
        mlp = predictions(NoGraph, features(tr), edge_index, y, tr, val, test)
        mlp_runs = [f1_score(y_test, p, average="macro") for p in mlp]
        curve.append(dict(train_users=len(tr), lr=round(f1_score(y_test, lr[0], average="macro"), 4),
                          mlp=round(float(np.mean(mlp_runs)), 4), mlp_std=round(float(np.std(mlp_runs)), 4)))
        print(curve[-1], flush=True)

    # 2. Predictions for the paired comparisons
    x_full, x_1300 = features(train), features(sample(train, y, 1300))
    tr_1300 = sample(train, y, 1300)
    P = {"MLP (all train)": predictions(NoGraph, x_full, edge_index, y, train, val, test),
         "GCN (all train)": predictions(GCNConv, x_full, edge_index, y, train, val, test),
         "SAGE (all train)": predictions(SAGEConv, x_full, edge_index, y, train, val, test),
         "FAGCN (all train)": predictions(FAConv, x_full, edge_index, y, train, val, test),
         "LR (all train)": lr_predictions(emb, y, train, val, test),
         "MLP (1,300)": predictions(NoGraph, x_1300, edge_index, y, tr_1300, val, test),
         "LR (1,300)": lr_predictions(emb, y, tr_1300, val, test)}
    for variant in ["dgp", "target"]:
        f = RESULTS_DIR / f"step6_Qwen3-1.7B_{variant}_scores.npz"
        if f.exists():
            s = np.load(f)
            assert (s["test_nodes"] == test).all()
            P[f"LoRA {variant} (1,300)"] = (s["test"] > best_threshold(y[val], s["val"]))[None]

    pairs = [("GCN (all train)", "MLP (all train)"), ("SAGE (all train)", "MLP (all train)"),
             ("FAGCN (all train)", "MLP (all train)"), ("FAGCN (all train)", "GCN (all train)"),
             ("MLP (1,300)", "MLP (all train)"),
             ("LoRA dgp (1,300)", "MLP (1,300)"), ("LoRA dgp (1,300)", "LR (1,300)"),
             ("LoRA dgp (1,300)", "MLP (all train)"),
             ("LoRA target (1,300)", "MLP (1,300)"), ("LoRA dgp (1,300)", "LoRA target (1,300)")]
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(test), len(test)) for _ in range(N_BOOT)]
    every = np.arange(len(test))
    ci = []
    for a, b in pairs:
        if a not in P or b not in P:
            continue
        d = np.array([macro_f1(y_test, P[a], i) - macro_f1(y_test, P[b], i) for i in boots])
        lo, hi = np.percentile(d, [2.5, 97.5])
        ci.append(dict(a=a, b=b, diff=round(macro_f1(y_test, P[a], every) - macro_f1(y_test, P[b], every), 4),
                       ci_low=round(float(lo), 4), ci_high=round(float(hi), 4),
                       significant=bool(lo > 0 or hi < 0)))
        print(f"{a:22s} - {b:18s} {ci[-1]['diff']:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
              f"{'  *' if ci[-1]['significant'] else ''}", flush=True)

    out = RESULTS_DIR / "bootstrap_ci.json"
    json.dump(dict(learning_curve=curve, paired_bootstrap=ci, n_boot=N_BOOT, test_users=len(test)),
              open(out, "w"), indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
