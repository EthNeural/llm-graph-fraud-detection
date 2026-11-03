"""
Two checks on the headline comparisons (TwiBot-20, Qwen2.5-1.5B embeddings, full test set).

1. Learning curve: LR and MLP trained on 325 ... all training users (stratified samples,
   validation and test unchanged), so the LoRA point (1,300 users) can be read against it.
2. Paired bootstrap 95% CIs for macro-F1 differences: resample the test users 2,000 times,
   score BOTH models on the same resample, report the mean difference and its 2.5-97.5%
   range. A range that crosses 0 means the gap is not distinguishable from test-set noise.
   Seeded models (MLP, GCN, ...) are scored as the mean over their 3 seeds in each resample.
   The same resamples also give CIs for AUROC and PR-AUC differences (threshold-free).
LORA_ADAPTER=best (default: lowest validation loss, step6_*_scores.npz) or last (the final
check's adapter, step6_*_last_scores.npz from step6_rescore.py) picks which LoRA scores to use.

Run:  DATASET=twibot20 SETTING=A python my_build/bootstrap_ci.py     (CPU, ~15 min)
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch_geometric.nn import GCNConv, SAGEConv, FAConv

from common import load, load_embeddings, get_splits, best_threshold, RESULTS_DIR
from step3_embed import make_clf, tune_C
from step4_gnn import train_one, NoGraph, SEEDS
from step5_llm_reason import sample

N_BOOT = 2000
LORA_ADAPTER = os.environ.get("LORA_ADAPTER", "best")
assert LORA_ADAPTER in ("best", "last"), "LORA_ADAPTER must be best or last"
SIZES = [325, 650, 1300, 2600, 5200]                 # sizes >= the training split are skipped


def predictions(conv, x, edge_index, y, tr, val, test, probs=False):
    # (n_seeds, n_test) 0/1 predictions, each seed with its own validation cut-off
    # (and, with probs=True, the matching P(class 1) scores).
    pred, score = [], []
    for seed in SEEDS:
        prob = train_one(conv, x, edge_index, torch.as_tensor(y), torch.as_tensor(tr), torch.as_tensor(val), seed)
        pred.append(prob[test] > best_threshold(y[val], prob[val]))
        score.append(prob[test])
    return (np.array(pred), np.array(score)) if probs else np.array(pred)


def lr_predictions(emb, y, tr, val, test, probs=False):
    clf = make_clf(tune_C(emb, y, tr, val)).fit(emb[tr], y[tr])
    t = best_threshold(y[val], clf.predict_proba(emb[val])[:, 1])
    p = clf.predict_proba(emb[test])[:, 1]
    return ((p > t)[None], p[None]) if probs else (p > t)[None]


def macro_f1(y, preds, idx):
    return np.mean([f1_score(y[idx], p[idx], average="macro") for p in preds])


def ranked(metric, y, scores, idx):
    return np.mean([metric(y[idx], p[idx]) for p in scores])


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
    P, S = {}, {}                                 # 0/1 predictions and scores, (n_seeds, n_test)
    for name, args in [("MLP (all train)", (NoGraph, x_full, train)), ("GCN (all train)", (GCNConv, x_full, train)),
                       ("SAGE (all train)", (SAGEConv, x_full, train)), ("FAGCN (all train)", (FAConv, x_full, train)),
                       ("MLP (1,300)", (NoGraph, x_1300, tr_1300))]:
        P[name], S[name] = predictions(args[0], args[1], edge_index, y, args[2], val, test, probs=True)
    P["LR (all train)"], S["LR (all train)"] = lr_predictions(emb, y, train, val, test, probs=True)
    P["LR (1,300)"], S["LR (1,300)"] = lr_predictions(emb, y, tr_1300, val, test, probs=True)
    suffix = "_scores.npz" if LORA_ADAPTER == "best" else "_last_scores.npz"
    for variant in ["dgp", "target"]:
        f = RESULTS_DIR / f"step6_Qwen3-1.7B_{variant}{suffix}"
        if f.exists():
            s = np.load(f)
            assert (s["test_nodes"] == test).all()
            P[f"LoRA {variant} (1,300)"] = (s["test"] > best_threshold(y[val], s["val"]))[None]
            S[f"LoRA {variant} (1,300)"] = s["test"][None]

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
        row = dict(a=a, b=b, diff=round(macro_f1(y_test, P[a], every) - macro_f1(y_test, P[b], every), 4),
                   ci_low=round(float(lo), 4), ci_high=round(float(hi), 4), significant=bool(lo > 0 or hi < 0))
        for mname, metric in [("auroc", roc_auc_score), ("pr_auc", average_precision_score)]:
            dm = np.array([ranked(metric, y_test, S[a], i) - ranked(metric, y_test, S[b], i) for i in boots])
            ml, mh = np.percentile(dm, [2.5, 97.5])
            row[f"{mname}_diff"] = round(ranked(metric, y_test, S[a], every) - ranked(metric, y_test, S[b], every), 4)
            row[f"{mname}_ci"] = [round(float(ml), 4), round(float(mh), 4)]
        ci.append(row)
        print(f"{a:22s} - {b:18s} F1 {row['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if row['significant'] else ' '}"
              f"  AUROC {row['auroc_diff']:+.3f} {row['auroc_ci']}  PR-AUC {row['pr_auc_diff']:+.3f} {row['pr_auc_ci']}",
              flush=True)

    out = RESULTS_DIR / ("bootstrap_ci.json" if LORA_ADAPTER == "best" else "bootstrap_ci_last_adapter.json")
    json.dump(dict(learning_curve=curve, paired_bootstrap=ci, n_boot=N_BOOT, test_users=len(test), lora_adapter=LORA_ADAPTER),
              open(out, "w"), indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
