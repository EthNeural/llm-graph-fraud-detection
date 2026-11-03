"""
GADBench-style tree baseline (Tang et al., "GADBench", NeurIPS 2023 D&B): gradient-boosted
trees on a user's own features plus a simple neighbourhood aggregate often beat GNNs built
for graph anomaly detection.

Features (all fitted on TRAINING users only):
  own       PCA-64 of the user's Qwen2.5-1.5B embedding (train-standardised)
  nbr       PCA-64 of the mean of its neighbours' embeddings (zeros + an "isolated" flag if none)
  degree    log(1 + degree)
Models: sklearn HistGradientBoosting (class_weight balanced) and step 4's MLP on the same
features, 3 seeds each, cut-off chosen on validation. Also "own only" rows (no graph), and on
TwiBot-20 one labelled extra row adding the 4 log in/out friend/follow degrees, which carry
the crawl artefact (see REPORT.md, "who lists whom").

Run:  DATASET=twibot20 SETTING=A python my_build/tree_baseline.py      (CPU, about 20 min)
Writes results/<dataset>/<setting>/tree_baseline.json
"""
import json

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

from common import load, load_embeddings, get_splits, best_threshold, original_ids, DATA_DIR, DATASET, RESULTS_DIR
from step4_gnn import train_one, NoGraph, SEEDS


def directed_degrees(n):
    keep = original_ids()
    new_id = np.full(len(np.load(DATA_DIR / "y.npy")), -1)
    new_id[keep] = np.arange(len(keep))
    cols = []
    for r, edges in np.load(DATA_DIR / "edge_typed.npz").items():
        e = new_id[edges.T]
        e = e[:, (e >= 0).all(0)]
        cols += [np.bincount(e[0], minlength=n), np.bincount(e[1], minlength=n)]      # out, in
    return np.log1p(np.column_stack(cols)).astype(np.float32)


def main():
    _, e, y = load()
    n = len(y)
    train, val, test = get_splits(y)
    emb = load_embeddings("qwen").astype(np.float32)
    mu, sd = emb[train].mean(0), emb[train].std(0) + 1e-6
    z = (emb - mu) / sd
    A = sp.csr_matrix((np.ones(e.shape[1], dtype=np.float32), (e[0], e[1])), shape=(n, n))
    deg = np.asarray(A.sum(1)).ravel()
    nbr = (sp.diags(1 / np.maximum(deg, 1)) @ A @ z).astype(np.float32)
    own_p = PCA(64, random_state=0).fit(z[train])
    nbr_p = PCA(64, random_state=0).fit(nbr[train][deg[train] > 0])
    own = own_p.transform(z)
    nb = nbr_p.transform(nbr) * (deg > 0)[:, None]
    graph = np.column_stack([nb, (deg == 0).astype(np.float32), np.log1p(deg)])
    feats = {"own only": own, "own + neighbour mean + degree": np.hstack([own, graph])}
    if DATASET == "twibot20":
        feats["own + neighbour mean + degree + in/out degrees (crawl artefact)"] = np.hstack([own, graph, directed_degrees(n)])
    rows = []
    for fname, X in feats.items():
        X = X.astype(np.float32)
        for model in ["HistGradientBoosting", "MLP"]:
            f1s, aucs = [], []
            for seed in SEEDS:
                if model == "MLP":
                    m, s = X[train].mean(0), X[train].std(0) + 1e-6
                    p = train_one(NoGraph, torch.as_tensor((X - m) / s), torch.as_tensor(e), torch.as_tensor(y),
                                  torch.as_tensor(train), torch.as_tensor(val), seed)
                else:
                    clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed,
                                                         early_stopping=True, validation_fraction=0.1)
                    p = clf.fit(X[train], y[train]).predict_proba(X)[:, 1]
                t = best_threshold(y[val], p[val])
                f1s.append(f1_score(y[test], p[test] > t, average="macro"))
                aucs.append(roc_auc_score(y[test], p[test]))
            rows.append(dict(features=fname, model=model, macro_f1=round(float(np.mean(f1s)), 4),
                             macro_f1_std=round(float(np.std(f1s)), 4), auroc=round(float(np.mean(aucs)), 4)))
            print(f"{model:21s} {fname:62s} {rows[-1]['macro_f1']:.3f} ± {rows[-1]['macro_f1_std']:.3f}  "
                  f"AUROC {rows[-1]['auroc']:.3f}", flush=True)
    json.dump(rows, open(RESULTS_DIR / "tree_baseline.json", "w"), indent=2)
    print(f"saved {RESULTS_DIR / 'tree_baseline.json'}")


if __name__ == "__main__":
    main()
