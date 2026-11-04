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

WHY TWO PROCESSES (MODEL=tree / MODEL=mlp)
  The first runs (4 Nov 06:04) crashed silently: importing torch and then fitting sklearn's
  HistGradientBoosting in the same process exits with code 139 (SIGSEGV, an OpenMP / libomp
  clash in the anaconda env) and prints nothing. So:
    MODEL=tree  fits only the trees and NEVER imports torch or torch_geometric;
    MODEL=mlp   trains only step 4's MLP (imports torch, never fits HistGradientBoosting);
    MODEL=table reads both JSON files and prints the comparison table (no training).
  Run with OMP_NUM_THREADS=1 as a second safeguard (see run_tree_baseline.sh).

Metrics on the test set: macro-F1 (validation cut-off), AUROC and PR-AUC, mean ± std over 3 seeds,
for ALL test users and for NON-VERIFIED test users only (verified users are all human, so they
inflate every score; see check_verified_subset.py). The cut-off is chosen once on the full
validation set and applied unchanged to both test groups.

Run:  bash my_build/run_tree_baseline.sh        (settings A then B; logs exit codes)
  or  OMP_NUM_THREADS=1 DATASET=twibot20 SETTING=A MODEL=tree python my_build/tree_baseline.py
Writes results/<dataset>/<setting>/tree_baseline_{tree,mlp}.json
"""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from common import load, load_embeddings, get_splits, best_threshold, original_ids, DATA_DIR, DATASET, RESULTS_DIR

MODEL = os.environ.get("MODEL", "tree")
assert MODEL in ("tree", "mlp", "table"), "MODEL must be tree, mlp or table"
SEEDS = [0, 1, 2]              # same as step4_gnn.SEEDS (copied so MODEL=tree never imports step4_gnn)


def directed_degrees(n):
    # TwiBot-20 only: log(1 + out-degree) and log(1 + in-degree) for each typed relation
    # (friend, follow), on this setting's nodes. These carry the crawl artefact.
    keep = original_ids()
    new_id = np.full(len(np.load(DATA_DIR / "y.npy")), -1)
    new_id[keep] = np.arange(len(keep))
    cols = []
    for r, edges in np.load(DATA_DIR / "edge_typed.npz").items():
        e = new_id[edges.T]
        e = e[:, (e >= 0).all(0)]
        cols += [np.bincount(e[0], minlength=n), np.bincount(e[1], minlength=n)]      # out, in
    return np.log1p(np.column_stack(cols)).astype(np.float32)


def verified_mask():
    # True for verified users, in this setting's node order (metadata.csv is in ids.npy order).
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    assert (meta.id.to_numpy() == np.load(DATA_DIR / "ids.npy", allow_pickle=True)).all()
    v = meta.verified.astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()
    return v[original_ids()]


def build_features(e, y, train):
    n = len(y)
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
    return {k: v.astype(np.float32) for k, v in feats.items()}


def score(y, p, cut, idx):
    return (f1_score(y[idx], p[idx] > cut, average="macro"), roc_auc_score(y[idx], p[idx]),
            average_precision_score(y[idx], p[idx]))


def summarise(per_seed):
    # per_seed: list of (macro_f1, auroc, pr_auc) -> mean and std of each
    a = np.array(per_seed)
    keys = ["macro_f1", "auroc", "pr_auc"]
    out = {k: round(float(a[:, i].mean()), 4) for i, k in enumerate(keys)}
    out.update({f"{k}_std": round(float(a[:, i].std()), 4) for i, k in enumerate(keys)})
    return out


def run(model_name):
    _, e, y = load()
    train, val, test = get_splits(y)
    nv_test = test[~verified_mask()[test]]
    feats = build_features(e, y, train)
    if model_name == "mlp":
        import torch                                     # only in the MLP process (see docstring)
        from step4_gnn import train_one, NoGraph
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
    rows = []
    for fname, X in feats.items():
        all_s, nv_s = [], []
        for seed in SEEDS:
            if model_name == "mlp":
                m, s = X[train].mean(0), X[train].std(0) + 1e-6
                p = train_one(NoGraph, torch.as_tensor((X - m) / s), torch.as_tensor(e), torch.as_tensor(y),
                              torch.as_tensor(train), torch.as_tensor(val), seed)
            else:
                clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed,
                                                     early_stopping=True, validation_fraction=0.1)
                p = clf.fit(X[train], y[train]).predict_proba(X)[:, 1]
            cut = best_threshold(y[val], p[val])
            all_s.append(score(y, p, cut, test))
            nv_s.append(score(y, p, cut, nv_test))
        row = dict(features=fname, model="HistGradientBoosting" if model_name == "tree" else "MLP (same features)",
                   all_test=summarise(all_s), non_verified_test=summarise(nv_s),
                   n_test=int(len(test)), n_non_verified_test=int(len(nv_test)))
        rows.append(row)
        a, b = row["all_test"], row["non_verified_test"]
        print(f"{row['model']:21s} {fname:66s} all {a['macro_f1']:.3f}±{a['macro_f1_std']:.3f} "
              f"AUROC {a['auroc']:.3f} PR {a['pr_auc']:.3f} | non-verified {b['macro_f1']:.3f} "
              f"AUROC {b['auroc']:.3f} PR {b['pr_auc']:.3f}", flush=True)
    out = RESULTS_DIR / f"tree_baseline_{model_name}.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"saved {out}")


def table():
    # Comparison table: tree and same-feature MLP rows, next to step 4's Qwen2.5 + MLP
    # (default settings, step4_gnn.json; tuned, step4_tuned.json where it exists).
    # Step 4's MLP reads the full 1,536-dim embedding (+2 flags), not PCA-64, so the fair
    # comparison for the trees is the same-feature MLP row; step 4 is shown for reference.
    rows = []
    for m in ("tree", "mlp"):
        f = RESULTS_DIR / f"tree_baseline_{m}.json"
        assert f.exists(), f"{f.name} missing: run MODEL={m} first"
        rows += json.load(open(f))
    print(f"\nTwiBot-20 setting {os.environ.get('SETTING', 'A')}: test set (all users | non-verified users)")
    print(f"{'model':21s} {'features':66s} {'F1':>6s} {'AUROC':>6s} {'PR-AUC':>6s} | {'F1':>6s} {'AUROC':>6s} {'PR-AUC':>6s}")
    for r in sorted(rows, key=lambda r: (r["features"], r["model"])):
        a, b = r["all_test"], r["non_verified_test"]
        print(f"{r['model']:21s} {r['features']:66s} {a['macro_f1']:>6.3f} {a['auroc']:>6.3f} {a['pr_auc']:>6.3f} | "
              f"{b['macro_f1']:>6.3f} {b['auroc']:>6.3f} {b['pr_auc']:>6.3f}")
    for name, file in (("step 4 Qwen2.5 + MLP (default)", "step4_gnn.json"), ("step 4 Qwen2.5 + MLP (tuned)", "step4_tuned.json")):
        f = RESULTS_DIR / file
        ref = [r for r in json.load(open(f)) if r["model"] == "qwen + MLP"] if f.exists() else []
        if ref:
            r = ref[0]
            print(f"{name:21s} {'full embedding (reference; non-verified not saved)':66s} {r['macro_f1']:>6.3f} "
                  f"{r['auroc']:>6.3f} {r['pr_auc']:>6.3f} | {'–':>6s} {'–':>6s} {'–':>6s}")
        else:
            print(f"{name:21s} {'(not available for this setting)':66s}")


if __name__ == "__main__":
    table() if MODEL == "table" else run(MODEL)
