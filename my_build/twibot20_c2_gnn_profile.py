"""
TwiBot-20 to-do C2 (Fri 6 Nov ~04:50): GNNs WITH profile, so TwiBot-20 compares like for like with TwiBot-22 (H3).
Input = Qwen2.5-1.5B embeddings + 2 missing-text flags (as every TwiBot-20 model) + the 5 allowed profile fields
(metadata_check.fields() without `verified`: log followers / following / tweets / listed, account age),
train-standardised. Models MLP (no edges), GCN, SAGE, FAGCN, LINKX via step4_gnn.train_one, 3 seeds, validation cut-off.
Metrics on all test users and non-verified test users; paired bootstrap (2,000, seed-averaged scores) GNN - MLP.
Output: results/twibot20/<SETTING>/c2_gnn_profile.json
Run: DATASET=twibot20 SETTING=A OMP_NUM_THREADS=2 python my_build/twibot20_c2_gnn_profile.py
"""
import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.nn import FAConv, GCNConv, SAGEConv
from torch_geometric.nn.models import LINKX

from common import RESULTS_DIR, best_threshold, get_splits, load, load_embeddings
from metadata_check import fields
from step4_gnn import NoGraph, SEEDS, train_one
from tree_baseline import verified_mask


def main():
    _, e, y = load()
    train, val, test = get_splits(y)
    nv = test[~verified_mask()[test]]
    prof = fields().drop(columns=["verified"]).to_numpy(np.float32)
    X = np.hstack([load_embeddings("qwen").astype(np.float32), prof])
    x = torch.as_tensor((X - X[train].mean(0)) / (X[train].std(0) + 1e-6))
    ei, yt = torch.as_tensor(e), torch.as_tensor(y)
    S, res = {}, {"profile_fields": list(fields().drop(columns=["verified"]).columns), "models": {}, "pairs": {}}
    for name, conv in (("MLP", NoGraph), ("GCN", GCNConv), ("SAGE", SAGEConv), ("FAGCN", FAConv), ("LINKX", LINKX)):
        runs = {"all": [], "non_verified": []}
        ps = []
        for sd in SEEDS:
            p = train_one(conv, x, ei, yt, torch.as_tensor(train), torch.as_tensor(val), sd)
            c = best_threshold(y[val], p[val])
            ps.append(p)
            for sl, idx in (("all", test), ("non_verified", nv)):
                runs[sl].append([f1_score(y[idx], p[idx] > c, average="macro"), roc_auc_score(y[idx], p[idx]), average_precision_score(y[idx], p[idx])])
        S[name] = np.mean(ps, 0)
        res["models"][name] = {sl: dict(zip(("macro_f1", "auroc", "pr_auc"), np.round(np.mean(v, 0), 4).tolist())) for sl, v in runs.items()}
        print(f"{name:6s} {res['models'][name]}", flush=True)
    rng = np.random.default_rng(0)
    cut = {k: best_threshold(y[val], v[val]) for k, v in S.items()}
    for g in ("GCN", "SAGE", "FAGCN", "LINKX"):
        r = {}
        for sl, idx in (("all", test), ("non_verified", nv)):
            d = []
            for _ in range(2000):
                i = idx[rng.integers(0, len(idx), len(idx))]
                if y[i].min() == y[i].max():
                    continue
                d.append([f1_score(y[i], S[g][i] > cut[g], average="macro") - f1_score(y[i], S["MLP"][i] > cut["MLP"], average="macro"),
                          roc_auc_score(y[i], S[g][i]) - roc_auc_score(y[i], S["MLP"][i])])
            d = np.array(d)
            r[sl] = {m: [round(float(d[:, j].mean()), 4), round(float(np.percentile(d[:, j], 2.5)), 4), round(float(np.percentile(d[:, j], 97.5)), 4)]
                     for j, m in enumerate(("macro_f1", "auroc"))}
        res["pairs"][f"{g} - MLP"] = r
        print(f"  {g} - MLP: all {r['all']} | non-verified {r['non_verified']}", flush=True)
    (RESULTS_DIR / "c2_gnn_profile.json").write_text(json.dumps(res, indent=1))
    print("saved", RESULTS_DIR / "c2_gnn_profile.json")


if __name__ == "__main__":
    main()
