"""
Semi-synthetic heterophily sweep: how would each GNN do if the edges DID carry label signal?

Our real graphs have adjusted homophily near 0, so they cannot show whether a GNN built for
heterophily (FAGCN, LINKX) or for homophily (GCN) would win when the edges are informative.
This keeps TwiBot-20 A's users, texts (Qwen2.5-1.5B embeddings), splits and degree
distribution, and rebuilds the edges at a chosen level of label homophily: each edge starts at
a random edge end of the real graph (so degrees are kept on average) and goes to a same-label
node with probability q, else to an other-label node. q sets the edge homophily; the measured
adjusted homophily (Platonov et al.) is reported per graph.

Building the edges uses every user's label, test included, so a highly homophilous or
heterophilous graph LEAKS labels through its edges. That is the point of the diagnostic (it
asks what each model could extract from informative edges); it is never a model comparison on
real data. Run:  DATASET=twibot20 SETTING=A python my_build/heterophily_sweep.py   (CPU, ~1 h)
Writes results/twibot20/A/heterophily_sweep.json
"""
import json

import numpy as np
import torch
from torch_geometric.nn import GCNConv, SAGEConv, FAConv
from torch_geometric.nn.models import LINKX

from common import load, load_embeddings, get_splits, isolated_mask, RESULTS_DIR
from step4_gnn import run_seeds, NoGraph, SkipGCN
from graph_stats import measures, undirected

TARGETS = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]          # adjusted homophily asked for
MODELS = [("GCN", GCNConv), ("SAGE", SAGEConv), ("FAGCN", FAConv), ("skip-GCN", SkipGCN), ("LINKX", LINKX)]


def synthetic_edges(edge_index_np, y, h_adj, seed=0):
    rng = np.random.default_rng(seed)
    u, v = undirected(edge_index_np, len(y))
    stubs = np.concatenate([u, v])                       # every edge end of the real graph
    m = len(u)
    pbar = np.bincount(y[stubs], minlength=2) / len(stubs)
    chance = float((pbar ** 2).sum())
    q = float(np.clip(chance + h_adj * (1 - chance), 0, 1))   # target edge homophily
    by_class = [stubs[y[stubs] == k] for k in (0, 1)]
    a = rng.choice(stubs, m)
    same = rng.random(m) < q
    b = np.empty(m, dtype=np.int64)
    for k in (0, 1):
        for want_same in (True, False):
            sel = (y[a] == k) & (same == want_same)
            pool = by_class[k] if want_same else by_class[1 - k]
            b[sel] = rng.choice(pool, sel.sum())
    keep = a != b
    pairs = np.unique(np.sort(np.stack([a[keep], b[keep]]), axis=0), axis=1)
    return np.hstack([pairs, pairs[::-1]])


def main():
    _, e_real, y_np = load()
    train, val, test = get_splits(y_np)
    emb = load_embeddings("qwen").astype(np.float32)
    mu, sd = emb[train].mean(0), emb[train].std(0) + 1e-6
    x = torch.as_tensor((emb - mu) / sd)
    y = torch.as_tensor(y_np, dtype=torch.long)
    train_t, val_t = torch.as_tensor(train), torch.as_tensor(val)
    rows = []
    graphs = [("real", e_real)] + [(f"target {t:+.1f}", synthetic_edges(e_real, y_np, t)) for t in TARGETS]
    mlp = None
    for gname, ei in graphs:
        stats = measures(*undirected(ei, len(y_np)), y_np)
        iso_test = isolated_mask(ei, len(y_np))[test]
        args = (torch.as_tensor(ei, dtype=torch.long), y, y_np, train_t, val_t, val, test, iso_test)
        print(f"\n##### {gname}: {stats}", flush=True)
        if mlp is None:
            mlp = run_seeds("MLP", NoGraph, x, *args)
        for name, conv in MODELS:
            r = run_seeds(f"{name} | {gname}", conv, x, *args)
            rows.append(dict(graph=gname, model=name, adjusted_homophily=stats["adjusted_homophily"],
                             edge_homophily=stats["edge_homophily"], edges=stats["edges"],
                             macro_f1=r["macro_f1"], macro_f1_std=r["macro_f1_std"], auroc=r["auroc"]))
    out = dict(mlp=dict(macro_f1=mlp["macro_f1"], macro_f1_std=mlp["macro_f1_std"], auroc=mlp["auroc"]), rows=rows)
    json.dump(out, open(RESULTS_DIR / "heterophily_sweep.json", "w"), indent=2)
    print(f"\nMLP {mlp['macro_f1']:.3f}")
    for r in rows:
        print(f"{r['graph']:12s} h_adj {r['adjusted_homophily']:+.3f}  {r['model']:9s} {r['macro_f1']:.3f} ± {r['macro_f1_std']:.3f}")


if __name__ == "__main__":
    main()
