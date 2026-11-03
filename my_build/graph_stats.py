"""
How much label signal do the edges carry? Two measures from Platonov et al. (2022),
"Characterizing Graph Datasets for Node Classification" (arXiv 2209.06177), which correct
plain edge homophily for class balance:

  edge homophily     h      = share of edges whose two ends have the same label
  adjusted homophily h_adj  = (h - sum_k pbar_k^2) / (1 - sum_k pbar_k^2)
                              pbar_k = share of edge ENDS in class k (degree-weighted)
                              0 = no more same-label edges than chance, 1 = perfect, < 0 = heterophily
  label informativeness LI  = I(y_u; y_v) / H(y_u) over edges (u, v): how much a neighbour's
                              label tells you about yours. 0 = nothing, 1 = everything.

Uses ALL labels (test included): a description of the data, never a model input.
Run:  python my_build/graph_stats.py      (CPU, seconds)   -> results/graph_stats.json
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent


def undirected(e, n):
    # (2, E) -> each undirected edge once as (u, v), u < v, no self-loops, no duplicates.
    A = sp.coo_matrix((np.ones(e.shape[1]), (e[0], e[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    T = sp.triu(A, k=1).tocoo()
    return T.row, T.col


def measures(u, v, y):
    k = int(y.max()) + 1
    ends = np.concatenate([y[u], y[v]])                      # both directions
    pbar = np.bincount(ends, minlength=k) / len(ends)
    h = float((y[u] == y[v]).mean())
    chance = float((pbar ** 2).sum())
    joint = np.zeros((k, k))
    np.add.at(joint, (y[u], y[v]), 1)
    joint = joint + joint.T
    joint /= joint.sum()
    nz = joint > 0
    mutual = float((joint[nz] * np.log(joint[nz] / np.outer(pbar, pbar)[nz])).sum())
    entropy = float(-(pbar[pbar > 0] * np.log(pbar[pbar > 0])).sum())
    return dict(edges=int(len(u)), edge_homophily=round(h, 3), chance=round(chance, 3),
                adjusted_homophily=round((h - chance) / (1 - chance), 3),
                label_informativeness=round(mutual / entropy, 4))


def one_setting():
    # Called in a subprocess with DATASET / SETTING set, because common reads them at import.
    sys.path.insert(0, str(ROOT))
    from common import load, original_ids, DATASET, DATA_DIR
    _, e, y = load()
    n = len(y)
    out = {"all edges": measures(*undirected(e, n), y)}
    if DATASET == "twibot20":
        keep = original_ids()
        new_id = np.full(len(np.load(DATA_DIR / "y.npy")), -1)
        new_id[keep] = np.arange(len(keep))
        renum = lambda x: (lambda z: z[:, (z >= 0).all(0)])(new_id[x])
        for r, rel in np.load(DATA_DIR / "edge_typed.npz").items():
            out[r] = measures(*undirected(renum(rel.T), n), y)
        if (DATA_DIR / "support_edges.npz").exists():
            from step4_gnn import add_bridge_edges
            both = add_bridge_edges(e)
            out["all + bridge edges"] = measures(*undirected(both, n), y)
    print(json.dumps(out))


def main():
    if os.environ.get("_GRAPH_STATS_CHILD"):
        return one_setting()
    results = {}
    for ds in ["twibot20", "instagram"]:
        for s in ["A", "B"]:
            env = dict(os.environ, DATASET=ds, SETTING=s, _GRAPH_STATS_CHILD="1")
            res = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=True)
            results[f"{ds} {s}"] = json.loads(res.stdout.strip().splitlines()[-1])
    print(f"{'graph':38s} {'edges':>8s} {'h':>6s} {'chance':>7s} {'h_adj':>6s} {'LI':>7s}")
    for name, rows in results.items():
        for g, m in rows.items():
            print(f"{name + ' | ' + g:38s} {m['edges']:8,d} {m['edge_homophily']:6.3f} {m['chance']:7.3f} "
                  f"{m['adjusted_homophily']:6.3f} {m['label_informativeness']:7.4f}")
    (ROOT / "results" / "graph_stats.json").write_text(json.dumps(results, indent=2))
    print("saved results/graph_stats.json")


if __name__ == "__main__":
    main()
