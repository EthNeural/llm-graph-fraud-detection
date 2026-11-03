"""
Step 4: Add the graph. GCN vs GraphSAGE on top of each LLM's embeddings.

Run:  python my_build/step4_gnn.py      (needs step 3 embeddings)

TwiBot-20 only: GRAPH=bridges adds an edge between two labelled users who share a support-user
neighbour (a "bridge", see playground_bridges.py, which must run first). Uses no labels.
Results go to step4_gnn_bridges.json; "isolated" still means isolated in the labelled-only
graph, so the connected / isolated groups are the same users in both runs.
  e.g.  DATASET=twibot20 GRAPH=bridges python my_build/step4_gnn.py
GRAPH=shuffled keeps every user's degree but rewires who links to whom at random (a control:
if GNNs score the same on it, the real edges carry no usable signal). Any dataset.

The heterophily test:
  GCN   : h_v = mean over {v and its neighbours} of W x      -> self and neighbours MIXED
  SAGE  : h_v = W1 x_v  +  W2 mean(x_neighbours)             -> self and neighbours SEPARATE
On a heterophilous graph, mixing a class-1 user with mostly class-0 neighbours blurs them,
so we expect GCN <= text-only, and SAGE to hold up better.

Each model is trained with 3 seeds; we report mean ± std on the test set,
using the epoch with the best VALIDATION macro-F1 (early stopping).

FAGCN (Bo et al., AAAI 2021) is built for heterophily: each edge learns a weight in [-1, 1],
so a neighbour can be added (low-frequency, homophily) or SUBTRACTED (high-frequency,
heterophily), and the layer input is mixed back in (eps). Same 2 layers, hidden size, dropout.

FLAG-lite (FLAG, Yang et al., KDD 2025, without its LLM): SKIPGCN is FLAG's skip-GNN,
GCN(X, A) + Linear(X) (its eq. 6), and GRAPH=similar10 keeps, for each user, the 10 neighbours
whose text embedding is most similar (cosine > 0 on train-standardised vectors; FLAG's top-10,
threshold 0, here over 1-hop neighbours). The pruning uses each encoder's own embeddings.
  e.g.  DATASET=twibot20 GRAPH=similar10 MODELS=GCN,SKIPGCN python my_build/step4_gnn.py

LINKX (Lim et al., NeurIPS 2021; PyG `LINKX`) is a non-homophilous baseline that never averages
neighbours: one MLP reads each user's adjacency row (who it links to), another its features,
and a third combines them. Same hidden size (128), 2 layers, dropout 0.5, training as above.

MODELS=FAGCN (comma-separated) runs only those models and merges their rows into the existing
results file, e.g.  DATASET=twibot20 MODELS=FAGCN python my_build/step4_gnn.py

Two controls make the comparison clean (same layers, hidden size, dropout, optimiser, seeds):
  MLP         : the same 2-layer network with the edges ignored  -> text only
  graph-only  : GCN / SAGE on structure features alone (degree)   -> graph only, no text
So "text-only vs hybrid" differs ONLY in whether the graph is used.
"""
import json
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, FAConv
from torch_geometric.nn.models import LINKX
from sklearn.metrics import f1_score

from common import (load, load_embeddings, get_splits, isolated_mask, evaluate, best_threshold,
                    original_ids, RESULTS_DIR, SETTING, DATASET, DATA_DIR)
from step3_embed import ENCODERS, emb_file

SEEDS = [0, 1, 2]
EPOCHS = 300
PATIENCE = 50
FAGCN_EPS = 0.3                 # share of the layer input kept each layer (FAGCN paper, heterophilous graphs)
MODELS = os.environ.get("MODELS", "MLP,GCN,SAGE,FAGCN").split(",")
assert set(MODELS) <= {"MLP", "GCN", "SAGE", "FAGCN", "SKIPGCN", "LINKX"}, f"unknown name in MODELS={MODELS}"
GRAPH = os.environ.get("GRAPH", "labelled")
SHUFFLE_SEED = int(os.environ.get("SHUFFLE_SEED", 0))     # GRAPH=shuffled: which random rewiring
assert GRAPH in ("labelled", "shuffled", "similar10") or (GRAPH == "bridges" and DATASET == "twibot20"), \
    "GRAPH must be labelled, shuffled, or bridges (TwiBot-20 only)"


class NoGraph(torch.nn.Module):
    # A "conv" that ignores the edges: GNN(NoGraph, ...) is a plain 2-layer MLP.
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        return self.lin(x)


class GNN(torch.nn.Module):
    def __init__(self, conv, in_dim, hidden=128, out_dim=2, dropout=0.5):
        super().__init__()
        self.conv1 = conv(in_dim, hidden)
        self.conv2 = conv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, self.training)
        return self.conv2(x, edge_index)


class FAGCN(torch.nn.Module):
    # Linear in -> 2 FAConv layers (signed edge weights, input mixed back in) -> linear out.
    def __init__(self, in_dim, hidden=128, out_dim=2, dropout=0.5):
        super().__init__()
        self.lin_in = torch.nn.Linear(in_dim, hidden)
        self.convs = torch.nn.ModuleList([FAConv(hidden, eps=FAGCN_EPS, dropout=dropout) for _ in range(2)])
        self.lin_out = torch.nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, self.training)
        x = x_0 = F.relu(self.lin_in(x))
        for conv in self.convs:
            x = conv(F.dropout(x, self.dropout, self.training), x_0, edge_index)
        return self.lin_out(F.dropout(x, self.dropout, self.training))


class SkipGCN(torch.nn.Module):
    # FLAG's skip-GNN (eq. 6): the 2-layer GCN's output plus a linear map of the node's own input.
    def __init__(self, in_dim, out_dim=2, dropout=0.5):
        super().__init__()
        self.gnn = GNN(GCNConv, in_dim, dropout=dropout)
        self.skip = torch.nn.Linear(in_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        return self.gnn(x, edge_index) + self.skip(F.dropout(x, self.dropout, self.training))


def train_one(conv, x, edge_index, y, train, val, seed):
    torch.manual_seed(seed)
    if conv is FAConv:
        model = FAGCN(x.shape[1])
    elif conv is SkipGCN:
        model = SkipGCN(x.shape[1])
    elif conv is LINKX:
        model = LINKX(num_nodes=x.shape[0], in_channels=x.shape[1], hidden_channels=128, out_channels=2,
                      num_layers=2, dropout=0.5)
    else:
        model = GNN(conv, x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # Class weights = inverse frequency, same idea as class_weight="balanced" in step 2/3.
    counts = torch.bincount(y[train])
    weight = (counts.sum() / (2 * counts)).float()

    best_f1, best_state, wait = -1, None, 0
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, edge_index)[train], y[train], weight=weight)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index).argmax(1)
        f1 = f1_score(y[val], pred[val], average="macro")
        if f1 > best_f1:
            best_f1, wait = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x, edge_index).softmax(1)[:, 1].numpy()      # P(class 1) for every node


def run_seeds(label, conv, x, edge_index, y, y_np, train_t, val_t, val, test, iso_test):
    # Train with every seed, score each on test, return mean and std of every metric.
    print(f"\n=== {label} ===")
    runs = []
    for seed in SEEDS:
        prob = train_one(conv, x, edge_index, y, train_t, val_t, seed)
        t = best_threshold(y_np[val], prob[val])          # cut-off chosen on val
        runs.append(evaluate(f"  seed {seed}", y_np[test], (prob[test] > t).astype(int),
                             prob[test], iso_test))
    summary = {"model": label}
    for k in runs[0]:
        if k != "model":
            vals = [r[k] for r in runs]
            summary[k] = round(float(np.mean(vals)), 4)
            summary[k + "_std"] = round(float(np.std(vals)), 4)
    print(f"  -> mean macroF1 {summary['macro_f1']:.3f} ± {summary['macro_f1_std']:.3f}")
    return summary


def structure_features(edge_index_np, n, train):
    # Graph-only input: no text, no metadata. Own degree + mean neighbour degree (log scale).
    deg = np.bincount(edge_index_np[0], minlength=n).astype(np.float32)
    nbr = np.bincount(edge_index_np[0], weights=np.log1p(deg[edge_index_np[1]]), minlength=n)
    f = np.stack([np.log1p(deg), nbr / np.maximum(deg, 1), np.ones(n)], 1).astype(np.float32)
    mu, sd = f[train].mean(0), f[train].std(0) + 1e-6
    sd[2], mu[2] = 1.0, 0.0                               # keep the constant column at 1
    return torch.as_tensor((f - mu) / sd)


def add_bridge_edges(edge_index_np):
    # Link two labelled users when they share a support-user neighbour: S = B B^T, where
    # B[labelled, support] = 1 for any friend/follow link. Then renumber to this setting.
    c = np.load(DATA_DIR / "support_edges.npz")
    n_all = len(np.load(DATA_DIR / "y.npy"))
    B = sp.csr_matrix((np.ones(len(c["lab"]), dtype=np.float32), (c["lab"], c["sup"])),
                      shape=(n_all, int(c["sup"].max()) + 1))
    B.data[:] = 1                                         # friend + follow to one user -> 1
    S = (B @ B.T).tocoo()                                 # symmetric, so both directions included
    e = np.stack([S.row, S.col])[:, S.row != S.col]
    keep = original_ids()
    new_id = np.full(n_all, -1)
    new_id[keep] = np.arange(len(keep))
    e = new_id[e]
    e = e[:, (e >= 0).all(0)]                             # setting B: both ends must be kept
    merged = np.unique(np.hstack([edge_index_np, e]), axis=1)
    print(f"GRAPH=bridges: {edge_index_np.shape[1]:,} edge rows + {e.shape[1]:,} bridge rows "
          f"-> {merged.shape[1]:,} (isolated {isolated_mask(merged, len(keep)).sum():,})")
    return merged


def shuffle_edges(edge_index_np, n, seed=0):
    # Degree-preserving rewiring: list both ends of every undirected edge ("stubs"), shuffle,
    # pair them up again. Self-loops and duplicate pairs are dropped (a few %), then both
    # directions are stored, as in the real graph.
    u, v = edge_index_np
    u, v = u[u < v], v[u < v]
    stubs = np.random.default_rng(seed).permutation(np.concatenate([u, v]))
    a, b = stubs[0::2], stubs[1::2]
    a, b = np.minimum(a, b)[a != b], np.maximum(a, b)[a != b]
    pairs = np.unique(np.stack([a, b]), axis=1)
    out = np.hstack([pairs, pairs[::-1]])
    print(f"GRAPH=shuffled: {edge_index_np.shape[1]:,} edge rows -> {out.shape[1]:,} after rewiring")
    return out


def similar_topk(edge_index_np, x, k=10):
    # FLAG's semantic-similarity sampling: per user, keep up to k neighbours with cosine > 0,
    # most similar first. x is train-standardised, so cosines are centred. Kept both ways.
    u = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    src, dst = edge_index_np
    cos = (u[src] * u[dst]).sum(1)
    order = np.lexsort((-cos, src))                       # by source, most similar first
    src, dst, cos = src[order], dst[order], cos[order]
    rank = np.arange(len(src)) - np.searchsorted(src, src)
    keep = (rank < k) & (cos > 0)
    pairs = np.stack([src[keep], dst[keep]])
    out = np.unique(np.hstack([pairs, pairs[::-1]]), axis=1)
    return out


def main():
    _, edge_index_np, y_np = load()
    train, val, test = get_splits(y_np)
    iso_test = isolated_mask(edge_index_np, len(y_np))[test]   # labelled-only graph, for every GRAPH
    if GRAPH == "bridges":
        edge_index_np = add_bridge_edges(edge_index_np)
    elif GRAPH == "shuffled":
        edge_index_np = shuffle_edges(edge_index_np, len(y_np), SHUFFLE_SEED)

    edge_index = torch.as_tensor(edge_index_np, dtype=torch.long)
    y = torch.as_tensor(y_np, dtype=torch.long)
    train_t, val_t = torch.as_tensor(train), torch.as_tensor(val)

    args = (edge_index, y, y_np, train_t, val_t, val, test, iso_test)
    results = []
    x_struct = structure_features(edge_index_np, len(y_np), train)
    for conv_name, conv in [("GCN", GCNConv), ("SAGE", SAGEConv), ("FAGCN", FAConv), ("SKIPGCN", SkipGCN),
                            ("LINKX", LINKX)]:
        if conv_name in MODELS and GRAPH != "similar10":          # similar10 needs text embeddings
            results.append(run_seeds(f"graph-only + {conv_name}", conv, x_struct, *args))

    for name in ENCODERS:
        if not emb_file(name).exists():
            print(f"\n(skipping {name}: no embeddings yet, run step 3)")
            continue
        emb = load_embeddings(name).astype(np.float32)
        # Standardise with TRAIN statistics only (no test leakage).
        mu, sd = emb[train].mean(0), emb[train].std(0) + 1e-6
        x = torch.as_tensor((emb - mu) / sd)
        enc_args = args
        if GRAPH == "similar10":
            ei = similar_topk(edge_index_np, x.numpy())
            print(f"\nGRAPH=similar10 ({name}): {edge_index_np.shape[1]:,} edge rows -> {ei.shape[1]:,}")
            enc_args = (torch.as_tensor(ei, dtype=torch.long),) + args[1:]

        for conv_name, conv in [("MLP", NoGraph), ("GCN", GCNConv), ("SAGE", SAGEConv), ("FAGCN", FAConv),
                                ("SKIPGCN", SkipGCN), ("LINKX", LINKX)]:
            if conv_name in MODELS:
                results.append(run_seeds(f"{name} + {conv_name}", conv, x, *enc_args))

    tag = f"{GRAPH}_s{SHUFFLE_SEED}" if GRAPH == "shuffled" and SHUFFLE_SEED else GRAPH
    out = RESULTS_DIR / ("step4_gnn.json" if GRAPH == "labelled" else f"step4_gnn_{tag}.json")
    # Merge: rows of the models run now replace their old rows; other rows are kept.
    new = {r["model"] for r in results}
    old = [r for r in json.load(open(out)) if r["model"] not in new] if out.exists() else []
    with open(out, "w") as f:
        json.dump(old + results, f, indent=2)
    print(f"\nSetting {SETTING} done. Full table: python my_build/compare.py")


def print_table():
    # One table with every model so far: baselines, text-only, GNNs.
    rows = []
    for fname in ["step2_baselines.json", "step3_text_only.json", "step4_gnn.json"]:
        p = RESULTS_DIR / fname
        if p.exists():
            rows += json.load(open(p))
    print(f"\n{'model':20s} {'macroF1':>8s} {'F1_1':>6s} {'AUROC':>6s} {'conn':>6s} {'iso':>6s}")
    for r in rows:
        std = f"±{r['macro_f1_std']:.3f}" if "macro_f1_std" in r else ""
        print(f"{r['model']:20s} {r['macro_f1']:8.3f} {r['f1_class1']:6.3f} "
              f"{r.get('auroc', float('nan')):6.3f} {r['macro_f1_connected']:6.3f} "
              f"{r['macro_f1_isolated']:6.3f}  {std}")


if __name__ == "__main__":
    main()
