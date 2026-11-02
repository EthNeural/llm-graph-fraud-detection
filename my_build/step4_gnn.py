"""
Step 4: Add the graph. GCN vs GraphSAGE on top of each LLM's embeddings.

Run:  python my_build/step4_gnn.py      (needs step 3 embeddings)

The heterophily test:
  GCN   : h_v = mean over {v and its neighbours} of W x      -> self and neighbours MIXED
  SAGE  : h_v = W1 x_v  +  W2 mean(x_neighbours)             -> self and neighbours SEPARATE
On a heterophilous graph, mixing a class-1 user with mostly class-0 neighbours blurs them,
so we expect GCN <= text-only, and SAGE to hold up better.

Each model is trained with 3 seeds; we report mean ± std on the test set,
using the epoch with the best VALIDATION macro-F1 (early stopping).

Two controls make the comparison clean (same layers, hidden size, dropout, optimiser, seeds):
  MLP         : the same 2-layer network with the edges ignored  -> text only
  graph-only  : GCN / SAGE on structure features alone (degree)   -> graph only, no text
So "text-only vs hybrid" differs ONLY in whether the graph is used.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.metrics import f1_score

from common import (load, load_embeddings, get_splits, isolated_mask, evaluate, best_threshold,
                    RESULTS_DIR, SETTING)
from step3_embed import ENCODERS, emb_file

SEEDS = [0, 1, 2]
EPOCHS = 300
PATIENCE = 50


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


def train_one(conv, x, edge_index, y, train, val, seed):
    torch.manual_seed(seed)
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


def main():
    _, edge_index_np, y_np = load()
    train, val, test = get_splits(y_np)
    iso_test = isolated_mask(edge_index_np, len(y_np))[test]

    edge_index = torch.as_tensor(edge_index_np, dtype=torch.long)
    y = torch.as_tensor(y_np, dtype=torch.long)
    train_t, val_t = torch.as_tensor(train), torch.as_tensor(val)

    args = (edge_index, y, y_np, train_t, val_t, val, test, iso_test)
    results = []
    x_struct = structure_features(edge_index_np, len(y_np), train)
    for conv_name, conv in [("GCN", GCNConv), ("SAGE", SAGEConv)]:
        results.append(run_seeds(f"graph-only + {conv_name}", conv, x_struct, *args))

    for name in ENCODERS:
        if not emb_file(name).exists():
            print(f"\n(skipping {name}: no embeddings yet, run step 3)")
            continue
        emb = load_embeddings(name).astype(np.float32)
        # Standardise with TRAIN statistics only (no test leakage).
        mu, sd = emb[train].mean(0), emb[train].std(0) + 1e-6
        x = torch.as_tensor((emb - mu) / sd)

        for conv_name, conv in [("MLP", NoGraph), ("GCN", GCNConv), ("SAGE", SAGEConv)]:
            results.append(run_seeds(f"{name} + {conv_name}", conv, x, *args))

    with open(RESULTS_DIR / "step4_gnn.json", "w") as f:
        json.dump(results, f, indent=2)
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
