"""
A small, equal tuning budget for every step 4 model (Luo et al., "Classic GNNs are Strong
Baselines", NeurIPS 2024: under-tuned baselines can explain claimed gaps).

Step 4 trains every model with one fixed setting (hidden 128, lr 0.01, dropout 0.5; FAGCN eps
0.3). Here MLP, GCN, SAGE and FAGCN each get the SAME grid, which contains that setting so a
tuned model can never be worse on validation than its untuned self:
  lr {1e-3, 1e-2} x hidden {64, 128, 256} x dropout {0.2, 0.5}   (+ FAGCN eps {0.1, 0.3, 0.5})
Selection uses seed 0 and VALIDATION macro-F1 only (cut-off chosen on validation, as in every
step). The chosen setting is then trained with 3 fresh seeds (1, 2, 3) and scored once on test,
so the selection seed never touches the reported numbers. Weight decay, epochs, early
stopping and class weights are unchanged. Qwen2.5-1.5B embeddings, labelled-only graph.

Run:  DATASET=twibot20 SETTING=A python my_build/tune_step4.py     (CPU, about 1 h)
Writes results/<dataset>/<setting>/step4_tuned.json
"""
import itertools
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.nn import GCNConv, SAGEConv, FAConv

import step4_gnn as s4
from common import load, load_embeddings, get_splits, isolated_mask, evaluate, best_threshold, RESULTS_DIR

GRID = dict(lr=[1e-3, 1e-2], hidden=[64, 128, 256], dropout=[0.2, 0.5])
EPS = [0.1, 0.3, 0.5]
MODELS = [("MLP", s4.NoGraph), ("GCN", GCNConv), ("SAGE", SAGEConv), ("FAGCN", FAConv)]
TEST_SEEDS = [1, 2, 3]


def build(conv, in_dim, hidden, dropout, eps):
    if conv is FAConv:
        s4.FAGCN_EPS = eps                                   # read when the FAConv layers are built
        return s4.FAGCN(in_dim, hidden=hidden, dropout=dropout)
    return s4.GNN(conv, in_dim, hidden=hidden, dropout=dropout)


def train(conv, x, ei, y, train_idx, val_idx, seed, lr, hidden, dropout, eps):
    # Same loop as step4_gnn.train_one, with the tuned settings passed in.
    torch.manual_seed(seed)
    model = build(conv, x.shape[1], hidden, dropout, eps)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    counts = torch.bincount(y[train_idx])
    weight = (counts.sum() / (2 * counts)).float()
    best_f1, best_state, wait = -1, None, 0
    for _ in range(s4.EPOCHS):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x, ei)[train_idx], y[train_idx], weight=weight).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, ei).argmax(1)
        f1 = f1_score(y[val_idx], pred[val_idx], average="macro")
        if f1 > best_f1:
            best_f1, wait = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= s4.PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x, ei).softmax(1)[:, 1].numpy()


def main():
    _, e, y_np = load()
    train_idx, val, test = get_splits(y_np)
    iso_test = isolated_mask(e, len(y_np))[test]
    emb = load_embeddings("qwen").astype(np.float32)
    mu, sd = emb[train_idx].mean(0), emb[train_idx].std(0) + 1e-6
    x = torch.as_tensor((emb - mu) / sd)
    ei, y = torch.as_tensor(e, dtype=torch.long), torch.as_tensor(y_np, dtype=torch.long)
    tr_t, va_t = torch.as_tensor(train_idx), torch.as_tensor(val)
    untuned = {r["model"]: r for r in json.load(open(RESULTS_DIR / "step4_gnn.json"))}
    out = []
    for name, conv in MODELS:
        configs = [dict(lr=lr, hidden=h, dropout=d, eps=eps)
                   for lr, h, d in itertools.product(GRID["lr"], GRID["hidden"], GRID["dropout"])
                   for eps in (EPS if conv is FAConv else [0.3])]
        scored = []
        for c in configs:
            p = train(conv, x, ei, y, tr_t, va_t, 0, **c)
            vf1 = f1_score(y_np[val], p[val] > best_threshold(y_np[val], p[val]), average="macro")
            scored.append((vf1, c))
            print(f"  {name} {c} val macro-F1 {vf1:.4f}", flush=True)
        vbest, cbest = max(scored, key=lambda t: t[0])
        default = dict(lr=1e-2, hidden=128, dropout=0.5, eps=0.3)
        vdefault = next(v for v, c in scored if c == default)
        runs = []
        for seed in TEST_SEEDS:
            p = train(conv, x, ei, y, tr_t, va_t, seed, **cbest)
            t = best_threshold(y_np[val], p[val])
            runs.append(evaluate(f"  seed {seed}", y_np[test], (p[test] > t).astype(int), p[test], iso_test))
        row = dict(model=f"qwen + {name}", tuned_config=cbest, val_macro_f1_tuned=round(vbest, 4),
                   val_macro_f1_default=round(vdefault, 4), configs_tried=len(configs))
        for k in runs[0]:
            if k != "model":
                vals = [r[k] for r in runs]
                row[k] = round(float(np.mean(vals)), 4)
                row[k + "_std"] = round(float(np.std(vals)), 4)
        u = untuned[f"qwen + {name}"]
        row["untuned_macro_f1"], row["untuned_macro_f1_std"] = u["macro_f1"], u["macro_f1_std"]
        out.append(row)
        print(f"== {name}: best {cbest} (val {vbest:.4f}, default val {vdefault:.4f}) -> test "
              f"{row['macro_f1']:.3f} ± {row['macro_f1_std']:.3f}  (untuned {u['macro_f1']:.3f})", flush=True)
    json.dump(out, open(RESULTS_DIR / "step4_tuned.json", "w"), indent=2)
    print(f"saved {RESULTS_DIR / 'step4_tuned.json'}")


if __name__ == "__main__":
    main()
