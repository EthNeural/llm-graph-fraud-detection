"""
TwiBot-22 GNNs on the FULL 1M-user graph (checklist H3: expensive models WITH profile; all 7 relations).
Plan (builder, 5 Nov): all 5 GNNs with 1 seed, then the 2 best by VALIDATION macro-F1 with 3 seeds;
tables mark which models have 1 seed and which have 3. The MLP control (same network, edges ignored)
runs alongside so "does the graph help" has a like-for-like answer.

Node features: the user's own columns of features_full_e5.npz (15 profile fields, no "verified", +
multilingual-E5 bio PCA 32 = 47), standardised on train. Edges: following, followers, mention, retweet,
reply, quote, like, undirected; one union graph for GCN / SAGE / FAGCN / LINKX, 7 typed graphs for RGCN.

Same settings as TwiBot-20 step 4: 2 layers, hidden 128, dropout 0.5, Adam lr 0.01, weight decay 5e-4,
class-weighted cross-entropy, up to 300 epochs, early stopping on val macro-F1 (patience 50), cut-off
on val (twibot22_metrics). Written with sparse matrix products, not PyG message passing: PyG builds one
message per edge (13.4M edges x 128 floats ~ 7 GB per layer), too much next to the LLM in 16 GB.
  MLP    h = W2 relu(W1 x)
  GCN    h = A_hat relu(A_hat x W1) W2,  A_hat = D^-1/2 (A + I) D^-1/2           (Kipf & Welling)
  SAGE   h = W_self x + W_nbr mean_nbr(x), per layer                            (Hamilton et al., mean)
  FAGCN  linear in; 2 layers h' = eps h0 + sum_j tanh(g [h_i, h_j]) h_j / sqrt(d_i d_j); linear out
         (Bo et al., eps 0.3 as on TwiBot-20)
  LINKX  MLP_A(adjacency row) + MLP_X(x), combined by an MLP                    (Lim et al.)
  RGCN   h = W_0 x + sum_r W_r mean_{nbr in r}(x), per layer, 7 relations       (Schlichtkrull et al.)
Train labels only (B7); sealed users never scored (guard). Scores -> runs/full_gnn/<model>_s<seed>.npz
Run: MODELS=GCN,SAGE SEEDS=0 THREADS=6 python my_build/twibot22_gnn.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import twibot22_guard as guard
from twibot22_metrics import labels, evaluate, save_scores

torch.set_num_threads(int(os.environ.get("THREADS", 6)))
DATA = ROOT.parent / "data" / "twibot-22"
RELATIONS = ["following", "followers", "mention", "retweet", "reply", "quote", "like"]
MODELS = os.environ.get("MODELS", "MLP,GCN,SAGE,FAGCN,LINKX,RGCN").split(",")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]
EPOCHS, PATIENCE, HIDDEN, DROPOUT, FAGCN_EPS = 300, 50, 128, 0.5, 0.3
SMOKE = os.environ.get("SMOKE") == "1"            # 3 epochs per model, timing only, nothing saved
EPOCHS_FULL = EPOCHS
if SMOKE:
    EPOCHS = 3
RESULTS = ROOT / "results" / "twibot22" / "gnn_full.json"


def to_torch(M):
    M = M.tocoo()
    return torch.sparse_coo_tensor(np.vstack([M.row, M.col]), M.data.astype(np.float32), M.shape).coalesce()


def graphs(n):
    rels = np.load(DATA / "relations_uu.npz")
    typed, union = [], sp.csr_matrix((n, n), dtype=np.float32)
    for r in RELATIONS:
        e = rels[r]
        A = sp.csr_matrix((np.ones(e.shape[1], np.float32), (e[0], e[1])), shape=(n, n))
        A = ((A + A.T) > 0).astype(np.float32)
        A.setdiag(0)
        A.eliminate_zeros()
        typed.append(A)
        union = union + A
    union = (union > 0).astype(np.float32).tocsr()
    deg = np.asarray(union.sum(1)).ravel()
    mean = lambda M: sp.diags(1 / np.maximum(np.asarray(M.sum(1)).ravel(), 1)) @ M
    Ai = union + sp.eye(n, dtype=np.float32)
    di = np.asarray(Ai.sum(1)).ravel() ** -0.5
    return dict(union=union, deg=deg, gcn=to_torch(sp.diags(di) @ Ai @ sp.diags(di)), mean=to_torch(mean(union)),
                adj=to_torch(union), typed=[to_torch(mean(A)) for A in typed])


CHUNK = 1_000_000


def fagcn_chunk(h, a, b, row, col, norm, training):
    val = torch.tanh(a[row] + b[col]) * norm
    val = F.dropout(val, DROPOUT, bool(training))
    return torch.zeros_like(h).index_add_(0, row, h[col] * val[:, None])


class Net(torch.nn.Module):
    def __init__(self, kind, d, G, n):
        super().__init__()
        self.kind, self.G = kind, G
        L = torch.nn.Linear
        if kind in ("MLP", "GCN"):
            self.l1, self.l2 = L(d, HIDDEN), L(HIDDEN, 2)
        elif kind == "SAGE":
            self.s1, self.n1, self.s2, self.n2 = L(d, HIDDEN), L(d, HIDDEN, bias=False), L(HIDDEN, 2), L(HIDDEN, 2, bias=False)
        elif kind == "RGCN":
            self.s1, self.s2 = L(d, HIDDEN), L(HIDDEN, 2)
            self.r1 = torch.nn.ModuleList([L(d, HIDDEN, bias=False) for _ in G["typed"]])
            self.r2 = torch.nn.ModuleList([L(HIDDEN, 2, bias=False) for _ in G["typed"]])
        elif kind == "FAGCN":
            self.lin_in, self.lin_out = L(d, HIDDEN), L(HIDDEN, 2)
            self.gates = torch.nn.ModuleList([L(2 * HIDDEN, 1) for _ in range(2)])
            A = G["union"].tocoo()
            self.row, self.col = torch.as_tensor(A.row, dtype=torch.long), torch.as_tensor(A.col, dtype=torch.long)
            dd = np.maximum(G["deg"], 1)
            self.norm = torch.as_tensor((dd[A.row] * dd[A.col]) ** -0.5, dtype=torch.float32)
        elif kind == "LINKX":
            self.wa = torch.nn.Parameter(torch.randn(n, HIDDEN) * (1 / np.sqrt(HIDDEN)))   # MLP_A first layer = A @ W_A
            self.ba = torch.nn.Parameter(torch.zeros(HIDDEN))
            self.lx = L(d, HIDDEN)
            self.lc, self.out = L(2 * HIDDEN, HIDDEN), L(HIDDEN, 2)

    def drop(self, x):
        return F.dropout(x, DROPOUT, self.training)

    def forward(self, x):
        G, k, mm = self.G, self.kind, torch.sparse.mm
        if k == "MLP":
            return self.l2(self.drop(F.relu(self.l1(self.drop(x)))))
        if k == "GCN":
            h = F.relu(mm(G["gcn"], self.l1(self.drop(x))))
            return mm(G["gcn"], self.l2(self.drop(h)))
        if k == "SAGE":
            x = self.drop(x)
            h = self.drop(F.relu(self.s1(x) + self.n1(mm(G["mean"], x))))
            return self.s2(h) + self.n2(mm(G["mean"], h))
        if k == "RGCN":
            x = self.drop(x)
            h = self.s1(x) + sum(W(mm(A, x)) for W, A in zip(self.r1, G["typed"]))
            h = self.drop(F.relu(h))
            return self.s2(h) + sum(W(mm(A, h)) for W, A in zip(self.r2, G["typed"]))
        if k == "FAGCN":
            h = h0 = F.relu(self.lin_in(self.drop(x)))
            for g in self.gates:
                h = self.drop(h)
                w = g.weight[0]
                a = h @ w[:HIDDEN] + g.bias                      # tanh(g [h_i, h_j]) = tanh(a_i + b_j)
                b = h @ w[HIDDEN:]
                agg = torch.zeros_like(h)
                for c in range(0, len(self.row), CHUNK):        # edge chunks, recomputed in backward:
                    sl = slice(c, c + CHUNK)                    # memory ~ one chunk x hidden, not E x hidden
                    agg = agg + checkpoint(fagcn_chunk, h, a, b, self.row[sl], self.col[sl], self.norm[sl],
                                           torch.tensor(self.training), use_reentrant=False)
                h = FAGCN_EPS * h0 + agg
            return self.lin_out(self.drop(h))
        if k == "LINKX":
            ha = mm(G["adj"], self.wa) + self.ba
            hx = self.lx(self.drop(x))
            h = F.relu(self.lc(self.drop(torch.cat([ha, hx], 1))) + ha + hx)
            return self.out(self.drop(h))


def train(kind, x, G, y, tr, va, seed):
    torch.manual_seed(seed)
    model = Net(kind, x.shape[1], G, x.shape[0])
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    yt = torch.as_tensor(y)
    cnt = torch.bincount(yt[tr])
    weight = (cnt.sum() / (2 * cnt)).float()
    best, state, wait, t0 = -1, None, 0, time.time()
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x)[tr], yt[tr], weight=weight)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x).argmax(1).numpy()
        f1 = f1_score(y[va], pred[va], average="macro")
        if f1 > best:
            best, wait = f1, 0
            state = {k_: v.clone() for k_, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
        if ep % 25 == 0:
            print(f"    {kind} seed {seed} epoch {ep}: loss {loss.item():.4f} val F1 {f1:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return model(x).softmax(1)[:, 1].numpy(), ep + 1, (time.time() - t0) / 60


def main():
    y = labels()
    n = len(y)
    roles = guard.roles()
    tr, va, te = (np.where(roles == r)[0] for r in ("train", "val", "test_work"))
    f = np.load(DATA / "features_full_e5.npz")
    names = list(f["names"])
    own = [j for j, k in enumerate(names) if k.startswith("self:")]
    guard.check_columns([names[j] for j in own])
    X = f["X"][:, own].astype(np.float32)
    X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
    x = torch.as_tensor(X)
    t0 = time.time()
    G = graphs(n)
    print(f"features {X.shape[1]} (own profile + bio PCA); union edges {G['union'].nnz:,}; graphs built in {(time.time() - t0) / 60:.1f} min", flush=True)
    res = json.load(open(RESULTS)) if RESULTS.exists() else {}
    for kind in MODELS:
        for seed in SEEDS:
            key = f"{kind}_s{seed}"
            if key in res:
                continue
            s, epochs, mins = train(kind, x, G, y, tr, va, seed)
            if SMOKE:
                print(f"  SMOKE {kind}: {mins / epochs * 60:.1f} s per epoch -> up to {mins / epochs * EPOCHS_FULL:.0f} min for 300 epochs", flush=True)
                continue
            guard.assert_not_sealed(te)
            r = evaluate(y, s.astype(np.float32), va)
            r.update(epochs=epochs, minutes=round(mins, 1))
            save_scores("full_gnn", kind, seed, s)
            res[key] = r
            json.dump(res, open(RESULTS, "w"), indent=1)
            print(f"  {key}: val F1 {r['val_macro_f1']:.3f} | test F1 {r['all']['macro_f1']:.3f} AUROC {r['all']['auroc']:.3f} "
                  f"PR-AUC {r['all']['pr_auc']:.3f} | non-verified F1 {r['non_verified']['macro_f1']:.3f} | {epochs} epochs, {mins:.1f} min", flush=True)


if __name__ == "__main__":
    main()
