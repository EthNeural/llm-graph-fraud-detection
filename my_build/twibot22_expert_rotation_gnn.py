"""
Weak -> expert second training stage for the GNNs (checklist G7 extension, builder 5 Nov ~20:35; rules
fixed before any expert score). Same folds as the LoRA and tree rotations (StratifiedKFold(5, shuffle,
random_state 42) on the expert label, experts in label_manual.csv order).

  models   FAGCN, RGCN (the 2 best GNNs by validation macro-F1) and the MLP control (same network, no edges)
  weak     trained exactly as twibot22_gnn.py (seed 0, early stopping on val macro-F1, weak labels)
  expert   per fold: continue from the weak weights for 30 epochs, Adam lr 1e-3 (10x lower), weight
           decay 5e-4, class-weighted cross-entropy on the ~789 experts of the other 4 folds (expert
           labels); no early stopping (no clean validation set); full-graph forward as in training
  cut-off  for macro-F1: chosen on the fold model's scores of its own 4 training folds
  also     the weak-only model's scores on all 986 experts (cut-off: its weak validation cut-off)
BLIND: the experts are sealed users, so no metric is printed; read once, in the final run.
Pauses automatically while a LoRA trains (launcher). Output: results/twibot22/expert_rotation_gnn.npz
Run: THREADS=4 python my_build/twibot22_expert_rotation_gnn.py
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import twibot22_guard as guard
import twibot22_gnn as g
from twibot22_metrics import labels, val_cutoff

OUT = ROOT / "results" / "twibot22" / "expert_rotation_gnn.npz"
RAW = ROOT.parent / "data" / "twibot-22-raw"
EXPERT_EPOCHS, EXPERT_LR = 30, 1e-3


def train_weak(kind, x, G, y, tr, va):
    # twibot22_gnn.train, but returning the model at its best validation epoch
    from sklearn.metrics import f1_score
    torch.manual_seed(0)
    model = g.Net(kind, x.shape[1], G, x.shape[0])
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    yt = torch.as_tensor(y)
    cnt = torch.bincount(yt[tr])
    weight = (cnt.sum() / (2 * cnt)).float()
    best, state, wait = -1, None, 0
    for ep in range(g.EPOCHS):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x)[tr], yt[tr], weight=weight).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            f1 = f1_score(y[va], model(x).argmax(1).numpy()[va], average="macro")
        if f1 > best:
            best, wait, state = f1, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= g.PATIENCE:
                break
    model.load_state_dict(state)
    return model


def scores(model, x):
    model.eval()
    with torch.no_grad():
        return model(x).softmax(1)[:, 1].numpy()


def main():
    assert not OUT.exists(), f"{OUT.name} exists: never overwritten"
    y, roles = labels(), guard.roles()
    tr, va = np.where(roles == "train")[0], np.where(roles == "val")[0]
    lab = pd.read_csv(RAW / "label.csv")
    exp = pd.read_csv(RAW / "expert_labels" / "label_manual.csv")
    e_rows = pd.Series(np.arange(len(lab)), index=lab.id)[exp.id].to_numpy()
    y_exp = (exp.label.values == "bot").astype(int)
    assert (roles[e_rows] == "test_sealed").all()
    folds = np.zeros(len(e_rows), int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(e_rows, y_exp)):
        folds[te] = k
    f = np.load(g.DATA / "features_full_e5.npz")
    names = list(f["names"])
    own = [j for j, k in enumerate(names) if k.startswith("self:")]
    guard.check_columns([names[j] for j in own])
    X = f["X"][:, own].astype(np.float32)
    X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
    x = torch.as_tensor(X)
    G = g.graphs(len(y))
    out = dict(rows=e_rows, y_expert=y_exp, y_weak=y[e_rows], fold=folds)
    for kind in ("MLP", "RGCN", "FAGCN"):
        t0 = time.time()
        weak = train_weak(kind, x, G, y, tr, va)
        s = scores(weak, x)
        out[f"{kind}_p_weak_only"], out[f"{kind}_cut_weak_only"] = s[e_rows], val_cutoff(y[va], s[va])
        p_rot, cut_rot = np.empty(len(e_rows)), np.empty(len(e_rows))
        for k in range(5):
            trk, tek = e_rows[folds != k], e_rows[folds == k]
            yk = torch.zeros(len(y), dtype=torch.long)
            yk[torch.as_tensor(e_rows)] = torch.as_tensor(y_exp)
            m = copy.deepcopy(weak)
            opt = torch.optim.Adam(m.parameters(), lr=EXPERT_LR, weight_decay=5e-4)
            cnt = torch.bincount(yk[trk], minlength=2)
            weight = (cnt.sum() / (2 * cnt)).float()
            for _ in range(EXPERT_EPOCHS):
                m.train()
                opt.zero_grad()
                F.cross_entropy(m(x)[trk], yk[trk], weight=weight).backward()
                opt.step()
            sk = scores(m, x)
            p_rot[folds == k] = sk[tek]
            cut_rot[folds == k] = val_cutoff(y_exp[folds != k], sk[trk])
        out[f"{kind}_p_rotated"], out[f"{kind}_cut_rotated"] = p_rot, cut_rot
        np.savez(OUT.with_suffix(".partial.npz"), **out)
        print(f"{kind}: weak + 5 folds x {EXPERT_EPOCHS} expert epochs done in {(time.time() - t0) / 60:.1f} min; saved (blind)", flush=True)
    np.savez(OUT, **out)
    print("Saved", OUT, "(blind: read in the final run)")


if __name__ == "__main__":
    main()
