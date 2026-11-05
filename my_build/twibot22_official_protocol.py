"""
E11: TwiBot-22 leaderboard reconciliation (designed 5 Nov ~19:50 before any number; checklist E11).
The TwiBot-22 paper ranks graph models first by accuracy on the official split; we find graph ~ no graph
on our random split without `verified`. This tests whether the split or the `verified` shortcut explains it.

Cells: SPLIT in {ours, official} x VERIFIED in {0, 1}   (ours / 0 = the main results, not rerun here)
  ours      roles.npy: train / val / test_work (14% bots each)
  official  split.csv: train 7.8% / val 28.0% / test 29.4% bots
  Our SEALED users (incl. the 986 experts) are removed from every part of both splits: the seal holds.
  VERIFIED=1 adds the `verified` flag as one input column. REPLICATION ONLY: these models are never used
  in the main results (explicit override of the guard, REPLICATION=1).
Models (1M, bio-only own features = 15 profile fields + multilingual-E5 bio PCA 32; neighbours from the full
graph): KIND=trees -> own-feature tree, XGB-Graph (twibot22_cheap_models.hgb, seed 0); KIND=gnn -> MLP control,
FAGCN, RGCN (twibot22_gnn, seed 0, same settings).
Metrics on the cell's test users: accuracy and bot-class F1 at 0.5 (the paper's convention) and at the
validation cut-off (our rule), macro-F1, AUROC, test bot share and the all-human accuracy.
Output: results/twibot22/official_protocol_<KIND>.json (one key per cell / model)
Run: REPLICATION=1 KIND=gnn SPLIT=official VERIFIED=1 python my_build/twibot22_official_protocol.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

assert os.environ.get("REPLICATION") == "1", "replication-only script: set REPLICATION=1"
KIND, SPLIT, VERIFIED = os.environ["KIND"], os.environ["SPLIT"], os.environ.get("VERIFIED", "0") == "1"
assert KIND in ("trees", "gnn") and SPLIT in ("ours", "official")
if KIND == "trees":
    assert os.environ.get("OMP_NUM_THREADS") == "1", "trees: OMP_NUM_THREADS=1 (G1)"
os.environ.setdefault("MODEL", "xgb_graph")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import twibot22_guard as guard
from twibot22_metrics import labels, val_cutoff

DATA = ROOT.parent / "data" / "twibot-22"
RAW = ROOT.parent / "data" / "twibot-22-raw"
OUT = ROOT / "results" / "twibot22" / f"official_protocol_{os.environ['KIND']}.json"


def parts(y):
    roles = guard.roles()
    sealed = roles == "test_sealed"
    if SPLIT == "ours":
        tr, va, te = (np.where(roles == r)[0] for r in ("train", "val", "test_work"))
    else:
        lab = pd.read_csv(RAW / "label.csv")
        sp_ = pd.read_csv(RAW / "split.csv").set_index("id").split.reindex(lab.id).to_numpy()
        tr, va, te = (np.where((sp_ == r) & ~sealed)[0] for r in ("train", "val", "test"))
    guard.assert_not_sealed(np.concatenate([tr, va, te]))
    return tr, va, te


def report(y, s, va, te):
    cut = val_cutoff(y[va], s[va])
    yt, st = y[te], s[te]
    return dict(test_users=int(len(te)), test_bot_share=float(yt.mean()), all_human_accuracy=float(1 - yt.mean()),
                acc_at_0_5=float(accuracy_score(yt, st > 0.5)), bot_f1_at_0_5=float(f1_score(yt, st > 0.5)),
                acc_at_val_cut=float(accuracy_score(yt, st > cut)), bot_f1_at_val_cut=float(f1_score(yt, st > cut)),
                macro_f1_at_val_cut=float(f1_score(yt, st > cut, average="macro")), auroc=float(roc_auc_score(yt, st)),
                train_bot_share=None, val_cut=float(cut))


def main():
    y = labels()
    tr, va, te = parts(y)
    f = np.load(DATA / "features_full_e5.npz")
    names = list(f["names"])
    ver = guard.verified(pd.read_csv(RAW / "label.csv").id).astype(np.float32)[:, None]
    cell = f"{SPLIT}_verified{int(VERIFIED)}"
    res = json.load(open(OUT)) if OUT.exists() else {}
    print(f"cell {cell}: train {len(tr):,} ({y[tr].mean():.3f} bots), val {len(va):,} ({y[va].mean():.3f}), "
          f"test {len(te):,} ({y[te].mean():.3f})", flush=True)
    if KIND == "trees":
        from twibot22_cheap_models import hgb
        own = np.array([k.startswith("self:") for k in names])
        for model, cols in (("tree_self_prof", own), ("xgb_graph_prof", np.ones(len(names), bool))):
            key = f"{cell}/{model}"
            if key in res:
                continue
            X = f["X"][:, cols].astype(np.float32)
            if VERIFIED:
                X = np.hstack([X, ver])
            t0 = time.time()
            m = hgb(0).fit(X[tr], y[tr])
            s = m.predict_proba(X)[:, 1]
            r = report(y, s, va, te) | dict(train_bot_share=float(y[tr].mean()), minutes=round((time.time() - t0) / 60, 1))
            res[key] = r
            json.dump(res, open(OUT, "w"), indent=1)
            print(f"  {key}: acc@0.5 {r['acc_at_0_5']:.3f} botF1@0.5 {r['bot_f1_at_0_5']:.3f} | macroF1 {r['macro_f1_at_val_cut']:.3f} "
                  f"AUROC {r['auroc']:.3f} | all-human acc {r['all_human_accuracy']:.3f} ({r['minutes']} min)", flush=True)
    else:
        import torch
        import twibot22_gnn as g
        own = [j for j, k in enumerate(names) if k.startswith("self:")]
        X = f["X"][:, own].astype(np.float32)
        if VERIFIED:
            X = np.hstack([X, ver])
        X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
        x = torch.as_tensor(X)
        G = g.graphs(len(y))
        for model in ("MLP", "FAGCN", "RGCN"):
            key = f"{cell}/{model}"
            if key in res:
                continue
            s, epochs, mins = g.train(model, x, G, y, tr, va, 0)
            r = report(y, s, va, te) | dict(train_bot_share=float(y[tr].mean()), epochs=epochs, minutes=round(mins, 1))
            res[key] = r
            json.dump(res, open(OUT, "w"), indent=1)
            print(f"  {key}: acc@0.5 {r['acc_at_0_5']:.3f} botF1@0.5 {r['bot_f1_at_0_5']:.3f} | macroF1 {r['macro_f1_at_val_cut']:.3f} "
                  f"AUROC {r['auroc']:.3f} | all-human acc {r['all_human_accuracy']:.3f} ({epochs} epochs, {mins:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
