"""
Weak -> expert second training stage for the tree models, 5-fold rotation over the 986 experts
(checklist G7, decided by the builder 5 Nov ~13:20). Same folds as the LoRA rotation
(StratifiedKFold(5, shuffle, random_state 42) on the expert label, experts in label_manual.csv order).

Rules, fixed before any expert score (5 Nov ~19:10):
  - weak model: the model's seed-0 settings (twibot22_cheap_models.hgb), trained on our TRAIN users (weak labels)
  - per fold: a copy of the weak model CONTINUES boosting for 50 more rounds (warm start, no early
    stopping, same learning rate) on the ~789 experts of the other 4 folds, expert labels
  - cut-off for macro-F1: chosen on the fold model's scores of its own 4 training folds
  - the weak-only model scores all 986 experts too (cut-off: its validation cut-off), for the paired comparison
Models: sample design profile_only, tree_self_prof, xgb_graph_prof; full design xgb_graph_prof.
BLIND: experts are sealed users, so no metric is printed; scores are read once, in the final run.
Output: results/twibot22/expert_rotation_trees.npz (one entry per design_model). No torch (G1).
Run: OMP_NUM_THREADS=1 python my_build/twibot22_expert_rotation_trees.py
"""
import copy
import os
import sys
from pathlib import Path

assert os.environ.get("OMP_NUM_THREADS") == "1"
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MODEL", "xgb_graph")
import twibot22_guard as guard
from twibot22_cheap_models import group, hgb
from twibot22_metrics import labels, val_cutoff

DATA = ROOT.parent / "data" / "twibot-22"
OUT = ROOT / "results" / "twibot22" / "expert_rotation_trees.npz"
EXTRA_ROUNDS = 50
JOBS = [("sample", "profile_only"), ("sample", "tree_self_prof"), ("sample", "xgb_graph_prof"), ("full", "xgb_graph_prof")]


def columns(names, model):
    g = np.array([group(k) for k in names])
    own = np.array([k.startswith("self:") for k in names])
    return {"profile_only": own & (g == "profile"), "tree_self_prof": own, "xgb_graph_prof": np.ones(len(names), bool)}[model]


def main():
    assert not OUT.exists(), f"{OUT.name} exists: never overwritten"
    y_weak, roles = labels(), guard.roles()
    lab = pd.read_csv(ROOT.parent / "data" / "twibot-22-raw" / "label.csv")
    exp = pd.read_csv(ROOT.parent / "data" / "twibot-22-raw" / "expert_labels" / "label_manual.csv")
    row_of = pd.Series(np.arange(len(lab)), index=lab.id)
    e_rows = row_of[exp.id].to_numpy()
    y_exp = (exp.label.values == "bot").astype(int)
    assert (roles[e_rows] == "test_sealed").all()
    folds = np.zeros(len(e_rows), int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(e_rows, y_exp)):
        folds[te] = k
    out = dict(rows=e_rows, y_expert=y_exp, y_weak=y_weak[e_rows], fold=folds)
    for design, model in JOBS:
        f = np.load(DATA / ("features_sample_e5tw.npz" if design == "sample" else "features_full_e5.npz"))
        rows = f["rows"] if "rows" in f.files else np.arange(len(y_weak))
        pos = np.full(len(y_weak), -1)
        pos[rows] = np.arange(len(rows))
        assert (pos[e_rows] >= 0).all()
        X = f["X"][:, columns(list(f["names"]), model)].astype(np.float32)
        tr, va = rows[roles[rows] == "train"], rows[roles[rows] == "val"]
        weak = hgb(0).fit(X[pos[tr]], y_weak[tr])
        Xe = X[pos[e_rows]]
        key = f"{design}_{model}"
        out[f"{key}_p_weak_only"] = weak.predict_proba(Xe)[:, 1]
        out[f"{key}_cut_weak_only"] = val_cutoff(y_weak[va], weak.predict_proba(X[pos[va]])[:, 1])
        p_rot, cut_rot = np.empty(len(e_rows)), np.empty(len(e_rows))
        for k in range(5):
            trk, tek = folds != k, folds == k
            m = copy.deepcopy(weak)
            m.set_params(warm_start=True, early_stopping=False, max_iter=weak.n_iter_ + EXTRA_ROUNDS)
            m.fit(Xe[trk], y_exp[trk])
            p_rot[tek] = m.predict_proba(Xe[tek])[:, 1]
            cut_rot[tek] = val_cutoff(y_exp[trk], m.predict_proba(Xe[trk])[:, 1])
        out[f"{key}_p_rotated"], out[f"{key}_cut_rotated"] = p_rot, cut_rot
        print(f"{key}: weak model {weak.n_iter_} rounds, 5 folds x {EXTRA_ROUNDS} expert rounds; saved (blind)", flush=True)
    np.savez(OUT, **out)
    print("Saved", OUT, "(blind: read in the final run)")


if __name__ == "__main__":
    main()
