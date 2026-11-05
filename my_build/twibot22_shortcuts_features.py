"""
Shortcut rule on every column of the model feature table (checklist D2 extension). CPU, no torch.
Same rule as twibot22_shortcuts.py: a column is a SHORTCUT if a depth-1 tree on it, fitted on OUR
TRAIN users, has a leaf with >= 200 training users that is >= 99% one class. Scored on VAL only.
Covers own fields, bio PCA, and every neighbour average / count (per relation, in/out, 1-2 hop).
Input:  data/twibot-22/features_full.npz (814 columns; MiniLM bio PCA, 7 relations)
Output: results/twibot22/shortcuts_features.json
Run:    python my_build/twibot22_shortcuts_features.py      (~5 min, 8 processes)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from twibot22_shortcuts import check, MIN_LEAF_USERS, PURITY

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "twibot-22"


def main():
    y = (pd.read_csv(ROOT.parent / "data" / "twibot-22-raw" / "label.csv").label.values == "bot").astype(int)
    split = np.load(DATA / "split_ours.npy")
    tr, va = split == "train", split == "val"
    import os
    f = np.load(DATA / os.environ.get("FEATURES", "features_full_e5.npz"))
    X, names = f["X"], list(f["names"])
    if "rows" in f.files:                                  # a sample table (H10): restrict labels and split to its rows
        y, tr, va = y[f["rows"]], tr[f["rows"]], va[f["rows"]]
    res = Parallel(n_jobs=int(os.environ.get('JOBS', 8)))(delayed(check)(X[:, j].astype(np.float32), y, tr, va) for j in range(X.shape[1]))
    rows = {}
    for name, r in zip(names, res):
        leaf = r["purest_leaf"]
        r["verdict"] = (f"SHORTCUT: {leaf['users']:,} train users ({leaf['share']:.1%}) are {leaf['purity']:.1%} {leaf['majority']}"
                        if leaf and leaf["purity"] >= PURITY else "allowed")
        rows[name] = r
    out = dict(rule=f"depth-1 leaf with >= {MIN_LEAF_USERS} train users at >= {PURITY:.0%} purity; scored on val",
               n_columns=len(names), shortcuts=[k for k, r in rows.items() if r["verdict"] != "allowed"], columns=rows)
    (ROOT / "results" / "twibot22" / f"shortcuts_{os.environ.get('FEATURES', 'features_full_e5.npz').replace('.npz', '')}.json").write_text(json.dumps(out, indent=1))
    top = sorted(rows.items(), key=lambda kv: -abs(kv[1]["val_auroc"] - 0.5))[:15]
    print(f"{len(names)} columns; shortcuts: {out['shortcuts'] or 'none'}\nstrongest single columns (val AUROC):")
    for k, r in top:
        print(f"  {k:55s} {r['val_auroc']:.3f}  purest big leaf {r['purest_leaf']}")


if __name__ == "__main__":
    main()
