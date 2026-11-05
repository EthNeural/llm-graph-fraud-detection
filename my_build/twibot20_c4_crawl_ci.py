"""
TwiBot-20 to-do C4 (Fri 6 Nov ~04:40): paired bootstrap CIs (2,000) on the tree baseline's crawl-artefact result
(4 Nov rerun: + in/out friend/follow degrees gave +0.025 macro-F1 on all test users, -0.009 on non-verified).
Reuses tree_baseline.py exactly (features, HistGradientBoosting, 3 seeds, validation cut-off); never imports torch
(OpenMP clash, see tree_baseline.py). Scores are averaged over the 3 seeds per user before the bootstrap.
Pairs: (crawl artefact) - (own + neighbour mean + degree); (own + neighbour mean + degree) - (own only).
Run: OMP_NUM_THREADS=1 DATASET=twibot20 SETTING=A MODEL=tree python my_build/twibot20_c4_crawl_ci.py
"""
import json
import os

os.environ.setdefault("MODEL", "tree")
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from common import RESULTS_DIR, best_threshold, get_splits, load
from tree_baseline import SEEDS, build_features, verified_mask

_, e, y = load()
train, val, test = get_splits(y)
nv = test[~verified_mask()[test]]
feats = build_features(e, y, train)
P, C = {}, {}
for name, X in feats.items():
    ps = []
    for sd in SEEDS:
        clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=sd, early_stopping=True, validation_fraction=0.1)
        ps.append(clf.fit(X[train], y[train]).predict_proba(X)[:, 1])
    P[name] = np.mean(ps, 0)
    C[name] = best_threshold(y[val], P[name][val])
names = list(feats)
rng = np.random.default_rng(0)
out = {}
for a, b in ((names[2], names[1]), (names[1], names[0])):
    r = {}
    for sl, idx in (("all", test), ("non_verified", nv)):
        d = []
        for _ in range(2000):
            i = idx[rng.integers(0, len(idx), len(idx))]
            if y[i].min() == y[i].max():
                continue
            d.append([f1_score(y[i], P[a][i] > C[a], average="macro") - f1_score(y[i], P[b][i] > C[b], average="macro"),
                      roc_auc_score(y[i], P[a][i]) - roc_auc_score(y[i], P[b][i]),
                      average_precision_score(y[i], P[a][i]) - average_precision_score(y[i], P[b][i])])
        d = np.array(d)
        r[sl] = {m: [round(float(d[:, j].mean()), 4), round(float(np.percentile(d[:, j], 2.5)), 4), round(float(np.percentile(d[:, j], 97.5)), 4)]
                 for j, m in enumerate(("macro_f1", "auroc", "pr_auc"))} | {"n": int(len(idx))}
        print(f"{a[:40]} - {b[:40]} [{sl}]: " + "  ".join(f"{m} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for m, v in r[sl].items() if m != "n"), flush=True)
    out[f"{a} - {b}"] = r
(RESULTS_DIR / "c4_crawl_artefact_ci.json").write_text(json.dumps(out, indent=1))
print("saved", RESULTS_DIR / "c4_crawl_artefact_ci.json")
