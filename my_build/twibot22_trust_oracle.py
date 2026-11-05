"""
Exploratory pilot, step 0 (decision 6 Nov ~01:05, artifacts/04_decisions_and_reasons.md): oracle ceiling
for a per-user neighbour-trust switch on TwiBot-22 (1M, working test, seed 0 scores).
Switch: own-feature tree (profile + bio) where the user's TRUE local homophily (E9: share of train-labelled
union-graph neighbours with the user's label) is below a threshold, XGB-Graph + profile elsewhere (incl. users
with no train neighbour). A = threshold 0.5, B = 0.25. Oracle = upper bound (uses the user's own label).
Primary: AUROC switch - XGB-Graph on all working-test users, paired bootstrap 2,000. Sealed users not used.
Output: results/twibot22/trust_oracle.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import twibot22_guard as guard
from twibot22_metrics import labels, val_cutoff
from twibot22_local_homophily import RELATIONS, DATA, local_h
import scipy.sparse as sp

OUT = ROOT / "results" / "twibot22" / "trust_oracle.json"
RUNS = ROOT / "results" / "twibot22" / "runs" / "full"


def main():
    assert not OUT.exists(), "never overwritten"
    y, roles = labels(), guard.roles()
    n = len(y)
    rels = np.load(DATA / "relations_uu.npz")
    U = sp.csr_matrix((n, n), dtype=np.float32)
    for r in RELATIONS:
        e = rels[r]
        M = sp.csr_matrix((np.ones(e.shape[1], np.float32), (e[0], e[1])), shape=(n, n))
        U = U + M + M.T
    U = (U > 0).astype(np.float32).tocsr()
    U.setdiag(0)
    U.eliminate_zeros()
    k, h = local_h(U, y.astype(np.float32), roles == "train")
    va, te = np.where(roles == "val")[0], np.where(roles == "test_work")[0]
    guard.assert_not_sealed(np.concatenate([va, te]))
    own = np.load(RUNS / "tree_self_prof_s0.npz")["scores"].astype(np.float64)
    xgb = np.load(RUNS / "xgb_graph_prof_s0.npz")["scores"].astype(np.float64)
    out = {}
    models = {"own-feature tree": own, "XGB-Graph": xgb}
    for name, thr in (("switch A (<0.5 -> own tree)", 0.5), ("switch B (<0.25 -> own tree)", 0.25)):
        use_own = (k > 0) & (h < thr)
        models[name] = np.where(use_own, own, xgb)
        out[name + " share switched (test)"] = float(use_own[te].mean())
    res = {}
    for name, s in models.items():
        c = val_cutoff(y[va], s[va])
        res[name] = dict(macro_f1=float(f1_score(y[te], s[te] > c, average="macro")), auroc=float(roc_auc_score(y[te], s[te])),
                         pr_auc=float(average_precision_score(y[te], s[te])), cut=float(c))
        print(f"{name:30s} macro-F1 {res[name]['macro_f1']:.3f}  AUROC {res[name]['auroc']:.4f}  PR-AUC {res[name]['pr_auc']:.3f}", flush=True)
    out["metrics"] = res
    rng = np.random.default_rng(0)
    yt = y[te]
    for name in ("switch A (<0.5 -> own tree)", "switch B (<0.25 -> own tree)"):
        a, b = models[name][te], xgb[te]
        ca, cb = res[name]["cut"], res["XGB-Graph"]["cut"]
        d = []
        for _ in range(2000):
            i = rng.integers(0, len(te), len(te))
            d.append([roc_auc_score(yt[i], a[i]) - roc_auc_score(yt[i], b[i]),
                      f1_score(yt[i], a[i] > ca, average="macro") - f1_score(yt[i], b[i] > cb, average="macro"),
                      average_precision_score(yt[i], a[i]) - average_precision_score(yt[i], b[i])])
        d = np.array(d)
        out[name + " - XGB-Graph"] = {m: [float(d[:, j].mean()), float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))]
                                      for j, m in enumerate(("auroc", "macro_f1", "pr_auc"))}
        r = out[name + " - XGB-Graph"]
        print(f"{name} - XGB-Graph: AUROC {r['auroc'][0]:+.4f} [{r['auroc'][1]:+.4f}, {r['auroc'][2]:+.4f}]  macro-F1 {r['macro_f1'][0]:+.3f} "
              f"[{r['macro_f1'][1]:+.3f}, {r['macro_f1'][2]:+.3f}]  PR-AUC {r['pr_auc'][0]:+.3f} [{r['pr_auc'][1]:+.3f}, {r['pr_auc'][2]:+.3f}]  "
              f"(switched {out[name + ' share switched (test)']:.1%})", flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("Saved", OUT.name)


if __name__ == "__main__":
    main()
