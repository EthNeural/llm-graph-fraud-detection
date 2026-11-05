"""
Q2 (checklist E9; designed 5 Nov ~19:20 before any number): local heterogeneity on TwiBot-22.

Local homophily of user v = share of v's TRAIN-labelled neighbours (both directions) that have v's
label. Neighbour labels come from train users only; v's own label is used only to compute / slice it
(the standard per-node homophily analysis), never as a model input. Sealed users are never scored.

(a) Distribution on VAL users, union graph and per relation: users with >= 1 train neighbour, deciles,
    share >= 0.5, separately for bots and humans.
(b) Where graph models gain, on the WORKING test users (1M design, seed 0 scores), per bin
    (no train neighbour, [0, .25), [.25, .5), [.5, .75), [.75, 1]):
      FAGCN vs MLP control, RGCN vs MLP control  (same network, edges on / off)
      XGB-Graph + profile vs own-feature tree + profile
    AUROC of each and the paired bootstrap difference (1,000 resamples), macro-F1 at each model's own
    validation cut-off, bin size and bot share.
Output: results/twibot22/local_homophily.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import twibot22_guard as guard
from twibot22_metrics import labels, val_cutoff

DATA = ROOT.parent / "data" / "twibot-22"
RUNS = ROOT / "results" / "twibot22" / "runs"
RELATIONS = ["following", "followers", "mention", "retweet", "reply", "quote", "like"]
BINS = [("no train neighbour", None), ("[0, 0.25)", (0, .25)), ("[0.25, 0.5)", (.25, .5)), ("[0.5, 0.75)", (.5, .75)), ("[0.75, 1]", (.75, 1.01))]


def local_h(A, y, is_train):
    # number of train neighbours, and number of train-labelled bot neighbours
    t = is_train.astype(np.float32)
    k = A @ t
    kb = A @ (t * y)
    same = np.where(y == 1, kb, k - kb)
    return k, np.where(k > 0, same / np.maximum(k, 1), np.nan)


def main():
    y, roles = labels(), guard.roles()
    n = len(y)
    is_train = roles == "train"
    rels = np.load(DATA / "relations_uu.npz")
    union = sp.csr_matrix((n, n), dtype=np.float32)
    per_rel = {}
    for r in RELATIONS:
        e = rels[r]
        M = sp.csr_matrix((np.ones(e.shape[1], np.float32), (e[0], e[1])), shape=(n, n))
        M = ((M + M.T) > 0).astype(np.float32).tocsr()
        per_rel[r] = M
        union = union + M
    union = (union > 0).astype(np.float32).tocsr()
    union.setdiag(0)
    union.eliminate_zeros()

    out = {"a_distribution_val": {}, "b_gain_by_bin_test_work": {}}
    va = np.where(roles == "val")[0]
    for name, A in [("union", union)] + list(per_rel.items()):
        k, h = local_h(A, y.astype(np.float32), is_train)
        row = {}
        for cls, c in (("bot", 1), ("human", 0)):
            idx = va[y[va] == c]
            hv = h[idx][~np.isnan(h[idx])]
            row[cls] = dict(users=int(len(idx)), with_train_neighbour=float(len(hv) / len(idx)),
                            deciles=[round(float(q), 3) for q in np.quantile(hv, np.linspace(0.1, 0.9, 9))] if len(hv) else [],
                            share_at_least_half=float(np.mean(hv >= 0.5)) if len(hv) else None,
                            mean=float(hv.mean()) if len(hv) else None)
        out["a_distribution_val"][name] = row
        print(f"(a) {name:10s} bots: {row['bot']['with_train_neighbour']:.1%} have a train nbr, mean local h {row['bot']['mean']:.3f}, "
              f">=0.5: {row['bot']['share_at_least_half']:.1%} | humans: mean {row['human']['mean']:.3f}, >=0.5: {row['human']['share_at_least_half']:.1%}", flush=True)

    k, h = local_h(union, y.astype(np.float32), is_train)
    te = np.where(roles == "test_work")[0]
    guard.assert_not_sealed(te)
    load = lambda d, m: np.load(RUNS / d / f"{m}_s0.npz")["scores"].astype(np.float64)
    pairs = {"FAGCN_vs_MLP": (load("full_gnn", "FAGCN"), load("full_gnn", "MLP")),
             "RGCN_vs_MLP": (load("full_gnn", "RGCN"), load("full_gnn", "MLP")),
             "XGBGraph_vs_ownTree (profile)": (load("full", "xgb_graph_prof"), load("full", "tree_self_prof"))}
    rng = np.random.default_rng(0)
    for pname, (a, b) in pairs.items():
        ca, cb = val_cutoff(y[va], a[va]), val_cutoff(y[va], b[va])
        res = {}
        for bname, lim in BINS:
            m = (k[te] == 0) if lim is None else (k[te] > 0) & (h[te] >= lim[0]) & (h[te] < lim[1])
            idx = te[m]
            r = dict(users=int(len(idx)), bot_share=float(y[idx].mean()) if len(idx) else None)
            if len(idx) and 0 < y[idx].sum() < len(idx):
                r.update(auroc_a=float(roc_auc_score(y[idx], a[idx])), auroc_b=float(roc_auc_score(y[idx], b[idx])),
                         f1_a=float(f1_score(y[idx], a[idx] > ca, average="macro")), f1_b=float(f1_score(y[idx], b[idx] > cb, average="macro")))
                d = []
                for _ in range(1000):
                    i = rng.integers(0, len(idx), len(idx))
                    if 0 < y[idx][i].sum() < len(i):
                        d.append(roc_auc_score(y[idx][i], a[idx][i]) - roc_auc_score(y[idx][i], b[idx][i]))
                r["auroc_diff_ci"] = [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            else:
                # one class only: AUROC undefined; report accuracy at each model's cut-off instead
                if len(idx):
                    r.update(acc_a=float(np.mean((a[idx] > ca) == y[idx])), acc_b=float(np.mean((b[idx] > cb) == y[idx])))
            res[bname] = r
            msg = (f"AUROC {r['auroc_a']:.3f} vs {r['auroc_b']:.3f}, diff {r['auroc_diff_ci'][0]:+.3f} [{r['auroc_diff_ci'][1]:+.3f}, {r['auroc_diff_ci'][2]:+.3f}]"
                   if "auroc_a" in r else (f"one class; accuracy {r['acc_a']:.3f} vs {r['acc_b']:.3f}" if "acc_a" in r else "empty"))
            print(f"(b) {pname:30s} {bname:20s} n {r['users']:>6,} bots {r['bot_share'] if r['bot_share'] is not None else float('nan'):.2f}  {msg}", flush=True)
        out["b_gain_by_bin_test_work"][pname] = res
    (ROOT / "results" / "twibot22" / "local_homophily.json").write_text(json.dumps(out, indent=1))
    print("Saved local_homophily.json")


if __name__ == "__main__":
    main()
