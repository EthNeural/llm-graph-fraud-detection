"""
TwiBot-22, exploratory (decision 6 Nov ~13:40): (1) LLM + graph fusion and (4) error analysis. Working-test users
and the 986 experts only (the sealed run is finished and is not touched again).

(1) FUSION. Inputs per user: the LoRA `target` score (mean of seeds 0 and 1; 1,000 validation + 5,000 test users)
    and a tree score (sample models, mean of 3 seeds): profile_only, tree_self_prof (profile + text),
    xgb_graph_prof (profile + text + neighbours). Stacker: logistic regression on [logit(tree), logit(LoRA)] fitted
    on the 1,000 validation users; cut-off from 5-fold cross-fitted validation scores. Baseline: the same tree with its
    cut-off chosen on the same 1,000 users. Paired bootstrap (2,000) on the 5,000 test users; slices: bio / no bio,
    >= 1 tweet, tree-uncertain (tree score between its 40th and 60th validation percentiles of bots... see code).
    Also: XGB-Graph + DGP vs XGB-Graph + target on the 1,500-user DGP subset (seed 0 both), and the stacker applied
    to the weak-only expert scores (expert labels).
(4) ERRORS. On the 5,000 users: bots missed by ALL of XGB-Graph, profile + text tree, LoRA and FAGCN (1M), and humans
    flagged by all, each at its own cut-off; profile statistics vs the caught bots / correct humans. On the experts:
    for the weak-label bots that all weak-trained models miss, the share the experts call human.
Output: results/twibot22/fusion_errors.json (aggregates only).  Run: OMP_NUM_THREADS=1 python my_build/twibot22_fusion_errors.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import twibot22_guard as guard
from twibot22_metrics import labels, val_cutoff

RES = ROOT / "results" / "twibot22"
RUNS = RES / "runs"
DATA = ROOT.parent / "data" / "twibot-22"
RAW = ROOT.parent / "data" / "twibot-22-raw"
OUT = RES / "fusion_errors.json"
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def seed_mean(folder, model):
    s = [np.load(RUNS / folder / f"{model}_s{k}.npz")["scores"].astype(float) for k in range(3) if (RUNS / folder / f"{model}_s{k}.npz").exists()]
    return np.mean(s, 0)


def lora_mean(n):
    s = np.full(n, np.nan)
    acc = []
    for tag in ("", "_s1"):
        z = np.load(RES / "A" / f"step6_Qwen3-1.7B_target{tag}_scores.npz")
        a = np.full(n, np.nan)
        a[z["test_rows"]], a[z["val_rows"]] = z["test"], z["val"]
        acc.append(a)
    return np.mean(acc, 0)


def boot(y, a, b, ca, cb, idx, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(2000):
        i = idx[rng.integers(0, len(idx), len(idx))]
        if y[i].min() == y[i].max():
            continue
        d.append([f1_score(y[i], a[i] > ca, average="macro") - f1_score(y[i], b[i] > cb, average="macro"),
                  roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]),
                  average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i])])
    d = np.array(d)
    return {m: [round(float(d[:, j].mean()), 4), round(float(np.percentile(d[:, j], 2.5)), 4), round(float(np.percentile(d[:, j], 97.5)), 4)]
            for j, m in enumerate(("macro_f1", "auroc", "pr_auc"))}


def fit_stacker(Z, y, va):
    m = LogisticRegression(C=1.0).fit(Z[va], y[va])
    oof = np.zeros(len(va))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(va, y[va]):
        oof[b] = LogisticRegression(C=1.0).fit(Z[va[a]], y[va[a]]).predict_proba(Z[va[b]])[:, 1]
    return m, val_cutoff(y[va], oof)


def main():
    assert not OUT.exists(), "never overwritten"
    y, roles = labels(), guard.roles()
    n = len(y)
    L = lora_mean(n)
    va = np.where(~np.isnan(L) & (roles == "val"))[0]
    te = np.where(~np.isnan(L) & (roles == "test_work"))[0]
    guard.assert_not_sealed(te)
    f = np.load(DATA / "features_sample_e5tw.npz")
    names, rows = list(f["names"]), f["rows"]
    pos = np.full(n, -1)
    pos[rows] = np.arange(len(rows))
    col = lambda k: f["X"][:, names.index(k)][pos]
    bio_empty, tweets = col("self:prof_bio empty"), np.load(DATA / "tweet_count_files.npy")
    out = {"n_val": int(len(va)), "n_test": int(len(te)), "test_bots": int(y[te].sum()), "fusion": {}, "slices": {}, "errors": {}}
    print(f"validation {len(va)}, test {len(te)} ({y[te].mean():.3f} bots)", flush=True)

    trees = {k: seed_mean("sample", k) for k in ("profile_only", "tree_self_prof", "xgb_graph_prof")}
    fused = {}
    for k, T in trees.items():
        Z = np.column_stack([logit(T), logit(L)])
        m, c = fit_stacker(Z, y, va)
        S = np.full(n, np.nan)
        idx = np.where(~np.isnan(T) & ~np.isnan(L))[0]
        S[idx] = m.predict_proba(Z[idx])[:, 1]
        ct = val_cutoff(y[va], T[va])
        fused[k] = (S, c, T, ct)
        r = {"weights_tree_llm": [round(float(w), 3) for w in m.coef_[0]],
             "tree_alone": {"macro_f1": round(float(f1_score(y[te], T[te] > ct, average="macro")), 4), "auroc": round(float(roc_auc_score(y[te], T[te])), 4)},
             "fused": {"macro_f1": round(float(f1_score(y[te], S[te] > c, average="macro")), 4), "auroc": round(float(roc_auc_score(y[te], S[te])), 4),
                       "pr_auc": round(float(average_precision_score(y[te], S[te])), 4)},
             "fused_minus_tree": boot(y, S, T, c, ct, te)}
        out["fusion"][k] = r
        print(f"(1) {k:16s} weights {r['weights_tree_llm']}  tree {r['tree_alone']}  fused {r['fused']}  diff {r['fused_minus_tree']}", flush=True)
    cl = val_cutoff(y[va], L[va])
    out["fusion"]["llm_alone"] = {"macro_f1": round(float(f1_score(y[te], L[te] > cl, average="macro")), 4), "auroc": round(float(roc_auc_score(y[te], L[te])), 4)}

    # slices for the XGB-Graph fusion
    S, c, T, ct = fused["xgb_graph_prof"]
    q_lo, q_hi = np.quantile(T[va], [0.4, 0.9])
    sl = {"no bio": te[bio_empty[te] > 0.5], "bio": te[bio_empty[te] < 0.5], ">= 1 tweet": te[tweets[te] > 0],
          "no tweet": te[tweets[te] == 0], "tree-uncertain (val 40th-90th pct)": te[(T[te] >= q_lo) & (T[te] <= q_hi)]}
    for k, idx in sl.items():
        if len(idx) > 50 and 0 < y[idx].sum() < len(idx):
            out["slices"][k] = {"n": int(len(idx)), "bots": int(y[idx].sum()), "auroc_tree": round(float(roc_auc_score(y[idx], T[idx])), 4),
                                "auroc_fused": round(float(roc_auc_score(y[idx], S[idx])), 4), "diff": boot(y, S, T, c, ct, idx)["auroc"]}
            print(f"    slice {k:36s} {out['slices'][k]}", flush=True)

    # DGP on its 1,500-user subset (seed 0), XGB-Graph + LLM
    z = np.load(RES / "A" / "step6_Qwen3-1.7B_dgp_scores.npz")
    D = np.full(n, np.nan)
    D[z["test_rows"]], D[z["val_rows"]] = z["test"], z["val"]
    z0 = np.load(RES / "A" / "step6_Qwen3-1.7B_target_scores.npz")
    T0 = np.full(n, np.nan)
    T0[z0["test_rows"]], T0[z0["val_rows"]] = z0["test"], z0["val"]
    sub = np.where(~np.isnan(D) & (roles == "test_work"))[0]
    Tx = trees["xgb_graph_prof"]
    res_d = {}
    for nm, Lx in (("target", T0), ("dgp", D)):
        Z = np.column_stack([logit(Tx), logit(Lx)])
        m, c2 = fit_stacker(Z, y, va)
        Sx = np.full(n, np.nan)
        ok = np.where(~np.isnan(Lx) & ~np.isnan(Tx))[0]
        Sx[ok] = m.predict_proba(Z[ok])[:, 1]
        res_d[nm] = (Sx, c2)
    out["fusion"]["dgp_subset"] = {"n": int(len(sub)), "xgb_plus_dgp_minus_xgb_plus_target": boot(y, res_d["dgp"][0], res_d["target"][0], res_d["dgp"][1], res_d["target"][1], sub)}
    print(f"(1) DGP subset: (XGB+dgp) - (XGB+target) {out['fusion']['dgp_subset']}", flush=True)

    # experts: apply the validation-fitted stacker to the weak-only expert scores
    lab = pd.read_csv(RAW / "label.csv")
    exp = pd.read_csv(RAW / "expert_labels" / "label_manual.csv")
    e_rows = pd.Series(np.arange(n), index=lab.id)[exp.id].to_numpy()
    y_e = (exp.label.values == "bot").astype(int)
    lo = np.load(RES / "A" / "expert_rotation_target.npz")
    order = pd.Series(np.arange(len(lo["rows"])), index=lo["rows"])[e_rows].to_numpy()
    Le = lo["p_weak_only"][order]
    tr_ = np.load(RES / "expert_rotation_trees.npz")
    for k, key in (("tree_self_prof", "sample_tree_self_prof"), ("xgb_graph_prof", "sample_xgb_graph_prof")):
        Te = tr_[f"{key}_p_weak_only"]
        Z = np.column_stack([logit(trees[k]), logit(L)])
        m = LogisticRegression(C=1.0).fit(Z[va], y[va])
        Se = m.predict_proba(np.column_stack([logit(Te), logit(Le)]))[:, 1]
        out["fusion"][f"experts_{k}"] = {"auroc_tree": round(float(roc_auc_score(y_e, Te)), 4), "auroc_fused": round(float(roc_auc_score(y_e, Se)), 4),
                                         "auroc_llm": round(float(roc_auc_score(y_e, Le)), 4)}
        print(f"(1) experts {k}: {out['fusion'][f'experts_{k}']}", flush=True)

    # (4) errors on the 5,000 users
    F = seed_mean("full_gnn", "FAGCN")
    models = {"XGB-Graph": trees["xgb_graph_prof"], "profile+text tree": trees["tree_self_prof"], "LoRA": L, "FAGCN": F}
    cuts = {k: val_cutoff(y[va], v[va]) for k, v in models.items()}
    flag = np.column_stack([models[k][te] > cuts[k] for k in models])
    yb = y[te] == 1
    miss_all, catch_all = yb & ~flag.any(1), yb & flag.all(1)
    fp_all, tn_all = ~yb & flag.all(1), ~yb & ~flag.any(1)
    out["errors"]["counts"] = {"bots": int(yb.sum()), "missed_by_all": int(miss_all.sum()), "caught_by_all": int(catch_all.sum()),
                               "humans": int((~yb).sum()), "flagged_by_all": int(fp_all.sum()), "cleared_by_all": int(tn_all.sum())}
    print("(4) counts", out["errors"]["counts"], flush=True)
    stats = ["self:prof_bio empty", "self:prof_log followers", "self:prof_log following", "self:prof_log tweet count",
             "self:prof_log listed count", "self:prof_account age (days)", "self:prof_default profile image", "self:prof_bio length"]
    groups = {"bots missed by all": te[miss_all], "bots caught by all": te[catch_all], "humans flagged by all": te[fp_all], "humans cleared by all": te[tn_all]}
    tab = {}
    for g, idx in groups.items():
        tab[g] = {"n": int(len(idx)), **{k.replace("self:prof_", ""): round(float(np.mean(col(k)[idx])), 3) for k in stats},
                  "has tweets": round(float(np.mean(tweets[idx] > 0)), 3)}
        print(f"    {g:24s} {tab[g]}", flush=True)
    out["errors"]["profile_by_group"] = tab

    # experts: are the shared misses label disagreements?
    Te_x = tr_["sample_xgb_graph_prof_p_weak_only"]
    Te_t = tr_["sample_tree_self_prof_p_weak_only"]
    flags_e = np.column_stack([Te_x > tr_["sample_xgb_graph_prof_cut_weak_only"], Te_t > tr_["sample_tree_self_prof_cut_weak_only"], Le > lo["cut_weak_only"]])
    yw = tr_["y_weak"]
    wb = yw == 1
    miss_e, catch_e = wb & ~flags_e.any(1), wb & flags_e.all(1)
    fp_e = (yw == 0) & flags_e.all(1)
    out["errors"]["experts"] = {"weak_bots": int(wb.sum()), "weak_bots_missed_by_all": int(miss_e.sum()),
                                "of_those_expert_human": int((miss_e & (y_e == 0)).sum()),
                                "weak_bots_caught_by_all": int(catch_e.sum()), "of_those_expert_human_caught": int((catch_e & (y_e == 0)).sum()),
                                "weak_humans_flagged_by_all": int(fp_e.sum()), "of_those_expert_bot": int((fp_e & (y_e == 1)).sum())}
    print("(4) experts", out["errors"]["experts"], flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("Saved", OUT.name)


if __name__ == "__main__":
    main()
