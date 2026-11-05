"""
TwiBot-22 FINAL RUN (rules fixed Fri 6 Nov ~03:20, artifacts/04_decisions_and_reasons.md, before any sealed score).
The builder authorised it to run unattended after the GPU queue (~13:00). Read ONCE; no model or rule changes after.

Users: the 49,951 SEALED NON-EXPERT users (test_sealed minus the 986 experts; the experts were read early, 5 Nov 23:50).
Models: the saved scores of every finished CPU model, unchanged (no retraining):
  1M (bio only): profile_only, tree_self_prof, tree_self_noprof, xgb_graph_prof, xgb_graph_noprof, links_only,
                 GNNs MLP, GCN, SAGE, FAGCN, LINKX, RGCN (all seeds saved);
  sample (bio + 20 tweets): the same 6 tree models, on the ~5,000 sealed non-expert sample users.
  The LLMs (LoRA target / dgp) were scored on working-test users only and are not part of the sealed run.
Metrics: each model's own validation cut-off (unchanged rule); macro-F1, AUROC, PR-AUC, PR-AUC reweighted to 5% / 1%
  bots; mean over seeds. Paired bootstrap (2,000) on seed 0 for the key pairs (as ci_pairs.json, plus FAGCN - MLP,
  RGCN - MLP). Working-test values printed alongside, to check that nothing was tuned to the working set.
D7 (checklist): graph gain under weak labels (sealed non-expert users) vs under expert labels (986 experts, the same
  weak-trained scores): XGB-Graph - own tree (1M and sample), FAGCN - MLP, RGCN - MLP; AUROC and macro-F1; CI of the
  difference from independent bootstraps (2,000).
DRY=1: same code on the WORKING test users instead (tests the script; never reads a sealed user).
Output: results/twibot22/final_run.json (DRY: final_run_dry.json)
Run: FINAL_RUN=1 OMP_NUM_THREADS=2 python my_build/twibot22_final_run.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import twibot22_guard as guard
from twibot22_metrics import labels, metrics, val_cutoff

DRY = os.environ.get("DRY") == "1"
assert DRY or os.environ.get("FINAL_RUN") == "1", "set FINAL_RUN=1 (or DRY=1 to test on working users)"
RES = ROOT / "results" / "twibot22"
RUNS = RES / "runs"
RAW = ROOT.parent / "data" / "twibot-22-raw"
OUT = RES / ("final_run_dry.json" if DRY else "final_run.json")
B = 2000
TREES = ["profile_only", "tree_self_prof", "tree_self_noprof", "xgb_graph_prof", "xgb_graph_noprof", "links_only"]
GNNS = ["MLP", "GCN", "SAGE", "FAGCN", "LINKX", "RGCN"]


def load(folder, model, seed):
    f = RUNS / folder / f"{model}_s{seed}.npz"
    return np.load(f)["scores"].astype(np.float64) if f.exists() else None


def boot_pair(y, a, b, ca, cb, idx, seed=0):
    rng = np.random.default_rng(seed)
    ya, sa, sb = y[idx], a[idx], b[idx]
    pa, pb = sa > ca, sb > cb
    d = []
    for _ in range(B):
        i = rng.integers(0, len(idx), len(idx))
        if not 0 < ya[i].sum() < len(i):
            continue
        d.append([f1_score(ya[i], pa[i], average="macro") - f1_score(ya[i], pb[i], average="macro"),
                  roc_auc_score(ya[i], sa[i]) - roc_auc_score(ya[i], sb[i]),
                  average_precision_score(ya[i], sa[i]) - average_precision_score(ya[i], sb[i])])
    d = np.array(d)
    return {m: [float(d[:, j].mean()), float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))]
            for j, m in enumerate(("macro_f1", "auroc", "pr_auc"))}


def boot_gain(y, a, b, ca, cb, idx, rng):
    ya, sa, sb, pa, pb = y[idx], a[idx], b[idx], a[idx] > ca, b[idx] > cb
    out = []
    for _ in range(B):
        i = rng.integers(0, len(idx), len(idx))
        if not 0 < ya[i].sum() < len(i):
            out.append([np.nan, np.nan])
            continue
        out.append([roc_auc_score(ya[i], sa[i]) - roc_auc_score(ya[i], sb[i]),
                    f1_score(ya[i], pa[i], average="macro") - f1_score(ya[i], pb[i], average="macro")])
    return np.array(out)


def main():
    assert not OUT.exists(), f"{OUT.name} exists: the final run is read once and never overwritten"
    t0 = time.time()
    y, roles = labels(), guard.roles()
    lab = pd.read_csv(RAW / "label.csv")
    exp = pd.read_csv(RAW / "expert_labels" / "label_manual.csv")
    e_rows = pd.Series(np.arange(len(lab)), index=lab.id)[exp.id].to_numpy()
    y_exp = (exp.label.values == "bot").astype(int)
    is_exp = np.zeros(len(y), bool)
    is_exp[e_rows] = True
    if DRY:
        target = np.where(roles == "test_work")[0]
    else:
        target = np.where(guard.score_mask() & ~is_exp)[0]          # sealed, minus the experts read early
        assert len(target) == 49951, len(target)
    work = np.where(roles == "test_work")[0]
    va_all = np.where(roles == "val")[0]
    out = {"users": int(len(target)), "bots": int(y[target].sum()), "dry": DRY, "models": {}, "pairs": {}, "d7": {}}
    print(f"{'DRY RUN (working users)' if DRY else 'FINAL RUN'}: {len(target):,} users, {y[target].mean():.3f} bots", flush=True)

    cache = {}
    for folder, models in (("full", TREES), ("full_gnn", GNNS), ("sample", TREES)):
        for m in models:
            seeds = []
            for sd in range(3):
                s = load(folder, m, sd)
                if s is None:
                    continue
                va = va_all[~np.isnan(s[va_all])]
                tg = target[~np.isnan(s[target])]
                wk = work[~np.isnan(s[work])]
                c = val_cutoff(y[va], s[va])
                seeds.append((metrics(y[tg], s[tg], c), metrics(y[wk], s[wk], c)))
                if sd == 0:
                    cache[(folder, m)] = (s, c)
            if not seeds:
                continue
            key = f"{folder}/{m}"
            mean = {k: float(np.mean([r[0][k] for r in seeds])) for k in seeds[0][0] if k not in ("n", "bots")}
            mean_w = {k: float(np.mean([r[1][k] for r in seeds])) for k in ("macro_f1", "auroc", "pr_auc")}
            out["models"][key] = {"seeds": len(seeds), "n": seeds[0][0]["n"], "bots": seeds[0][0]["bots"], "target": mean, "working": mean_w}
            print(f"  {key:26s} {len(seeds)} seed(s) n {seeds[0][0]['n']:>6,}: macro-F1 {mean['macro_f1']:.3f} AUROC {mean['auroc']:.3f} "
                  f"PR-AUC {mean['pr_auc']:.3f} (5% {mean['pr_auc_at_5pct_bots']:.3f}, 1% {mean['pr_auc_at_1pct_bots']:.3f}) | "
                  f"working {mean_w['macro_f1']:.3f} / {mean_w['auroc']:.3f}", flush=True)

    pairs = [("full", "text_on_top_of_profile", ("full", "tree_self_prof"), ("full", "profile_only")),
             ("full", "neighbours_with_profile", ("full", "xgb_graph_prof"), ("full", "tree_self_prof")),
             ("full", "neighbours_no_profile", ("full", "xgb_graph_noprof"), ("full", "tree_self_noprof")),
             ("full", "FAGCN_minus_MLP", ("full_gnn", "FAGCN"), ("full_gnn", "MLP")),
             ("full", "RGCN_minus_MLP", ("full_gnn", "RGCN"), ("full_gnn", "MLP")),
             ("sample", "text_on_top_of_profile", ("sample", "tree_self_prof"), ("sample", "profile_only")),
             ("sample", "neighbours_with_profile", ("sample", "xgb_graph_prof"), ("sample", "tree_self_prof")),
             ("sample", "neighbours_no_profile", ("sample", "xgb_graph_noprof"), ("sample", "tree_self_noprof"))]
    for design, name, a, b in pairs:
        (sa, ca), (sb, cb) = cache[a], cache[b]
        idx = target[~np.isnan(sa[target]) & ~np.isnan(sb[target])]
        r = boot_pair(y, sa, sb, ca, cb, idx)
        out["pairs"][f"{design}/{name}"] = r | {"n": int(len(idx))}
        print(f"  pair {design}/{name:24s} n {len(idx):>6,}: " + "  ".join(f"{m} {r[m][0]:+.3f} [{r[m][1]:+.3f}, {r[m][2]:+.3f}]" for m in r), flush=True)

    rng = np.random.default_rng(1)
    y_e = y.copy()
    y_e[e_rows] = y_exp                                   # expert labels on the expert rows
    for name, a, b in (("XGB-Graph - own tree (1M)", ("full", "xgb_graph_prof"), ("full", "tree_self_prof")),
                       ("XGB-Graph - own tree (sample)", ("sample", "xgb_graph_prof"), ("sample", "tree_self_prof")),
                       ("FAGCN - MLP (1M)", ("full_gnn", "FAGCN"), ("full_gnn", "MLP")),
                       ("RGCN - MLP (1M)", ("full_gnn", "RGCN"), ("full_gnn", "MLP"))):
        (sa, ca), (sb, cb) = cache[a], cache[b]
        iw = target[~np.isnan(sa[target]) & ~np.isnan(sb[target])]
        ie = e_rows[~np.isnan(sa[e_rows]) & ~np.isnan(sb[e_rows])]
        gw, ge = boot_gain(y, sa, sb, ca, cb, iw, rng), boot_gain(y_e, sa, sb, ca, cb, ie, rng)
        pt_w = [roc_auc_score(y[iw], sa[iw]) - roc_auc_score(y[iw], sb[iw]),
                f1_score(y[iw], sa[iw] > ca, average="macro") - f1_score(y[iw], sb[iw] > cb, average="macro")]
        pt_e = [roc_auc_score(y_e[ie], sa[ie]) - roc_auc_score(y_e[ie], sb[ie]),
                f1_score(y_e[ie], sa[ie] > ca, average="macro") - f1_score(y_e[ie], sb[ie] > cb, average="macro")]
        r = {}
        for j, m in enumerate(("auroc", "macro_f1")):
            dw, de = gw[:, j], ge[:, j]
            ok = ~np.isnan(dw) & ~np.isnan(de)
            r[m] = {"weak": [float(pt_w[j]), float(np.nanpercentile(dw, 2.5)), float(np.nanpercentile(dw, 97.5))],
                    "expert": [float(pt_e[j]), float(np.nanpercentile(de, 2.5)), float(np.nanpercentile(de, 97.5))],
                    "weak_minus_expert": [float(pt_w[j] - pt_e[j]), float(np.percentile(dw[ok] - de[ok], 2.5)), float(np.percentile(dw[ok] - de[ok], 97.5))]}
        out["d7"][name] = r | {"n_weak": int(len(iw)), "n_expert": int(len(ie))}
        a_ = r["auroc"]
        print(f"  D7 {name:30s} AUROC gain: weak {a_['weak'][0]:+.3f} [{a_['weak'][1]:+.3f}, {a_['weak'][2]:+.3f}] (n {len(iw):,}) | "
              f"expert {a_['expert'][0]:+.3f} [{a_['expert'][1]:+.3f}, {a_['expert'][2]:+.3f}] (n {len(ie)}) | "
              f"weak - expert {a_['weak_minus_expert'][0]:+.3f} [{a_['weak_minus_expert'][1]:+.3f}, {a_['weak_minus_expert'][2]:+.3f}]", flush=True)

    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"Saved {OUT.name} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
