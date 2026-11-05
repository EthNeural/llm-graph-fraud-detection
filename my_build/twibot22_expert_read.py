"""
Early read of the expert rotation scores and D8 (builder's decision, Thu 5 Nov ~23:50; recorded in
artifacts/04_decisions_and_reasons.md, commit 851a08b, before anything was read; no model or rule changed after).

Reads ONLY the 986 expert users (all saved rotation scores) and, for D8, the TwiBot-22 annotation-study users
that are NOT sealed non-expert users. The other sealed users are never read.

Part 1 (G7): per model, weak-only vs after the expert stage, on the 986 experts with EXPERT labels:
  macro-F1 at the model's cut-off (weak-only: its validation cut-off; rotated: the fold model's cut-off from
  its own 4 training folds), AUROC, PR-AUC; 95% bootstrap CIs (2,000) and the paired difference.
  Models finished: LoRA target, trees (sample: profile only, profile + text, XGB-Graph + profile; 1M: XGB-Graph
  + profile), GNNs (MLP control, RGCN). FAGCN still running: left out.
  Reference: the dataset's weak label itself as a 0/1 prediction of the expert label.
Part 2 (D8, the builder's plan 5 Nov ~20:15): empty-bio lift = P(bot | empty bio) / P(bot | bio), under the
  weak labels and under the expert labels, same 986 users; bootstrap CIs on each lift and on the difference.
  Second human-label set: TwiBot-22 annotation study (3 experts, 0-4 scale), median >= 3 = bot, <= 1 = human,
  2 dropped; sealed non-expert users removed.
Output: results/twibot22/expert_read_early.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import twibot22_guard as guard

RES = ROOT / "results" / "twibot22"
RAW = ROOT.parent / "data" / "twibot-22-raw"
DATA = ROOT.parent / "data" / "twibot-22"
B = 2000
# after the GNN run finishes: WITH_FAGCN=1 reads the final file (adds FAGCN) and writes expert_read_with_fagcn.json
import os
WITH_FAGCN = os.environ.get("WITH_FAGCN") == "1"
GNN_FILE = RES / ("expert_rotation_gnn.npz" if WITH_FAGCN else "expert_rotation_gnn.partial.npz")
OUT_NAME = "expert_read_with_fagcn.json" if WITH_FAGCN else "expert_read_early.json"


def metrics(y, p, cut):
    return (f1_score(y, p > cut, average="macro"), roc_auc_score(y, p), average_precision_score(y, p))


def boot(y, p, cut, p2=None, cut2=None, seed=0):
    rng = np.random.default_rng(seed)
    n, out = len(y), []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if not 0 < y[i].sum() < n:
            continue
        c1 = cut[i] if np.ndim(cut) else cut
        m = np.array(metrics(y[i], p[i], c1))
        if p2 is not None:
            c2 = cut2[i] if np.ndim(cut2) else cut2
            m = m - np.array(metrics(y[i], p2[i], c2))
        out.append(m)
    out = np.array(out)
    return np.percentile(out, 2.5, 0), np.percentile(out, 97.5, 0)


def fmt(v, lo, hi):
    return [round(float(v), 3), round(float(lo), 3), round(float(hi), 3)]


def lift(y, e):
    return y[e].mean() / max(y[~e].mean(), 1e-9)


def lift_ci(e, ys, seed=0):
    rng = np.random.default_rng(seed)
    n, out = len(e), []
    for _ in range(B):
        i = rng.integers(0, n, n)
        out.append([lift(y[i], e[i]) for y in ys])
    out = np.array(out)
    if len(ys) == 2:
        out = np.column_stack([out, out[:, 0] - out[:, 1]])
    return np.percentile(out, 2.5, 0), np.percentile(out, 97.5, 0)


def main():
    lab = pd.read_csv(RAW / "label.csv")
    roles = guard.roles()
    exp = pd.read_csv(RAW / "expert_labels" / "label_manual.csv")
    rows = pd.Series(np.arange(len(lab)), index=lab.id)[exp.id].to_numpy()
    y_exp = (exp.label.values == "bot").astype(int)
    y_weak = (lab.label.values[rows] == "bot").astype(int)
    assert (roles[rows] == "test_sealed").all() and len(rows) == 986

    models = {}
    lo = np.load(RES / "A" / "expert_rotation_target.npz")
    order = pd.Series(np.arange(len(lo["rows"])), index=lo["rows"])[rows].to_numpy()
    assert (lo["y_expert"][order] == y_exp).all()
    models["LoRA target (text, sample)"] = (lo["p_weak_only"][order], lo["cut_weak_only"], lo["p_rotated"][order], lo["cut_rotated"][order])
    for f, names in ((RES / "expert_rotation_trees.npz", [("sample_profile_only", "Profile only (tree, sample)"),
                                                         ("sample_tree_self_prof", "Profile + text (tree, sample)"),
                                                         ("sample_xgb_graph_prof", "XGB-Graph + profile (sample)"),
                                                         ("full_xgb_graph_prof", "XGB-Graph + profile (1M)")]),
                     (GNN_FILE, [("MLP", "MLP control, no edges (1M)"), ("RGCN", "RGCN (1M)")] + ([("FAGCN", "FAGCN (1M)")] if WITH_FAGCN else []))):
        d = np.load(f)
        assert (d["rows"] == rows).all() and (d["y_expert"] == y_exp).all()
        for k, name in names:
            models[name] = (d[f"{k}_p_weak_only"], d[f"{k}_cut_weak_only"], d[f"{k}_p_rotated"], d[f"{k}_cut_rotated"])

    out = {"users": 986, "expert_bots": int(y_exp.sum()), "part1": {}, "part2": {}}
    wf1 = f1_score(y_exp, y_weak, average="macro")
    out["part1"]["dataset weak label as prediction"] = {"macro_f1": round(float(wf1), 3),
        "agreement": round(float((y_exp == y_weak).mean()), 3), "weak_bots": int(y_weak.sum()), "both_bot": int((y_exp & y_weak).sum())}
    print(f"986 experts, {y_exp.sum()} expert bots. Dataset weak label as prediction: macro-F1 {wf1:.3f}, "
          f"agreement {(y_exp == y_weak).mean():.3f} (weak bots {y_weak.sum()}, both bot {(y_exp & y_weak).sum()})\n")
    print(f"{'model':32s} {'stage':9s} {'macro-F1 [95% CI]':24s} {'AUROC [95% CI]':24s} {'PR-AUC [95% CI]':24s}")
    for name, (pw, cw, pr, cr) in models.items():
        r = {}
        for stage, p, c in (("weak", pw, cw), ("expert", pr, cr)):
            m = metrics(y_exp, p, c if not np.ndim(c) else c)
            if np.ndim(c):
                m = (f1_score(y_exp, p > c, average="macro"),) + m[1:]
            l, h = boot(y_exp, p, c)
            r[stage] = {k: fmt(m[j], l[j], h[j]) for j, k in enumerate(("macro_f1", "auroc", "pr_auc"))}
            print(f"{name:32s} {stage:9s} " + " ".join(f"{m[j]:.3f} [{l[j]:.3f}, {h[j]:.3f}]   " for j in range(3)))
        dm = np.array([f1_score(y_exp, pr > cr, average="macro"), roc_auc_score(y_exp, pr), average_precision_score(y_exp, pr)]) - \
             np.array([f1_score(y_exp, pw > cw, average="macro"), roc_auc_score(y_exp, pw), average_precision_score(y_exp, pw)])
        l, h = boot(y_exp, pr, cr, pw, cw)
        r["expert_minus_weak"] = {k: fmt(dm[j], l[j], h[j]) for j, k in enumerate(("macro_f1", "auroc", "pr_auc"))}
        print(f"{'':32s} {'diff':9s} " + " ".join(f"{dm[j]:+.3f} [{l[j]:+.3f}, {h[j]:+.3f}] " for j in range(3)))
        out["part1"][name] = r

    # D8
    f = np.load(DATA / "features_full_e5.npz")
    j = list(f["names"]).index("self:bio empty")
    empty_all = f["X"][:, j] > 0.5
    e = empty_all[rows]
    lw, le = lift(y_weak, e), lift(y_exp, e)
    l, h = lift_ci(e, [y_weak, y_exp])
    d8 = {"empty_bio_users": int(e.sum()), "with_bio_users": int((~e).sum()),
          "weak": {"bot_rate_empty": round(float(y_weak[e].mean()), 3), "bot_rate_bio": round(float(y_weak[~e].mean()), 3), "lift": fmt(lw, l[0], h[0])},
          "expert": {"bot_rate_empty": round(float(y_exp[e].mean()), 3), "bot_rate_bio": round(float(y_exp[~e].mean()), 3), "lift": fmt(le, l[1], h[1])},
          "weak_minus_expert_lift": fmt(lw - le, l[2], h[2])}
    out["part2"]["986 experts"] = d8
    print(f"\nD8 on the 986 experts: empty bio {e.sum()}, with bio {(~e).sum()}")
    for k in ("weak", "expert"):
        print(f"  {k:6s} labels: bot rate empty {d8[k]['bot_rate_empty']:.3f} vs with bio {d8[k]['bot_rate_bio']:.3f} -> lift {d8[k]['lift'][0]:.2f} [{d8[k]['lift'][1]:.2f}, {d8[k]['lift'][2]:.2f}]")
    print(f"  weak - expert lift: {d8['weak_minus_expert_lift'][0]:+.2f} [{d8['weak_minus_expert_lift'][1]:+.2f}, {d8['weak_minus_expert_lift'][2]:+.2f}]")

    ids22 = {u["id"] for u in json.load(open(RAW / "annotation_study" / "Twibot22-anno_study.json"))}
    a = pd.read_csv(RAW / "annotation_study" / "annotation_combine.csv")
    a = a[a.id.isin(ids22)]
    med = a.groupby("id").label.median()
    idx = pd.Series(np.arange(len(lab)), index=lab.id)
    med = med[med.index.isin(idx.index)]
    r_a = idx[med.index].to_numpy()
    is_expert = np.isin(r_a, rows)
    keep = ~((roles[r_a] == "test_sealed") & ~is_expert)
    dropped_sealed = int((~keep).sum())
    med, r_a = med[keep], r_a[keep]
    sure = (med.values >= 3) | (med.values <= 1)
    med, r_a = med[sure], r_a[sure]
    y_ann = (med.values >= 3).astype(int)
    y_w = (lab.label.values[r_a] == "bot").astype(int)
    ea = empty_all[r_a]
    lw, la = lift(y_w, ea), lift(y_ann, ea)
    l, h = lift_ci(ea, [y_w, y_ann])
    d8b = {"users": int(len(r_a)), "sealed_non_expert_removed": dropped_sealed, "not_sure_dropped": int((~sure).sum()),
           "annotation_bots": int(y_ann.sum()), "weak_bots": int(y_w.sum()), "empty_bio_users": int(ea.sum()),
           "weak": {"bot_rate_empty": round(float(y_w[ea].mean()), 3), "bot_rate_bio": round(float(y_w[~ea].mean()), 3), "lift": fmt(lw, l[0], h[0])},
           "annotation": {"bot_rate_empty": round(float(y_ann[ea].mean()), 3), "bot_rate_bio": round(float(y_ann[~ea].mean()), 3), "lift": fmt(la, l[1], h[1])},
           "weak_minus_annotation_lift": fmt(lw - la, l[2], h[2]),
           "agreement_weak_vs_annotation": round(float((y_w == y_ann).mean()), 3)}
    out["part2"]["annotation study (TwiBot-22)"] = d8b
    print(f"\nD8 on the annotation study: {len(r_a)} users ({dropped_sealed} sealed non-expert removed, {(~sure).sum()} 'not sure' dropped); "
          f"annotation bots {y_ann.sum()}, weak bots {y_w.sum()}, empty bio {ea.sum()}, weak-vs-annotation agreement {d8b['agreement_weak_vs_annotation']:.3f}")
    for k in ("weak", "annotation"):
        print(f"  {k:10s} labels: bot rate empty {d8b[k]['bot_rate_empty']:.3f} vs with bio {d8b[k]['bot_rate_bio']:.3f} -> lift {d8b[k]['lift'][0]:.2f} [{d8b[k]['lift'][1]:.2f}, {d8b[k]['lift'][2]:.2f}]")
    print(f"  weak - annotation lift: {d8b['weak_minus_annotation_lift'][0]:+.2f} [{d8b['weak_minus_annotation_lift'][1]:+.2f}, {d8b['weak_minus_annotation_lift'][2]:+.2f}]")
    (RES / OUT_NAME).write_text(json.dumps(out, indent=1))
    print("\nSaved", OUT_NAME)


if __name__ == "__main__":
    main()
