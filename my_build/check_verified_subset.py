"""
Check the "verified users" claims for TwiBot-20, setting A, from the raw data. CPU only, no GPU,
no torch (nothing here trains a model; it only re-reads saved labels, splits and scores).

Claims under test (from reports/twibot20_lessons_for_twibot22.md, lesson 2):
  C1  326 of the 1,183 test users are verified
  C2  0 of those verified test users are bots
  C3  among the non-verified test users, about 75% are bots
  C4  target LoRA (run 0): macro-F1 0.704 -> 0.641 and AUROC 0.782 -> 0.754 (all -> non-verified)
  C5  dgp LoRA (run 0): macro-F1 0.649 and AUROC 0.754 on non-verified test users

What it reports:
  1. Bot share in train / val / test (official TwiBot-20 split, as stored in setting A's splits.npz).
  2-3. Verified users per split, how many are bots, and the bot share among non-verified users.
  4. For every saved LoRA score file (target and dgp; run 0 and seed 1; best-loss and last adapter):
     macro-F1, AUROC and PR-AUC on all test users vs non-verified test users.
     The cut-off is chosen ONCE on the full validation set (same rule as common.best_threshold)
     and applied unchanged to both test groups, so only the population changes.
     LR, MLP and GCN test scores are not saved anywhere (bootstrap_ci.py retrains them), so they
     are listed as "not saved" rather than recomputed.

Inputs:  data/twibot-20/{ids.npy, y.npy, split.npy, metadata.csv}
         my_build/results/twibot20/A/splits.npz
         my_build/results/twibot20/A/step6_Qwen3-1.7B_{target,dgp}[_s1][_last]_scores.npz
Output:  my_build/results/twibot20/A/verified_subset.json
Run:     python my_build/check_verified_subset.py          (a few seconds)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "twibot-20"
RES = ROOT / "results" / "twibot20" / "A"


def best_threshold(y_true, prob):
    # Copy of common.best_threshold (common.py), kept here so this script imports no torch:
    # the cut-off that maximises macro-F1 on the VALIDATION set, over 99 score quantiles.
    grid = np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 99)))
    return max(grid, key=lambda t: f1_score(y_true, prob > t, average="macro"))


def metrics(y, score, cut):
    # The three metrics used throughout REPORT.md. Macro-F1 needs the cut-off; AUROC and PR-AUC
    # (bot = positive class) are threshold-free.
    return dict(n=int(len(y)), bot_share=round(float(y.mean()), 3),
                macro_f1=round(float(f1_score(y, score > cut, average="macro")), 3),
                auroc=round(float(roc_auc_score(y, score)), 3),
                pr_auc=round(float(average_precision_score(y, score)), 3))


def verdict(real, claim, tol):
    return "CONFIRMED" if abs(real - claim) <= tol else "WRONG"


def main():
    # --- Load labels and the verified flag, and check that every file uses the same row order.
    ids = np.load(DATA / "ids.npy", allow_pickle=True)
    y = np.load(DATA / "y.npy").astype(int)                       # 1 = bot, 0 = human
    meta = pd.read_csv(DATA / "metadata.csv")
    assert (meta.id.to_numpy() == ids).all(), "metadata.csv rows are not in ids.npy order"
    assert ((meta.label == "bot").astype(int).to_numpy() == y).all(), "metadata labels differ from y.npy"
    verified = meta.verified.astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()

    # --- The split. Setting A's splits.npz holds index arrays; data/twibot-20/split.npy holds the
    # official split as one string per user. Confirm they are the same split.
    s = np.load(RES / "splits.npz")
    parts = {k: s[k] for k in ("train", "val", "test")}
    official = np.load(DATA / "split.npy", allow_pickle=True)
    same_as_official = all((official[idx] == k).all() for k, idx in parts.items())

    # --- 1-3. Bot share and verified users per split.
    splits = {}
    for k, idx in parts.items():
        v = verified[idx]
        splits[k] = dict(n=int(len(idx)), bot_share=round(float(y[idx].mean()), 3),
                         verified=int(v.sum()), verified_share=round(float(v.mean()), 3),
                         verified_bots=int(y[idx][v].sum()),
                         non_verified=int((~v).sum()),
                         non_verified_bot_share=round(float(y[idx][~v].mean()), 3))

    # --- 4. Saved LoRA scores: all test users vs non-verified test users.
    test, val = parts["test"], parts["val"]
    nv = ~verified[test]
    models = {}
    for variant in ("target", "dgp"):
        for run, tag in (("run0", ""), ("seed1", "_s1")):
            for adapter, suffix in (("best_loss", ""), ("last", "_last")):
                f = RES / f"step6_Qwen3-1.7B_{variant}{tag}{suffix}_scores.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                # The score file stores its own test-node order; it must match the split.
                assert (d["test_nodes"] == test).all(), f"{f.name}: test_nodes differ from splits.npz"
                cut = best_threshold(y[val], d["val"])
                models[f"{variant}_{run}_{adapter}"] = dict(
                    file=f.name, val_cut=round(float(cut), 4),
                    all_test=metrics(y[test], d["test"], cut),
                    non_verified_test=metrics(y[test][nv], d["test"][nv], cut))
    not_saved = ["LR", "MLP", "GCN"]

    # --- Claims table.
    te = splits["test"]
    tgt, dgp = models.get("target_run0_best_loss"), models.get("dgp_run0_best_loss")
    claims = [
        ("C1 verified test users = 326", te["verified"], 326, 0),
        ("C2 verified test users who are bots = 0", te["verified_bots"], 0, 0),
        ("C3 bot share among non-verified test users ~ 0.75", te["non_verified_bot_share"], 0.75, 0.01),
    ]
    if tgt:
        claims += [("C4 target run0 macro-F1, all test = 0.704", tgt["all_test"]["macro_f1"], 0.704, 0.0015),
                   ("C4 target run0 macro-F1, non-verified = 0.641", tgt["non_verified_test"]["macro_f1"], 0.641, 0.0015),
                   ("C4 target run0 AUROC, all test = 0.782", tgt["all_test"]["auroc"], 0.782, 0.0015),
                   ("C4 target run0 AUROC, non-verified = 0.754", tgt["non_verified_test"]["auroc"], 0.754, 0.0015)]
    if dgp:
        claims += [("C5 dgp run0 macro-F1, non-verified = 0.649", dgp["non_verified_test"]["macro_f1"], 0.649, 0.0015),
                   ("C5 dgp run0 AUROC, non-verified = 0.754", dgp["non_verified_test"]["auroc"], 0.754, 0.0015)]
    claim_rows = [dict(claim=c, real=r, claimed=v, verdict=verdict(r, v, t)) for c, r, v, t in claims]

    out = dict(split_is_official=bool(same_as_official), splits=splits, lora=models,
               not_saved=not_saved, claims=claim_rows,
               note="Cut-off chosen on the full validation set and applied unchanged to both test groups. "
                    "'best_loss' = adapter with the lowest validation loss; 'last' = final check's adapter.")
    (RES / "verified_subset.json").write_text(json.dumps(out, indent=1))

    # --- Print.
    print(f"splits.npz == official split.npy: {same_as_official}\n")
    print(f"{'split':6s} {'n':>6s} {'bots':>6s} {'verified':>9s} {'ver. bots':>9s} {'non-ver bots':>12s}")
    for k, r in splits.items():
        print(f"{k:6s} {r['n']:>6d} {r['bot_share']:>6.3f} {r['verified']:>5d} ({r['verified_share']:.1%}) "
              f"{r['verified_bots']:>5d} {r['non_verified_bot_share']:>12.3f}")
    print(f"\n{'model':28s} {'all: F1':>8s} {'AUROC':>6s} {'PR-AUC':>7s} | {'non-ver: F1':>11s} {'AUROC':>6s} {'PR-AUC':>7s}")
    for k, r in models.items():
        a, b = r["all_test"], r["non_verified_test"]
        print(f"{k:28s} {a['macro_f1']:>8.3f} {a['auroc']:>6.3f} {a['pr_auc']:>7.3f} | "
              f"{b['macro_f1']:>11.3f} {b['auroc']:>6.3f} {b['pr_auc']:>7.3f}")
    print(f"not saved (not checked): {', '.join(not_saved)}\n")
    for r in claim_rows:
        print(f"{r['verdict']:9s}  {r['claim']:52s} real {r['real']}")
    print(f"\nsaved {RES / 'verified_subset.json'}")


if __name__ == "__main__":
    main()
