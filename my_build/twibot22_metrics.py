"""
TwiBot-22 shared scoring and selection rules (checklist G3, G4, G5, H6). Fixed 4 Nov 2026, before any
model score. Every TwiBot-22 model script scores through this module, so all models follow one rule.

RULES
  Cut-off     chosen on VALIDATION only: the score threshold that maximises macro-F1 on val (over 199 val
              quantiles), then applied unchanged to test. Same idea as TwiBot-20's common.best_threshold.
  Selection   model / epoch / hyper-parameters chosen by best VALIDATION macro-F1; never by a test number.
  Seeds       3 seeds (0, 1, 2) per model; tables report the mean and the seed std.
  Metrics     macro-F1 (at the val cut-off), AUROC, PR-AUC (bot = positive class).
              + PR-AUC with test users reweighted so bots make up 5% and 1% of the weight (G5);
                graph, training and cut-off unchanged (evaluation only).
  CIs         paired bootstrap over test users, 2,000 resamples, 95%: the same resample for every model,
              so differences between models get their own CI (bootstrap_diff).
  Slices (G4) all working-test users; non-verified; with a bio; with >= 1 tweet.
              Sealed users (incl. the 986 experts) only when FINAL_RUN=1 (twibot22_guard.score_mask).
  Saving (H6) scores are saved per design / model / seed under results/twibot22/runs/<design>/ and an
              existing file is never overwritten.

Use:
  from twibot22_metrics import evaluate, save_scores
  res = evaluate(y, scores, val_idx)          # scores: P(bot) for all 1M users (label.csv order)
  save_scores("sample_bio_tweets", "mlp_profile", seed, scores)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import twibot22_guard as guard

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "twibot-22"
RAW = ROOT.parent / "data" / "twibot-22-raw"
RUNS = ROOT / "results" / "twibot22" / "runs"
SEEDS = [0, 1, 2]
N_BOOT = 2000
REWEIGHT_TO = (0.05, 0.01)


def labels():
    return (pd.read_csv(RAW / "label.csv").label.values == "bot").astype(int)


def val_cutoff(y_val, s_val):
    grid = np.unique(np.quantile(s_val, np.linspace(0.005, 0.995, 199)))
    return float(max(grid, key=lambda t: f1_score(y_val, s_val > t, average="macro")))


def reweighted_pr_auc(y, s, target):
    # Weight bots so that they carry `target` of the total weight; humans weight 1.
    nb, nh = y.sum(), (1 - y).sum()
    if nb == 0 or nh == 0:
        return float("nan")
    w = np.where(y == 1, target / (1 - target) * nh / nb, 1.0)
    return float(average_precision_score(y, s, sample_weight=w))


def metrics(y, s, cut):
    out = dict(n=int(len(y)), bots=int(y.sum()),
               macro_f1=float(f1_score(y, s > cut, average="macro")),
               auroc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)))
    for r in REWEIGHT_TO:
        out[f"pr_auc_at_{int(r * 100)}pct_bots"] = reweighted_pr_auc(y, s, r)
    return out


def slices(final=False):
    # Boolean masks over all users (label.csv order) for every reported test group.
    lab = pd.read_csv(RAW / "label.csv")
    test = guard.score_mask(final)
    verified = guard.verified(lab.id)
    bio = pd.read_parquet(DATA / "users.parquet", columns=["id", "description"]).set_index("id").description
    has_bio = (bio.reindex(lab.id).fillna("").str.strip() != "").to_numpy()
    has_tweets = np.load(DATA / "tweet_count_files.npy") > 0
    sample = np.zeros(len(lab), bool)
    sample[np.load(DATA / "sample_idx.npy")] = True
    out = {"all": test, "non_verified": test & ~verified, "with_bio": test & has_bio, "with_tweets": test & has_tweets}
    return out, sample


def evaluate(y, scores, val_idx, in_sample_only=False, final=False):
    # scores: P(bot) for every user (label.csv order; NaN where a model gave no score).
    # in_sample_only=True restricts every slice to the 100k sample (headline comparison, F6).
    cut = val_cutoff(y[val_idx], scores[val_idx])
    masks, sample = slices(final)
    res = dict(val_cutoff=cut, val_macro_f1=float(f1_score(y[val_idx], scores[val_idx] > cut, average="macro")))
    for name, m in masks.items():
        m = m & ~np.isnan(scores)
        if in_sample_only:
            m &= sample
        idx = np.where(m)[0]
        guard.assert_not_sealed(idx)
        res[name] = metrics(y[idx], scores[idx], cut)
    return res


def bootstrap_diff(y, s_a, s_b, cut_a, cut_b, idx, seed=0):
    # Paired bootstrap 95% CI of (model A - model B) on the same test users.
    guard.assert_not_sealed(idx)
    rng = np.random.default_rng(seed)
    yy, a, b = y[idx], s_a[idx], s_b[idx]
    d = {k: [] for k in ("macro_f1", "auroc", "pr_auc")}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(idx), len(idx))
        if yy[i].min() == yy[i].max():
            continue
        d["macro_f1"].append(f1_score(yy[i], a[i] > cut_a, average="macro") - f1_score(yy[i], b[i] > cut_b, average="macro"))
        d["auroc"].append(roc_auc_score(yy[i], a[i]) - roc_auc_score(yy[i], b[i]))
        d["pr_auc"].append(average_precision_score(yy[i], a[i]) - average_precision_score(yy[i], b[i]))
    return {k: dict(mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
            for k, v in d.items()}


def save_scores(design, model, seed, scores, info=None):
    # One file per design / model / seed; never overwrites (lessons 18-20).
    out = RUNS / design
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{model}_s{seed}.npz"
    if f.exists():
        raise FileExistsError(f"{f} exists: results are never overwritten; use a new design or model name")
    np.savez(f, scores=scores.astype(np.float32))
    if info is not None:
        (out / f"{model}_s{seed}.json").write_text(json.dumps(info, indent=1))
    return f


if __name__ == "__main__":
    # Self-test on random scores (no model): checks the rules run end to end and refuse sealed users.
    y = labels()
    split = np.load(DATA / "split_ours.npy")
    val = np.where(split == "val")[0]
    rng = np.random.default_rng(0)
    s = rng.random(len(y)) * 0.3 + y * rng.random(len(y)) * 0.5        # a fake, weakly informative score
    r = evaluate(y, s, val)
    print(json.dumps({k: ({kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()} if isinstance(v, dict) else round(v, 3))
                      for k, v in r.items()}, indent=1))
    try:
        guard.assert_not_sealed(np.where(guard.roles() == "test_sealed")[0][:3])
        print("ERROR: sealed users not refused")
    except RuntimeError:
        print("sealed users refused: OK")
    # A purely random score must give PR-AUC near the bot share (0.14), and near 0.05 / 0.01 when reweighted.
    rnd = evaluate(y, rng.random(len(y)), val)["all"]
    print("pure random score: PR-AUC", round(rnd["pr_auc"], 3), "(expect ~0.14) | reweighted to 5% / 1%:",
          round(rnd["pr_auc_at_5pct_bots"], 3), round(rnd["pr_auc_at_1pct_bots"], 3), "(expect ~0.05 / ~0.01)")
