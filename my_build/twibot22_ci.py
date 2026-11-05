"""
TwiBot-22 paired bootstrap CIs (2,000 resamples, twibot22_metrics.bootstrap_diff) for the key pairs,
working-test users only (sealed never scored). Seed 0 scores; each model keeps its own validation cut-off.
  sample (5,000 users, the same as LoRA):
    text adds on top of profile        tree_self_prof    - profile_only
    neighbours add (with profile)      xgb_graph_prof    - tree_self_prof
    neighbours add (no profile)        xgb_graph_noprof  - tree_self_noprof
    profile vs fine-tuned LLM          profile_only      - LoRA target
    tweets vs bio only (same users)    tree_self_noprof  - bio_on_sample tree_self_noprof
  full (49,951 users): the same first three pairs.
Output: results/twibot22/ci_pairs.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import twibot22_guard as guard
from twibot22_metrics import labels, val_cutoff, bootstrap_diff

RUNS = ROOT / "results" / "twibot22" / "runs"
y = labels()
roles = guard.roles()


def load(design, model):
    return np.load(RUNS / design / f"{model}_s0.npz")["scores"].astype(np.float64)


def lora():
    z = np.load(ROOT / "results" / "twibot22" / "A" / "step6_Qwen3-1.7B_target_scores.npz")
    s = np.full(len(y), np.nan)
    s[z["test_rows"]] = z["test"]
    s[z["val_rows"]] = z["val"]
    return s


def pair(sa, sb, idx_eval):
    va = np.where((roles == "val") & ~np.isnan(sa) & ~np.isnan(sb))[0]
    ca, cb = val_cutoff(y[va], sa[va]), val_cutoff(y[va], sb[va])
    return bootstrap_diff(y, sa, sb, ca, cb, idx_eval)


out = {}
for design in ("sample", "full"):
    s = {m: load(design, m) for m in ("profile_only", "tree_self_prof", "tree_self_noprof", "xgb_graph_prof", "xgb_graph_noprof")}
    te = np.where((roles == "test_work") & ~np.isnan(s["profile_only"]))[0]
    pairs = {"text_on_top_of_profile": ("tree_self_prof", "profile_only"),
             "neighbours_with_profile": ("xgb_graph_prof", "tree_self_prof"),
             "neighbours_no_profile": ("xgb_graph_noprof", "tree_self_noprof")}
    out[design] = {k: pair(s[a], s[b], te) | {"a": a, "b": b, "n_test": int(len(te))} for k, (a, b) in pairs.items()}
    if design == "sample":
        L = lora()
        te_l = te[~np.isnan(L[te])]
        assert len(te_l) == 5000
        # LoRA's cut-off comes from its own 1,000 validation users (pair() uses val users both models scored)
        out[design]["profile_minus_lora_target"] = pair(s["profile_only"], L, te_l) | {"n_test": int(len(te_l))}
        out[design]["bio_tweets_minus_bio_only"] = pair(s["tree_self_noprof"], load("bio_on_sample", "tree_self_noprof"), te) | {"n_test": int(len(te))}
json.dump(out, open(ROOT / "results" / "twibot22" / "ci_pairs.json", "w"), indent=1)
for d, v in out.items():
    for k, r in v.items():
        print(f"{d:6s} {k:28s} " + "  ".join(f"{m} {r[m]['mean']:+.3f} [{r[m]['lo']:+.3f}, {r[m]['hi']:+.3f}]" for m in ("macro_f1", "auroc", "pr_auc")))
