"""
TwiBot-22 guard: keep the label and the "verified" shortcut out of every model input.

Layout (after migrate(), run once on 4 Nov 2026):
  data/twibot-22/users.parquet             profile fields only: NO verified, NO label / y / split
  data/twibot-22/eval_only/verified.parquet  id, verified  -- read ONLY to score the non-verified
                                             slice and by the shortcut checks, never by a model
  data/twibot-22-raw/label.csv             labels (models read train labels only)

verified(ids)          verified flag for the given user ids, in that order (evaluation and checks only)
check_columns(names)   stop with an error if any model input column could carry the label or "verified"
migrate()              one-off: move verified out of users.parquet and drop the label columns

Run:  python my_build/twibot22_guard.py        (runs migrate(); safe to rerun)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "twibot-22"
EVAL = DATA / "eval_only"
FORBIDDEN = {"verified", "label", "y", "split"}


def verified(ids):
    v = pd.read_parquet(EVAL / "verified.parquet").set_index("id").verified
    out = v.reindex(pd.Index(ids))
    assert out.notna().all(), "some ids have no verified flag"
    return out.to_numpy(bool)


def check_columns(names):
    # Column names may carry prefixes ("self:", "following_in:mean_..."), so test every part.
    bad = [n for n in names if any(part.strip().lower() in FORBIDDEN or "verified" in part.lower()
                                   for part in str(n).replace(":", " ").replace("_", " ").split())]
    if bad:
        raise ValueError(f"model input contains forbidden columns (label or verified shortcut): {bad[:10]}")


def migrate():
    p = DATA / "users.parquet"
    users = pd.read_parquet(p)
    EVAL.mkdir(parents=True, exist_ok=True)
    if "verified" in users.columns:
        users[["id", "verified"]].to_parquet(EVAL / "verified.parquet")
    drop = [c for c in ("verified", "label", "y", "split") if c in users.columns]
    if drop:
        users.drop(columns=drop).to_parquet(p)
    left = pd.read_parquet(p).columns.tolist()
    check_columns(left)
    print(f"users.parquet columns now: {left}")
    print(f"eval_only/verified.parquet: {int(pd.read_parquet(EVAL / 'verified.parquet').verified.sum()):,} verified users")


if __name__ == "__main__":
    migrate()
