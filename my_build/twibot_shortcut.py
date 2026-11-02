"""
TwiBot-20 metadata shortcut: how far do profile numbers alone get, with NO text and NO graph?

Run:  DATASET=twibot20 SETTING=A python my_build/twibot_shortcut.py      (and SETTING=B)

This is NOT one of our models. It is the reference line that shows why metadata is excluded
(report Section 4.6): "verified" alone, and the other popularity fields together, nearly
solve the task without saying anything about how bots write or connect.
Same split, same validation-tuned threshold and same metrics as every other step.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (load, original_ids, get_splits, isolated_mask, evaluate, best_threshold,
                    DATA_DIR, RESULTS_DIR, DATASET)

assert DATASET == "twibot20", "run with DATASET=twibot20"


def features():
    m = pd.read_csv(DATA_DIR / "metadata.csv").iloc[original_ids()].reset_index(drop=True)
    f = pd.DataFrame({"verified": (m.verified.astype(str).str.strip() == "True").astype(float)})
    for c in ["followers_count", "following_count", "tweet_count", "listed_count"]:
        f["log_" + c] = np.log1p(m[c].fillna(0))
    created = pd.to_datetime(m.created_at, format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
    f["account_age_years"] = (pd.Timestamp("2020-09-01", tz="UTC") - created).dt.days / 365.25
    return f.fillna(0)


def main():
    _, edge_index, y = load()
    train, val, test = get_splits(y)
    iso_test = isolated_mask(edge_index, len(y))[test]
    f = features()
    no_verified = [c for c in f.columns if c != "verified"]
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    gb = lambda: HistGradientBoostingClassifier(max_depth=3, class_weight="balanced", random_state=0)

    results = []
    for name, cols, make in [("meta: verified only", ["verified"], lr),
                             ("meta: followers only", ["log_followers_count"], lr),
                             ("meta: verified + followers", ["verified", "log_followers_count"], lr),
                             ("meta: all except verified", no_verified, gb)]:
        X = f[cols].to_numpy()
        clf = make().fit(X[train], y[train])
        t = best_threshold(y[val], clf.predict_proba(X[val])[:, 1])
        p = clf.predict_proba(X[test])[:, 1]
        results.append(evaluate(name, y[test], (p > t).astype(int), p, iso_test))

    with open(RESULTS_DIR / "metadata_shortcut.json", "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
