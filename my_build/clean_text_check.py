"""
Would cleaning the text have changed the results? A sensitivity check on step 2's TF-IDF + LR.

We feed the models raw text (bio + tweets for TwiBot-20, bio for Instagram). 94% of TwiBot-20
users have a URL and an @mention, and 33% of their tweets are retweets. This reruns step 2's
balanced TF-IDF + LR (same vectoriser, cut-off chosen on validation) on three versions:
  raw         the text the models saw
  normalised  URLs -> "URL", @mentions -> "@USER", retweet prefixes ("RT @name:") removed
  stripped    URLs and @mentions deleted, retweet prefixes removed
and reports macro-F1 / AUROC on the full test set with a paired bootstrap 95% CI against raw.

Run:  python my_build/clean_text_check.py      (CPU, a few minutes)
Writes results/clean_text_check.json
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
N_BOOT = 2000


def clean(t, mode):
    lines = [re.sub(r"^RT @\w+:\s*", "", line) for line in t.split("\n")]
    t = "\n".join(lines)
    if mode == "normalised":
        t = re.sub(r"https?://\S+", " URL ", t)
        t = re.sub(r"@\w+", " @USER ", t)
    else:
        t = re.sub(r"https?://\S+", " ", t)
        t = re.sub(r"@\w+", " ", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def one_setting():
    sys.path.insert(0, str(ROOT))
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from common import load, get_splits, best_threshold
    x_text, _, y = load()
    train, val, test = get_splits(y)
    preds = {}
    for mode in ["raw", "normalised", "stripped"]:
        texts = x_text if mode == "raw" else np.array([clean(t, mode) for t in x_text])
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X_tr = vec.fit_transform(texts[train])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_tr, y[train])
        p_val = clf.predict_proba(vec.transform(texts[val]))[:, 1]
        p = clf.predict_proba(vec.transform(texts[test]))[:, 1]
        preds[mode] = (p > best_threshold(y[val], p_val), p)
    yt = y[test]
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(test), len(test)) for _ in range(N_BOOT)]
    f1 = lambda pr, i: f1_score(yt[i], pr[i], average="macro")
    out = {}
    for mode, (pr, p) in preds.items():
        row = dict(macro_f1=round(f1(pr, np.arange(len(yt))), 4), auroc=round(roc_auc_score(yt, p), 4))
        if mode != "raw":
            d = np.array([f1(pr, i) - f1(preds["raw"][0], i) for i in boots])
            row["minus_raw"] = round(row["macro_f1"] - out["raw"]["macro_f1"], 4)
            row["ci"] = [round(float(v), 4) for v in np.percentile(d, [2.5, 97.5])]
        out[mode] = row
    print(json.dumps(out))


def main():
    if os.environ.get("_CLEAN_CHILD"):
        return one_setting()
    results = {}
    for d in ["twibot20", "instagram"]:
        for s in ["A", "B"]:
            env = dict(os.environ, DATASET=d, SETTING=s, _CLEAN_CHILD="1")
            res = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=True)
            results[f"{d} {s}"] = r = json.loads(res.stdout.strip().splitlines()[-1])
            print(f"{d} {s}: raw {r['raw']['macro_f1']:.3f} | normalised {r['normalised']['macro_f1']:.3f} "
                  f"({r['normalised']['minus_raw']:+.3f} {r['normalised']['ci']}) | stripped {r['stripped']['macro_f1']:.3f} "
                  f"({r['stripped']['minus_raw']:+.3f} {r['stripped']['ci']})", flush=True)
    (ROOT / "results" / "clean_text_check.json").write_text(json.dumps(results, indent=2))
    print("saved results/clean_text_check.json")


if __name__ == "__main__":
    main()
