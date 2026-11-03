"""
Leakage check: do validation/test users share their text with training users?

Platonov et al. (ICLR 2023) found duplicate nodes across splits in popular heterophily
benchmarks, which inflates scores. For each dataset (setting A, all users) this counts
validation/test users whose text exactly matches a training user's, after normalising
(lower case, URLs and @mentions replaced, whitespace collapsed), separately for the full text
and the bio alone; how often duplicates share the label; and Qwen2.5-1.5B + LR on the test set
with and without the duplicated test users (same training users, C and cut-off as step 3).

Run:  python my_build/duplicate_text.py      (CPU, about a minute per dataset)
Writes results/duplicate_text.json
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def norm(t):
    t = re.sub(r"https?://\S+", "URL", t.lower())
    t = re.sub(r"@\w+", "@USER", t)
    return re.sub(r"\s+", " ", t).strip()


def one_dataset():
    sys.path.insert(0, str(ROOT))
    from sklearn.metrics import f1_score
    from common import load, load_embeddings, get_splits, best_threshold, DATA_DIR, DATASET
    from step3_embed import make_clf, tune_C
    x_text, _, y = load()
    train, val, test = get_splits(y)
    out = {}
    fields = {"full text": x_text}
    if DATASET == "twibot20":
        fields["bio only"] = np.load(DATA_DIR / "x_bio.npy").astype(str)
    for fname, texts in fields.items():
        t = np.array([norm(s) for s in texts])
        nonempty = t != ""
        first = {}
        for i in train:
            if nonempty[i]:
                first.setdefault(t[i], []).append(i)
        for split_name, idx in [("val", val), ("test", test)]:
            dup = np.array([nonempty[i] and t[i] in first for i in idx])
            same = [np.mean(y[first[t[i]]] == y[i]) for i, d in zip(idx, dup) if d]
            out[f"{fname}: {split_name} users with an exact train duplicate"] = [int(dup.sum()), int(len(idx)),
                                                                                round(float(dup.mean()), 4)]
            out[f"{fname}: {split_name} duplicates sharing the label"] = round(float(np.mean(same)), 3) if same else None
            if fname == "full text" and split_name == "test":
                test_dup = dup
        out[f"{fname}: empty texts (all users)"] = int((~nonempty).sum())
    emb = load_embeddings("qwen").astype(np.float32)
    clf = make_clf(tune_C(emb, y, train, val)).fit(emb[train], y[train])
    t_cut = best_threshold(y[val], clf.predict_proba(emb[val])[:, 1])
    pred = clf.predict_proba(emb[test])[:, 1] > t_cut
    out["Qwen2.5 + LR macro-F1, full test"] = round(f1_score(y[test], pred, average="macro"), 4)
    keep = ~test_dup
    out["Qwen2.5 + LR macro-F1, test without full-text duplicates"] = round(f1_score(y[test][keep], pred[keep], average="macro"), 4)
    print(json.dumps(out))


def main():
    if os.environ.get("_DUP_CHILD"):
        return one_dataset()
    results = {}
    for d in ["twibot20", "instagram"]:
        env = dict(os.environ, DATASET=d, SETTING="A", _DUP_CHILD="1")
        res = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=True)
        results[d] = json.loads(res.stdout.strip().splitlines()[-1])
        print(d, json.dumps(results[d], indent=1))
    (ROOT / "results" / "duplicate_text.json").write_text(json.dumps(results, indent=2))
    print("saved results/duplicate_text.json")


if __name__ == "__main__":
    main()
