"""
Which TwiBot-20 profile fields can we use, and what do they add to text?

Rule, fixed before looking at the results (used for every field): a field is a SHORTCUT, and
is excluded, if
  (a) it is a documented collection/labelling artefact (Hays et al., WWW 2023, §4.1: TwiBot-20
      "verified accounts were automatically labeled human"), or
  (b) one split on it labels a large group almost perfectly: a depth-1 tree leaf holding at
      least 10% of training users that is at least 99% one class. Real behaviour is never that
      clean; a labelling rule is.
For each field we report its single-field depth-1 tree (macro-F1 / AUROC on test), its purest
large leaf, and its Spearman correlation with the crawl artefact (in-friend / in-follow
counts, see link_direction.py). Then LR and step 4's MLP on Qwen2.5-1.5B embeddings + the
allowed fields, against text only and against text + all fields (reference only), with a
paired bootstrap CI against text only. Full test set, cut-offs chosen on validation.

Run:  DATASET=twibot20 SETTING=A python my_build/metadata_check.py     (CPU, about 10 min)
Writes results/twibot20/<setting>/metadata_check.json
"""
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from common import load, load_embeddings, original_ids, get_splits, best_threshold, DATA_DIR, RESULTS_DIR, DATASET
from step3_embed import make_clf, tune_C
from step4_gnn import train_one, NoGraph, SEEDS

assert DATASET == "twibot20"
N_BOOT = 2000
DOCUMENTED = {"verified": "Hays et al. (WWW 2023, §4.1): verified accounts were automatically labelled human"}


def fields():
    m = pd.read_csv(DATA_DIR / "metadata.csv").iloc[original_ids()].reset_index(drop=True)
    f = pd.DataFrame({"verified": (m.verified.astype(str).str.strip() == "True").astype(float)})
    for c in ["followers_count", "following_count", "tweet_count", "listed_count"]:
        f["log_" + c] = np.log1p(m[c].fillna(0))
    created = pd.to_datetime(m.created_at, format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
    f["account_age_years"] = (pd.Timestamp("2020-09-01", tz="UTC") - created).dt.days / 365.25
    return f.fillna(0)


def purest_leaf(x, y, train):
    tree = DecisionTreeClassifier(max_depth=1, class_weight="balanced").fit(x[train, None], y[train])
    leaf = tree.apply(x[train, None])
    best = None
    for l in np.unique(leaf):
        m = leaf == l
        share, purity = m.mean(), max(y[train][m].mean(), 1 - y[train][m].mean())
        if share >= 0.10 and (best is None or purity > best[1]):
            best = (round(float(share), 3), round(float(purity), 4))
    return tree, best


def main():
    _, e, y = load()
    train, val, test = get_splits(y)
    f = fields()
    n = len(y)
    keep = original_ids()
    new_id = np.full(len(np.load(DATA_DIR / "y.npy")), -1)
    new_id[keep] = np.arange(len(keep))
    indeg = {}
    for r, edges in np.load(DATA_DIR / "edge_typed.npz").items():
        ee = new_id[edges.T]
        ee = ee[:, (ee >= 0).all(0)]
        indeg[f"in-{r}"] = np.bincount(ee[1], minlength=n)

    out, allowed = {"fields": {}}, []
    for c in f.columns:
        x = f[c].to_numpy()
        tree, leaf = purest_leaf(x, y, train)
        p = tree.predict_proba(x[:, None])[:, 1]
        t = best_threshold(y[val], p[val])
        row = dict(depth1_macro_f1=round(f1_score(y[test], p[test] > t, average="macro"), 4),
                   depth1_auroc=round(roc_auc_score(y[test], p[test]), 4), purest_large_leaf=leaf,
                   spearman_in_friend=round(float(spearmanr(x, indeg["in-friend"])[0]), 3),
                   spearman_in_follow=round(float(spearmanr(x, indeg["in-follow"])[0]), 3))
        reason = DOCUMENTED.get(c) or (f"a leaf with {leaf[0]:.0%} of users is {leaf[1]:.1%} one class"
                                       if leaf and leaf[1] >= 0.99 else None)
        row["shortcut"] = reason
        if not reason:
            allowed.append(c)
        out["fields"][c] = row
        print(f"{c:22s} depth-1 F1 {row['depth1_macro_f1']:.3f} AUROC {row['depth1_auroc']:.3f} "
              f"leaf {leaf}  rho(in-friend) {row['spearman_in_friend']:+.2f}  -> {'SHORTCUT: ' + reason if reason else 'allowed'}")
    out["allowed"] = allowed

    emb = load_embeddings("qwen").astype(np.float32)
    meta = f.to_numpy().astype(np.float32)
    mm, ms = meta[train].mean(0), meta[train].std(0) + 1e-6
    meta = (meta - mm) / ms
    idx = {c: i for i, c in enumerate(f.columns)}
    sets = {"text only": emb, "text + allowed metadata": np.hstack([emb, meta[:, [idx[c] for c in allowed]]]),
            "text + all metadata (incl. shortcuts; reference only)": np.hstack([emb, meta]),
            "allowed metadata only": meta[:, [idx[c] for c in allowed]]}
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(test), len(test)) for _ in range(N_BOOT)]
    yt = y[test]
    preds, out["models"] = {}, []
    for sname, X in sets.items():
        clf = make_clf(tune_C(X, y, train, val)).fit(X[train], y[train])
        p = clf.predict_proba(X)[:, 1]
        preds[("LR", sname)] = [p[test] > best_threshold(y[val], p[val])]
        auc_lr = roc_auc_score(yt, p[test])
        mlp = []
        for seed in SEEDS:
            xm, xs = X[train].mean(0), X[train].std(0) + 1e-6          # train-standardised, as in step 4
            pm = train_one(NoGraph, torch.as_tensor((X - xm) / xs), torch.as_tensor(e), torch.as_tensor(y),
                           torch.as_tensor(train), torch.as_tensor(val), seed)
            mlp.append((pm[test] > best_threshold(y[val], pm[val]), roc_auc_score(yt, pm[test])))
        preds[("MLP", sname)] = [m[0] for m in mlp]
        for model, auc in [("LR", auc_lr), ("MLP", float(np.mean([m[1] for m in mlp])))]:
            pr = preds[(model, sname)]
            f1 = float(np.mean([f1_score(yt, q, average="macro") for q in pr]))
            row = dict(model=model, features=sname, macro_f1=round(f1, 4), auroc=round(auc, 4))
            if sname != "text only":
                base = preds[(model, "text only")]
                d = [np.mean([f1_score(yt[i], q[i], average="macro") for q in pr]) -
                     np.mean([f1_score(yt[i], q[i], average="macro") for q in base]) for i in boots]
                row["minus_text_only"] = round(f1 - float(np.mean([f1_score(yt, q, average="macro") for q in base])), 4)
                row["ci"] = [round(float(v), 4) for v in np.percentile(d, [2.5, 97.5])]
            out["models"].append(row)
            print(f"{model:4s} {sname:55s} {row['macro_f1']:.3f}  AUROC {row['auroc']:.3f}  "
                  f"{row.get('minus_text_only', ''):+} {row.get('ci', '')}" if "ci" in row else
                  f"{model:4s} {sname:55s} {row['macro_f1']:.3f}  AUROC {row['auroc']:.3f}", flush=True)
    json.dump(out, open(RESULTS_DIR / "metadata_check.json", "w"), indent=2)
    print(f"saved {RESULTS_DIR / 'metadata_check.json'}")


if __name__ == "__main__":
    main()
