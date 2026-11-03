"""
Where is the label signal in TwiBot-20's links, and is it a property of the graph or of the crawl?

TwiBot-20 stored, for each labelled user, a sample of about 10 accounts they follow ("friend")
and 10 accounts following them ("follow"), as source -> target rows. Our pipelines symmetrise
and (in steps 4-5) merge the two types. This script, on setting A, measures:
  1. how the stored lists are capped (out-degree per relation) and how often links are mutual;
  2. incoming vs outgoing links per relation, for bots and humans, and the test AUROC of a
     training-free score (in-follow minus in-friend);
  3. how in-degree tracks followers_count, a metadata field we exclude as a shortcut;
  4. whether a user's own lists were recorded at all (any link to a support user);
  5. whether step 6 dgp beats target only through the link-type cue: AUROC within groups of
     equal flags, and target-LoRA score + 3 flags (LR fitted on validation) against dgp;
  6. the lift from adding link flags or log in/out degrees to text-only LR, with paired
     bootstrap 95% CIs (2,000 test resamples).
All labels are used only to DESCRIBE the data (1-4) or to score test predictions (5-6).
Run:  DATASET=twibot20 SETTING=A python my_build/link_direction.py   (CPU, about 2 min)
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from common import load, load_embeddings, get_splits, best_threshold, DATA_DIR, RESULTS_DIR, SETTING
from step3_embed import make_clf, tune_C
from step6_dgp import typed_relations, metapath_adjacency, METAPATHS

N_BOOT = 2000


def lr_preds(X, y, train, val, test):
    clf = make_clf(tune_C(X, y, train, val)).fit(X[train], y[train])
    t = best_threshold(y[val], clf.predict_proba(X[val])[:, 1])
    p = clf.predict_proba(X[test])[:, 1]
    return p, p > t


def paired_ci(y, a, b, boots):
    f = lambda p, i: f1_score(y[i], p[i], average="macro")
    d = np.array([f(a, i) - f(b, i) for i in boots])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(diff=round(f(a, np.arange(len(y))) - f(b, np.arange(len(y))), 4), ci=[round(lo, 4), round(hi, 4)])


def main():
    assert SETTING == "A", "run on setting A (B's edges depend on the labels through its subsampling)"
    _, _, y = load()
    n = len(y)
    train, val, test = get_splits(y)
    typed = {r: e.T for r, e in np.load(DATA_DIR / "edge_typed.npz").items()}      # (2, E) source -> target
    out = {}

    # 1. caps and reciprocity
    for r, (s, t) in typed.items():
        od = np.bincount(s, minlength=n)
        pairs = set(zip(s.tolist(), t.tolist()))
        out[f"{r}: out-degree median / p90 / max"] = [int(np.median(od[od > 0])), int(np.percentile(od[od > 0], 90)), int(od.max())]
        out[f"{r}: mutual share"] = round(sum((b, a) in pairs for a, b in pairs) / len(pairs), 4)

    # 2. in vs out, by label; training-free score
    deg = {}
    for r, (s, t) in typed.items():
        deg[f"out-{r}"], deg[f"in-{r}"] = np.bincount(s, minlength=n), np.bincount(t, minlength=n)
    for k, d in deg.items():
        out[f"has {k}: bots / humans"] = [round(float((d[y == 1] > 0).mean()), 3), round(float((d[y == 0] > 0).mean()), 3)]
        out[f"{k}: test AUROC of the count"] = round(roc_auc_score(y[test], d[test]), 3)
    score = deg["in-follow"] - deg["in-friend"]
    out["in-follow minus in-friend: test AUROC (no training)"] = round(roc_auc_score(y[test], score[test]), 3)

    # 3. metadata proxy
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    fol = pd.to_numeric(meta.followers_count, errors="coerce").to_numpy()
    ok = ~np.isnan(fol)
    for k in ["in-friend", "in-follow"]:
        out[f"Spearman({k}, followers_count)"] = round(float(spearmanr(deg[k][ok], fol[ok])[0]), 3)
        h = ok & (y == 0)
        out[f"Spearman({k}, followers_count), humans"] = round(float(spearmanr(deg[k][h], fol[h])[0]), 3)

    # 4. were the user's own lists recorded? (any stored link to a support user)
    sup = np.load(DATA_DIR / "support_edges.npz")
    for r in ["friend", "follow"]:
        has = np.bincount(sup["lab"][sup["rel"] == r], minlength=n) > 0
        out[f"own {r} list recorded (any support link): bots / humans"] = [round(float(has[y == 1].mean()), 3), round(float(has[y == 0].mean()), 3)]

    # 5. step 6: is dgp's gain the cue?
    rel = typed_relations()
    flags = np.stack([np.asarray(metapath_adjacency(rel, path, n).sum(1)).ravel() > 0
                      for path in METAPATHS.values()], 1).astype(np.float32)
    sc = {v: np.load(RESULTS_DIR / f"step6_Qwen3-1.7B_{v}_scores.npz") for v in ["dgp", "target"]}
    assert (sc["dgp"]["test_nodes"] == test).all() and (sc["target"]["test_nodes"] == test).all()
    groups = {}
    for j, name in enumerate(METAPATHS):
        for val_ in [0, 1]:
            m = flags[test, j] == val_
            if 0 < y[test][m].sum() < m.sum():
                groups[f"{'has' if val_ else 'no'} {name} ({int(m.sum())} users)"] = {
                    v: round(roc_auc_score(y[test][m], sc[v]["test"][m]), 3) for v in sc}
    out["AUROC within flag groups (dgp vs target)"] = groups
    Xv = np.column_stack([sc["target"]["val"], flags[val]])
    Xt = np.column_stack([sc["target"]["test"], flags[test]])
    stack = LogisticRegression(max_iter=1000).fit(Xv, y[val])
    t = best_threshold(y[val], stack.predict_proba(Xv)[:, 1])
    p_stack = stack.predict_proba(Xt)[:, 1]
    p_dgp = sc["dgp"]["test"] > best_threshold(y[val], sc["dgp"]["val"])
    p_tgt = sc["target"]["test"] > best_threshold(y[val], sc["target"]["val"])
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(test), len(test)) for _ in range(N_BOOT)]
    yt = y[test]
    out["target LoRA + flags (LR fitted on validation)"] = dict(
        macro_f1=round(f1_score(yt, p_stack > t, average="macro"), 4), auroc=round(roc_auc_score(yt, p_stack), 4),
        minus_dgp=paired_ci(yt, p_stack > t, p_dgp, boots), minus_target=paired_ci(yt, p_stack > t, p_tgt, boots))
    out["dgp LoRA"] = dict(macro_f1=round(f1_score(yt, p_dgp, average="macro"), 4), auroc=round(roc_auc_score(yt, sc["dgp"]["test"]), 4))

    # 6. lift over text-only LR
    emb = load_embeddings("qwen").astype(np.float32)
    logdeg = np.log1p(np.column_stack([deg[k] for k in ["in-friend", "out-friend", "in-follow", "out-follow"]])).astype(np.float32)
    base_p, base = lr_preds(emb, y, train, val, test)
    for name, X in [("text + 3 link flags", np.hstack([emb, flags])), ("text + 4 log in/out degrees", np.hstack([emb, logdeg]))]:
        p, pred = lr_preds(X, y, train, val, test)
        out[f"{name} - text (LR)"] = dict(**paired_ci(yt, pred, base, boots), auroc=round(roc_auc_score(yt, p), 4),
                                          auroc_text=round(roc_auc_score(yt, base_p), 4))
    for name, X in [("4 log in/out degrees only", logdeg), ("3 link flags only", flags)]:
        clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[train], y[train])
        p = clf.predict_proba(X[test])[:, 1]
        t = best_threshold(y[val], clf.predict_proba(X[val])[:, 1])
        out[name] = dict(macro_f1=round(f1_score(yt, p > t, average="macro"), 4), auroc=round(roc_auc_score(yt, p), 4))

    print(json.dumps(out, indent=1))
    json.dump(out, open(RESULTS_DIR / "link_direction.json", "w"), indent=2)


if __name__ == "__main__":
    main()
