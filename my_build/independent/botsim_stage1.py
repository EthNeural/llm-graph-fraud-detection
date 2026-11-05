"""
Independent study, stage 1 (CPU, rules in reports/independent_study/DECISIONS.md 02:35): BotSim-24, working test only.
Reuses the main study's code: twibot22_metrics.val_cutoff / metrics (cut-off on validation; macro-F1, AUROC, PR-AUC,
PR-AUC reweighted to 5% / 1% bots), twibot22_cheap_models.hgb (the same HistGradientBoosting tree), XGB-Graph-style
neighbour means, paired bootstrap 2,000 as in twibot22_metrics.bootstrap_diff. 3 seeds for trees.

Models (no leak fields anywhere; see data_inspection_botsim24.md):
  profile_timing  : 18 clean features (counts, subreddits, timing, text lengths)
  tfidf_lr        : TF-IDF (1-2 grams, 50k) + logistic regression on all of a user's posts + comments
  in_degree_probe : minus in-degree, untrained (the leak, for reference)
  xgbgraph_out    : profile_timing + means of out-neighbours' features per edge type + out-degree per type (G-out)
  xgbgraph_full   : the same + in-neighbour means + in-degree per type (G-full, as the paper's GNNs see)
  fusion_out      : profile_timing + TF-IDF-LR score (cross-fitted on train) + G-out neighbour features
  leak_contrast   : profile_timing + share of zero-score posts/comments (a leak; reported only to show its size)
Output: results/independent/botsim_stage1.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MODEL", "xgb_graph")
from twibot22_metrics import metrics, val_cutoff          # reused from the main study
from twibot22_cheap_models import hgb                      # the same tree

DATA = ROOT.parent / "data" / "botsim24"
RAW = ROOT.parent / "data" / "botsim24-raw"
OUT = ROOT / "results" / "independent" / "botsim_stage1.json"


def neigh_means(X, src, dst, et, n, direction):
    cols, names = [], []
    for t in range(3):
        m = et == t
        a, b = (src[m], dst[m]) if direction == "out" else (dst[m], src[m])
        deg = np.bincount(a, minlength=n).astype(float)
        S = np.zeros_like(X)
        np.add.at(S, a, X[b])
        cols += [S / np.maximum(deg, 1)[:, None], np.log1p(deg)[:, None]]
        names += [f"{direction}{t}:mean_{k}" for k in range(X.shape[1])] + [f"{direction}{t}:log_deg"]
    return np.hstack(cols), names


def boot(y, a, b, ca, cb, idx, seed=0):
    rng = np.random.default_rng(seed)
    yy, sa, sb = y[idx], a[idx], b[idx]
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(idx), len(idx))
        if yy[i].min() == yy[i].max():
            continue
        d.append([f1_score(yy[i], sa[i] > ca, average="macro") - f1_score(yy[i], sb[i] > cb, average="macro"),
                  roc_auc_score(yy[i], sa[i]) - roc_auc_score(yy[i], sb[i]),
                  average_precision_score(yy[i], sa[i]) - average_precision_score(yy[i], sb[i])])
    d = np.array(d)
    return {m: [round(float(d[:, j].mean()), 4), round(float(np.percentile(d[:, j], 2.5)), 4), round(float(np.percentile(d[:, j], 97.5)), 4)]
            for j, m in enumerate(("macro_f1", "auroc", "pr_auc"))}


def main():
    assert not OUT.exists(), "never overwritten"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    u, e, sp = np.load(DATA / "users.npz"), np.load(DATA / "edges.npz"), np.load(DATA / "split.npz")
    y, X0, uid = u["y"], u["X_prof"].astype(np.float64), list(u["user_id"])
    n = len(y)
    tr, va, te = sp["train"], sp["val"], sp["test_work"]
    src, dst, et = e["src"], e["dst"], e["etype"]
    texts = json.load(open(DATA / "texts.json"))
    docs = [" ".join(texts[k]["posts"] + texts[k]["comments"]) for k in uid]

    # TF-IDF + LR: cross-fitted on train (5 folds) so the fusion model never sees in-sample text scores
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2, sublinear_tf=True)
    T = vec.fit_transform([docs[i] for i in tr])
    Tall = vec.transform(docs)
    lr = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(T, y[tr])
    s_text = lr.predict_proba(Tall)[:, 1]
    oof = s_text.copy()
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y[tr])):
        m = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(Tall[tr[a]], y[tr[a]])
        oof[tr[b]] = m.predict_proba(Tall[tr[b]])[:, 1]

    Nout, _ = neigh_means(X0, src, dst, et, n, "out")
    Nin, _ = neigh_means(X0, src, dst, et, n, "in")
    # leak contrast: share of zero-score items (simulation artefact)
    jraw = json.load(open(RAW / "user_post_comment.json"))
    zero = np.zeros(n)
    for i, k in enumerate(uid):
        r = jraw[k]
        sc = [p["score"] for p in r["posts"]] + [c["comment_score"] for l in ("comment_1", "comment_2") for c in r.get(l, [])]
        zero[i] = np.mean(np.array(sc) == 0) if sc else 0.0

    designs = {"profile_timing": X0, "xgbgraph_out": np.hstack([X0, Nout]), "xgbgraph_full": np.hstack([X0, Nout, Nin]),
               "fusion_out": np.hstack([X0, oof[:, None], Nout]), "leak_contrast": np.hstack([X0, zero[:, None]])}
    res, scores = {}, {}
    indeg = np.bincount(dst, minlength=n).astype(float)
    for name, s in (("tfidf_lr", s_text), ("in_degree_probe", -indeg)):
        c = val_cutoff(y[va], s[va])
        res[name] = {"seeds": 1, **{k: round(v, 4) for k, v in metrics(y[te], s[te], c).items() if k not in ("n", "bots")}}
        scores[name] = (s, c)
    for name, X in designs.items():
        if name == "fusion_out":
            X = X.copy()
            X[:, X0.shape[1]] = np.where(np.isin(np.arange(n), tr), oof, s_text)   # out-of-fold for train, full model elsewhere
        runs = []
        for sd in range(3):
            m = hgb(sd).fit(X[tr], y[tr])
            s = m.predict_proba(X)[:, 1]
            c = val_cutoff(y[va], s[va])
            runs.append(metrics(y[te], s[te], c))
            if sd == 0:
                scores[name] = (s, c)
        res[name] = {"seeds": 3, **{k: round(float(np.mean([r[k] for r in runs])), 4) for k in runs[0] if k not in ("n", "bots")},
                     "auroc_sd": round(float(np.std([r["auroc"] for r in runs])), 4)}
    for k, v in res.items():
        print(f"{k:16s} macro-F1 {v['macro_f1']:.3f}  AUROC {v['auroc']:.3f}  PR-AUC {v['pr_auc']:.3f}  ({v['seeds']} seed(s))", flush=True)
    pairs = {"G-out - profile_timing": ("xgbgraph_out", "profile_timing"), "G-full - G-out (in-edge artefact)": ("xgbgraph_full", "xgbgraph_out"),
             "text - profile_timing": ("tfidf_lr", "profile_timing"), "fusion_out - best single (text)": ("fusion_out", "tfidf_lr"),
             "fusion_out - G-out": ("fusion_out", "xgbgraph_out")}
    out_pairs = {}
    for pname, (a, b) in pairs.items():
        (sa, ca), (sb, cb) = scores[a], scores[b]
        out_pairs[pname] = boot(y, sa, sb, ca, cb, te)
        r = out_pairs[pname]
        print(f"  {pname:36s} " + "  ".join(f"{m} {r[m][0]:+.3f} [{r[m][1]:+.3f}, {r[m][2]:+.3f}]" for m in r), flush=True)
    OUT.write_text(json.dumps({"test_work_users": int(len(te)), "bots": int(y[te].sum()), "models": res, "pairs": out_pairs}, indent=1))
    print("Saved", OUT.name)


if __name__ == "__main__":
    main()
