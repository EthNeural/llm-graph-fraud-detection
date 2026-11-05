"""
Independent study, stage 1b (CPU; designed 6 Nov ~02:55 after stage 1 gave TF-IDF AUROC 1.000 -> shortcut checks).
Working test users only; split, cut-off rule and metrics as in stage 1 (reused from the main study).

(a) Genre check: TF-IDF + LR on COMMENTS ONLY (human posts are news headlines, bot posts are opinion sentences).
(b) Amount-of-text curve: the user is represented by k randomly drawn comments, k in {1, 2, 5, 10, 20, all}
    (train and test alike; 3 draws), TF-IDF + LR. Answers: how much text does it take to spot an LLM bot?
(c) Single-text detection: each comment scored on its own (trained on train users' comments, scored on test users'
    comments), AUROC at the comment level.
(d) LLM focal-word rate: share of a user's comment tokens that are in the published LLM-overused lists
    (Juzek & Ward 2024; Kobak et al. 2025; listed in reports/independent_study/notes_writing_style.md), untrained.
(e) Profile/timing: per-feature permutation importance of the stage-1 tree on validation (which clean feature carries it).
Output: results/independent/botsim_stage1b.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MODEL", "xgb_graph")
from twibot22_metrics import metrics, val_cutoff
from twibot22_cheap_models import hgb

DATA = ROOT.parent / "data" / "botsim24"
OUT = ROOT / "results" / "independent" / "botsim_stage1b.json"
FOCAL = {"delve", "delves", "delved", "delving", "showcasing", "showcases", "boasts", "underscore", "underscores", "underscoring",
         "comprehending", "intricacies", "intricate", "surpassing", "surpasses", "garnered", "emphasizing", "realm", "groundbreaking",
         "advancements", "aligns", "potential", "findings", "crucial", "notably", "across", "additionally", "comprehensive",
         "enhancing", "exhibited", "insights", "particularly", "within"}


def tfidf_lr(train_docs, ytr, all_docs):
    v = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2, sublinear_tf=True)
    T = v.fit_transform(train_docs)
    m = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(T, ytr)
    return m.predict_proba(v.transform(all_docs))[:, 1]


def main():
    assert not OUT.exists(), "never overwritten"
    u, sp = np.load(DATA / "users.npz"), np.load(DATA / "split.npz")
    y, uid, X0, names = u["y"], list(u["user_id"]), u["X_prof"].astype(float), list(u["names"])
    tr, va, te = sp["train"], sp["val"], sp["test_work"]
    texts = json.load(open(DATA / "texts.json"))
    coms = [texts[k]["comments"] for k in uid]
    has = np.array([len(c) > 0 for c in coms])
    out = {}

    def score(s, idx_te=te, idx_va=va):
        c = val_cutoff(y[idx_va], s[idx_va])
        r = metrics(y[idx_te], s[idx_te], c)
        return {k: round(v, 4) for k, v in r.items() if k in ("n", "bots", "macro_f1", "auroc", "pr_auc")}

    # (a) comments only (users with >= 1 comment)
    trc, vac, tec = tr[has[tr]], va[has[va]], te[has[te]]
    docs = [" ".join(c) for c in coms]
    s = tfidf_lr([docs[i] for i in trc], y[trc], docs)
    out["a_comments_only"] = score(s, tec, vac)
    print("(a) comments only:", out["a_comments_only"], flush=True)

    # (b) amount-of-text curve
    curve = {}
    for k in (1, 2, 5, 10, 20, "all"):
        runs = []
        for d in range(3):
            rng = np.random.default_rng(d)
            sub = [" ".join(c if k == "all" or len(c) <= k else list(rng.choice(c, k, replace=False))) for c in coms]
            s = tfidf_lr([sub[i] for i in trc], y[trc], sub)
            runs.append(score(s, tec, vac))
        curve[str(k)] = {m: round(float(np.mean([r[m] for r in runs])), 4) for m in ("macro_f1", "auroc", "pr_auc")}
        print(f"(b) k={k}: {curve[str(k)]}", flush=True)
    out["b_amount_curve_comments"] = curve

    # (c) single-comment detection
    rows_tr = [(i, c) for i in trc for c in coms[i]]
    rows_te = [(i, c) for i in tec for c in coms[i]]
    v = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2, sublinear_tf=True)
    T = v.fit_transform([c for _, c in rows_tr])
    m = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(T, y[[i for i, _ in rows_tr]])
    st = m.predict_proba(v.transform([c for _, c in rows_te]))[:, 1]
    yt = y[[i for i, _ in rows_te]]
    out["c_single_comment_auroc"] = {"comments": len(rows_te), "auroc": round(float(roc_auc_score(yt, st)), 4)}
    print("(c) single comment:", out["c_single_comment_auroc"], flush=True)

    # (d) focal-word rate, untrained
    rate = np.zeros(len(y))
    for i, c in enumerate(coms):
        toks = " ".join(c).lower().split()
        rate[i] = sum(t.strip(".,;:!?()\"'") in FOCAL for t in toks) / max(len(toks), 1)
    out["d_focal_word_rate_auroc_test"] = round(float(roc_auc_score(y[tec], rate[tec])), 4)
    print("(d) focal-word rate AUROC (untrained):", out["d_focal_word_rate_auroc_test"], flush=True)

    # (e) permutation importance of the clean profile/timing tree
    mdl = hgb(0).fit(X0[tr], y[tr])
    pi = permutation_importance(mdl, X0[va], y[va], scoring="roc_auc", n_repeats=10, random_state=0)
    imp = sorted(zip(names, pi.importances_mean), key=lambda t: -t[1])
    out["e_profile_timing_importance"] = [[k, round(float(v), 4)] for k, v in imp]
    print("(e) top features:", imp[:6], flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("Saved", OUT.name)


if __name__ == "__main__":
    main()
