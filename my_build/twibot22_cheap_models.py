"""
TwiBot-22 cheap models (checklist H3: with and without profile), one model per process (G1: no torch,
OMP_NUM_THREADS=1, the launcher logs the exit code, G2). Train labels only (B7): fitted on our TRAIN
users, cut-off on VAL (twibot22_metrics), scored on the WORKING test users; sealed users never scored.

DESIGN
  sample   features_sample_e5tw.npz: the 100,986-user study sample, multilingual-E5 bio + 20 tweets,
           neighbours from the full graph (headline, F6). Test = the 5,000 sample working-test users,
           the same users LoRA target is scored on.
  full     features_full_e5.npz: all 1M users, bio only (full-scale check). Test = 49,951 working-test users.

MODEL (fixed settings, no tuning; 3 seeds unless deterministic)
  profile_only     HistGB on the user's own profile fields
  links_only       HistGB on the neighbour counts only (log count per relation and direction): how much
                   the crawl structure gives away
  tree_self        HistGB on the user's own text PCA (+ profile)        -> "_prof" / "_noprof"
  xgb_graph        HistGB on own + all neighbour features (GADBench XGB-Graph style) -> "_prof" / "_noprof"
  mlp_text         sklearn MLP (256) on the text embedding (sample: full 768-dim E5 bio + tweets;
                   full: bio PCA 32) (+ own profile)                     -> "_prof" / "_noprof"
  flags            no training: score = empty bio flag, or no-tweets flag (deterministic, seed 0)
Column groups by name (H10): base name in {log_count, share_with_text, has_text} = graph; text PCs
(txt_*, bio_pc*) = text; everything else = profile.
Run: OMP_NUM_THREADS=1 DESIGN=sample MODEL=xgb_graph PROFILE=1 python my_build/twibot22_cheap_models.py
"""
import json
import os
import sys
import time
from pathlib import Path

assert os.environ.get("OMP_NUM_THREADS") == "1", "set OMP_NUM_THREADS=1 (G1)"
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import twibot22_guard as guard
from twibot22_metrics import labels, evaluate, save_scores, SEEDS

DATA = ROOT.parent / "data" / "twibot-22"
DESIGN = os.environ.get("DESIGN", "sample")
MODEL = os.environ["MODEL"]
PROFILE = os.environ.get("PROFILE", "1") == "1"
assert DESIGN in ("sample", "full") and MODEL in ("profile_only", "links_only", "tree_self", "xgb_graph", "mlp_text", "flags")
GRAPH_BASES = {"log_count", "share_with_text", "has_text"}


def group(name):
    base = name.split(":", 1)[1]
    if base in GRAPH_BASES:
        return "graph"
    base = base[5:] if base.startswith("mean_") else base
    return "text" if base.startswith(("txt_", "bio_pc")) else "profile"


def hgb(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=20, random_state=seed)


def mlp(seed):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200, random_state=seed)


def main():
    y = labels()
    roles = guard.roles()
    f = np.load(DATA / ("features_sample_e5tw.npz" if DESIGN == "sample" else "features_full_e5.npz"))
    names = list(f["names"])
    guard.check_columns(names)
    rows = f["rows"] if "rows" in f.files else np.arange(len(y))
    tr, va = rows[roles[rows] == "train"], rows[roles[rows] == "val"]
    te = rows[roles[rows] == "test_work"]
    pos = np.full(len(y), -1)
    pos[rows] = np.arange(len(rows))

    g = np.array([group(k) for k in names])
    own = np.array([k.startswith("self:") for k in names])
    if MODEL == "profile_only":
        cols, tag = own & (g == "profile"), "profile_only"
    elif MODEL == "links_only":
        cols, tag = np.array([k.endswith(":log_count") for k in names]), "links_only"
    elif MODEL == "tree_self":
        cols, tag = own & ((g != "profile") | PROFILE), "tree_self_" + ("prof" if PROFILE else "noprof")
    elif MODEL == "xgb_graph":
        cols, tag = (g != "profile") | PROFILE, "xgb_graph_" + ("prof" if PROFILE else "noprof")
    else:
        cols, tag = None, None

    out = {}
    if MODEL == "flags":
        bio = names.index("self:prof_bio empty" if DESIGN == "sample" else "self:bio empty")
        X = f["X"]
        for fname, s_rows in (("flag_empty_bio", X[:, bio].astype(np.float32)),
                              ("flag_no_tweets", (np.load(DATA / "tweet_count_files.npy")[rows] == 0).astype(np.float32))):
            s = np.full(len(y), np.nan, np.float32)
            s[rows] = s_rows
            out[fname] = [evaluate(y, s, va, in_sample_only=DESIGN == "sample")]
            save_scores(DESIGN, fname, 0, s)
    else:
        if MODEL == "mlp_text":
            tag = "mlp_text_" + ("prof" if PROFILE else "noprof")
            if DESIGN == "sample":
                d = DATA / "sample110k_e5_bio20tw_raw"
                ep = np.full(len(y), -1)
                ep[np.load(d / "ids.npy")] = np.arange(len(np.load(d / "ids.npy")))
                Xt = np.load(d / "emb.npy").astype(np.float32)[ep[rows]]
            else:
                Xt = f["X"][:, own & (g == "text")].astype(np.float32)
            X = np.hstack([Xt, f["X"][:, own & (g == "profile")].astype(np.float32)]) if PROFILE else Xt
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit(X[pos[tr]]).transform(X).astype(np.float32)
        else:
            X = f["X"][:, cols].astype(np.float32)
        print(f"{DESIGN} {tag}: {X.shape[1]} columns; train {len(tr):,} val {len(va):,} test {len(te):,}", flush=True)
        out[tag] = []
        for seed in SEEDS:
            t0 = time.time()
            m = (mlp if MODEL == "mlp_text" else hgb)(seed).fit(X[pos[tr]], y[tr])
            s = np.full(len(y), np.nan, np.float32)
            s[rows] = m.predict_proba(X)[:, 1]
            guard.assert_not_sealed(te)
            r = evaluate(y, s, va, in_sample_only=DESIGN == "sample")
            r["minutes"] = round((time.time() - t0) / 60, 2)
            out[tag].append(r)
            save_scores(DESIGN, tag, seed, s)
            print(f"  seed {seed}: val F1 {r['val_macro_f1']:.3f} | test all F1 {r['all']['macro_f1']:.3f} "
                  f"AUROC {r['all']['auroc']:.3f} PR-AUC {r['all']['pr_auc']:.3f} | non-verified F1 "
                  f"{r['non_verified']['macro_f1']:.3f} ({r['minutes']} min)", flush=True)
    res = ROOT / "results" / "twibot22" / f"cheap_{DESIGN}.json"
    allres = json.load(open(res)) if res.exists() else {}
    allres.update(out)
    json.dump(allres, open(res, "w"), indent=1)


if __name__ == "__main__":
    main()
