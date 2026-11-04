"""
TwiBot-22: our split and the full-graph feature table for the cheap models (MLP, XGB-Graph, DEC-lite).
No model is trained here and no test score is computed: this only prepares data.

1. SPLIT (fixed before any score is seen; see reports/twibot22_plan.md)
   Stratified 70/20/10 train/val/test over all 1,000,000 users, seed 42. All 986 expert-labelled
   users (data/twibot-22-raw/expert_labels/label_manual.csv) are forced into TEST, so no model ever
   trains or tunes on them. The official split.csv is kept alongside for comparison only.
   -> data/twibot-22/split_ours.npy  ('train' / 'val' / 'test', label.csv row order)

2. BASE FEATURES per user (no "verified": it is a labelling function and a shortcut)
   profile: protected, has pinned tweet, default profile image, has url, has location, bio empty,
            bio length, urls in bio, name length, username digits, log followers, log following,
            log tweet count, log listed count, account age (days)        (twibot22_profile_check.py)
   text:    MiniLM bio embedding (emb_bio_minilm_all.npy) reduced to 32 dims with PCA fitted on TRAIN
   Numeric columns are standardised with TRAIN statistics only.

3. NEIGHBOUR FEATURES (DEC / GADBench XGB-Graph style), computed on the FULL graph
   For each relation r in RELATIONS and each direction (in = who points at me, out = whom I point at):
       count of neighbours (log1p), mean of the neighbours' base features
   plus a 2-hop mean over the union graph (all relations, both directions), as in GADBench's default.
   Uses neighbours' FEATURES only, never their labels.

Output: data/twibot-22/features_full.npz  (X float16 [1M x d], names, base_dim)
Run:    python my_build/twibot22_features.py         (CPU, ~5-10 min, ~4 GB RAM)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "twibot-22"
RAW = ROOT.parent / "data" / "twibot-22-raw"
RELATIONS = ["following", "followers", "mention", "retweet", "reply", "quote", "like"]
SEED = 42
PCA_DIMS = 32


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_split(lab):
    y = (lab.label.values == "bot").astype(int)
    expert = set(pd.read_csv(RAW / "expert_labels" / "label_manual.csv").id)
    is_exp = lab.id.isin(expert).values
    idx = np.where(~is_exp)[0]
    train, rest = train_test_split(idx, train_size=0.7, stratify=y[idx], random_state=SEED)
    val, test = train_test_split(rest, train_size=2 / 3, stratify=y[rest], random_state=SEED)
    split = np.empty(len(y), dtype=object)
    split[train], split[val], split[test] = "train", "val", "test"
    split[is_exp] = "test"
    np.save(DATA / "split_ours.npy", split.astype(str))
    for s in ("train", "val", "test"):
        m = split == s
        log(f"  {s:5s} {m.sum():>8,} users, bot share {y[m].mean():.3f}")
    log(f"  expert users in test: {is_exp.sum()}")
    return split


def base_features(lab, split):
    import sys
    sys.path.insert(0, str(ROOT))
    from twibot22_profile_check import features
    users = pd.read_parquet(DATA / "users.parquet").set_index("id").loc[lab.id].reset_index()
    prof = features(users).drop(columns=["verified", "withheld"])     # withheld: 0 users, constant
    tr = split == "train"
    P = prof.to_numpy(np.float32)
    mu, sd = P[tr].mean(0), P[tr].std(0) + 1e-6
    P = (P - mu) / sd
    emb = np.load(ROOT / "results" / "twibot22" / "emb_bio_minilm_all.npy").astype(np.float32)
    pca = PCA(n_components=PCA_DIMS, random_state=SEED).fit(emb[tr][::5])   # 1/5 of train is plenty
    E = pca.transform(emb).astype(np.float32)
    E = (E - E[tr].mean(0)) / (E[tr].std(0) + 1e-6)
    log(f"  profile {P.shape[1]} cols, bio PCA {PCA_DIMS} dims (explained {pca.explained_variance_ratio_.sum():.2f})")
    names = list(prof.columns) + [f"bio_pc{i}" for i in range(PCA_DIMS)]
    return np.hstack([P, E]), names


def mean_agg(A, X):
    deg = np.asarray(A.sum(1)).ravel()
    M = sp.diags(1 / np.maximum(deg, 1)) @ A
    return np.log1p(deg).astype(np.float32)[:, None], (M @ X).astype(np.float32)


def main():
    lab = pd.read_csv(RAW / "label.csv")
    n = len(lab)
    log("split")
    split = make_split(lab)
    log("base features")
    X, base_names = base_features(lab, split)
    rels = np.load(DATA / "relations_uu.npz")
    blocks, names = [X], [f"self:{c}" for c in base_names]
    union = sp.csr_matrix((n, n), dtype=np.float32)
    for r in RELATIONS:
        e = rels[r]
        A = sp.csr_matrix((np.ones(e.shape[1], np.float32), (e[0], e[1])), shape=(n, n))
        A.data[:] = 1
        union = union + A + A.T
        for d, M in (("out", A), ("in", A.T.tocsr())):
            c, m = mean_agg(M, X)
            blocks += [c, m]
            names += [f"{r}_{d}:log_count"] + [f"{r}_{d}:mean_{b}" for b in base_names]
        log(f"  {r:10s} done ({e.shape[1]:,} edges)")
    union.data[:] = 1
    c1, m1 = mean_agg(union, X)
    _, m2 = mean_agg(union, m1)
    blocks += [c1, m1, m2]
    names += ["union:log_count"] + [f"union_hop1:mean_{b}" for b in base_names] + [f"union_hop2:mean_{b}" for b in base_names]
    F = np.hstack(blocks).astype(np.float16)
    np.savez(DATA / "features_full.npz", X=F, names=np.array(names), base_dim=X.shape[1])
    log(f"saved features_full.npz: {F.shape[0]:,} users x {F.shape[1]} features ({F.nbytes / 1e9:.1f} GB)")
    (ROOT / "results" / "twibot22" / "features_info.json").write_text(json.dumps(dict(
        n_users=n, n_features=F.shape[1], base_dim=X.shape[1], relations=RELATIONS,
        split_counts={s: int((split == s).sum()) for s in ("train", "val", "test")}), indent=1))


if __name__ == "__main__":
    main()
