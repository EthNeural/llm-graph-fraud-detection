"""
H10: feature table for the 100,986 study-sample users (headline design, F6), neighbour features from the
FULL 1M graph. Like twibot22_features.py, but the text is multilingual-E5 on bio + 20 most recent tweets
(sample110k_e5_bio20tw_raw: the 110,057 sample + hub users), where it exists:

  self:      profile (15 fields, no "verified"), text PCA 32 (bio + tweets), has_text
  per relation r (7) and direction (out / in):
             log count of neighbours (all 1M), mean neighbour profile (all have one),
             mean neighbour text PCA over neighbours WITH text, share of neighbours with text
  union 1-hop (all relations, both directions): log count, profile mean, text mean, share with text
  union 2-hop: mean of the bio-only features from features_full_e5.npz (2-hop neighbours are mostly
             not embedded: 18%, checked 5 Nov 00:50), as in the full-graph table

Column groups, used by the model runner for "with / without profile" (H3):
  names containing "prof_" -> profile; "txt_" -> text; counts / shares -> graph.
PCA and standardisation fitted on SAMPLE TRAIN users only. Neighbours' features only, never labels.
Output: data/twibot-22/features_sample_e5tw.npz (X float16 [100,986 x d], names, rows = label.csv rows)
Run:    python my_build/twibot22_features_sample.py   (CPU, ~5 min)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from twibot22_guard import check_columns
from twibot22_features import RELATIONS, PCA_DIMS, SEED, mean_agg

DATA = ROOT.parent / "data" / "twibot-22"
RAW = ROOT.parent / "data" / "twibot-22-raw"
OUT = DATA / "features_sample_e5tw.npz"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    assert not OUT.exists(), f"{OUT.name} exists: never overwritten (H6)"
    lab = pd.read_csv(RAW / "label.csv")
    n = len(lab)
    roles = np.load(DATA / "roles.npy")
    sample = np.load(DATA / "sample_idx.npy")
    tr_s = sample[roles[sample] == "train"]

    # profile for all 1M (standardised on sample train)
    from twibot22_profile_check import features
    users = pd.read_parquet(DATA / "users.parquet").set_index("id").loc[lab.id].reset_index()
    prof = features(users).drop(columns=["withheld"])
    check_columns(prof.columns)
    P = prof.to_numpy(np.float32)
    P = (P - P[tr_s].mean(0)) / (P[tr_s].std(0) + 1e-6)
    pnames = [f"prof_{c}" for c in prof.columns]

    # text (bio + tweets) for the 110,057 embedded users; zero vector = no text
    d = DATA / "sample110k_e5_bio20tw_raw"
    erows, emb = np.load(d / "ids.npy"), np.load(d / "emb.npy").astype(np.float32)
    has = np.zeros(n, bool)
    has[erows[np.linalg.norm(emb, axis=1) > 0]] = True
    pos = np.full(n, -1)
    pos[erows] = np.arange(len(erows))
    fit = pos[tr_s][has[tr_s]]
    pca = PCA(n_components=PCA_DIMS, random_state=SEED).fit(emb[fit])
    T = np.zeros((n, PCA_DIMS), np.float32)
    T[erows] = pca.transform(emb)
    T[~has] = 0
    mu, sd = T[tr_s][has[tr_s]].mean(0), T[tr_s][has[tr_s]].std(0) + 1e-6
    T[has] = (T[has] - mu) / sd
    tnames = [f"txt_pc{i}" for i in range(PCA_DIMS)]
    log(f"text PCA {PCA_DIMS} (explained {pca.explained_variance_ratio_.sum():.2f}); users with text {has.sum():,}")
    H = has.astype(np.float32)[:, None]

    def nbr_block(A, tag):
        deg = np.asarray(A.sum(1)).ravel()
        c, mp = mean_agg(A, P)
        k = A @ H                                                   # neighbours with text
        mt = (A @ T) / np.maximum(k, 1)
        share = k / np.maximum(deg, 1)[:, None]
        cols = [c[sample], mp[sample], mt[sample], share[sample]]
        nm = [f"{tag}:log_count"] + [f"{tag}:mean_{p}" for p in pnames] + [f"{tag}:mean_{t}" for t in tnames] + [f"{tag}:share_with_text"]
        return cols, nm

    blocks = [P[sample], T[sample], H[sample]]
    names = [f"self:{p}" for p in pnames] + [f"self:{t}" for t in tnames] + ["self:has_text"]
    rels = np.load(DATA / "relations_uu.npz")
    union = sp.csr_matrix((n, n), dtype=np.float32)
    for r in RELATIONS:
        e = rels[r]
        A = sp.csr_matrix((np.ones(e.shape[1], np.float32), (e[0], e[1])), shape=(n, n))
        A.data[:] = 1
        union = union + A + A.T
        for dname, M in (("out", A), ("in", A.T.tocsr())):
            c, nm = nbr_block(M, f"{r}_{dname}")
            blocks += c
            names += nm
        log(f"  {r:10s} done")
    union.data[:] = 1
    c, nm = nbr_block(union, "union_hop1")
    blocks += c
    names += nm
    full = np.load(DATA / "features_full_e5.npz")
    fn = list(full["names"])
    hop2 = [j for j, k in enumerate(fn) if k.startswith("union_hop2:")]
    blocks.append(full["X"][sample][:, hop2].astype(np.float32))
    # bio-only 2-hop: rename to the same groups (profile fields keep "prof_", bio PCs become "txt_bio_")
    names += [fn[j].replace("union_hop2:mean_bio_pc", "union_hop2:mean_txt_bio_pc").replace("union_hop2:mean_", "union_hop2:mean_prof_")
              .replace("mean_prof_txt_bio_pc", "mean_txt_bio_pc") for j in hop2]

    X = np.hstack(blocks).astype(np.float16)
    check_columns(names)
    assert X.shape[1] == len(names) and np.isfinite(X).all()
    np.savez(OUT, X=X, names=np.array(names), rows=sample)
    info = dict(n_users=int(len(sample)), n_features=int(X.shape[1]), text="multilingual-E5 bio + 20 tweets (2-hop: bio only)",
                profile_columns=sum("prof_" in k for k in names), text_columns=sum("txt_" in k for k in names),
                graph_columns=sum(("prof_" not in k) and ("txt_" not in k) for k in names),
                pca_explained=float(pca.explained_variance_ratio_.sum()))
    (ROOT / "results" / "twibot22" / "features_info_sample_e5tw.json").write_text(json.dumps(info, indent=1))
    log(f"saved {OUT.name}: {X.shape[0]:,} x {X.shape[1]}  {info}")


if __name__ == "__main__":
    main()
