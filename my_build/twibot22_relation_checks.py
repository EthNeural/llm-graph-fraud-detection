"""
TwiBot-22 relation checks (checklist E8, E10; designed 5 Nov ~19:20 before any number).

Q1 (E8) Homophily by relation, direction and class. Edges u -> v as stored in relations_uu.npz
   ("out" = whom u points at). Only users with a TRAIN or VAL label count (no test, no sealed).
   Per relation and direction: P(nbr bot | node bot), P(nbr bot | node human), base rate, and adjusted
   homophily (Platonov et al. 2023) on those directed pairs.
   -> results/twibot22/relation_homophily_by_class.json

Q3 (E10) Cross-relation inconsistency features, label-free, for all 1M users. Two relation groups,
   follow = following + followers and interaction = mention + retweet + reply + quote + like, both
   directions. Per user: cos distance between the groups' mean neighbour bio embedding (bio PCA 32 from
   features_full_e5.npz, all users have it), Euclidean distance between their mean neighbour profiles,
   cos(own bio, follow mean), cos(own bio, interaction mean) and their difference, log degree ratio,
   and flags for an empty group. -> data/twibot-22/inconsistency.npz (X [1M x 8], names)
Run: python my_build/twibot22_relation_checks.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import twibot22_guard as guard
from twibot22_metrics import labels

DATA = ROOT.parent / "data" / "twibot-22"
RELATIONS = ["following", "followers", "mention", "retweet", "reply", "quote", "like"]
GROUPS = {"follow": ["following", "followers"], "interaction": ["mention", "retweet", "reply", "quote", "like"]}


def adjusted_homophily(y, a, b):
    # Platonov et al. 2023: (h_edge - sum_k p_k^2) / (1 - sum_k p_k^2), p_k = degree-weighted class share
    h = np.mean(y[a] == y[b])
    deg_share = np.bincount(np.concatenate([y[a], y[b]]), minlength=2) / (2 * len(a))
    s = (deg_share ** 2).sum()
    return float((h - s) / (1 - s))


def q1(y, roles, rels):
    known = np.isin(roles, ["train", "val"])
    out = {"base_rate_bot": float(y[known].mean()), "relations": {}}
    for r in RELATIONS:
        e = rels[r]
        m = known[e[0]] & known[e[1]] & (e[0] != e[1])
        u, v = e[0][m], e[1][m]
        res = {}
        for d, (a, b) in (("out", (u, v)), ("in", (v, u))):        # in: who points at me
            res[d] = dict(pairs=int(len(a)),
                          p_nbr_bot_given_bot=float(y[b][y[a] == 1].mean()) if (y[a] == 1).any() else None,
                          p_nbr_bot_given_human=float(y[b][y[a] == 0].mean()) if (y[a] == 0).any() else None,
                          nodes_with_edge_bot=int(len(np.unique(a[y[a] == 1]))),
                          nodes_with_edge_human=int(len(np.unique(a[y[a] == 0]))))
        res["adjusted_homophily"] = adjusted_homophily(y, u, v)
        out["relations"][r] = res
        print(f"{r:10s} adj.h {res['adjusted_homophily']:+.3f} | out: bot->bot {res['out']['p_nbr_bot_given_bot']:.3f} "
              f"human->bot {res['out']['p_nbr_bot_given_human']:.3f} | in: bot<-bot {res['in']['p_nbr_bot_given_bot']:.3f} "
              f"human<-bot {res['in']['p_nbr_bot_given_human']:.3f} | pairs {len(u):,}", flush=True)
    (ROOT / "results" / "twibot22" / "relation_homophily_by_class.json").write_text(json.dumps(out, indent=1))


def q3(n, rels):
    f = np.load(DATA / "features_full_e5.npz")
    names = list(f["names"])
    own = [j for j, k in enumerate(names) if k.startswith("self:")]
    bio = [j for j in own if "bio_pc" in names[j]]
    prof = [j for j in own if "bio_pc" not in names[j]]
    B, P = f["X"][:, bio].astype(np.float32), f["X"][:, prof].astype(np.float32)
    mats, deg = {}, {}
    for g, members in GROUPS.items():
        A = sp.csr_matrix((n, n), dtype=np.float32)
        for r in members:
            e = rels[r]
            M = sp.csr_matrix((np.ones(e.shape[1], np.float32), (e[0], e[1])), shape=(n, n))
            A = A + M + M.T
        A = (A > 0).astype(np.float32).tocsr()
        A.setdiag(0)
        A.eliminate_zeros()
        deg[g] = np.asarray(A.sum(1)).ravel()
        D = sp.diags(1 / np.maximum(deg[g], 1))
        mats[g] = (D @ A @ B, D @ A @ P)
    cos = lambda a, b: (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-6)
    has_f, has_i = deg["follow"] > 0, deg["interaction"] > 0
    both = has_f & has_i
    feats = {
        "inc:cos_dist_follow_vs_interaction": np.where(both, 1 - cos(mats["follow"][0], mats["interaction"][0]), 0),
        "inc:profile_dist_follow_vs_interaction": np.where(both, np.linalg.norm(mats["follow"][1] - mats["interaction"][1], axis=1), 0),
        "inc:own_cos_follow": np.where(has_f, cos(B, mats["follow"][0]), 0),
        "inc:own_cos_interaction": np.where(has_i, cos(B, mats["interaction"][0]), 0),
        "inc:own_cos_difference": np.where(both, cos(B, mats["follow"][0]) - cos(B, mats["interaction"][0]), 0),
        "inc:log_degree_ratio": np.log1p(deg["follow"]) - np.log1p(deg["interaction"]),
        "inc:no_follow_group": (~has_f).astype(np.float32),
        "inc:no_interaction_group": (~has_i).astype(np.float32),
    }
    names_out = list(feats)
    guard.check_columns(names_out)
    X = np.vstack([feats[k] for k in names_out]).T.astype(np.float32)
    assert np.isfinite(X).all()
    out = DATA / "inconsistency.npz"
    assert not out.exists(), f"{out.name} exists: never overwritten"
    np.savez(out, X=X, names=np.array(names_out))
    print(f"Q3 features saved: {X.shape}; users with both groups {both.sum():,}, only follow {(has_f & ~has_i).sum():,}, "
          f"only interaction {(has_i & ~has_f).sum():,}, neither {(~has_f & ~has_i).sum():,}", flush=True)


def main():
    y, roles = labels(), guard.roles()
    rels = np.load(DATA / "relations_uu.npz")
    print("Q1: homophily by relation, direction and class (train + val users only)")
    q1(y, roles, rels)
    print("\nQ3: cross-relation inconsistency features (label-free)")
    q3(len(y), rels)


if __name__ == "__main__":
    main()
