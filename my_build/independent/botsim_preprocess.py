"""
Independent study, step 1: BotSim-24 preprocessing (Fri 6 Nov ~02:25, before any model).

Leaks found in the inspection (reports/independent_study/data_inspection_botsim24.md) are EXCLUDED here and never
written to the feature table: post score / upvote_ratio / num_comments, comment_score, post and comment ids,
description presence and text, character_setting.

Outputs (data/botsim24/, git-ignored):
  users.npz   y (1 = bot), names, X_prof (allowed profile counts + subreddit membership + timing), feature names
  texts.json  per user: list of post titles, list of comment bodies (for text encoders / LLMs later)
  edges.npz   directed typed edges commenter -> parent author: type 0 = level-1 comment on a post,
              1 = level-2 comment on a post's thread author, 2 = level-2 reply to a level-1 commenter; weight = count
  split.npz   stratified random split, seed 42: train 60% / val 20% / test_work 10% / test_sealed 10%
Prints: edge counts by class direction (incl. human -> bot), edge homophily and adjusted homophily per type.
Run: python my_build/independent/botsim_preprocess.py
"""
import ast
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW, OUT = ROOT / "data" / "botsim24-raw", ROOT / "data" / "botsim24"
SUBS = ["worldnews", "politics", "news", "GlobalTalk", "InternationalNews", "UkrainianConflict"]


def adjusted_homophily(y, a, b):
    h = np.mean(y[a] == y[b])
    p = np.bincount(np.concatenate([y[a], y[b]]), minlength=2) / (2 * len(a))
    s = (p ** 2).sum()
    return float((h - s) / (1 - s))


def main():
    OUT.mkdir(exist_ok=True)
    u = pd.read_csv(RAW / "Users.csv")
    j = json.load(open(RAW / "user_post_comment.json"))
    n = len(u)
    y = (np.arange(n) >= 1907).astype(int)
    uid = u.user_id.astype(str).tolist()
    idx = {k: i for i, k in enumerate(uid)}

    # who wrote each post / comment (ids used ONLY to resolve reply links, never as features)
    post_author, com_author = {}, {}
    for k, r in j.items():
        for p in r["posts"]:
            post_author[p["submission_id"]] = idx[k]
        for lvl in ("comment_1", "comment_2"):
            for c in r.get(lvl, []):
                com_author[c["comment_id"]] = idx[k]

    edges = collections.Counter()
    unresolved = collections.Counter()
    for k, r in j.items():
        a = idx[k]
        for lvl in ("comment_1", "comment_2"):
            for c in r.get(lvl, []):
                par, link = str(c["parent_id"]), str(c["link_id"])
                if par.startswith("t3_"):
                    b, t = post_author.get(par[3:]), 0 if lvl == "comment_1" else 1
                else:
                    b, t = com_author.get(par[3:]), 2
                if b is None:
                    unresolved[lvl] += 1
                    continue
                if b != a:
                    edges[(a, b, t)] += 1
                if lvl == "comment_2":                      # also link to the thread's post author
                    pb = post_author.get(link[3:])
                    if pb is not None and pb != a and par.startswith("t1_"):
                        edges[(a, pb, 1)] += 1
    E = np.array([[a, b, t, w] for (a, b, t), w in edges.items()], dtype=np.int64)
    np.savez(OUT / "edges.npz", src=E[:, 0], dst=E[:, 1], etype=E[:, 2], weight=E[:, 3])
    print(f"edges: {len(E):,} directed typed pairs ({E[:, 3].sum():,} comments); unresolved parents {dict(unresolved)}")
    for t, name in enumerate(["L1 comment -> post author", "L2 comment -> post author", "L2 reply -> L1 commenter"]):
        m = E[:, 2] == t
        s, d = E[m, 0], E[m, 1]
        dirs = collections.Counter(zip(y[s], y[d]))
        print(f"  type {t} {name:28s} pairs {m.sum():>6,} | h->h {dirs[(0,0)]:,} h->b {dirs[(0,1)]:,} b->h {dirs[(1,0)]:,} b->b {dirs[(1,1)]:,} "
              f"| edge homophily {np.mean(y[s] == y[d]):.3f} adjusted {adjusted_homophily(y, s, d):+.3f}")
    s, d = E[:, 0], E[:, 1]
    dirs = collections.Counter(zip(y[s], y[d]))
    print(f"  all types: h->b {dirs[(0,1)]:,} of {len(E):,} (the paper: none, because human activity is replayed)")

    # allowed features
    feats = {}
    for c in ("submission_num", "comment_num", "comment_num_1", "comment_num_2"):
        feats[f"prof:log_{c}"] = np.log1p(u[c].values.astype(float))
    feats["prof:comment_per_post"] = np.log1p(u.comment_num.values / np.maximum(u.submission_num.values, 1))
    subl = u.subreddit.apply(lambda s: list(ast.literal_eval(s)) if isinstance(s, str) else [])   # safe parse, never eval
    for sname in SUBS:
        feats[f"sub:{sname}"] = subl.apply(lambda L: sname in L).astype(float).values
    feats["sub:count"] = subl.apply(len).values.astype(float)
    texts = {}
    tf = collections.defaultdict(list)
    for k, r in j.items():
        i = idx[k]
        posts = [p["posts"] or "" for p in r["posts"]]
        coms = [c["comment_body"] or "" for lvl in ("comment_1", "comment_2") for c in r.get(lvl, [])]
        texts[uid[i]] = {"posts": posts, "comments": coms}
        ts = sorted(pd.to_datetime([p["created_utc"] for p in r["posts"]] +
                                   [c["created_utc"] for lvl in ("comment_1", "comment_2") for c in r.get(lvl, [])]))
        gaps = np.diff([t.value / 1e9 for t in ts]) if len(ts) > 1 else np.array([])
        tf["time:log_median_gap_s"].append(np.log1p(np.median(gaps)) if len(gaps) else 0.0)
        tf["time:gap_cv"].append(float(gaps.std() / (gaps.mean() + 1e-9)) if len(gaps) > 1 else 0.0)
        hours = np.array([t.hour for t in ts]) if ts else np.array([])
        tf["time:hour_entropy"].append(float(-(lambda p: (p[p > 0] * np.log(p[p > 0])).sum())(np.bincount(hours, minlength=24) / max(len(hours), 1))) if len(hours) else 0.0)
        tf["time:active_days"].append(float(len({t.date() for t in ts})))
        tf["text:mean_post_len"].append(np.log1p(np.mean([len(x) for x in posts])) if posts else 0.0)
        tf["text:mean_comment_len"].append(np.log1p(np.mean([len(x) for x in coms])) if coms else 0.0)
    order = np.argsort([idx[k] for k in j])                  # rows already in idx order via uid
    for k2, v in tf.items():
        arr = np.zeros(n)
        for kk, val in zip(j.keys(), v):
            arr[idx[kk]] = val
        feats[k2] = arr
    names = list(feats)
    X = np.vstack([feats[k] for k in names]).T.astype(np.float32)
    assert np.isfinite(X).all()
    np.savez(OUT / "users.npz", y=y, X_prof=X, names=np.array(names), user_id=np.array(uid))
    json.dump(texts, open(OUT / "texts.json", "w"))

    # split (fixed before any model)
    i_all = np.arange(n)
    tr, rest = train_test_split(i_all, train_size=0.6, stratify=y, random_state=42)
    va, te = train_test_split(rest, train_size=0.5, stratify=y[rest], random_state=42)
    tw, ts_ = train_test_split(te, train_size=0.5, stratify=y[te], random_state=42)
    np.savez(OUT / "split.npz", train=np.sort(tr), val=np.sort(va), test_work=np.sort(tw), test_sealed=np.sort(ts_))
    print(f"features {X.shape} {names}")
    print(f"split: train {len(tr)} val {len(va)} test_work {len(tw)} test_sealed {len(ts_)}; bot share {y[tr].mean():.3f}/{y[va].mean():.3f}/{y[tw].mean():.3f}/{y[ts_].mean():.3f}")


if __name__ == "__main__":
    main()
