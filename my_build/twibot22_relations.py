"""
TwiBot-22: which user-user relations exist, and how much label signal (heterophily) does each carry?

TwiBot-22's edge.csv (6.6 GB, ~170M rows: source_id, relation, target_id) has 14 relation types
between 4 entity types (user u..., tweet t..., list l..., hashtag h...). Only `following` and
`followers` join two users directly; the official BotRGCN code uses just those two. Every other
user-user relation has to be built through a tweet or a list:

  following     u -> v                          u follows v                      (direct)
  followers     u -> v                          u is followed by v               (direct)
  mention       author(t) -> v                  t mentions v
  retweet       author(t) -> author(t')         t retweets t'
  quote         author(t) -> author(t')         t quotes t'
  reply         author(t) -> author(t')         t replies to t'
  like          u -> author(t)                  u likes t
  co_list       u -- v                          u and v are members of the same list
author(t) comes from the `post` relation (user -> tweet). Endpoints outside the 1M labelled users
are dropped, and so are self-loops.

For every relation it reports (on all 1M users and inside each of the paper's 10 sub-communities):
  - edges, and the share of bots / humans with at least one edge (does isolation itself carry the label?)
  - edge homophily, chance, adjusted homophily, label informativeness (graph_stats.measures,
    the same numbers as TwiBot-20's table, so the two datasets are comparable)
  - per-class mean node homophily against its chance level (FLAG's "camouflage" test)
  - test of the crawl-artefact pattern found on TwiBot-20: AUROC of log in-degree and of log
    out-degree for "is a bot", using no model at all.
Uses ALL labels: a description of the data, never a model input.

One pass over edge.csv with pyarrow's streaming reader (~10-20 min, ~3-4 GB RAM). The user-user
edges are saved to data/twibot-22/relations_uu.npz for the later prep script.

Run:  python my_build/twibot22_relations.py
      RAW=<folder with edge.csv, label.csv, split.csv, sub-communities/> to override the input.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RAW = Path(os.environ.get("RAW", PROJECT / "data" / "twibot-22-raw"))
OUT_DIR = PROJECT / "data" / "twibot-22"
RES = ROOT / "results" / "twibot22"
sys.path.insert(0, str(ROOT))
from graph_stats import measures  # noqa: E402  (same edge homophily / h_adj / LI as TwiBot-20)

REPLY_NAMES = {"replied", "reply_to", "replied_to"}      # the readme and the paper disagree on the name
TWEET_TWEET = {"retweeted": "retweet", "quoted": "quote"} | {r: "reply" for r in REPLY_NAMES}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- users and labels
def load_users():
    lab = pacsv.read_csv(RAW / "label.csv").to_pandas()
    ids = pa.array(lab["id"].astype(str).values)
    y = (lab["label"].values == "bot").astype(np.int64)
    log(f"{len(y):,} labelled users, {y.sum():,} bots ({y.mean():.3f})")
    return ids, y


def tweet_int(col):
    # "t1234..." -> 1234... as int64 (tweet ids are Twitter snowflakes, < 2^63)
    return pc.cast(pc.utf8_slice_codeunits(col, 1), pa.int64()).to_numpy(zero_copy_only=False)


# ---------------------------------------------------------------- one streaming pass
def read_edges(user_ids):
    path = RAW / "edge.csv"
    reader = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=256 << 20),
                            convert_options=pacsv.ConvertOptions(
                                column_types={"source_id": pa.string(), "target_id": pa.string(),
                                              "relation": pa.string()}))
    lst = {k: [] for k in ["following", "followers", "post", "mentioned", "like",
                           "retweet", "quote", "reply", "membership"]}
    counts = {}
    rows = 0
    uidx = lambda col: pc.index_in(col, value_set=user_ids).to_numpy(zero_copy_only=False)
    for batch in reader:
        rel = batch.column("relation")
        src, dst = batch.column("source_id"), batch.column("target_id")
        rows += batch.num_rows
        vc = pc.value_counts(rel)
        for name, c in zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()):
            counts[name] = counts.get(name, 0) + c
        for name in counts:
            m = pc.equal(rel, name)
            if not pc.any(m).as_py():
                continue
            s, d = pc.filter(src, m), pc.filter(dst, m)
            if name in ("following", "followers"):
                lst[name].append(np.stack([uidx(s), uidx(d)]))          # -1 = not a labelled user
            elif name == "post":
                lst["post"].append((uidx(s), tweet_int(d)))
            elif name == "mentioned":
                lst["mentioned"].append((tweet_int(s), uidx(d)))
            elif name == "like":
                lst["like"].append((uidx(s), tweet_int(d)))
            elif name in TWEET_TWEET:
                lst[TWEET_TWEET[name]].append((tweet_int(s), tweet_int(d)))
            elif name == "membership":                                   # list -> user
                lst["membership"].append((pc.utf8_slice_codeunits(s, 1).to_numpy(zero_copy_only=False),
                                          uidx(d)))
        if rows % (20_000_000) < batch.num_rows:
            log(f"  {rows:,} rows read")
    log(f"edge.csv: {rows:,} rows; relation counts: {json.dumps(counts)}")
    return lst, counts


def cat_pairs(parts):
    if not parts:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    if isinstance(parts[0], np.ndarray):
        a = np.concatenate(parts, axis=1)
        return a[0], a[1]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def build_relations(lst, n):
    """Directed user->user edges (2, E) per relation, both ends labelled users, no self-loops."""
    post_u, post_t = cat_pairs(lst["post"])
    keep = post_u >= 0
    post_u, post_t = post_u[keep], post_t[keep]
    order = np.argsort(post_t, kind="stable")
    post_t, post_u = post_t[order], post_u[order]
    log(f"post: {len(post_t):,} tweets by labelled users")

    def author(t):
        i = np.searchsorted(post_t, t).clip(0, max(len(post_t) - 1, 0))
        found = (len(post_t) > 0) & (post_t[i] == t) if len(post_t) else np.zeros(len(t), bool)
        return np.where(found, post_u[i], -1)

    rel = {}
    for name in ("following", "followers"):
        rel[name] = cat_pairs(lst[name])
    t, v = cat_pairs(lst["mentioned"])
    rel["mention"] = (author(t), v)
    for name in ("retweet", "quote", "reply"):
        t, t2 = cat_pairs(lst[name])
        rel[name] = (author(t), author(t2))
    u, t = cat_pairs(lst["like"])
    rel["like"] = (u, author(t))

    # co-list: members of the same list, via the sparse product B^T B (list x user incidence)
    lid, u = cat_pairs(lst["membership"])
    ok = u >= 0
    if ok.any():
        _, lcode = np.unique(lid[ok], return_inverse=True)
        B = sp.csr_matrix((np.ones(ok.sum(), np.int8), (lcode, u[ok])), shape=(lcode.max() + 1, n))
        C = sp.triu((B.T @ B), k=1).tocoo()
        rel["co_list"] = (C.row.astype(np.int64), C.col.astype(np.int64))
        log(f"co_list: {B.shape[0]:,} lists with labelled members -> {C.nnz:,} member pairs")
    else:
        rel["co_list"] = (np.zeros(0, np.int64), np.zeros(0, np.int64))

    out = {}
    for name, (a, b) in rel.items():
        m = (a >= 0) & (b >= 0) & (a != b)
        e = np.unique(np.stack([a[m], b[m]]).astype(np.int64), axis=1)   # drop repeated (a, b)
        out[name] = e
        log(f"{name:10s} {m.mean() if len(m) else 0:.3f} of rows kept -> {e.shape[1]:,} distinct directed edges")
    a = np.concatenate([out["following"], out["followers"][::-1]], axis=1)  # followers u->v == v follows u
    out["follow_any"] = np.unique(a, axis=1)
    return out


# ---------------------------------------------------------------- measures
def undirected(e, n):
    if e.shape[1] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    A = sp.coo_matrix((np.ones(e.shape[1], np.int8), (e[0], e[1])), shape=(n, n)).tocsr()
    T = sp.triu(((A + A.T) > 0).astype(np.int8), k=1).tocoo()
    return T.row.astype(np.int64), T.col.astype(np.int64)


def class_homophily(u, v, y):
    n = len(y)
    deg = np.bincount(u, minlength=n) + np.bincount(v, minlength=n)
    same = np.bincount(u, weights=(y[u] == y[v]), minlength=n) + np.bincount(v, weights=(y[u] == y[v]), minlength=n)
    ends = np.concatenate([y[u], y[v]])
    out = {}
    for k, name in [(1, "bot"), (0, "human")]:
        m = (y == k) & (deg > 0)
        out[name] = dict(mean_node_homophily=round(float((same[m] / deg[m]).mean()), 3) if m.any() else None,
                         chance=round(float((ends == k).mean()), 3),
                         share_with_edge=round(float(((deg > 0) & (y == k)).sum() / max((y == k).sum(), 1)), 3))
    return out


def degree_auroc(e, y):
    # The TwiBot-20 crawl-artefact test: can in- or out-degree alone rank bots above humans?
    n = len(y)
    res = {}
    for name, idx in [("in_degree", e[1]), ("out_degree", e[0])]:
        d = np.log1p(np.bincount(idx, minlength=n))
        res[name] = round(float(roc_auc_score(y, d)), 3) if d.std() > 0 else None
    return res


def describe(e, y, directed=True):
    u, v = undirected(e, len(y))
    if len(u) == 0:
        return {"edges": 0}
    out = measures(u, v, y)
    out["per_class"] = class_homophily(u, v, y)
    if directed:
        out["auroc_from_degree"] = degree_auroc(e, y)
    return out


def subgraph(e, keep):
    new = np.full(len(keep), -1)
    new[keep] = np.arange(keep.sum())
    a, b = new[e[0]], new[e[1]]
    m = (a >= 0) & (b >= 0)
    return np.stack([a[m], b[m]])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    user_ids, y = load_users()
    n = len(y)
    cache = OUT_DIR / "relations_uu.npz"
    if cache.exists() and not os.environ.get("FORCE"):
        log(f"using cached {cache}")
        rels = dict(np.load(cache))
        counts = json.loads((OUT_DIR / "edge_relation_counts.json").read_text())
    else:
        lst, counts = read_edges(user_ids)
        rels = build_relations(lst, n)
        del lst
        np.savez_compressed(cache, **rels)
        (OUT_DIR / "edge_relation_counts.json").write_text(json.dumps(counts, indent=1))
        log(f"saved {cache}")

    results = {"edge_csv_relation_counts": counts, "all_users": {}, "sub_communities": {}}
    log("all 1M users:")
    for name, e in rels.items():
        r = describe(e, y, directed=name != "co_list")
        results["all_users"][name] = r
        pc_ = r.get("per_class", {})
        log(f"  {name:10s} edges {r['edges']:>11,}  h {r.get('edge_homophily')}  chance {r.get('chance')}  "
            f"h_adj {r.get('adjusted_homophily')}  LI {r.get('label_informativeness')}  "
            f"with-edge bot/human {pc_.get('bot', {}).get('share_with_edge')}/{pc_.get('human', {}).get('share_with_edge')}  "
            f"AUROC in/out {r.get('auroc_from_degree')}")

    import pandas as pd
    id_list = user_ids.to_pylist()
    pos = {u: i for i, u in enumerate(id_list)}
    for k in range(10):
        f = RAW / "sub-communities" / f"user{k}.json"
        if not f.exists():
            continue
        keep = np.zeros(n, bool)
        keep[[pos[u] for u in json.loads(f.read_text()) if u in pos]] = True
        ys = y[keep]
        results["sub_communities"][k] = {name: describe(subgraph(e, keep), ys, directed=name != "co_list")
                                         for name, e in rels.items()}
    if results["sub_communities"]:
        log("sub-communities (10k users, 50% bots): mean over the 10 of h_adj / LI / edges")
        for name in rels:
            rows = [results["sub_communities"][k][name] for k in results["sub_communities"]]
            g = lambda key: np.mean([r[key] for r in rows if r.get(key) is not None]) if any(r.get(key) is not None for r in rows) else float("nan")
            log(f"  {name:10s} edges {g('edges'):>10,.0f}  h_adj {g('adjusted_homophily'):+.3f}  LI {g('label_informativeness'):.4f}")
    out = RES / "relations_heterophily.json"
    out.write_text(json.dumps(results, indent=1, default=float))
    log(f"saved {out}")


if __name__ == "__main__":
    main()
