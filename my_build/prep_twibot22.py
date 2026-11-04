"""
TwiBot-22 -> the same compact format as TwiBot-20 (data/twibot-22/), so steps 2-6 run unchanged.

Inputs (all made earlier, see the Data preparation section of REPORT.md):
  data/twibot-22-raw/label.csv, sub-communities/user<k>.json
  data/twibot-22/users.parquet           twibot22_profile_check.py (flattened user.json)
  data/twibot-22/relations_uu.npz        twibot22_relations.py (user-user edges per relation,
                                         indices = row order of label.csv)
  data/twibot-22/tweets_subset.parquet   twibot22_tweets.py (optional: without it, text = bio only)

Choices, each set by an environment variable:
  SUBSET=sub<k>        which users: one of the paper's 10 sub-communities (10,000 users, 5,000 bots)
  RELATIONS=a,b,...    which user-user relations become edges (see twibot22_relations.py)
  N_TWEETS=20          tweets per user in the text, most recent first (TwiBot-20 used 20)

Writes data/twibot-22/:
  x_text.npy      bio + N_TWEETS tweets, one per line, cut at MAX_CHARS   <- model input with edges
  x_bio.npy       bio alone
  y.npy           1 = bot, 0 = human
  edge_index.npy  (E, 2) union of the chosen relations, undirected (both directions), no self-loops
  edge_typed.npz  one (E_r, 2) array per relation, directed as stored (for step 6 metapaths)
  split.npy       'train' / 'val' / 'test', stratified 70/20/10, seed 42 (the paper's 7:2:1 ratio;
                  the official split.csv is not random: train 7.8% bots vs test 29.4%)
  ids.npy         original user ids
  metadata.csv    profile fields -- NOT a model input, only for shortcut checks ("verified" is one)
  prep_info.json  the choices above and the counts
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "twibot-22-raw"
OUT = ROOT / "data" / "twibot-22"
SUBSET = os.environ.get("SUBSET", "sub6")
RELATIONS = os.environ.get("RELATIONS", "following,followers,mention,retweet,reply").split(",")
N_TWEETS = int(os.environ.get("N_TWEETS", 20))
MAX_CHARS = 2000
SEED = 42


def clean(s):
    return " ".join((s or "").split())                  # trim, and line breaks inside a text -> spaces


def main():
    lab = pd.read_csv(RAW / "label.csv")                 # row order = index used in relations_uu.npz
    pos = pd.Series(np.arange(len(lab)), index=lab.id)
    assert SUBSET.startswith("sub"), "SUBSET must be sub0 ... sub9"
    ids = json.loads((RAW / "sub-communities" / f"user{SUBSET[3:]}.json").read_text())
    ids = sorted(set(ids) & set(lab.id))
    full_idx = pos[ids].to_numpy()
    y = (lab.label.to_numpy()[full_idx] == "bot").astype(np.int64)
    print(f"{SUBSET}: {len(ids):,} users, {y.sum():,} bots")

    users = pd.read_parquet(OUT / "users.parquet").set_index("id").loc[ids]
    bio = users.description.map(clean).to_numpy()

    tweets = {u: [] for u in ids}
    tw_file = OUT / "tweets_subset.parquet"
    if tw_file.exists():
        tw = pd.read_parquet(tw_file)
        tw = tw[tw.user_id.isin(set(ids))].sort_values("created_at", ascending=False)
        for u, g in tw.groupby("user_id", sort=False):
            tweets[u] = [clean(t) for t in g.text.head(N_TWEETS) if clean(t)]
    else:
        print("no tweets_subset.parquet: text = bio only")
    x_text = np.array([("\n".join([b] + tweets[u])).strip()[:MAX_CHARS] for u, b in zip(ids, bio)], dtype=object)

    # Edges: induced subgraph of each chosen relation on the subset.
    rels = np.load(OUT / "relations_uu.npz")
    new = np.full(len(lab), -1)
    new[full_idx] = np.arange(len(ids))
    typed, und = {}, []
    for r in RELATIONS:
        e = rels[r]
        a, b = new[e[0]], new[e[1]]
        m = (a >= 0) & (b >= 0) & (a != b)
        typed[r] = np.unique(np.stack([a[m], b[m]], 1), axis=0) if m.any() else np.zeros((0, 2), np.int64)
        und.append(typed[r])
        print(f"  {r:10s} {len(typed[r]):,} directed edges inside the subset")
    und = np.concatenate(und) if und else np.zeros((0, 2), np.int64)
    und = np.unique(np.concatenate([und, und[:, ::-1]]), axis=0)

    idx = np.arange(len(ids))
    train, rest = train_test_split(idx, train_size=0.7, stratify=y, random_state=SEED)
    val, test = train_test_split(rest, train_size=2 / 3, stratify=y[rest], random_state=SEED)
    split = np.empty(len(ids), dtype=object)
    split[train], split[val], split[test] = "train", "val", "test"

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "x_text.npy", x_text.astype(str))
    np.save(OUT / "x_bio.npy", bio.astype(str))
    np.save(OUT / "y.npy", y)
    np.save(OUT / "edge_index.npy", und.astype(np.int64))
    np.savez(OUT / "edge_typed.npz", **{r: v.astype(np.int64) for r, v in typed.items()})
    np.save(OUT / "split.npy", split.astype(str))
    np.save(OUT / "ids.npy", np.array(ids).astype(str))
    import twibot22_guard
    meta = users[["protected", "pinned", "default_image", "url", "location", "created_at",
                  "followers", "following", "tweets", "listed"]].reset_index()
    meta.insert(1, "verified", twibot22_guard.verified(meta.id))       # eval_only/: checks only
    meta.insert(1, "label", np.where(y == 1, "bot", "human"))
    meta.to_csv(OUT / "metadata.csv", index=False)
    deg = np.bincount(und[:, 0], minlength=len(ids)) if len(und) else np.zeros(len(ids), int)
    info = dict(subset=SUBSET, relations=RELATIONS, n_tweets=N_TWEETS, users=len(ids), bots=int(y.sum()),
                undirected_edges=int(len(und) // 2), isolated=int((deg == 0).sum()),
                users_with_tweets=int(sum(len(v) > 0 for v in tweets.values())),
                bio_empty=int((bio == "").sum()), text_at_cap=int(sum(len(t) >= MAX_CHARS for t in x_text)),
                split={s: int((split == s).sum()) for s in ("train", "val", "test")})
    (OUT / "prep_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
