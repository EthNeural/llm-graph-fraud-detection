"""
One-off preprocessing: TwiBot-20 (TwiBot-22 format zip) -> the same compact format as Instagram.

Run:  python my_build/prep_twibot20.py [path/to/TwiBot-20-Format22.zip]     (~20-30 min)

Reads straight from the zip (no 7.4 GB unzip) and writes data/twibot-20/:
  x_text.npy      bio + first N_TWEETS tweets per user        <- the ONLY model input besides edges
  x_bio.npy       bio alone
  y.npy           1 = bot, 0 = human
  edge_index.npy  (E, 2) follow/friend links between labelled users, both directions
  split.npy       official split: 'train' / 'val' / 'test'
  ids.npy         original user ids
  metadata.csv    verified, followers, ... -- NOT a model input, only for the shortcut check

Only labelled users are kept (the 217k unlabelled "support" users are dropped).
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ZIP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "TwiBot-20-Format22.zip"
OUT = Path(__file__).resolve().parents[1] / "data" / "twibot-20"
N_TWEETS = 20            # tweets kept per user (TwiBot-20 has up to ~200)
CHUNK = 1 << 20
MAX_CHARS = 2000         # models read <= 256 tokens (~1,000 chars); keeps x_text.npy small


def stream_nodes(zf):
    # node.json is one 5.8 GB JSON array; decode it object by object.
    dec = json.JSONDecoder()
    with zf.open("Twibot-20/node.json") as raw:
        f = io.TextIOWrapper(raw, encoding="utf-8")
        buf = f.read(CHUNK).lstrip()[1:]
        while True:
            buf = buf.lstrip().lstrip(",").lstrip()
            try:
                obj, end = dec.raw_decode(buf)
            except json.JSONDecodeError:
                more = f.read(CHUNK)
                if not more:
                    return
                buf += more
                continue
            buf = buf[end:]
            yield obj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(ZIP)
    lab = pd.read_csv(zf.open("Twibot-20/label.csv"))
    split = pd.read_csv(zf.open("Twibot-20/split.csv"))
    lab = lab.merge(split, on="id")
    ids = lab.id.to_numpy()
    index = {u: i for i, u in enumerate(ids)}
    print(f"labelled users: {len(ids):,}")

    # Edges: tweets each user posted (first N_TWEETS) + follow/friend links between labelled users.
    tweet_owner, n_kept, pairs = {}, {}, set()
    for ch in pd.read_csv(zf.open("Twibot-20/edge.csv"), chunksize=5_000_000):
        post = ch[(ch.relation == "post") & ch.source_id.isin(index)]
        for u, t in zip(post.source_id, post.target_id):
            if n_kept.get(u, 0) < N_TWEETS:
                tweet_owner[t] = u
                n_kept[u] = n_kept.get(u, 0) + 1
        soc = ch[(ch.relation != "post") & ch.source_id.isin(index) & ch.target_id.isin(index)]
        for a, b in zip(soc.source_id, soc.target_id):
            if a != b:
                pairs.add((index[a], index[b]))
                pairs.add((index[b], index[a]))          # undirected: store both directions
    print(f"tweets to collect: {len(tweet_owner):,}   edge rows: {len(pairs):,}")

    # One pass over node.json: profiles first, then tweets.
    meta, bio, tweets = {}, {}, {u: [] for u in ids}
    found = 0
    for obj in stream_nodes(zf):
        i = obj["id"]
        if i.startswith("u"):
            if i in index:
                pm = obj.get("public_metrics") or {}
                bio[i] = (obj.get("description") or "").strip()
                meta[i] = dict(verified=str(obj.get("verified")).strip(), created_at=str(obj.get("created_at")).strip(),
                               followers_count=pm.get("followers_count"), following_count=pm.get("following_count"),
                               tweet_count=pm.get("tweet_count"), listed_count=pm.get("listed_count"))
        elif i in tweet_owner:
            tweets[tweet_owner[i]].append((obj.get("text") or "").strip().replace("\n", " "))
            found += 1
            if found % 50_000 == 0:
                print(f"  tweets found {found:,} / {len(tweet_owner):,}", flush=True)
            if found == len(tweet_owner):
                break

    x_bio = np.array([bio.get(u, "") for u in ids], dtype=object)
    x_text = np.array([(bio.get(u, "") + "\n" + "\n".join(tweets[u])).strip()[:MAX_CHARS] for u in ids], dtype=object)
    np.save(OUT / "x_text.npy", x_text.astype(str))
    np.save(OUT / "x_bio.npy", x_bio.astype(str))
    np.save(OUT / "y.npy", (lab.label == "bot").astype(np.int64).to_numpy())
    np.save(OUT / "edge_index.npy", np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2))
    np.save(OUT / "split.npy", lab.split.to_numpy().astype(str))
    np.save(OUT / "ids.npy", ids.astype(str))
    pd.DataFrame([dict(id=u, label=l, **meta.get(u, {})) for u, l in zip(ids, lab.label)]).to_csv(OUT / "metadata.csv", index=False)
    print(f"saved to {OUT}: {len(ids):,} users, {len(pairs):,} edge rows, "
          f"{sum(len(v) > 0 for v in tweets.values()):,} users with tweets")


if __name__ == "__main__":
    main()
