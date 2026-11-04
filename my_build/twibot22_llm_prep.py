"""
TwiBot-22 -> a small dataset folder in TwiBot-20's format, so steps 5-6 (zero-shot LLM, LoRA, DGP)
run on it with DATASET=twibot22 (checklist H7). CPU, ~1 min. Decided 5 Nov ~02:15.

Nodes: the 110,057 users of the study sample (100,986) and the hub users (10,039), in label.csv row
order (the same order as data/twibot-22/sample110k_e5_bio20tw_raw/ids.npy).
  x_text.npy   bio, then the 20 most recent tweets, one per line (text rules C5: retweets kept,
               links and @mentions raw; step 6 keeps as many whole lines as fit in 480 tokens)
  x_bio.npy    bio only (for the "no bio" / "no tweets" flags)
  y.npy        1 = bot
  split.npy    'train' / 'val' / 'test' for SAMPLE users in train / val / test_work;
               'sealed' for sample users in test_sealed (never in any split: get_splits only reads
               train / val / test); 'none' for hub users outside the sample (neighbours only)
  ids.npy      user ids;  rows.npy  label.csv row numbers
  edge_typed.npz  the two DGP relation groups (directed pairs between these nodes, step 6 makes
               them undirected): follow = following + followers; interaction = mention + retweet
               + reply + quote + like
  edge_index.npy  both groups merged, undirected, (E, 2)
Also copies the multilingual-E5 bio + tweet embedding to results/twibot22/emb_e5.npy (step 6's
diffusion features X, as on TwiBot-20).

Run: python my_build/twibot22_llm_prep.py      (refuses to overwrite)
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "twibot-22"
RAW = ROOT.parent / "data" / "twibot-22-raw"
OUT = ROOT.parent / "data" / "twibot-22-llm"
GROUPS = {"follow": ["following", "followers"],
          "interaction": ["mention", "retweet", "reply", "quote", "like"]}


def clean(t):
    return " ".join(str(t).split())


def main():
    assert not OUT.exists(), f"{OUT} exists: never overwritten"
    lab = pd.read_csv(RAW / "label.csv")
    rows = np.load(DATA / "sample110k_e5_bio20tw_raw" / "ids.npy")
    assert (rows == np.union1d(np.load(DATA / "sample_idx.npy"), np.load(DATA / "hub_idx.npy"))).all()
    ids = lab.id.to_numpy()[rows]
    y = (lab.label.values[rows] == "bot").astype(np.int64)

    roles = np.load(DATA / "roles.npy")[rows]
    in_sample = np.isin(rows, np.load(DATA / "sample_idx.npy"))
    split = np.full(len(rows), "none", dtype=object)
    for role, part in (("train", "train"), ("val", "val"), ("test_work", "test"), ("test_sealed", "sealed")):
        split[in_sample & (roles == role)] = part
    split = split.astype(str)

    bios = pd.read_parquet(DATA / "users.parquet", columns=["id", "description"]).set_index("id").description
    bio = bios.reindex(ids).fillna("").map(clean).to_numpy()
    tw = pd.concat([pd.read_parquet(f, columns=["user_id", "text"], filters=[("user_id", "in", list(ids))])
                    for f in sorted(glob.glob(str(DATA / "tweets_top20" / "part_*.parquet")))])
    tw["text"] = tw.text.map(clean)
    joined = tw[tw.text != ""].groupby("user_id").text.agg("\n".join).reindex(ids).fillna("").to_numpy()
    x_text = np.array([(b + "\n" + t).strip("\n") for b, t in zip(bio, joined)], dtype=object)

    pos = np.full(len(lab), -1)
    pos[rows] = np.arange(len(rows))
    rels = np.load(DATA / "relations_uu.npz")
    typed, merged = {}, []
    for g, members in GROUPS.items():
        e = np.hstack([rels[r] for r in members])
        e = pos[e]
        e = e[:, (e >= 0).all(0) & (e[0] != e[1])]
        e = np.unique(e.T, axis=0)
        typed[g] = e.astype(np.int64)
        merged.append(e)
    und = np.unique(np.vstack([np.vstack(merged), np.vstack(merged)[:, ::-1]]), axis=0)

    OUT.mkdir(parents=True)
    np.save(OUT / "x_text.npy", x_text.astype(str))
    np.save(OUT / "x_bio.npy", bio.astype(str))
    np.save(OUT / "y.npy", y)
    np.save(OUT / "split.npy", split)
    np.save(OUT / "ids.npy", ids.astype(str))
    np.save(OUT / "rows.npy", rows)
    np.savez(OUT / "edge_typed.npz", **typed)
    np.save(OUT / "edge_index.npy", und.astype(np.int64))
    emb_out = ROOT / "results" / "twibot22" / "emb_e5.npy"
    assert not emb_out.exists(), f"{emb_out} exists"
    np.save(emb_out, np.load(DATA / "sample110k_e5_bio20tw_raw" / "emb.npy").astype(np.float32))

    print(f"{len(rows):,} nodes; split counts: {pd.Series(split).value_counts().to_dict()}")
    print(f"bots: {y.mean():.3f} overall; " + ", ".join(f"{p} {y[split == p].mean():.3f}" for p in ("train", "val", "test")))
    print("edges (directed pairs): " + ", ".join(f"{g} {len(e):,}" for g, e in typed.items()) + f"; undirected merged {len(und) // 2:,}")
    print(f"users with no text: {(x_text == '').sum():,}")


if __name__ == "__main__":
    main()
