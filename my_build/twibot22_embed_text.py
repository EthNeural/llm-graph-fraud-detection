"""
Embed bio + 20 most recent tweets for the TwiBot-22 study sample and the hub users (checklist F4, F6).
Text rules fixed 4 Nov (checklist C5): retweets kept; links and @mentions raw; no total length cap.
Same method as TwiBot-20's step 3: one piece per line (the bio, then each tweet), each piece cut at
128 tokens, each piece embedded and L2-normalised, the user's vector = mean of their pieces.
Users with no text at all get a zero vector (the "no bio" / "no tweets" flags are added at load time).

Users: sample_idx.npy (100,986) union hub_idx.npy (10,039) = 110,057, sorted (label.csv row numbers).
  ENC=e5 (multilingual-E5, "query: " prefix; headline, F6) | minilm (check)
  TIMING=10000   timing test only: a random 10,000 of those users (seed 0); prints the measured speed
                 and the estimated total time; saves nothing.

Output: data/twibot-22/sample110k_<ENC>_bio20tw_raw/emb.npy  (float16, one row per user)
        .../ids.npy (label.csv row numbers of those rows) and .../user_ids.npy; row order asserted after saving (F2, F3)
Run:    TIMING=10000 ENC=e5 python my_build/twibot22_embed_text.py      (GPU, a few minutes)
        ENC=e5 python my_build/twibot22_embed_text.py                   (GPU, only after the builder's go)
"""
import glob
import os
import time
from pathlib import Path

os.environ.setdefault("DATASET", "twibot20")     # reuse step 3's encoder functions (see twibot22_embed_bios.py)
import numpy as np
import pandas as pd

import step3_embed as s3

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "twibot-22"
RES = ROOT / "results" / "twibot22"
ENC = os.environ.get("ENC", "e5")
TIMING = int(os.environ.get("TIMING", "0"))


def clean(t):
    return " ".join(str(t).split())              # one line per piece; no other change (C5)


def build_texts(rows, lab_ids):
    # bio + tweets of the given users, joined by "\n" (one piece per line), in `rows` order
    ids = lab_ids[rows]
    bios = pd.read_parquet(DATA / "users.parquet", columns=["id", "description"]).set_index("id").description
    bio = bios.reindex(ids).fillna("").map(clean)
    want = set(ids)
    tw = pd.concat([pd.read_parquet(f, columns=["user_id", "text"], filters=[("user_id", "in", list(want))])
                    for f in sorted(glob.glob(str(DATA / "tweets_top20" / "part_*.parquet")))])
    tw["text"] = tw.text.map(clean)
    joined = tw[tw.text != ""].groupby("user_id").text.agg("\n".join)
    tweets = joined.reindex(ids).fillna("")
    return [(b + "\n" + t).strip("\n") for b, t in zip(bio.values, tweets.values)]


def main():
    lab_ids = pd.read_csv(ROOT.parent / "data" / "twibot-22-raw" / "label.csv").id.to_numpy()
    rows = np.union1d(np.load(DATA / "sample_idx.npy"), np.load(DATA / "hub_idx.npy"))
    total_users = len(rows)
    if TIMING:
        rows = np.sort(np.random.default_rng(0).choice(rows, TIMING, replace=False))
    outdir = DATA / f"sample110k_{ENC}_bio20tw_raw"          # named by the text recipe (F3)
    if not TIMING:
        assert not outdir.exists(), f"{outdir.name} exists: never overwritten (H6)"
    t0 = time.time()
    texts = build_texts(rows, lab_ids)
    pieces, owner = s3.split_pieces(texts)
    print(f"{len(rows):,} users, {len(pieces):,} pieces ({len(pieces) / len(rows):.1f} per user); "
          f"text built in {time.time() - t0:.0f} s; encoder {ENC} = {s3.ENCODERS[ENC]}", flush=True)
    t1 = time.time()
    prefix = "query: " if ENC in ("e5", "e5en") else ""
    emb = s3.encode_sentence_model(s3.ENCODERS[ENC], pieces, prefix=prefix)
    secs = time.time() - t1
    user_emb = s3.average_by_user(emb, owner, len(rows)).astype(np.float16)
    if TIMING:
        est = secs / len(rows) * total_users
        print(f"TIMING: {len(pieces):,} pieces in {secs / 60:.1f} min ({len(pieces) / secs:.0f} pieces/s). "
              f"Estimated total for {total_users:,} users: {est / 3600:.1f} h (encoding only). Nothing saved.", flush=True)
        return
    outdir.mkdir(parents=True)
    np.save(outdir / "emb.npy", user_emb)
    np.save(outdir / "ids.npy", rows)
    np.save(outdir / "user_ids.npy", lab_ids[rows].astype(str))
    # F2: re-read and assert the row order: same length, same ids, users without text are zero rows
    e, i, u = np.load(outdir / "emb.npy"), np.load(outdir / "ids.npy"), np.load(outdir / "user_ids.npy")
    assert len(e) == len(i) == len(u) == total_users and (i == rows).all() and (u == lab_ids[rows]).all()
    no_text = np.array([t == "" for t in texts])
    assert ((~e.any(1)) == no_text).all(), "zero rows do not match users without text"
    print(f"saved {outdir.name}/emb.npy {e.shape}; row order asserted ({int(no_text.sum())} users without text = zero rows); "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
