"""
TwiBot-22: the 20 most recent tweets of EVERY labelled user, from the nine tweet_*.json files
(~104 GB). Replaces twibot22_tweets.py for the plan fixed 4 Nov 20:58 (checklist C1-C3, C8, H4, H5).
CPU only, no torch.

Why two passes: "most recent 20" must be decided across all 9 files (a user's tweets are spread over
several files), and decoding all ~88M tweets as JSON would take hours. So:

  PASS 1 (index)  For every tweet, read only author_id and created_at with byte-level search, and
                  remember where the tweet sits in its file (byte offset and length). Saved per file
                  to data/twibot-22/tweet_index/tweet_<i>.npz, so the pass can resume.
  SELECT          Over all files: keep tweets of labelled users only; per user, the 20 with the latest
                  created_at (ties broken by file and position, so the choice is deterministic).
  PASS 2 (decode) JSON-decode only the selected tweets, file by file, reading each at its offset.

File facts (checked on the files): each file is one JSON array of tweet objects with keys in alphabetical
order, so every object starts with {"attachments": , then "author_id": <int>, and "created_at": "YYYY-MM-DD
hh:mm:ss+00:00" is the 5th key. The marker {"attachments": cannot occur inside tweet text (quotes in
strings are escaped), so splitting on it is safe.

Checks written to the summary (data/twibot-22/tweets_top20_summary.json):
  per file: complete (ends with "]"), objects found, author misses, date misses, timezone suffix other
            than +00:00, tweets of unlabelled authors, decode errors in pass 2;
  overall:  total tweets vs the 88,217,457 `post` edges in edge.csv; date range; tweets after the
            collection end (15 Mar 2022); users with 0 tweets, by class.

Outputs (git-ignored, CC BY-NC-ND data):
  data/twibot-22/tweet_index/tweet_<i>.npz          pass-1 index per file
  data/twibot-22/tweet_count_files.npy              tweets per user in label.csv order (all, not just 20)
  data/twibot-22/tweets_top20/part_<i>.parquet      selected tweets, columns:
        user_id, tweet_id, created_at (UTC), lang, source, is_retweet, is_reply, text
  data/twibot-22/tweets_top20_summary.json
Run:  python my_build/twibot22_tweets_all.py          (PASS=1|2|all, default all; FILES=0,1 to limit)
"""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT.parent / "data" / "twibot-22-raw"
OUT = ROOT.parent / "data" / "twibot-22"
IDX = OUT / "tweet_index"
PARTS = OUT / "tweets_top20"
N_KEEP = 20
CHUNK = 256 << 20
MARK = b'{"attachments": '
AUTHOR = re.compile(rb'"author_id": (\d+),')
DATE_KEY = b'"created_at": "'
COLLECTION_END = np.datetime64("2022-03-15T23:59:59", "s")
POST_EDGES = 88_217_457          # `post` rows in edge.csv (edge_relation_counts.json)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def files():
    sel = os.environ.get("FILES")
    paths = sorted(RAW.glob("tweets/tweet_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    return [p for p in paths if sel is None or p.stem.split("_")[1] in sel.split(",")]


def complete(path):
    # H4: a file still being written must not be indexed. The array must end with "]".
    with open(path, "rb") as f:
        f.seek(-64, 2)
        return f.read().rstrip().endswith(b"]")


def index_file(path):
    # PASS 1 for one file: author, created_at (seconds, UTC), byte offset and length of every object.
    out = IDX / f"{path.stem}.npz"
    if out.exists():
        return
    assert complete(path), f"{path.name} does not end with ']': incomplete, not indexed"
    authors, dates, offs, lens = [], [], [], []
    miss_author = miss_date = 0
    tz_other = 0
    t0 = time.time()
    with open(path, "rb") as f:
        base, buf = 0, b""                    # base = file offset of buf[0]
        while True:
            chunk = f.read(CHUNK)
            buf += chunk
            starts = [m.start() for m in re.finditer(re.escape(MARK), buf)]
            # all objects except the last are complete; at end of file the last one is complete too
            last = len(starts) if not chunk else len(starts) - 1
            for k in range(last):
                s = starts[k]
                e = starts[k + 1] if k + 1 < len(starts) else len(buf)
                m = AUTHOR.match(buf, buf.find(b'"author_id"', s, e))
                d = buf.find(DATE_KEY, s, e)
                if not m:
                    miss_author += 1
                    continue
                if d < 0:
                    miss_date += 1
                    continue
                d += len(DATE_KEY)
                if buf[d + 19:d + 25] != b"+00:00":
                    tz_other += 1
                authors.append(int(m.group(1)))
                dates.append(buf[d:d + 19])
                offs.append(base + s)
                lens.append(e - s)
            if not chunk:
                break
            cut = starts[-1] if starts else len(buf)
            base += cut
            buf = buf[cut:]
    dt = np.array(dates, dtype="S19")
    ts = np.char.replace(dt, b" ", b"T").astype("M8[s]").astype(np.int64)
    np.savez(out, author=np.array(authors, np.int64), ts=ts, offset=np.array(offs, np.int64),
             length=np.array(lens, np.int32),
             stats=np.array([len(authors), miss_author, miss_date, tz_other], np.int64))
    log(f"  {path.name}: {len(authors):,} tweets indexed in {time.time() - t0:.0f} s "
        f"(author misses {miss_author}, date misses {miss_date}, timezone not +00:00: {tz_other})")


def select(lab_int):
    # SELECT: per labelled user, the N_KEEP most recent tweets across all files.
    cols = {k: [] for k in ("uidx", "ts", "file", "offset", "length")}
    per_file = {}
    for p in files():
        z = np.load(IDX / f"{p.stem}.npz")
        a = z["author"]
        pos = np.searchsorted(lab_int, a)
        pos[pos >= len(lab_int)] = 0
        ok = lab_int[pos] == a
        st = z["stats"]
        per_file[p.name] = dict(tweets=int(st[0]), author_misses=int(st[1]), date_misses=int(st[2]),
                                timezone_not_utc=int(st[3]), unlabelled_author=int((~ok).sum()))
        cols["uidx"].append(pos[ok]); cols["ts"].append(z["ts"][ok])
        cols["file"].append(np.full(ok.sum(), int(p.stem.split("_")[1]), np.int8))
        cols["offset"].append(z["offset"][ok]); cols["length"].append(z["length"][ok])
    c = {k: np.concatenate(v) for k, v in cols.items()}
    count = np.bincount(c["uidx"], minlength=len(lab_int))
    # sort by user, then newest first, then file and offset (deterministic ties)
    order = np.lexsort((c["offset"], c["file"], -c["ts"], c["uidx"]))
    u = c["uidx"][order]
    first = np.r_[0, np.flatnonzero(np.diff(u)) + 1]
    rank = np.arange(len(u)) - np.repeat(first, np.diff(np.r_[first, len(u)]))
    keep = order[rank < N_KEEP]
    sel = {k: v[keep] for k, v in c.items()}
    ts_all = c["ts"]
    dates = dict(min=str(np.datetime64(int(ts_all.min()), "s")), max=str(np.datetime64(int(ts_all.max()), "s")),
                 after_collection_end=int((ts_all > COLLECTION_END.astype(np.int64)).sum()))
    return sel, count, per_file, dates


def decode_file(fid, offsets, lengths, uidx, lab_ids):
    # PASS 2 for one file: decode the selected objects.
    path = RAW / "tweets" / f"tweet_{fid}.json"
    order = np.argsort(offsets)
    rows, errors = [], 0
    with open(path, "rb") as f:
        for i in order:
            f.seek(int(offsets[i]))
            raw = f.read(int(lengths[i])).rstrip().rstrip(b"]").rstrip().rstrip(b",")
            try:
                t = json.loads(raw)
            except json.JSONDecodeError:
                errors += 1
                continue
            text = t.get("text") or ""
            ref = [r.get("type") for r in (t.get("referenced_tweets") or [])]
            rows.append(dict(user_id=lab_ids[uidx[i]], tweet_id=t.get("id"), created_at=t.get("created_at"),
                             lang=t.get("lang"), source=re.sub(r"<[^>]+>", "", t.get("source") or ""),
                             is_retweet=text.startswith("RT @") or "retweeted" in ref,
                             is_reply=t.get("in_reply_to_user_id") is not None or "replied_to" in ref,
                             text=text))
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df.created_at, utc=True)       # H5: parsed, not compared as strings
    df.to_parquet(PARTS / f"part_{fid}.parquet")
    return len(rows), errors


def main():
    IDX.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(RAW / "label.csv")
    lab_ids = lab.id.to_numpy()
    lab_num = np.array([int(u[1:]) for u in lab_ids], np.int64)       # "u123" -> 123 = author_id
    srt = np.argsort(lab_num)
    lab_sorted = lab_num[srt]
    which = os.environ.get("PASS", "all")
    if which in ("1", "all"):
        log("PASS 1: index author and date of every tweet")
        for p in files():
            index_file(p)
    if which in ("2", "all"):
        assert all((IDX / f"{p.stem}.npz").exists() for p in files()), "pass 1 not finished"
        log("SELECT: 20 most recent per labelled user")
        sel, count_sorted, per_file, dates = select(lab_sorted)
        count = np.empty_like(count_sorted)
        count[srt] = count_sorted                                     # back to label.csv order
        np.save(OUT / "tweet_count_files.npy", count)
        uidx_label = srt[sel["uidx"]]                                 # sorted position -> label.csv row
        log(f"  {len(sel['uidx']):,} tweets selected for {np.unique(uidx_label).size:,} users")
        log("PASS 2: decode the selected tweets")
        for fid in np.unique(sel["file"]):
            m = sel["file"] == fid
            n, err = decode_file(int(fid), sel["offset"][m], sel["length"][m], uidx_label[m], lab_ids)
            per_file[f"tweet_{fid}.json"]["decoded"] = n
            per_file[f"tweet_{fid}.json"]["decode_errors"] = err
            log(f"  tweet_{fid}.json: {n:,} decoded, {err} decode errors")
        y = (lab.label.values == "bot")
        total = sum(v["tweets"] for v in per_file.values())
        summary = dict(
            files=per_file, total_tweets=int(total), post_edges_in_edge_csv=POST_EDGES,
            labelled_author_tweets=int(count.sum()), dates=dates, n_keep=N_KEEP,
            users_with_tweets=int((count > 0).sum()),
            zero_tweet_share=dict(bot=round(float((count[y] == 0).mean()), 4), human=round(float((count[~y] == 0).mean()), 4)),
            median_tweets_per_user=dict(all=float(np.median(count)), with_tweets=float(np.median(count[count > 0]))))
        (OUT / "tweets_top20_summary.json").write_text(json.dumps(summary, indent=1))
        log(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=1))


if __name__ == "__main__":
    main()
