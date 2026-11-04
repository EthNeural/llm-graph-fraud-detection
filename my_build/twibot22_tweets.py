"""
Pull the tweets of a chosen set of TwiBot-22 users out of the nine tweet_i.json files (~12 GB each).

Each file is one JSON array of tweet objects whose keys are in a fixed (alphabetical) order and
start with {"attachments": ... , "author_id": 16989141, ...}. Parsing 88M tweets fully would take
hours, so we split the stream on that object marker, read author_id with a regex, and json-decode
only the tweets of users we keep (~1-10% of them). About 2-4 min per file on the Mac.

Users kept: the union of the paper's 10 sub-communities (91,719 users) unless IDS=<json list>.
At most MAX_PER_USER tweets per user are kept (the first ones met). TwiBot-20 used 20 per user.

Output: data/twibot-22/tweets_subset.parquet with
        user_id, tweet_id, created_at, lang, source, is_retweet, is_reply, text
Files already processed are listed in tweets_done.json, so the script can be rerun as more
tweet files finish downloading.

Run:  python my_build/twibot22_tweets.py
"""
import json
import os
import re
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT.parent / "data" / "twibot-22-raw"
OUT = ROOT.parent / "data" / "twibot-22"
MAX_PER_USER = 50
CHUNK = 64 << 20
MARK = '{"attachments": '
AUTHOR = re.compile(r'"author_id": (\d+),')


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def wanted_ids():
    if os.environ.get("IDS"):
        ids = json.loads(Path(os.environ["IDS"]).read_text())
    else:
        ids = set()
        for k in range(10):
            ids |= set(json.loads((RAW / "sub-communities" / f"user{k}.json").read_text()))
    return {int(u[1:]) for u in ids}                 # "u123" -> 123, matching author_id


def objects(path):
    # Yield the raw text of each tweet object in a JSON array, without parsing it.
    with open(path, encoding="utf-8") as f:
        buf = ""
        while True:
            chunk = f.read(CHUNK)
            buf += chunk
            parts = buf.split(MARK)
            buf = parts.pop() if chunk else ""
            for p in parts:
                if p.strip() not in ("", "["):
                    yield MARK + p
            if not chunk:
                if buf.strip():
                    yield MARK + buf if not buf.startswith(MARK) else buf
                return


def decode(raw):
    raw = raw.rstrip().rstrip("]").rstrip().rstrip(",")
    return json.loads(raw)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    keep = wanted_ids()
    log(f"{len(keep):,} users wanted")
    done_file = OUT / "tweets_done.json"
    done = json.loads(done_file.read_text()) if done_file.exists() else []
    out_file = OUT / "tweets_subset.parquet"
    old = pd.read_parquet(out_file) if out_file.exists() else None
    count = old.user_id.value_counts().to_dict() if old is not None else {}
    rows = []
    for path in sorted((RAW / "tweets").glob("tweet_*.json")):
        if path.name in done:
            continue
        log(f"{path.name} ({path.stat().st_size / 1e9:.1f} GB)")
        seen = kept = 0
        for raw in objects(path):
            seen += 1
            m = AUTHOR.search(raw, 0, 200)
            if not m or int(m.group(1)) not in keep:
                continue
            uid = "u" + m.group(1)
            if count.get(uid, 0) >= MAX_PER_USER:
                continue
            try:
                t = decode(raw)
            except json.JSONDecodeError:
                continue
            ref = [r.get("type") for r in (t.get("referenced_tweets") or [])]
            rows.append(dict(user_id=uid, tweet_id=t["id"], created_at=t.get("created_at"),
                             lang=t.get("lang"), source=re.sub(r"<[^>]+>", "", t.get("source") or ""),
                             is_retweet=(t.get("text") or "").startswith("RT @") or "retweeted" in ref,
                             is_reply=t.get("in_reply_to_user_id") is not None or "replied_to" in ref,
                             text=t.get("text") or ""))
            count[uid] = count.get(uid, 0) + 1
            kept += 1
            if seen % 2_000_000 == 0:
                log(f"  {seen:,} tweets read, {kept:,} kept")
        log(f"  done: {seen:,} tweets read, {kept:,} kept")
        done.append(path.name)
        df = pd.DataFrame(rows)
        if old is not None:
            df = pd.concat([old, df], ignore_index=True)
        df.to_parquet(out_file)
        old, rows = df, []
        done_file.write_text(json.dumps(done))
        log(f"  saved {out_file}: {len(df):,} tweets from {df.user_id.nunique():,} users")


if __name__ == "__main__":
    main()
