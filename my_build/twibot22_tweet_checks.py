"""
TwiBot-22 tweet checks before embedding (checklist C6, C7, D4, plus the posting app). CPU only, no torch.
Labels are read for TRAIN and VAL users only; no test user (working or sealed) is touched.

1. C7 COPY-PASTE ACROSS SPLITS. Exact duplicate tweets (same text after trimming, original tweets of
   >= 30 characters, retweets excluded because sharing the same retweet is normal) between a VAL user and
   any TRAIN user. For val users with such a copy: share by class, and how often the train users who wrote
   the same text have the same label (vs the 14% / 86% chance). Bot farms posting identical text would
   show up as val bots whose copies are written by train bots.
2. D4 WORD SHORTCUTS (covers the labelling recipe's spam keywords, whose list is not published).
   Per user: the set of lowercase words in bio + 20 tweets. Same rule as every other shortcut check:
   a word is a SHORTCUT if >= 200 training users use it and >= 99% of them are one class.
3. POSTING APP (tweet "source", e.g. "Twitter for iPhone"): same rule per app, on training users.
4. C6 LANGUAGE MIX: each user's most common tweet language; top languages by class (descriptive).

Output: results/twibot22/tweet_checks.json
Run:    python my_build/twibot22_tweet_checks.py      (~15-25 min, ~8 GB RAM)
"""
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "twibot-22"
RAW = ROOT.parent / "data" / "twibot-22-raw"
RES = ROOT / "results" / "twibot22"
MIN_USERS, PURITY = 200, 0.99


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pure(counts_bot, counts_all, names):
    # entries used by >= MIN_USERS training users that are >= PURITY one class
    out = []
    for name, b, n in zip(names, counts_bot, counts_all):
        if n >= MIN_USERS:
            share = b / n
            if share >= PURITY or share <= 1 - PURITY:
                out.append(dict(item=str(name), train_users=int(n), bot_share=round(float(share), 4)))
    return sorted(out, key=lambda r: -r["train_users"])


def main():
    lab = pd.read_csv(RAW / "label.csv")
    split = np.load(DATA / "split_ours.npy")
    keep = split != "test"                                            # train + val only
    role = pd.Series(split[keep], index=lab.id[keep])
    bot = pd.Series((lab.label.values == "bot")[keep], index=lab.id[keep])
    p_bot = float(bot[role == "train"].mean())
    log("loading tweets (train + val users)")
    df = pd.concat([pd.read_parquet(f, columns=["user_id", "text", "lang", "source", "is_retweet"])
                    for f in sorted(glob.glob(str(DATA / "tweets_top20" / "part_*.parquet")))])
    df = df[df.user_id.isin(role.index)]
    df["role"] = df.user_id.map(role)
    df["bot"] = df.user_id.map(bot)
    out = dict(train_bot_share=round(p_bot, 4), tweets_checked=int(len(df)))

    # 1. copy-paste across splits
    log("C7: duplicate tweets between val and train users")
    o = df[~df.is_retweet].copy()
    o["t"] = o.text.str.strip()
    o = o[o.t.str.len() >= 30]
    o["h"] = pd.util.hash_pandas_object(o.t, index=False).values
    tr = o[o.role == "train"].groupby("h").bot.agg(["mean", "size"])
    va = o[o.role == "val"][["user_id", "h", "bot"]].drop_duplicates()
    va = va[va.h.isin(tr.index)]
    va["train_bot_share"] = va.h.map(tr["mean"])
    per_user = va.groupby("user_id").agg(bot=("bot", "first"), copies=("h", "size"), train_bot_share=("train_bot_share", "mean"))
    n_val = bot[role == "val"]
    c7 = {}
    for cls, name in ((True, "bot"), (False, "human")):
        u = per_user[per_user.bot == cls]
        same = u.train_bot_share if cls else 1 - u.train_bot_share
        c7[f"val_{name}"] = dict(
            users_with_a_copy_in_train=int(len(u)),
            share_of_val_class=round(len(u) / int((n_val == cls).sum()), 4),
            mean_same_label_share_of_copy_writers=round(float(same.mean()), 4) if len(u) else None,
            chance=round(p_bot if cls else 1 - p_bot, 4))
    out["c7_duplicates_val_vs_train"] = c7
    log(f"  {c7}")

    # 2. word shortcuts on training users
    log("D4: word shortcuts (training users)")
    t = df[df.role == "train"]
    docs = t.groupby("user_id").text.apply(lambda s: " ".join(s))
    bios = pd.read_parquet(DATA / "users.parquet", columns=["id", "description"]).set_index("id").description
    train_ids = role.index[role == "train"]
    docs = docs.reindex(train_ids).fillna("") + " " + bios.reindex(train_ids).fillna("")
    vec = CountVectorizer(binary=True, lowercase=True, min_df=MIN_USERS, token_pattern=r"(?u)\b\w\w+\b|[#@]\w+")
    X = vec.fit_transform(docs.values)
    yb = bot.reindex(train_ids).to_numpy().astype(np.float32)
    n_all = np.asarray(X.sum(0)).ravel()
    n_bot = X.T @ yb
    out["d4_word_shortcuts"] = dict(vocabulary_checked=int(len(n_all)), shortcuts=pure(n_bot, n_all, vec.get_feature_names_out())[:50])
    log(f"  {len(n_all):,} words used by >= {MIN_USERS} train users; shortcuts: {len(out['d4_word_shortcuts']['shortcuts'])}")

    # 3. posting app
    log("posting app (training users)")
    s = t[["user_id", "source", "bot"]].drop_duplicates(["user_id", "source"])
    g = s.groupby("source").bot.agg(["sum", "size"])
    out["source_shortcuts"] = pure(g["sum"].values, g["size"].values, g.index)
    top = g.sort_values("size", ascending=False).head(12)
    out["source_top"] = [dict(source=k, train_users=int(r["size"]), bot_share=round(float(r["sum"] / r["size"]), 4)) for k, r in top.iterrows()]
    log(f"  app shortcuts: {len(out['source_shortcuts'])}")

    # 4. language mix (descriptive)
    log("C6: language mix")
    main_lang = t.groupby("user_id").lang.agg(lambda s: s.mode().iat[0] if len(s) else None)
    ul = pd.DataFrame(dict(lang=main_lang, bot=bot.reindex(main_lang.index)))
    mix = {}
    for cls, name in ((True, "bot"), (False, "human")):
        mix[name] = ul[ul.bot == cls].lang.value_counts(normalize=True).head(8).round(3).to_dict()
    out["c6_main_language_share"] = mix

    (RES / "tweet_checks.json").write_text(json.dumps(out, indent=1, default=str))
    log(json.dumps({k: v for k, v in out.items() if k != "d4_word_shortcuts"}, indent=1, default=str))
    log("word shortcuts (top 20):", out["d4_word_shortcuts"]["shortcuts"][:20])


if __name__ == "__main__":
    main()
