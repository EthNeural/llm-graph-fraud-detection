"""
TwiBot-22 profile-side data checks (no edges, no tweets needed). The same checks as TwiBot-20's
Data preparation section, so the two datasets can be compared line by line.

1. Flatten user.json (780 MB, 1M users) once into data/twibot-22/users.parquet (~2 min).
2. Shortcut test for every profile field, with the rule fixed for TwiBot-20 (metadata_check.py):
   a field is a SHORTCUT if (a) it is a documented collection or labelling artefact, or
   (b) a depth-1 tree on it has a leaf with >= 10% of training users that is >= 99% one class.
   TwiBot-22's labels were made partly by labelling functions (paper §3.2): "verified" and spam
   keywords in tweets and bios. So "verified" is a labelling artefact by (a).
3. Bio text: empty, length, duplicates across users, non-Latin script share.
Computed on all 1M users (official split) and inside each of the paper's 10 sub-communities
(50% bots each; a stratified 70/20/10 train/val/test split of the sub-community, seed 42, the
same proportions as prep_twibot22.py).

Run:  python my_build/twibot22_profile_check.py      -> results/twibot22/profile_check.json
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
RAW = ROOT.parent / "data" / "twibot-22-raw"
OUT = ROOT.parent / "data" / "twibot-22"
RES = ROOT / "results" / "twibot22"
COLLECTED = pd.Timestamp("2022-03-15", tz="UTC")     # end of the paper's collection (App. A.2)
NON_LATIN = re.compile(r"[^\x00-ɏ\s\d\W]")        # letters outside Latin / Latin Extended


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def flatten():
    path = OUT / "users.parquet"
    if path.exists():
        return pd.read_parquet(path)
    log("reading user.json ...")
    users = json.load(open(RAW / "user.json"))
    rows = []
    for u in users:
        pm = u.get("public_metrics") or {}
        ent = u.get("entities") or {}
        rows.append(dict(
            id=u["id"], description=(u.get("description") or "").strip(), name=u.get("name") or "",
            username=u.get("username") or "", location=(u.get("location") or "").strip(),
            url=u.get("url") or "", created_at=u.get("created_at"),
            verified=bool(u.get("verified")), protected=bool(u.get("protected")),
            withheld=u.get("withheld") is not None, pinned=u.get("pinned_tweet_id") is not None,
            default_image="default_profile" in (u.get("profile_image_url") or ""),
            n_desc_urls=len(((ent.get("description") or {}).get("urls")) or []),
            followers=pm.get("followers_count"), following=pm.get("following_count"),
            tweets=pm.get("tweet_count"), listed=pm.get("listed_count")))
    del users
    df = pd.DataFrame(rows)
    df = df[df.id.isin(pd.read_csv(RAW / "label.csv").id)]
    OUT.mkdir(parents=True, exist_ok=True)
    # "verified" goes to eval_only/ and labels are never stored here (twibot22_guard.py)
    (OUT / "eval_only").mkdir(exist_ok=True)
    df[["id", "verified"]].to_parquet(OUT / "eval_only" / "verified.parquet")
    df = df.drop(columns=["verified"])
    df.to_parquet(path)
    log(f"saved {path} ({len(df):,} users)")
    return df


def with_labels(df):
    # Labels and the official split joined in memory for the checks; never written to users.parquet.
    lab = pd.read_csv(RAW / "label.csv").merge(pd.read_csv(RAW / "split.csv"), on="id")
    df = df.merge(lab, on="id", how="inner")
    df["y"] = (df.label == "bot").astype(int)
    return df


def features(df):
    # Profile features WITHOUT "verified" (it lives in eval_only/, see twibot22_guard.py).
    # "withheld" is legitimate but near-empty (59 of 1M users); model scripts drop it.
    created = pd.to_datetime(df.created_at, utc=True, errors="coerce")
    return pd.DataFrame({
        "protected": df.protected.astype(int),
        "withheld": df.withheld.astype(int),
        "has pinned tweet": df.pinned.astype(int),
        "default profile image": df.default_image.astype(int),
        "has url": (df.url != "").astype(int),
        "has location": (df.location != "").astype(int),
        "bio empty": (df.description == "").astype(int),
        "bio length": df.description.str.len(),
        "urls in bio": df.n_desc_urls,
        "name length": df.name.str.len(),
        "username digits": df.username.str.count(r"\d"),
        "log followers": np.log1p(df.followers.fillna(0)),
        "log following": np.log1p(df.following.fillna(0)),
        "log tweet count": np.log1p(df.tweets.fillna(0)),
        "log listed count": np.log1p(df.listed.fillna(0)),
        "account age (days)": (COLLECTED - created).dt.days,
    }).fillna(-1)


DOCUMENTED = {"verified": "labelling function in TwiBot-22 (paper §3.2) and documented shortcut (Hays et al., WWW 2023)"}


def field_table(X, y, tr, te):
    out = {}
    for col in X.columns:
        x = X[col].to_numpy().reshape(-1, 1)
        t = DecisionTreeClassifier(max_depth=1, class_weight="balanced").fit(x[tr], y[tr])
        p = t.predict_proba(x)[:, 1]
        leaves = t.apply(x[tr])
        best = None
        for leaf in np.unique(leaves):
            m = leaves == leaf
            share, purity = m.mean(), max(y[tr][m].mean(), 1 - y[tr][m].mean())
            if share >= 0.10 and (best is None or purity > best[1]):
                best = (round(float(share), 3), round(float(purity), 3))
        reason = DOCUMENTED.get(col) or (f"leaf {best[0]:.0%} of users, {best[1]:.1%} pure" if best and best[1] >= 0.99 else None)
        out[col] = dict(macro_f1=round(float(f1_score(y[te], p[te] >= 0.5, average="macro")), 3),
                        auroc=round(float(roc_auc_score(y[te], p[te])), 3) if len(np.unique(p[te])) > 1 else 0.5,
                        bot_mean=round(float(X[col][y == 1].mean()), 3), human_mean=round(float(X[col][y == 0].mean()), 3),
                        purest_large_leaf=best, verdict=("SHORTCUT: " + reason) if reason else "allowed")
    return out


def text_stats(df):
    d = df.description
    dup = d[d != ""].duplicated(keep=False).reindex(d.index, fill_value=False)
    nonlat = d.map(lambda s: len(NON_LATIN.findall(s)) > len(s) / 2 if s else False)
    res = {}
    for k, name in [(1, "bot"), (0, "human")]:
        m = df.y == k
        res[name] = dict(bio_empty=round(float((d[m] == "").mean()), 3),
                         bio_median_chars=float(d[m].str.len().median()),
                         bio_shared_with_another_user=round(float(dup[m].mean()), 3),
                         mostly_non_latin=round(float(nonlat[m].mean()), 3))
    return res


def main():
    RES.mkdir(parents=True, exist_ok=True)
    import twibot22_guard
    df = with_labels(flatten())
    X, y = features(df), df.y.to_numpy()
    X.insert(0, "verified", twibot22_guard.verified(df.id).astype(int))     # checked here, never a model input
    results = {}

    log("all 1M users, official split (train -> test)")
    tr, te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()
    results["all_users"] = dict(n=len(df), bot_share=round(float(y.mean()), 3),
                                fields=field_table(X, y, tr, te), text=text_stats(df))
    for col, r in results["all_users"]["fields"].items():
        log(f"  {col:22s} F1 {r['macro_f1']:.3f} AUROC {r['auroc']:.3f}  bot {r['bot_mean']:>8} human {r['human_mean']:>8}  {r['verdict']}")
    log("  text:", results["all_users"]["text"])

    pos = pd.Series(np.arange(len(df)), index=df.id)
    results["sub_communities"] = {}
    for k in range(10):
        ids = json.loads((RAW / "sub-communities" / f"user{k}.json").read_text())
        idx = pos.reindex(ids).dropna().astype(int).to_numpy()
        ys = y[idx]
        # same proportions as prep_twibot22.py (70/20/10): train on the 70%, score on the 10% test part
        a, rest = train_test_split(np.arange(len(idx)), train_size=0.7, stratify=ys, random_state=42)
        _, b = train_test_split(rest, train_size=2 / 3, stratify=ys[rest], random_state=42)
        trm, tem = np.zeros(len(idx), bool), np.zeros(len(idx), bool)
        trm[a], tem[b] = True, True
        f = field_table(X.iloc[idx].reset_index(drop=True), ys, trm, tem)
        results["sub_communities"][k] = dict(fields=f, text=text_stats(df.iloc[idx].reset_index(drop=True)))
        short = [c for c, r in f.items() if r["verdict"] != "allowed"]
        top = sorted(f.items(), key=lambda kv: -kv[1]["macro_f1"])[:3]
        log(f"  sub {k}: shortcuts {short}; best single fields "
            + ", ".join(f"{c} {r['macro_f1']:.3f}" for c, r in top)
            + f"; bio empty bot/human {results['sub_communities'][k]['text']['bot']['bio_empty']}/{results['sub_communities'][k]['text']['human']['bio_empty']}")
    (RES / "profile_check.json").write_text(json.dumps(results, indent=1, default=float))
    log(f"saved {RES / 'profile_check.json'}")


if __name__ == "__main__":
    main()
