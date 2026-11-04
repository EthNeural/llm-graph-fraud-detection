"""
TwiBot-22 shortcut check on OUR split (checklist rows D2, D3, E4). CPU only, no torch, ~2 min.

A "shortcut" is an input that gives away the label for reasons unrelated to bot behaviour
(how the data was collected or labelled). Rule, fixed for TwiBot-20 (metadata_check.py), with
the leaf-size test changed from a share (10%) to a count, so a small but pure group is not missed:
  a field is a SHORTCUT if
   (a) it is a documented collection or labelling artefact, or
   (b) a depth-1 tree on it, fitted on OUR TRAIN users, has a leaf with >= 200 training users
       that is >= 99% one class.
Each field is also scored alone on OUR VAL users (AUROC of the tree's score; macro-F1 at the cut-off
that is best on val). No test user is used to fit or score anything; only the descriptive
bot_mean / human_mean columns are averaged over all users (labels incl. test), as in the plan's tables.

Fields checked:
  profile  the 17 fields of twibot22_profile_check.features() (incl. verified and withheld)
  graph    per relation: has an incoming edge, has an outgoing edge, log in-degree, log out-degree
           (relations_uu.npz: following, followers, mention, retweet, reply, quote, like)
           These are crawl-design cues: only ~10k users have any outgoing edge.

Output: results/twibot22/shortcuts_ours.json, and a printed table.
Run:    python my_build/twibot22_shortcuts.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from twibot22_profile_check import features

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "twibot-22"
RAW = ROOT.parent / "data" / "twibot-22-raw"
RES = ROOT / "results" / "twibot22"
RELATIONS = ["following", "followers", "mention", "retweet", "reply", "quote", "like"]
MIN_LEAF_USERS, PURITY = 200, 0.99
DOCUMENTED = {"verified": "labelling function in TwiBot-22 (paper §3.2): 0 verified bots"}


def check(x, y, tr, va):
    # depth-1 tree on one field: purest leaf with >= MIN_LEAF_USERS training users, val scores
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    t = DecisionTreeClassifier(max_depth=1, class_weight="balanced").fit(x[tr], y[tr])
    leaves = t.apply(x[tr])
    best = None
    for leaf in np.unique(leaves):
        m = leaves == leaf
        n, bot = int(m.sum()), float(y[tr][m].mean())
        purity = max(bot, 1 - bot)
        if n >= MIN_LEAF_USERS and (best is None or purity > best["purity"]):
            best = dict(users=n, share=round(n / tr.sum(), 3), purity=round(purity, 4),
                        majority="bot" if bot > 0.5 else "human")
    p = t.predict_proba(x[va])[:, 1]
    auroc = float(roc_auc_score(y[va], p)) if len(np.unique(p)) > 1 else 0.5
    cuts = np.unique(p)
    f1 = max(f1_score(y[va], p >= c, average="macro") for c in cuts) if len(cuts) > 1 else \
        f1_score(y[va], np.zeros_like(y[va]), average="macro")
    return dict(val_auroc=round(auroc, 3), val_macro_f1=round(float(f1), 3), purest_leaf=best,
                bot_mean=round(float(x[y == 1].mean()), 3), human_mean=round(float(x[y == 0].mean()), 3))


def tail_scan(fields, y, tr):
    # Stricter extra check (not the pre-registered rule): a depth-1 tree picks ONE split that best
    # separates the classes overall, so it can miss a small pure group in a tail. Here every cut-off
    # of every profile field is tried: the largest low or high tail with >= MIN_LEAF_USERS training
    # users at >= PURITY. For each such tail, how much of it is verified, and how pure it is without them.
    ver = np.asarray(fields["profile: verified"])[tr] == 1
    yt, out = y[tr], {}
    for name, x in fields.items():
        if not name.startswith("profile: ") or name == "profile: verified":
            continue
        xs = np.asarray(x, dtype=np.float64)[tr]
        for side in ("low", "high"):
            vals = np.unique(xs)
            vals = vals if side == "low" else vals[::-1]
            best = None
            for v in vals:
                m = xs <= v if side == "low" else xs >= v
                n = int(m.sum())
                if n > 0.5 * len(xs):
                    break
                bot = yt[m].mean()
                if n >= MIN_LEAF_USERS and max(bot, 1 - bot) >= PURITY:
                    best = (v, m, bot)
            if best:
                v, m, bot = best
                nv = m & ~ver
                out[f"{name} {side} {'<=' if side == 'low' else '>='} {v:.4g}"] = dict(
                    users=int(m.sum()), share=round(float(m.mean()), 4), bot_share=round(float(bot), 4),
                    verified_share=round(float(ver[m].mean()), 3), non_verified_users=int(nv.sum()),
                    non_verified_bot_share=round(float(yt[nv].mean()), 4) if nv.any() else None)
    print("\nStricter tail scan (largest pure tail per field; not the pre-registered rule):")
    for k, r in out.items():
        print(f"  {k:42s} {r['users']:>7,} users ({r['share']:.1%}), bots {r['bot_share']:.3f}; "
              f"verified {r['verified_share']:.0%}; without verified: {r['non_verified_users']:,} users, bots {r['non_verified_bot_share']}")
    return out


def main():
    lab = pd.read_csv(RAW / "label.csv")
    y = (lab.label.values == "bot").astype(int)
    split = np.load(DATA / "split_ours.npy")
    tr, va = split == "train", split == "val"
    users = pd.read_parquet(DATA / "users.parquet").set_index("id").loc[lab.id].reset_index()
    import twibot22_guard
    fields = {"profile: verified": twibot22_guard.verified(lab.id).astype(int)}   # from eval_only/, checked only
    fields.update({f"profile: {c}": v.to_numpy() for c, v in features(users).items()})
    rels = np.load(DATA / "relations_uu.npz")
    n = len(y)
    for r in RELATIONS:
        e = rels[r]
        out_deg, in_deg = np.bincount(e[0], minlength=n), np.bincount(e[1], minlength=n)
        fields[f"graph: {r} has incoming edge"] = (in_deg > 0).astype(int)
        fields[f"graph: {r} has outgoing edge"] = (out_deg > 0).astype(int)
        fields[f"graph: {r} log in-degree"] = np.log1p(in_deg)
        fields[f"graph: {r} log out-degree"] = np.log1p(out_deg)

    rows = {}
    for name, x in fields.items():
        r = check(x, y, tr, va)
        short = name.split(": ", 1)[1]
        leaf = r["purest_leaf"]
        if short in DOCUMENTED:
            r["verdict"] = "SHORTCUT (documented): " + DOCUMENTED[short]
        elif leaf and leaf["purity"] >= PURITY:
            r["verdict"] = f"SHORTCUT: {leaf['users']:,} train users ({leaf['share']:.1%}) are {leaf['purity']:.1%} {leaf['majority']}"
        else:
            r["verdict"] = "allowed"
        rows[name] = r

    info = dict(split="ours (70/20/10, seed 42, experts in test)", n_train=int(tr.sum()), n_val=int(va.sum()),
                train_bot_share=round(float(y[tr].mean()), 3),
                rule=f"documented artefact, or depth-1 leaf with >= {MIN_LEAF_USERS} train users at >= {PURITY:.0%} purity",
                fields=rows)
    info["tail_scan"] = tail_scan(fields, y, tr)
    (RES / "shortcuts_ours.json").write_text(json.dumps(info, indent=1))
    print(f"train {tr.sum():,} users ({y[tr].mean():.1%} bots); scored on val {va.sum():,}\n")
    print(f"{'field':40s} {'val AUROC':>9s} {'val F1':>7s} {'bot mean':>9s} {'human mean':>10s}  verdict")
    for name, r in sorted(rows.items(), key=lambda kv: -abs(kv[1]["val_auroc"] - 0.5)):
        print(f"{name:40s} {r['val_auroc']:>9.3f} {r['val_macro_f1']:>7.3f} {r['bot_mean']:>9.3f} {r['human_mean']:>10.3f}  {r['verdict']}")
    print(f"\nsaved {RES / 'shortcuts_ours.json'}")


if __name__ == "__main__":
    main()
