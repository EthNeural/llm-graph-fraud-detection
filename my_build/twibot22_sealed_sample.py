"""
TwiBot-22: sealed test set, then the 100k study sample (checklist B4, B5, B8, H1, H8).
Decided by the builder 4 Nov 2026 ~22:05, before any model score: seal 50% of test + all experts.
CPU only, seconds. Seed 42 everywhere. Nothing here looks at features or scores.

1. ROLES (B4, B8). Starting from our split (split_ours.npy: train / val / test):
     test_sealed  all 986 expert-labelled users + a stratified random 50% of the other test users
     test_work    the other 50% of the non-expert test users
   Sealed users are scored ONCE, at the very end, after every choice is fixed (twibot22_guard).
   -> data/twibot-22/roles.npy   ('train' / 'val' / 'test_work' / 'test_sealed', label.csv order)

2. SAMPLE (B5). 100,000 users drawn at random from the non-expert users, stratified by role AND label,
   so each role keeps its share and every role keeps 14% bots. All 986 experts are then added (they are
   test_sealed). Neighbours still come from the full 1M graph; nothing is cut.
   -> data/twibot-22/sample_idx.npy   (row indices into label.csv, sorted)

3. HUBS (H1). The 10,039 users with any outgoing edge (7 relations). They get bio + tweet embeddings
   ONLY as neighbour-feature providers; they are not study users unless the random draw picked them.
   -> data/twibot-22/hub_idx.npy

Summary -> results/twibot22/sealed_sample.json
Run:  python my_build/twibot22_sealed_sample.py      (refuses to overwrite existing files)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "twibot-22"
RAW = ROOT.parent / "data" / "twibot-22-raw"
RES = ROOT / "results" / "twibot22"
SEED, SEAL_SHARE, SAMPLE_SIZE = 42, 0.5, 100_000
RELATIONS = ["following", "followers", "mention", "retweet", "reply", "quote", "like"]


def main():
    for f in ("roles.npy", "sample_idx.npy", "hub_idx.npy"):
        assert not (DATA / f).exists(), f"{f} exists: these files are fixed once and never redrawn"
    lab = pd.read_csv(RAW / "label.csv")
    n = len(lab)
    y = (lab.label.values == "bot").astype(int)
    split = np.load(DATA / "split_ours.npy")
    expert = lab.id.isin(pd.read_csv(RAW / "expert_labels" / "label_manual.csv").id).to_numpy()
    assert (split[expert] == "test").all()

    # 1. roles: seal the experts and a stratified half of the other test users
    roles = split.astype(object).copy()
    other_test = np.where((split == "test") & ~expert)[0]
    work, sealed = train_test_split(other_test, test_size=SEAL_SHARE, stratify=y[other_test], random_state=SEED)
    roles[work] = "test_work"
    roles[sealed] = "test_sealed"
    roles[expert] = "test_sealed"
    roles = roles.astype(str)
    np.save(DATA / "roles.npy", roles)

    # 2. sample: stratified by role and label among non-experts, then add every expert
    pool = np.where(~expert)[0]
    strata = np.char.add(roles[pool], y[pool].astype(str))
    take, _ = train_test_split(pool, train_size=SAMPLE_SIZE, stratify=strata, random_state=SEED)
    sample = np.sort(np.concatenate([take, np.where(expert)[0]]))
    np.save(DATA / "sample_idx.npy", sample)

    # 3. hubs: any outgoing edge in the 7 relations
    rels = np.load(DATA / "relations_uu.npz")
    out = np.zeros(n, bool)
    for r in RELATIONS:
        out[rels[r][0]] = True
    hubs = np.where(out)[0]
    np.save(DATA / "hub_idx.npy", hubs)

    def counts(idx):
        return {r: dict(users=int((roles[idx] == r).sum()), bot_share=round(float(y[idx][roles[idx] == r].mean()), 4))
                for r in ("train", "val", "test_work", "test_sealed")}
    in_sample = np.zeros(n, bool)
    in_sample[sample] = True
    summary = dict(seed=SEED, seal_share=SEAL_SHARE, sample_size=SAMPLE_SIZE,
                   all_users=counts(np.arange(n)), sample=counts(sample),
                   sample_total=int(len(sample)), sample_bot_share=round(float(y[sample].mean()), 4),
                   experts_in_sealed=int((roles[expert] == "test_sealed").sum()),
                   hubs=int(len(hubs)), hubs_in_sample=int(in_sample[hubs].sum()),
                   embed_users_sample_or_hub=int((in_sample | out).sum()))
    (RES / "sealed_sample.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
