"""
TwiBot-22 text-only baselines on the LLM study folder (data/twibot-22-llm), scored on the SAME 5,000
working-test users as LoRA target (step 6). Builder's queue, 5 Nov 13:20, item 1.

  TF-IDF + LR    word 1-2 grams on x_text (bio + 20 most recent tweets, the text LoRA reads before its
                 480-token cut), sublinear tf, min_df 3, 200k features
  MiniLM + MLP   the MiniLM bio + tweet embedding (pieces averaged per user), sklearn MLP 256, early stopping

Each is trained twice: on all 70,000 train users, and on the same 1,300 users LoRA was trained on
("matched"). Cut-off chosen on the 20,000 validation users (macro-F1), never on test. Sealed users are
never touched (guard). Paired bootstrap (2,000) of each model minus LoRA target on the 5,000 test users.
No torch import (the TwiBot-20 sklearn segfault, G1). Output: results/twibot22/text_baselines.json and
runs/llm_sample/<model>_s<seed>.npz (test scores).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import twibot22_guard
from twibot22_metrics import val_cutoff, metrics

D = ROOT.parent / "data" / "twibot-22-llm"
OUT = ROOT / "results" / "twibot22"
RUNS = OUT / "runs" / "llm_sample"
N_BOOT = 2000
SEED = 42   # common.SEED


def boot_diff(y, a, b, ca, cb, seed=0):
    from sklearn.metrics import f1_score, roc_auc_score
    rng = np.random.default_rng(seed)
    d = {"macro_f1": [], "auroc": []}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d["macro_f1"].append(f1_score(y[i], a[i] > ca, average="macro") - f1_score(y[i], b[i] > cb, average="macro"))
        d["auroc"].append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return {k: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()}


def main():
    y, split = np.load(D / "y.npy"), np.load(D / "split.npy")
    rows = np.load(D / "rows.npy")
    tr, va, te = (np.where(split == p)[0] for p in ("train", "val", "test"))
    twibot22_guard.assert_not_sealed(rows[np.concatenate([tr, va, te])])

    lora = np.load(OUT / "A" / "step6_Qwen3-1.7B_target_scores.npz")
    assert (lora["test_nodes"] == te).all(), "LoRA test users differ from split == 'test'"
    lora_val_nodes = lora["val_nodes"]
    lora_cut = val_cutoff(y[lora_val_nodes], lora["val"])
    # LoRA's 1,300 training users: step 5's sample() rule (stratified, random_state = common.SEED),
    # copied here so this script never imports torch.
    from sklearn.model_selection import train_test_split
    tr_matched = np.sort(train_test_split(tr, train_size=1300, stratify=y[tr], random_state=SEED)[0])
    assert np.isin(tr_matched, tr).all() and len(tr_matched) == 1300

    x_text = np.load(D / "x_text.npy")
    mini_dir = ROOT.parent / "data" / "twibot-22" / "sample110k_minilm_bio20tw_raw"
    assert (np.load(mini_dir / "ids.npy") == rows).all(), "MiniLM rows not in the study folder's order"
    E = np.load(mini_dir / "emb.npy").astype(np.float32)

    results = {"test_users": int(len(te)), "test_bots": int(y[te].sum()),
               "lora_target": metrics(y[te], lora["test"], lora_cut)}
    print(f"LoRA target (rescored here, cut-off on its 1,000 val users): {results['lora_target']}", flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    tfv = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
    Xt = tfv.fit_transform(x_text[tr]); Xall = tfv.transform(x_text)

    for train_name, trn in (("full70k", tr), ("matched1300", tr_matched)):
        # TF-IDF + LR (deterministic: one seed)
        Xi = Xall[trn] if train_name == "matched1300" else Xt
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xi, y[trn])
        s_va, s_te = lr.predict_proba(Xall[va])[:, 1], lr.predict_proba(Xall[te])[:, 1]
        cut = val_cutoff(y[va], s_va)
        key = f"tfidf_lr_{train_name}"
        results[key] = metrics(y[te], s_te, cut)
        results[key]["minus_lora_target"] = boot_diff(y[te], s_te, lora["test"], cut, lora_cut)
        np.savez(RUNS / f"{key}_s0.npz", val=s_va, test=s_te, val_nodes=va, test_nodes=te)
        print(key, results[key], flush=True)

        # MiniLM + MLP, 3 seeds
        sc = StandardScaler().fit(E[trn])
        per_seed = []
        for seed in (0, 1, 2):
            mlp = MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200,
                                random_state=seed).fit(sc.transform(E[trn]), y[trn])
            s_va = mlp.predict_proba(sc.transform(E[va]))[:, 1]
            s_te = mlp.predict_proba(sc.transform(E[te]))[:, 1]
            cut = val_cutoff(y[va], s_va)
            m = metrics(y[te], s_te, cut)
            m["minus_lora_target"] = boot_diff(y[te], s_te, lora["test"], cut, lora_cut, seed=seed)
            per_seed.append(m)
            np.savez(RUNS / f"minilm_mlp_{train_name}_s{seed}.npz", val=s_va, test=s_te, val_nodes=va, test_nodes=te)
            print(f"minilm_mlp_{train_name} seed {seed}", m, flush=True)
        results[f"minilm_mlp_{train_name}"] = {
            "seeds": per_seed,
            "mean": {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0] if k != "minus_lora_target" and k not in ("n", "bots")},
            "std": {k: float(np.std([p[k] for p in per_seed])) for k in ("macro_f1", "auroc", "pr_auc")}}

    json.dump(results, open(OUT / "text_baselines.json", "w"), indent=1)
    print("Saved", OUT / "text_baselines.json")


if __name__ == "__main__":
    main()
