"""
TwiBot-20 to-do C1 + C3 (reports/twibot20_todo.md; run Fri 6 Nov ~04:15).
C1: save the test scores of LR, MLP and the GNNs (bootstrap_ci.py retrains them and never saved them), then score
    every model on ALL test users and on NON-VERIFIED test users (TwiBot-20 A: 326 of 1,183 test users are verified,
    none of them a bot), so every headline can also be shown without the verified users.
C3: LoRA target / dgp averaged over the seeds that exist (unseeded run, s1, s2), each seed's adapter chosen by the
    full-validation macro-F1 rule ("chosen": best-loss vs last check), as the text-only models are chosen.
Same models, features, seeds and cut-off rule as bootstrap_ci.py (Qwen2.5-1.5B embeddings + 2 missing-text flags).
Paired bootstrap (2,000) on both slices for the headline pairs. Output: results/twibot20/<SETTING>/c1_scores/*.npz
and c1_nonverified.json.  Run: DATASET=twibot20 SETTING=A OMP_NUM_THREADS=2 python my_build/twibot20_c1_scores.py
"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.nn import FAConv, GCNConv, SAGEConv

from common import DATA_DIR, RESULTS_DIR, best_threshold, get_splits, load, load_embeddings, original_ids
from bootstrap_ci import lr_predictions, predictions
from step4_gnn import NoGraph
from step5_llm_reason import sample

OUT = RESULTS_DIR / "c1_scores"


def lora_seed_scores(variant, y, val, test):
    preds, scores, used = [], [], []
    for tag in ("", "_s1", "_s2"):
        best, last = (RESULTS_DIR / f"step6_Qwen3-1.7B_{variant}{tag}_{k}.npz" for k in ("scores", "last_scores"))
        if not best.exists():
            continue
        cands = [np.load(best)] + ([np.load(last)] if last.exists() else [])
        f1 = lambda s: f1_score(y[val], s["val"] > best_threshold(y[val], s["val"]), average="macro")
        s = max(cands, key=f1)
        assert (s["test_nodes"] == test).all()
        preds.append(s["test"] > best_threshold(y[val], s["val"]))
        scores.append(s["test"])
        used.append(tag or "unseeded")
    return (np.array(preds), np.array(scores), used) if preds else (None, None, [])


def summary(y, P, S, idx):
    return dict(macro_f1=round(float(np.mean([f1_score(y[idx], p[idx], average="macro") for p in P])), 4),
                auroc=round(float(np.mean([roc_auc_score(y[idx], s[idx]) for s in S])), 4),
                pr_auc=round(float(np.mean([average_precision_score(y[idx], s[idx]) for s in S])), 4))


def boot(y, Pa, Sa, Pb, Sb, idx, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    d = []
    for _ in range(2000):
        i = idx[rng.integers(0, len(idx), len(idx))]
        if y[i].min() == y[i].max():
            continue
        f = lambda P, S: (np.mean([f1_score(y[i], p[i], average="macro") for p in P]), np.mean([roc_auc_score(y[i], s[i]) for s in S]))
        a, b = f(Pa, Sa), f(Pb, Sb)
        d.append([a[0] - b[0], a[1] - b[1]])
    d = np.array(d)
    return {m: [round(float(d[:, j].mean()), 4), round(float(np.percentile(d[:, j], 2.5)), 4), round(float(np.percentile(d[:, j], 97.5)), 4)]
            for j, m in enumerate(("macro_f1", "auroc"))}


def main():
    OUT.mkdir(exist_ok=True)
    _, e, y = load()
    train, val, test = get_splits(y)
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    verified = meta.verified.astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()[original_ids()]
    emb = load_embeddings("qwen").astype(np.float32)
    edge_index = torch.as_tensor(e)
    feats = lambda tr: torch.as_tensor((emb - emb[tr].mean(0)) / (emb[tr].std(0) + 1e-6))
    tr_1300 = sample(train, y, 1300)
    x_full, x_1300 = feats(train), feats(tr_1300)
    P, S = {}, {}
    for name, (conv, x, tr) in {"MLP (all train)": (NoGraph, x_full, train), "GCN (all train)": (GCNConv, x_full, train),
                                "SAGE (all train)": (SAGEConv, x_full, train), "FAGCN (all train)": (FAConv, x_full, train),
                                "MLP (1,300)": (NoGraph, x_1300, tr_1300)}.items():
        P[name], S[name] = predictions(conv, x, edge_index, y, tr, val, test, probs=True)
        print(f"trained {name}", flush=True)
    P["LR (all train)"], S["LR (all train)"] = lr_predictions(emb, y, train, val, test, probs=True)
    P["LR (1,300)"], S["LR (1,300)"] = lr_predictions(emb, y, tr_1300, val, test, probs=True)
    seeds_used = {}
    for v in ("target", "dgp"):
        p, s, used = lora_seed_scores(v, y, val, test)
        if p is not None:
            P[f"LoRA {v} (1,300)"], S[f"LoRA {v} (1,300)"] = p, s
            seeds_used[v] = used
    for name in P:
        np.savez(OUT / (name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "") + ".npz"),
                 preds=P[name], scores=S[name], test_nodes=test)
    # positions within the test array
    all_pos = np.arange(len(test))
    nv_pos = np.where(~verified[test])[0]
    yt = y[test]
    res = {"test_users": int(len(test)), "non_verified_test_users": int(len(nv_pos)),
           "verified_test_bots": int(yt[verified[test]].sum()), "lora_seeds_used": seeds_used, "models": {}, "pairs": {}}
    for name in P:
        res["models"][name] = {"all": summary(yt, P[name], S[name], all_pos), "non_verified": summary(yt, P[name], S[name], nv_pos),
                               "seeds": int(len(P[name]))}
        r = res["models"][name]
        print(f"{name:22s} all {r['all']}  non-verified {r['non_verified']}  ({r['seeds']} run(s))", flush=True)
    for a, b in [("GCN (all train)", "MLP (all train)"), ("SAGE (all train)", "MLP (all train)"), ("FAGCN (all train)", "MLP (all train)"),
                 ("LoRA dgp (1,300)", "MLP (1,300)"), ("LoRA target (1,300)", "MLP (1,300)"), ("LoRA dgp (1,300)", "LoRA target (1,300)"),
                 ("LoRA dgp (1,300)", "MLP (all train)")]:
        if a in P and b in P:
            res["pairs"][f"{a} - {b}"] = {"all": boot(yt, P[a], S[a], P[b], S[b], all_pos), "non_verified": boot(yt, P[a], S[a], P[b], S[b], nv_pos)}
            r = res["pairs"][f"{a} - {b}"]
            print(f"  {a} - {b}: all F1 {r['all']['macro_f1']} | non-verified F1 {r['non_verified']['macro_f1']} AUROC {r['non_verified']['auroc']}", flush=True)
    (RESULTS_DIR / "c1_nonverified.json").write_text(json.dumps(res, indent=1))
    print("saved", RESULTS_DIR / "c1_nonverified.json")


if __name__ == "__main__":
    main()
