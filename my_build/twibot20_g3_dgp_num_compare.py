"""
TwiBot-20 to-do G3 (Sat 7 Nov): LoRA `dgp_num` (numeric profile summaries of neighbours) vs `target` and `dgp`,
same unseeded run (one seed each), each adapter chosen by the full-validation macro-F1 rule as in twibot20_c1_scores.py.
Paired bootstrap (2,000) on all and non-verified test users. Output: results/twibot20/A/g3_dgp_num.json
Run: DATASET=twibot20 SETTING=A OMP_NUM_THREADS=2 python my_build/twibot20_g3_dgp_num_compare.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from common import DATA_DIR, RESULTS_DIR, best_threshold, get_splits, load, original_ids
from twibot20_c1_scores import boot, summary


def chosen(variant, y, val, test):
    cands = [np.load(RESULTS_DIR / f"step6_Qwen3-1.7B_{variant}_{k}.npz") for k in ("scores", "last_scores")
             if (RESULTS_DIR / f"step6_Qwen3-1.7B_{variant}_{k}.npz").exists()]
    f1 = lambda s: f1_score(y[val], s["val"] > best_threshold(y[val], s["val"]), average="macro")
    s = max(cands, key=f1)
    assert (s["test_nodes"] == test).all()
    return np.array([s["test"] > best_threshold(y[val], s["val"])]), np.array([s["test"]]), len(cands)


_, e, y = load()
train, val, test = get_splits(y)
meta = pd.read_csv(DATA_DIR / "metadata.csv")
ver = meta.verified.astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()[original_ids()]
yt, allp, nvp = y[test], np.arange(len(test)), np.where(~ver[test])[0]
M = {v: chosen(v, y, val, test) for v in ("target", "dgp", "dgp_num")}
res = {"note": "one unseeded run per variant", "models": {}, "pairs": {}}
for v, (P, S, n) in M.items():
    res["models"][v] = {"all": summary(yt, P, S, allp), "non_verified": summary(yt, P, S, nvp), "candidates": n}
    print(v, res["models"][v], flush=True)
for a, b in (("dgp_num", "target"), ("dgp_num", "dgp")):
    res["pairs"][f"{a} - {b}"] = {sl: boot(yt, M[a][0], M[a][1], M[b][0], M[b][1], idx) for sl, idx in (("all", allp), ("non_verified", nvp))}
    print(a, "-", b, res["pairs"][f"{a} - {b}"], flush=True)
(RESULTS_DIR / "g3_dgp_num.json").write_text(json.dumps(res, indent=1))
