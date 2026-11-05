"""
TwiBot-22: LoRA dgp vs LoRA target on the SAME 1,500-user test subset (dgp_test_subset_1500.npy, drawn
5 Nov ~15:00, stratified from the 5,000 working-test users, seed 42, committed before any scoring).
target was scored on all 5,000, so its scores are restricted to the subset; each model keeps its own
cut-off from its 1,000 validation-check users. Paired bootstrap (2,000) of dgp - target. Any extra
seeds found (step6_..._s<seed>_scores.npz) are listed per seed. Output: results/twibot22/dgp_vs_target.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import twibot22_guard
from twibot22_metrics import val_cutoff, metrics
from twibot22_text_baselines import boot_diff

A = ROOT / "results" / "twibot22" / "A"
y = np.load(ROOT.parent / "data" / "twibot-22-llm" / "y.npy")
rows = np.load(ROOT.parent / "data" / "twibot-22-llm" / "rows.npy")
sub = np.load(A / "dgp_test_subset_1500.npy")
twibot22_guard.assert_not_sealed(rows[sub])


def on_subset(f):
    z = np.load(f)
    pos = {v: i for i, v in enumerate(z["test_nodes"])}
    assert all(v in pos for v in sub), f"{f.name} does not cover the subset"
    return z["test"][[pos[v] for v in sub]], val_cutoff(y[z["val_nodes"]], z["val"])


out = {"test_users": int(len(sub)), "test_bots": int(y[sub].sum()), "pairs": {}}
for tag in ["", "_s1", "_s2"]:
    ft, fd = A / f"step6_Qwen3-1.7B_target{tag}_scores.npz", A / f"step6_Qwen3-1.7B_dgp{tag}_scores.npz"
    for name, f in (("target" + tag, ft), ("dgp" + tag, fd)):
        if f.exists():
            s, c = on_subset(f)
            out[name] = metrics(y[sub], s, c)
            print(name, {k: round(v, 3) for k, v in out[name].items() if isinstance(v, float)})
    if ft.exists() and fd.exists():
        (sd, cd), (st, ct) = on_subset(fd), on_subset(ft)
        out["pairs"]["dgp_minus_target" + tag] = boot_diff(y[sub], sd, st, cd, ct)
        print("dgp - target" + tag, out["pairs"]["dgp_minus_target" + tag])
zs = A / "step6_Qwen3-1.7B_dgp_zeroshot.npz"
if zs.exists():
    z = np.load(zs)
    if len(z["test"]) == len(sub):
        out["zero_shot_dgp"] = metrics(y[sub], z["test"], val_cutoff(y[np.load(A / "step6_Qwen3-1.7B_target_scores.npz")["val_nodes"]], z["val"]))
json.dump(out, open(ROOT / "results" / "twibot22" / "dgp_vs_target.json", "w"), indent=1)
print("Saved dgp_vs_target.json")
