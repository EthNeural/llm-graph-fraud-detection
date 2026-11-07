"""TwiBot-20 C3 with all 3 LoRA seeds (Sat 7 Nov): reuses twibot20_c1_scores.lora_seed_scores (adapter chosen by
full-validation macro-F1 per seed) and the saved MLP (1,300) / MLP (all) scores in results/twibot20/A/c1_scores/.
Output: results/twibot20/A/c3_three_seeds.json.  Run: DATASET=twibot20 SETTING=A python my_build/twibot20_c3_three_seeds.py"""
import json
import numpy as np
import pandas as pd
from common import DATA_DIR, RESULTS_DIR, get_splits, load, original_ids
from twibot20_c1_scores import boot, lora_seed_scores, summary
_, _, y = load()
train, val, test = get_splits(y)
v = pd.read_csv(DATA_DIR / "metadata.csv").verified.astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()[original_ids()]
yt, allp, nvp = y[test], np.arange(len(test)), np.where(~v[test])[0]
P, S = {}, {}
for k in ("target", "dgp"):
    P[k], S[k], used = lora_seed_scores(k, y, val, test)
    print(k, "seeds", used)
for k, f in (("MLP (1,300)", "MLP_1300.npz"), ("MLP (all train)", "MLP_all_train.npz")):
    z = np.load(RESULTS_DIR / "c1_scores" / f); P[k], S[k] = z["preds"], z["scores"]
res = {"models": {k: {"all": summary(yt, P[k], S[k], allp), "non_verified": summary(yt, P[k], S[k], nvp), "runs": int(len(P[k]))} for k in P}, "pairs": {}}
for k, r in res["models"].items(): print(k, r)
for a, b in (("dgp", "target"), ("target", "MLP (1,300)"), ("dgp", "MLP (1,300)"), ("dgp", "MLP (all train)")):
    res["pairs"][f"{a} - {b}"] = {"all": boot(yt, P[a], S[a], P[b], S[b], allp), "non_verified": boot(yt, P[a], S[a], P[b], S[b], nvp)}
    print(f"{a} - {b}:", res["pairs"][f"{a} - {b}"])
(RESULTS_DIR / "c3_three_seeds.json").write_text(json.dumps(res, indent=1))
