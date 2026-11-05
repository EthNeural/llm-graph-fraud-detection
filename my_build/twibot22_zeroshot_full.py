"""
Zero-shot Qwen3-1.7B (VARIANT=target prompt, no training) on the SAME 5,000 working-test users as
LoRA target, so zero-shot, LoRA and the text baselines share one test set (builder's queue, 5 Nov, item 2).
step 6 scored zero-shot only on the 400-user step 5 sample.

Readout checked first (5 Nov 13:30): P(bot) = softmax(logit "Yes", logit "No")[Yes] for "Is this Twitter
account a bot ... ?". Same code gives zero-shot AUROC 0.60 on TwiBot-20, and LoRA trained through the
same readout reaches 0.71, so the sign is right. The untrained model says Yes ~99% for everyone; its
1,000-user validation AUROC is 0.458 (1 - p: 0.542) because it says Yes slightly less for no-text users,
who are 48% bots. On users with text, AUROC 0.50. Not a bug, so nothing was changed.

Validation: the 1,000 check users, already scored in step6_Qwen3-1.7B_target_zeroshot.npz (cut-off).
Run: DATASET=twibot22 VARIANT=target VAL_CHECK=1000 HF_HUB_OFFLINE=1 python -u my_build/twibot22_zeroshot_full.py
"""
import os
import sys
from pathlib import Path

import numpy as np

assert os.environ.get("DATASET") == "twibot22" and os.environ.get("VARIANT") == "target"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step6_dgp as s6
import twibot22_guard
from twibot22_metrics import val_cutoff, metrics

OUT = s6.RESULTS_DIR / "step6_Qwen3-1.7B_target_zeroshot_full.npz"


def main():
    assert not OUT.exists(), f"{OUT} exists: never overwritten"
    x_text, _, y = s6.load()
    _, val, test = s6.get_splits(y)
    rows = np.load(s6.DATA_DIR / "rows.npy")
    twibot22_guard.assert_not_sealed(rows[test])
    lora = np.load(s6.RESULTS_DIR / "step6_Qwen3-1.7B_target_scores.npz")
    assert (lora["test_nodes"] == test).all()
    zs = np.load(s6.RESULTS_DIR / "step6_Qwen3-1.7B_target_zeroshot.npz")
    val_s = s6.sample(val, y, s6.N_VAL)
    assert (val_s == lora["val_nodes"]).all() and len(zs["val"]) == len(val_s)

    tok, model = s6.load_llm(s6.torch.float16)
    yes = tok.encode("Yes", add_special_tokens=False)[0]
    no = tok.encode("No", add_special_tokens=False)[0]
    orig = s6.original_ids()
    seqs = [tok(s6.build_prompt(tok, v, x_text, {}, orig), add_special_tokens=False)["input_ids"] for v in test]
    p = s6.score(model, seqs, tok.pad_token_id, yes, no, desc="zero-shot full test")
    np.savez(OUT, val=zs["val"], test=p, val_nodes=val_s, test_nodes=test, test_rows=rows[test])
    cut = val_cutoff(y[val_s], zs["val"])
    print("zero-shot target, 5,000 working-test users:", metrics(y[test], p, cut))
    print("same with 1 - p (sign check only, not a result):", metrics(y[test], 1 - p, val_cutoff(y[val_s], 1 - zs["val"])))
    print("Saved", OUT)


if __name__ == "__main__":
    main()
