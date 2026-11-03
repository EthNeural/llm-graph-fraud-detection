"""
Step 6 checkpoint check: was LoRA stopped too early?

step6_dgp.py keeps the adapter with the lowest validation LOSS on the 200-user check sample
(best_adapter.pt). In both TwiBot-20 A runs that is epoch 1, yet validation AUROC was still
rising at epoch 2 (loss up, ranking better = over-confidence, not overfitting). The text-only
and GNN models instead keep the epoch with the best macro-F1 on the FULL validation set.

This script scores the LAST adapter (state.pt, epoch 2) on the same full validation and test
sets, and applies the text-only models' rule to choose between the two adapters:
  rule (fixed before looking at test): keep the adapter with the higher FULL-VALIDATION
  macro-F1 at its own best cut-off. Both adapters are reported either way.
No training, no new summaries: prompts are rebuilt from the cached metapath summaries.

Run:  caffeinate -ims env DATASET=twibot20 SETTING=A VARIANT=dgp python -u my_build/step6_rescore.py
GPU, about 40 min for TwiBot-20 A (full val 2,365 + test 1,183), less for B.
Results: results/<dataset>/<setting>/step6_rescore.json (one row per variant, merged).
"""
import json

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from sklearn.metrics import f1_score

import step6_dgp as s6
from common import load, get_splits, isolated_mask, evaluate, best_threshold, original_ids, RESULTS_DIR
from step3_embed import DEVICE
from step5_llm_reason import sample, N_TEST


def val_f1(y_val, p):
    return f1_score(y_val, p > best_threshold(y_val, p), average="macro")


def main():
    x_text, edge_index, y = load()
    train, val, test = get_splits(y)
    test_s = sample(test, y, N_TEST)
    iso, orig = isolated_mask(edge_index, len(y)), original_ids()
    best = np.load(RESULTS_DIR / f"step6_{s6.TAG}_scores.npz")        # epoch-1 adapter, already scored
    assert (best["test_nodes"] == test).all()

    out_file = RESULTS_DIR / f"step6_{s6.TAG}_last_scores.npz"
    if not out_file.exists():
        tok, model = s6.load_llm(torch.float16)
        yes = tok.encode("Yes", add_special_tokens=False)[0]
        no = tok.encode("No", add_special_tokens=False)[0]
        path_sum = json.load(open(s6.PATH_SUMMARY_FILE)) if s6.VARIANT == "dgp" else {}
        enc = lambda nodes: [tok(s6.build_prompt(tok, v, x_text, path_sum, orig), add_special_tokens=False)
                             ["input_ids"] for v in nodes]
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **s6.LORA))
        state = torch.load(s6.CKPT / "state.pt", weights_only=False)
        assert state["state"]["done"], "training has not finished for this variant"
        set_peft_model_state_dict(model, state["adapter"])
        model.eval()
        p_val = s6.score(model, enc(val), tok.pad_token_id, yes, no, desc="last adapter: full val")
        p_test = s6.score(model, enc(test), tok.pad_token_id, yes, no, desc="last adapter: full test")
        np.savez(out_file, val=p_val, test=p_test, test_nodes=test, last_epoch=state["state"]["epoch"])
        del model
        torch.mps.empty_cache() if DEVICE == "mps" else None
    last = np.load(out_file)

    rows, f1s = [], {}
    pos = {v: i for i, v in enumerate(test)}
    for name, sc in [("best val loss (epoch 1)", best), ("last (epoch 2)", last)]:
        f1s[name] = val_f1(y[val], sc["val"])
        t = best_threshold(y[val], sc["val"])
        r = evaluate(f"{s6.NAME} | {name} | full test", y[test], (sc["test"] > t).astype(int), sc["test"], iso[test])
        p_s = sc["test"][[pos[v] for v in test_s]]
        r_s = evaluate(f"{s6.NAME} | {name} | step 5 sample", y[test_s], (p_s > t).astype(int), p_s, iso[test_s])
        r["full_val_macro_f1"] = r_s["full_val_macro_f1"] = round(float(f1s[name]), 4)
        rows += [r, r_s]
    chosen = max(f1s, key=f1s.get)
    print(f"\nfull-validation macro-F1: " + ", ".join(f"{k} {v:.4f}" for k, v in f1s.items()) +
          f"  -> chosen by the text-only rule: {chosen}")
    for r in rows:
        r["chosen_by_val_f1"] = chosen in r["model"]

    f = RESULTS_DIR / "step6_rescore.json"
    names = {r["model"] for r in rows}
    old = [r for r in json.load(open(f)) if r["model"] not in names] if f.exists() else []
    json.dump(old + rows, open(f, "w"), indent=2)
    print(f"saved {f}")


if __name__ == "__main__":
    main()
