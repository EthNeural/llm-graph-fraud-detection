"""
TwiBot-20 to-do G4 (reports/twibot20_todo.md; written Fri 6 Nov ~04:25): matched-text control.
LoRA reads 480 Qwen tokens (step6_dgp.fit_target: the bio, then whole tweets while they fit); the encoders read up to
2,000 characters, and 81% of users are cut for LoRA. Here the headline encoder (Qwen2.5-1.5B, mean-pooled, the same
step3_embed code) reads EXACTLY LoRA's 480-token strings; LR and MLP are then trained as in bootstrap_ci.py (all train
and LoRA's 1,300 users). If the matched encoder stays above LoRA, "frozen encoder beats fine-tuning" is not a
text-length artefact. GPU ~20-30 min (11,826 users). Output: results/twibot20/<SETTING>/g4_matched_text.json
Run: DATASET=twibot20 SETTING=A VARIANT=target HF_HUB_OFFLINE=1 python my_build/twibot20_g4_matched_text.py
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from transformers import AutoTokenizer

from common import SHARED_DIR, RESULTS_DIR, best_threshold, get_splits, load, load_full, missing_text_flags, original_ids
from step3_embed import ENCODERS, average_by_user, encode_decoder_llm, split_pieces
from bootstrap_ci import lr_predictions, predictions
from step4_gnn import NoGraph
from step5_llm_reason import sample
import step6_dgp as s6

EMB = SHARED_DIR / "emb_qwen_matched480.npy"


def main():
    if not EMB.exists():
        tok = AutoTokenizer.from_pretrained(s6.MODEL_ID)                 # LoRA's tokenizer (Qwen3-1.7B)
        texts = [s6.fit_target(tok, t) for t in load_full()[0]]
        pieces, owner = split_pieces(texts)
        emb = average_by_user(encode_decoder_llm(ENCODERS["qwen"], pieces), owner, len(texts))
        np.save(EMB, emb)
        print("saved", EMB, emb.shape, flush=True)
    _, e, y = load()
    train, val, test = get_splits(y)
    out = {}
    for name, f in (("original (up to 2,000 chars)", SHARED_DIR / "emb_qwen.npy"), ("matched (LoRA's 480 tokens)", EMB)):
        emb = np.hstack([np.load(f), missing_text_flags()])[original_ids()].astype(np.float32)
        feats = lambda tr: torch.as_tensor((emb - emb[tr].mean(0)) / (emb[tr].std(0) + 1e-6))
        tr13 = sample(train, y, 1300)
        rows = {}
        for lab, tr in (("all train", train), ("1,300", tr13)):
            P, S = lr_predictions(emb, y, tr, val, test, probs=True)
            rows[f"LR ({lab})"] = (P, S)
            P, S = predictions(NoGraph, feats(tr), torch.as_tensor(e), y, tr, val, test, probs=True)
            rows[f"MLP ({lab})"] = (P, S)
        yt = y[test]
        out[name] = {k: dict(macro_f1=round(float(np.mean([f1_score(yt, p, average="macro") for p in P])), 4),
                             auroc=round(float(np.mean([roc_auc_score(yt, s) for s in S])), 4),
                             pr_auc=round(float(np.mean([average_precision_score(yt, s) for s in S])), 4), runs=int(len(P)))
                     for k, (P, S) in rows.items()}
        for k, v in out[name].items():
            print(f"{name:30s} {k:16s} {v}", flush=True)
    (RESULTS_DIR / "g4_matched_text.json").write_text(json.dumps(out, indent=1))
    print("saved", RESULTS_DIR / "g4_matched_text.json")


if __name__ == "__main__":
    main()
