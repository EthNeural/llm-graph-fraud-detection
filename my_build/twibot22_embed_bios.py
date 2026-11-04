"""
Embed the bios of ALL 1,000,000 TwiBot-22 users once (GPU), so any later subset, and the full-graph
GNN baselines, can reuse them. Same encoders, pooling and 128-token cap as step 3 on TwiBot-20.

  ENC=minilm | e5 | qwen      (default: minilm). e5 = multilingual-E5 (with the "query: " prefix).
Users with an empty bio get a zero vector (as TwiBot-20 users without text do); the "no bio" flag
is added later, at load time. Rows follow data/twibot-22-raw/label.csv. Saved as float16.

Output: my_build/results/twibot22/emb_bio_<ENC>_all.npy  (1M x d)
Run:    ENC=minilm python my_build/twibot22_embed_bios.py
"""
import os
import time
from pathlib import Path

os.environ.setdefault("DATASET", "twibot20")     # step3/common only accept known datasets; we reuse
import numpy as np                               # their encoder functions, not their data paths
import pandas as pd

import step3_embed as s3

ROOT = Path(__file__).resolve().parent
ENC = os.environ.get("ENC", "minilm")
OUT = ROOT / "results" / "twibot22" / f"emb_bio_{ENC}_all.npy"


def main():
    if OUT.exists():
        print(f"{OUT} exists")
        return
    lab = pd.read_csv(ROOT.parent / "data" / "twibot-22-raw" / "label.csv")
    users = pd.read_parquet(ROOT.parent / "data" / "twibot-22" / "users.parquet", columns=["id", "description"])
    bio = users.set_index("id").description.reindex(lab.id).fillna("").map(lambda s: " ".join(s.split())).to_numpy()
    has = np.where(bio != "")[0]
    print(f"{len(bio):,} users, {len(has):,} with a bio; encoder {ENC} = {s3.ENCODERS[ENC]}", flush=True)
    t0 = time.time()
    texts = list(bio[has])
    if ENC in s3.DECODERS:
        emb = s3.encode_decoder_llm(s3.ENCODERS[ENC], texts)
    elif ENC in ("e5", "e5en"):
        emb = s3.encode_sentence_model(s3.ENCODERS[ENC], texts, prefix="query: ")
    else:
        emb = s3.encode_sentence_model(s3.ENCODERS[ENC], texts)
    full = np.zeros((len(bio), emb.shape[1]), dtype=np.float16)
    full[has] = emb.astype(np.float16)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, full)
    print(f"saved {OUT} {full.shape} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
