"""
Step 3: LLM role (a) "encoder". Turn every bio into a vector with THREE different LLMs,
then train a text-only classifier on each.

Run:  python my_build/step3_embed.py

The three encoders test different things:
  minilm : small English sentence encoder (22M)        -> the cheap baseline
  e5     : multilingual sentence encoder (278M)         -> do Spanish/Dutch bios matter?
  qwen   : Qwen2.5-1.5B decoder LLM, mean-pooled (1.5B) -> does a "real" LLM read bios better?
           (open on Hugging Face, no token needed)

Embeddings are cached in results/emb_<name>.npy and become the node features in step 4.

TwiBot-20: a user's text (bio + ~20 tweets, ~570 tokens) is far longer than an encoder reads,
so each piece (the bio, each tweet) is encoded SEPARATELY and the vectors are averaged per user.
Every encoder then sees the whole text, as TF-IDF does. Each piece vector is L2-normalised
before averaging, so no single long or odd tweet dominates. A user with no text at all gets
a zero vector; common.load_embeddings adds 0/1 "no bio" / "no tweets" flags. Nobody is dropped.
Set SKIP_ENCODERS=qwen (comma-separated) to leave slow encoders out of a run.
"""
import os
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from common import (load, load_full, load_embeddings, get_splits, isolated_mask, evaluate,
                    best_threshold,
                    RESULTS_DIR, SHARED_DIR, TEXT_MAX_TOKENS, DATASET)

ENCODERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5": "intfloat/multilingual-e5-base",
    "qwen": "Qwen/Qwen2.5-1.5B",
}
if DATASET == "twibot20":
    # Model ladder (TwiBot-20): everything else fixed, only the encoder changes.
    # Size and type: small sentence model -> tweet-trained -> general embedder
    #                -> LLM-based embedder -> decoder LLM.
    ENCODERS = {
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",      # 22M
        "e5en": "intfloat/e5-base-v2",                           # English E5-base, 110M
        "twroberta": "cardiffnlp/twitter-roberta-base",          # 125M, pre-trained on tweets
        "e5": "intfloat/multilingual-e5-base",                   # multilingual-E5, 278M (192M is vocabulary)
        "qwen3emb": "Qwen/Qwen3-Embedding-0.6B",                 # 0.6B, LLM-based embedder
        "qwen": "Qwen/Qwen2.5-1.5B",                             # 1.5B decoder LLM
        "qwen3": "Qwen/Qwen3-1.7B",                              # 1.7B, the step 6 LLM, same checkpoint
    }
DECODERS = {"qwen", "qwen3", "twroberta"}      # give one vector per token -> need mean pooling, see below
_skip = set(os.environ.get("SKIP_ENCODERS", "").split(","))
ENCODERS = {k: v for k, v in ENCODERS.items() if k not in _skip}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def emb_file(name):
    return SHARED_DIR / f"emb_{name}.npy"            # shared by settings A and B


def encode_sentence_model(model_id, texts, prefix=""):
    # Sentence encoders are trained to output one vector per sentence directly.
    from sentence_transformers import SentenceTransformer
    # TwiBot-20 has ~197k pieces: run in half precision (about 2x faster on the Mac GPU).
    half = {"model_kwargs": {"dtype": torch.float16}} if DATASET != "instagram" else {}
    model = SentenceTransformer(model_id, device=DEVICE, **half)
    if DATASET != "instagram":                         # instagram keeps each model's default
        model.max_seq_length = TEXT_MAX_TOKENS
    # (sentence-transformers already sorts inputs by length, so batches carry little padding)
    return model.encode([prefix + t for t in texts], batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True)


@torch.no_grad()
def encode_decoder_llm(model_id, texts, batch_size=16 if DATASET == "instagram" else 32,
                       max_len=TEXT_MAX_TOKENS):
    # A decoder LLM (or a plain pre-trained model like RoBERTa) gives one vector PER TOKEN.
    # Average them (ignoring padding) = mean pooling.
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:                          # some LLMs ship without a pad token
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_id, dtype=torch.float16).to(DEVICE).eval()
    # An empty bio becomes ZERO tokens (Qwen adds no start token), so mean pooling would
    # divide 0 by 0 = NaN. Give those users a placeholder text instead.
    texts = [t if t.strip() else "[empty bio]" for t in texts]

    # Encode in order of length so each batch has little padding, then restore the order.
    order = np.argsort([len(t) for t in texts]) if DATASET != "instagram" else np.arange(len(texts))
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc=model_id):
        batch = tok([texts[j] for j in order[i:i + batch_size]], padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt").to(DEVICE)
        hidden = model(**batch).last_hidden_state.float()      # (B, T, D)
        mask = batch["attention_mask"].unsqueeze(-1).float()   # (B, T, 1), 0 on padding
        out.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
    emb = np.empty((len(texts), out[0].shape[1]), dtype=np.float32)
    emb[order] = np.concatenate(out)
    return emb


def split_pieces(texts):
    # TwiBot-20: one piece per line (the bio, then one tweet per line). Returns the pieces and,
    # for each piece, the index of the user it belongs to.
    pieces, owner = [], []
    for i, t in enumerate(texts):
        parts = [p.strip() for p in t.split("\n") if p.strip()]      # [] if no text at all
        pieces += parts
        owner += [i] * len(parts)
    return pieces, np.array(owner)


def average_by_user(piece_emb, owner, n):
    # Mean of each user's L2-normalised piece vectors; zero vector for a user with no pieces.
    unit = piece_emb / (np.linalg.norm(piece_emb, axis=1, keepdims=True) + 1e-12)
    total = np.zeros((n, piece_emb.shape[1]), dtype=np.float64)
    np.add.at(total, owner, unit)
    count = np.bincount(owner, minlength=n)
    return (total / np.maximum(count, 1)[:, None]).astype(np.float32)


def encode(name):
    # Encode ALL users once, then reuse the saved file (encoding is the slow part).
    if emb_file(name).exists():
        return
    texts = list(load_full()[0])
    if DATASET != "instagram":
        texts, owner = split_pieces(texts)
        n_users = len(load_full()[2])
        print(f"  {len(texts):,} pieces (bios + tweets) for {n_users:,} users")
    if name in DECODERS:
        emb = encode_decoder_llm(ENCODERS[name], texts)
    elif name in ("e5", "e5en"):
        emb = encode_sentence_model(ENCODERS[name], texts, prefix="query: ")  # E5 expects a prefix
    else:
        emb = encode_sentence_model(ENCODERS[name], texts)
    if DATASET != "instagram":
        emb = average_by_user(emb, owner, n_users)
    np.save(emb_file(name), emb)


def make_clf(C):
    # Scaling matters for decoder LLMs: their hidden states have a few huge "outlier" dimensions.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


def tune_C(X, y, train, val):
    # Pick regularisation strength C on the VALIDATION set, never on test.
    best_C, best_f1 = None, -1
    for C in [0.001, 0.01, 0.1, 1]:
        clf = make_clf(C).fit(X[train], y[train])
        p = clf.predict_proba(X[val])[:, 1]
        f1 = f1_score(y[val], p > best_threshold(y[val], p), average="macro")
        if f1 > best_f1:
            best_C, best_f1 = C, f1
    print(f"  best C = {best_C} (val macroF1 {best_f1:.3f})")
    return best_C


def main():
    _, edge_index, y = load()
    train, val, test = get_splits(y)
    iso_test = isolated_mask(edge_index, len(y))[test]

    results = []
    for name in ENCODERS:
        print(f"\n=== {name} ===")
        try:
            encode(name)
            emb = load_embeddings(name)             # rows for the current SETTING
        except Exception as e:                  # e.g. download failed / out of memory
            print(f"  SKIPPED {name}: {type(e).__name__}: {str(e)[:200]}")
            continue
        print(f"  embeddings {emb.shape}")
        clf = make_clf(tune_C(emb, y, train, val)).fit(emb[train], y[train])
        t = best_threshold(y[val], clf.predict_proba(emb[val])[:, 1])
        p = clf.predict_proba(emb[test])[:, 1]
        results.append(evaluate(f"{name} + LR", y[test], (p > t).astype(int), p, iso_test))

    with open(RESULTS_DIR / "step3_text_only.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
