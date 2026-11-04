"""
Shared helpers used by EVERY step from now on: load data, fixed splits, one evaluate().

Why a shared file?
  - Every model (TF-IDF, MiniLM, GCN, SAGE, LLM...) must be trained/tested on the SAME
    nodes and scored with the SAME metrics, or the comparison in your report is meaningless.
  - Swapping dataset later (e.g. TwiBot-20) = rewrite load() only.

Two experimental settings, picked with an environment variable:
  SETTING=A  natural ratio (7,224 normal : 4,115 commercial, ~64/36), all edges   [default]
  SETTING=B  FLAG-style 1:10: keep all normal users, randomly keep ~722 commercial users,
             and only the edges between the users that remain (induced subgraph).
  e.g.  SETTING=B python my_build/step4_gnn.py

Commercial accounts (class 1) play the role of the "fraud" class in both settings.
Embeddings and LLM summaries are computed ONCE on all users (results/) and shared;
everything setting-specific (splits, metrics) lives in results/A or results/B.

Two datasets, picked the same way:
  DATASET=instagram  [default]  class 1 = commercial; our stratified 60/20/20 split; results/
  DATASET=twibot20             class 1 = bot; the OFFICIAL TwiBot-20 split; results/twibot20/
                               text = bio + tweets only (no profile metadata, see report 4.6)
  e.g.  DATASET=twibot20 SETTING=B python my_build/step4_gnn.py
"""
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
DATASET = os.environ.get("DATASET", "instagram")
assert DATASET in ("instagram", "twibot20", "twibot22"), "DATASET must be instagram, twibot20 or twibot22"
# twibot22: the LLM-step folder built by twibot22_llm_prep.py (100k sample + hubs, sealed users excluded)
DATA_DIR = ROOT / "data" / {"instagram": "instagram", "twibot20": "twibot-20", "twibot22": "twibot-22-llm"}[DATASET]
TWITTER = DATASET in ("twibot20", "twibot22")       # bot or human; text = bio + tweets, one per line
SETTING = os.environ.get("SETTING", "A")
assert SETTING in ("A", "B"), "SETTING must be A or B"
MINORITY_RATIO = 0.1                                  # setting B: class 1 = 10% of class 0
TEXT_MAX_TOKENS = 128          # per text piece: one bio (Instagram), or one bio / one tweet (TwiBot-20)
SHARED_DIR = Path(__file__).resolve().parent / "results"
if DATASET != "instagram":
    SHARED_DIR = SHARED_DIR / DATASET                 # instagram keeps its original paths
RESULTS_DIR = SHARED_DIR / SETTING
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_FILE = RESULTS_DIR / "splits.npz"
SEED = 42


def load_full():
    # TODO 1: same as step 1, but return x_text as a str array (.astype(str)).
    # Return x_text, edge_index (2, E), y   -- ALL users, ignoring SETTING
    x_text = np.load(DATA_DIR / "x_text.npy").astype(str)
    y = np.load(DATA_DIR / "y.npy")
    edge_index = np.load(DATA_DIR / "edge_index.npy").T
    return x_text, edge_index, y


def original_ids():
    # Which of the original users are in this setting (their ids in the full dataset).
    y = np.load(DATA_DIR / "y.npy")
    if SETTING == "A":
        return np.arange(len(y))
    normal, commercial = np.where(y == 0)[0], np.where(y == 1)[0]
    n_keep = round(len(normal) * MINORITY_RATIO)
    kept = np.random.default_rng(SEED).choice(commercial, n_keep, replace=False)
    return np.sort(np.concatenate([normal, kept]))


def load():
    # The data for the current SETTING. Nodes are renumbered 0..n-1.
    x_text, edge_index, y = load_full()
    if SETTING == "A":
        return x_text, edge_index, y
    keep = original_ids()
    new_id = np.full(len(y), -1)
    new_id[keep] = np.arange(len(keep))
    both_kept = (new_id[edge_index[0]] >= 0) & (new_id[edge_index[1]] >= 0)
    return x_text[keep], new_id[edge_index[:, both_kept]], y[keep]


def missing_text_flags():
    # TwiBot-20: 0/1 columns "no bio" and "no tweets" (taken from the text, not from metadata).
    x, bio = np.load(DATA_DIR / "x_text.npy"), np.load(DATA_DIR / "x_bio.npy")
    no_bio = np.char.strip(bio) == ""
    no_tweets = np.char.strip(x) == np.char.strip(bio)
    return np.stack([no_bio, no_tweets], 1).astype(np.float32)


def load_embeddings(name):
    # Cached embeddings are for ALL users; pick the rows for this setting.
    emb = np.load(SHARED_DIR / f"emb_{name}.npy")
    if TWITTER:
        emb = np.hstack([emb, missing_text_flags()])   # zero vector + flags, nobody dropped
    return emb[original_ids()]


def make_splits(y, seed=SEED):
    # TODO 2: stratified 60 / 20 / 20 split of node indices into train / val / test.
    # Hint: call train_test_split twice.
    #   1st: all indices -> train (60%) and rest (40%), stratify=y
    #   2nd: rest -> val (50%) and test (50%), stratify=y[rest]
    # Use random_state=seed both times.
    # Then np.savez(SPLIT_FILE, train=..., val=..., test=...)
    if TWITTER:
        # TwiBot-20: the official split; TwiBot-22: split.npy of twibot22_llm_prep.py (sealed users
        # are marked "sealed" and so never enter train, val or test). Restricted to the users kept in this setting.
        s = np.load(DATA_DIR / "split.npy")[original_ids()]
        train, val, test = (np.where(s == part)[0] for part in ("train", "val", "test"))
        np.savez(SPLIT_FILE, train=train, val=val, test=test)
        return train, val, test
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, train_size=0.6, stratify=y, random_state=seed)
    val, test = train_test_split(rest, train_size=0.5, stratify=y[rest], random_state=seed)
    np.savez(SPLIT_FILE, train=train, val=val, test=test)
    return train, val, test


def get_splits(y):
    # TODO 3: if SPLIT_FILE exists, load it and return (train, val, test).
    # Otherwise call make_splits(y) first.
    # Why? Splits are made ONCE and reused by every later step.
    if not SPLIT_FILE.exists():
        return make_splits(y)
    s = np.load(SPLIT_FILE)
    return s["train"], s["val"], s["test"]


def isolated_mask(edge_index, n):
    # TODO 4: boolean array, True for nodes with degree 0. (You did this in step 1.)
    deg = np.bincount(edge_index[0], minlength=n)
    return deg == 0


def best_threshold(y_true, prob):
    # Scores are not calibrated (class weights and LLM logits shift them), and the default
    # 0.5 cut-off is badly placed when class 1 is rare (setting B). So every model picks the
    # cut-off that maximises macro-F1 on the VALIDATION set, then applies it unchanged to test.
    grid = np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 99)))
    return max(grid, key=lambda t: f1_score(y_true, prob > t, average="macro"))


def evaluate(name, y_true, y_pred, y_score=None, iso=None):
    """
    y_true, y_pred : labels for the test nodes
    y_score        : P(class 1) for each test node (needed for AUROC), optional
    iso            : isolated_mask restricted to the test nodes, optional
    Returns a dict of metrics and prints one neat line.
    """
    # TODO 5: compute
    #   acc       = accuracy
    #   macro_f1  = F1 averaged over both classes        <-- your MAIN metric
    #   f1_class1 = F1 of class 1 only                   <-- the minority / "suspicious" class
    #   auroc     = roc_auc_score(y_true, y_score)        (only if y_score is given)
    #   if iso is given: macro_f1 on connected nodes (~iso) and on isolated nodes (iso) separately
    # Print them on one line and return them in a dict.
    #
    # THINK: why is accuracy a bad main metric here? (Hint: what accuracy does
    #        "always predict 0" get?)
    m = {
        "model": name,
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "f1_class1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        m["auroc"] = roc_auc_score(y_true, y_score)
        # PR-AUC: precision/recall on the minority class. A random model scores the class-1
        # share (0.36 in A, ~0.09 in B), so it is much harsher than AUROC when fraud is rare.
        m["pr_auc"] = average_precision_score(y_true, y_score)
    if iso is not None:
        m["macro_f1_connected"] = f1_score(y_true[~iso], y_pred[~iso], average="macro")
        m["macro_f1_isolated"] = f1_score(y_true[iso], y_pred[iso], average="macro")

    line = (f"{name:18s} acc {m['acc']:.3f}  macroF1 {m['macro_f1']:.3f}  "
            f"F1_1 {m['f1_class1']:.3f}  AUROC {m.get('auroc', float('nan')):.3f}  "
            f"PR-AUC {m.get('pr_auc', float('nan')):.3f}")
    if iso is not None:
        line += f"  | conn {m['macro_f1_connected']:.3f}  iso {m['macro_f1_isolated']:.3f}"
    print(line)
    return {k: (round(float(v), 4) if not isinstance(v, str) else v) for k, v in m.items()}
