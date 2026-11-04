"""
TwiBot-22 graph and row-order checks before any model (checklist E4, E5, and the row-order test).
CPU only, no torch, ~2 min. Labels are read for TRAIN and VAL users only; no test label is used.

1. ROW ORDER: every per-user artefact must be in label.csv order (rows are joined by position).
   Checks: users.parquet ids (after the same reindex the feature script uses), split_ours.npy length,
   features_full.npz rows, MiniLM / multilingual-E5 bio embeddings rows, and that each embedding row
   for an empty bio is the zero vector (twibot22_embed_bios.py writes zeros for empty bios), which
   would fail if rows were shuffled.
2. E4, crawl design per relation (val users): AUROC of log in-degree and log out-degree for "bot",
   share of users with any in / out edge by class, max out-degree (a cap would show the crawl limit),
   and reciprocity (share of directed edges u->v whose reverse v->u also exists).
3. E5, train-to-val label leakage through edges: for val users with >= 1 neighbour (any relation,
   either direction), the share of their TRAIN neighbours that have the same label as them, against
   the chance level from the train bot share. If neighbours "give away" the label, a GNN would score
   well for a reason unrelated to the user. Reported for bots and humans separately.

Output: results/twibot22/graph_checks.json
Run:    python my_build/twibot22_graph_checks.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "twibot-22"
RAW = ROOT.parent / "data" / "twibot-22-raw"
RES = ROOT / "results" / "twibot22"
RELATIONS = ["following", "followers", "mention", "retweet", "reply", "quote", "like"]


def row_order(lab):
    n, out = len(lab), {}
    users = pd.read_parquet(DATA / "users.parquet", columns=["id", "description"])
    users = users.set_index("id").loc[lab.id].reset_index()          # the reindex used by the feature scripts
    out["users_parquet_reindexed_to_label_order"] = bool((users.id.values == lab.id.values).all())
    out["split_ours_len"] = int(len(np.load(DATA / "split_ours.npy"))) == n
    with np.load(DATA / "features_full.npz") as f:
        out["features_full_rows"] = int(f["X"].shape[0]) == n
    empty = (users.description.fillna("").str.strip() == "").values
    for enc in ("minilm", "e5"):
        p = RES / f"emb_bio_{enc}_all.npy"
        if not p.exists():
            out[f"emb_{enc}"] = "missing"
            continue
        E = np.load(p, mmap_mode="r")
        zero = np.zeros(n, bool)
        for i in range(0, n, 100_000):                                # in chunks: 1M x 768 float16
            zero[i:i + 100_000] = ~np.asarray(E[i:i + 100_000]).any(1)
        out[f"emb_{enc}"] = dict(rows_ok=int(E.shape[0]) == n,
                                 zero_rows_equal_empty_bios=bool((zero == empty).all()),
                                 zero_rows=int(zero.sum()), empty_bios=int(empty.sum()))
    return out


def crawl_design(rels, y, va, n):
    out = {}
    for r in RELATIONS:
        e = rels[r]
        outd, ind = np.bincount(e[0], minlength=n), np.bincount(e[1], minlength=n)
        A = sp.csr_matrix((np.ones(e.shape[1], np.int8), (e[0], e[1])), shape=(n, n))
        A.data[:] = 1
        recip = float(A.multiply(A.T).sum() / max(A.sum(), 1))
        yv = y[va]
        out[r] = dict(
            val_auroc_log_in_degree=round(float(roc_auc_score(yv, np.log1p(ind[va]))), 3),
            val_auroc_log_out_degree=round(float(roc_auc_score(yv, np.log1p(outd[va]))), 3),
            val_has_in_edge_bot_human=[round(float((ind[va][yv == 1] > 0).mean()), 3), round(float((ind[va][yv == 0] > 0).mean()), 3)],
            val_has_out_edge_bot_human=[round(float((outd[va][yv == 1] > 0).mean()), 3), round(float((outd[va][yv == 0] > 0).mean()), 3)],
            max_out_degree=int(outd.max()), max_in_degree=int(ind.max()),
            users_at_max_out_degree=int((outd == outd.max()).sum()),
            reciprocity=round(recip, 4))
    return out


def neighbour_leakage(rels, y, tr, va, n):
    U = sp.csr_matrix((n, n), dtype=np.float32)
    for r in RELATIONS:
        e = rels[r]
        A = sp.csr_matrix((np.ones(e.shape[1], np.float32), (e[0], e[1])), shape=(n, n))
        U = U + A + A.T
    U.data[:] = 1
    Ut = U[:, np.where(tr)[0]]                                       # neighbours restricted to TRAIN users
    ytr = y[tr].astype(np.float32)
    n_tr_nb = np.asarray(Ut.sum(1)).ravel()
    bot_tr_nb = Ut @ ytr
    p_bot = float(ytr.mean())
    out = dict(train_bot_share=round(p_bot, 4))
    for name, cls in (("bot", 1), ("human", 0)):
        m = va & (y == cls) & (n_tr_nb > 0)
        same = bot_tr_nb[m] / n_tr_nb[m] if cls == 1 else 1 - bot_tr_nb[m] / n_tr_nb[m]
        out[f"val_{name}"] = dict(
            users_with_train_neighbour=int(m.sum()),
            share_of_val_class_with_train_neighbour=round(float(m.sum() / (va & (y == cls)).sum()), 4),
            mean_same_label_share=round(float(same.mean()), 4),
            chance_same_label_share=round(p_bot if cls == 1 else 1 - p_bot, 4))
    return out


def main():
    lab = pd.read_csv(RAW / "label.csv")
    n = len(lab)
    y = (lab.label.values == "bot").astype(int)
    split = np.load(DATA / "split_ours.npy")
    tr, va = split == "train", split == "val"
    rels = np.load(DATA / "relations_uu.npz")
    res = dict(row_order=row_order(lab), crawl_design_val=crawl_design(rels, y, va, n),
               neighbour_leakage=neighbour_leakage(rels, y, tr, va, n))
    (RES / "graph_checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
