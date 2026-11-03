"""
How much is "which link types does this user have?" worth on its own?

The step 6 dgp prompt writes "none" for every metapath a user has no neighbours in, so the
LLM can see whether a user is isolated, whatever the summaries say. In setting B isolation is
tied to the label (41.8% of bots isolated vs 27.6% of humans; about equal in A). This checks
the size of that cue BEFORE the B dgp result is read: LR on Qwen2.5-1.5B embeddings, with and
without three 0/1 flags (has a friend, has a follow link, has a friend-of-friend), and the
flags alone. Same splits, validation-chosen C and cut-off, full test set.

Run:  DATASET=twibot20 SETTING=B python my_build/isolation_cue.py     (CPU, about a minute)
"""
import json

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from common import load, load_embeddings, get_splits, best_threshold, RESULTS_DIR
from step3_embed import make_clf, tune_C
from step6_dgp import typed_relations, metapath_adjacency, METAPATHS


def main():
    _, _, y = load()
    n = len(y)
    train, val, test = get_splits(y)
    rel = typed_relations()
    flags = np.stack([np.asarray(metapath_adjacency(rel, path, n).sum(1)).ravel() > 0
                      for path in METAPATHS.values()], 1).astype(np.float32)
    emb = load_embeddings("qwen").astype(np.float32)
    out = {"flag share, bots": flags[y == 1].mean(0).round(3).tolist(),
           "flag share, humans": flags[y == 0].mean(0).round(3).tolist(), "metapaths": list(METAPATHS)}
    for name, X in [("text", emb), ("text + link flags", np.hstack([emb, flags]))]:
        clf = make_clf(tune_C(X, y, train, val)).fit(X[train], y[train])
        t = best_threshold(y[val], clf.predict_proba(X[val])[:, 1])
        p = clf.predict_proba(X[test])[:, 1]
        out[name] = dict(macro_f1=round(f1_score(y[test], p > t, average="macro"), 4),
                         auroc=round(roc_auc_score(y[test], p), 4))
    clf = LogisticRegression(class_weight="balanced").fit(flags[train], y[train])
    p = clf.predict_proba(flags[test])[:, 1]
    t = best_threshold(y[val], clf.predict_proba(flags[val])[:, 1])
    out["link flags only"] = dict(macro_f1=round(f1_score(y[test], p > t, average="macro"), 4),
                                  auroc=round(roc_auc_score(y[test], p), 4))
    print(json.dumps(out, indent=1))
    json.dump(out, open(RESULTS_DIR / "isolation_cue.json", "w"), indent=2)


if __name__ == "__main__":
    main()
