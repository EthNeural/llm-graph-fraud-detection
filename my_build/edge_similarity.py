"""
Do real edges join users who WRITE alike, whatever their label?

GCN loses less on the real TwiBot-20 A edges than on randomly rewired ones (step 4, GRAPH=
shuffled), yet label informativeness is only 0.003. One explanation: linked users have similar
text, so averaging them blurs a user's own vector less. This compares the mean cosine
similarity of Qwen2.5-1.5B embeddings along real edges, along degree-preserving shuffled edges
and between random pairs, split by the labels at the two ends (bot-bot, bot-human, human-human).
Uses all labels: a description of the data, never a model input.

Run:  DATASET=twibot20 SETTING=A python my_build/edge_similarity.py     (CPU, seconds)
"""
import json

import numpy as np

from common import load, load_embeddings, RESULTS_DIR
from step4_gnn import shuffle_edges


def by_pair(emb, y, u, v):
    cos = (emb[u] * emb[v]).sum(1)
    out = {"all": round(float(cos.mean()), 4)}
    for name, (a, b) in {"bot-bot": (1, 1), "bot-human": (1, 0), "human-human": (0, 0)}.items():
        m = ((y[u] == a) & (y[v] == b)) | ((y[u] == b) & (y[v] == a))
        out[name] = round(float(cos[m].mean()), 4) if m.any() else None
        out[name + " edges"] = int(m.sum())
    return out


def main():
    _, e, y = load()
    emb = load_embeddings("qwen")[:, :-2].astype(np.float64)       # drop the 0/1 "no text" flags
    # Centre first: mean-pooled LLM vectors share one large common direction (anisotropy),
    # which pushes every raw cosine to ~0.9 and hides the differences.
    emb -= emb.mean(0)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    once = lambda ei: (ei[0][ei[0] < ei[1]], ei[1][ei[0] < ei[1]])
    rng = np.random.default_rng(0)
    rand = rng.integers(0, len(y), (2, 200_000))
    rand = rand[:, rand[0] != rand[1]]
    res = {"real edges": by_pair(emb, y, *once(e)),
           "shuffled edges (mean of 5 rewirings)": None,
           "random pairs": by_pair(emb, y, rand[0], rand[1])}
    sh = [by_pair(emb, y, *once(shuffle_edges(e, len(y), s))) for s in range(5)]
    res["shuffled edges (mean of 5 rewirings)"] = {k: (round(float(np.mean([d[k] for d in sh])), 4)
                                                        if sh[0][k] is not None else None) for k in sh[0]}
    for k, d in res.items():
        print(f"{k:38s} " + "  ".join(f"{n} {d[n]}" for n in ["all", "bot-bot", "bot-human", "human-human"]))
    json.dump(res, open(RESULTS_DIR / "edge_similarity.json", "w"), indent=2)
    print(f"saved {RESULTS_DIR / 'edge_similarity.json'}")


if __name__ == "__main__":
    main()
