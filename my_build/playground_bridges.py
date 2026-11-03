"""
Playground: how many TwiBot-20 support users are "bridges"?

Our graph keeps only the 11,826 labelled users (prep_twibot20.py). The benchmark also ships
217,754 unlabelled support users. A support user linked to ONE labelled user is a dead end
(no text, no label, connects nothing). A support user linked to TWO OR MORE labelled users is
a bridge: it creates labelled -> support -> labelled paths our graph cannot see.

This script answers, before we change any model:
  1. Which kinds of follow/friend edges exist (labelled-labelled, labelled-support, support-support)?
  2. How many support users are bridges, and how many labelled neighbours do they have (hubs)?
  3. How many labelled users that are ISOLATED in our graph would get a neighbour through a bridge?
  4. If bridges are collapsed into direct labelled-labelled edges, how many edges do we get,
     with and without a cap on hub size?
  5. Edge homophily (same-label share) of our graph vs the collapsed bridge edges.
     Uses ALL labels, test included: a description of the data, never a model input.

Run:  python my_build/playground_bridges.py     (~2-4 min first time; edge.csv is read once, then cached)
Safe next to a running step: CPU and disk only, no GPU, reads edge.csv in small chunks.
"""
# %%
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "twibot-20"
ZIP = Path.home() / "Downloads" / "TwiBot-20-Format22.zip"
CACHE = DATA_DIR / "support_edges.npz"          # labelled <-> support edges, reused later by step 4
OUT = ROOT / "my_build" / "results" / "twibot20" / "support_bridges.json"
CHUNK = 1_000_000                               # small chunks: step 6 is using most of the RAM
HUB_CAPS = [10, 50, 200, None]                  # max labelled neighbours a bridge may have (None = no cap)
MAX_PAIRS = 60_000_000                          # skip a cap whose collapsed graph would not fit in memory

# %% 1. Who is labelled, who is support
ids = np.load(DATA_DIR / "ids.npy")
y = np.load(DATA_DIR / "y.npy")
split = np.load(DATA_DIR / "split.npy")
n_lab = len(ids)

zf = zipfile.ZipFile(ZIP)
split_csv = pd.read_csv(zf.open("Twibot-20/split.csv"))
sup_ids = split_csv.id[split_csv.split == "support"].to_numpy()
n_sup = len(sup_ids)
lab_index, sup_index = pd.Index(ids), pd.Index(sup_ids)
print(f"labelled {n_lab:,}   support {n_sup:,}   (split.csv rows {len(split_csv):,})")

# %% 2. Read follow/friend edges once; classify each by its two ends
if CACHE.exists():
    c = np.load(CACHE, allow_pickle=True)
    ls_lab, ls_sup, ls_rel = c["lab"], c["sup"], c["rel"]
    kinds = json.loads(str(c["kinds"]))
    print(f"loaded cache {CACHE.name}")
else:
    kinds = {}                                   # relation -> {"L-L": n, "L-S": n, ...}
    lab_parts, sup_parts, rel_parts = [], [], []
    for ch in pd.read_csv(zf.open("Twibot-20/edge.csv"), chunksize=CHUNK):
        ch = ch[ch.relation != "post"]           # posts link users to tweets; not part of the user graph
        if ch.empty:
            continue
        sl, tl = lab_index.get_indexer(ch.source_id), lab_index.get_indexer(ch.target_id)
        ss, ts = sup_index.get_indexer(ch.source_id), sup_index.get_indexer(ch.target_id)
        end = lambda l, s: np.where(l >= 0, "L", np.where(s >= 0, "S", "?"))   # ? = not a known user
        kind = pd.Series(end(sl, ss)) + "-" + pd.Series(end(tl, ts))
        for (r, k), n in pd.crosstab(ch.relation.to_numpy(), kind.to_numpy()).stack().items():
            kinds.setdefault(r, {})[k] = kinds.get(r, {}).get(k, 0) + int(n)
        # keep labelled <-> support edges in either direction, as (labelled idx, support idx)
        a = (sl >= 0) & (ts >= 0)
        b = (ss >= 0) & (tl >= 0)
        lab_parts += [sl[a], tl[b]]
        sup_parts += [ts[a], ss[b]]
        rel_parts += [ch.relation.to_numpy()[a], ch.relation.to_numpy()[b]]
        print(f"  ... {sum(map(len, lab_parts)):,} labelled-support edges so far", flush=True)
    ls_lab, ls_sup = np.concatenate(lab_parts), np.concatenate(sup_parts)
    ls_rel = np.concatenate(rel_parts).astype(str)
    np.savez(CACHE, lab=ls_lab, sup=ls_sup, rel=ls_rel, kinds=json.dumps(kinds))

print("\nEdge kinds (L = labelled, S = support, ? = unknown id), counted as stored (directed):")
print(pd.DataFrame(kinds).fillna(0).astype(int).T.to_string())

# %% 3. Bridges: support users with >= 2 distinct labelled neighbours (any relation, either direction)
B = sp.csr_matrix((np.ones(len(ls_lab), dtype=np.float32), (ls_lab, ls_sup)), shape=(n_lab, n_sup))
B.data[:] = 1                                    # friend + follow to the same user summed to 2 -> back to 0/1
k = np.asarray(B.sum(0)).ravel().astype(int)     # labelled neighbours per support user

bins = [0, 1, 2, 3, 6, 21, 101, 1001, np.inf]
names = ["0", "1", "2", "3-5", "6-20", "21-100", "101-1000", ">1000"]
dist = pd.cut(k, bins=bins, right=False, labels=names).value_counts().reindex(names)
bridge = k >= 2
print(f"\nSupport users by number of labelled neighbours:\n{dist.to_string()}")
print(f"bridges (>= 2 labelled neighbours): {bridge.sum():,} of {n_sup:,} ({bridge.mean():.1%})")
print(f"largest hub: {k.max():,} labelled neighbours")

# %% 4. Who gains a neighbour? Isolated in our graph, but next to a bridge
E = np.load(DATA_DIR / "edge_index.npy")
A = sp.csr_matrix((np.ones(len(E), dtype=np.float32), (E[:, 0], E[:, 1])), shape=(n_lab, n_lab))
deg = np.asarray(A.sum(1)).ravel()
near_bridge = np.asarray(B[:, bridge].sum(1)).ravel() > 0
rows = []
for name, m in [("all", np.ones(n_lab, bool)), ("train", split == "train"),
                ("val", split == "val"), ("test", split == "test")]:
    iso = m & (deg == 0)
    rows.append(dict(split=name, users=int(m.sum()), isolated_now=int(iso.sum()),
                     isolated_but_next_to_bridge=int((iso & near_bridge).sum()),
                     any_support_neighbour=int((m & (np.asarray(B.sum(1)).ravel() > 0)).sum())))
gain = pd.DataFrame(rows).set_index("split")
print(f"\nIsolated labelled users who would gain a 2-hop neighbour through a bridge:\n{gain.to_string()}")

# %% 5. Collapse bridges into labelled-labelled edges, per hub cap; edge homophily of each graph
def homophily(M):
    # Share of edges (undirected, each counted once) whose ends have the same label.
    M = sp.triu(M, k=1).tocoo()
    return float((y[M.row] == y[M.col]).mean()) if M.nnz else float("nan"), int(M.nnz)

p_bot = y.mean()
baseline = p_bot ** 2 + (1 - p_bot) ** 2       # same-label share if edges ignored labels
h_now, e_now = homophily(A)
print(f"\nEdge homophily: random-edge baseline {baseline:.3f}   our graph {h_now:.3f} ({e_now:,} edges)")

collapsed = []
for cap in HUB_CAPS:
    keep = bridge & (k <= (cap or np.inf))
    pairs = int((k[keep] * (k[keep] - 1) // 2).sum())
    row = dict(cap="none" if cap is None else cap, bridges_used=int(keep.sum()), pairs_before_dedup=pairs)
    if pairs > MAX_PAIRS:
        print(f"  cap {row['cap']}: {pairs:,} pairs, too many to build here -- skipped")
        collapsed.append(row)
        continue
    Bk = B[:, keep]
    A2 = (Bk @ Bk.T).tocsr()
    A2.setdiag(0)
    A2.eliminate_zeros()
    A2.data[:] = 1
    h2, e2 = homophily(A2)
    union = ((A + A2) > 0).astype(np.float32)
    hu, eu = homophily(union)
    new = sp.triu(A2 - A2.multiply(A), k=1).nnz
    iso_after = int((np.asarray(union.sum(1)).ravel() == 0).sum())
    row.update(bridge_edges=e2, new_edges_not_in_our_graph=int(new), homophily_bridge_edges=round(h2, 3),
               union_edges=eu, homophily_union=round(hu, 3), isolated_after=iso_after)
    collapsed.append(row)
coll = pd.DataFrame(collapsed).set_index("cap")
print(f"\nCollapsed bridge edges by hub cap (isolated now: {int((deg == 0).sum()):,}):\n{coll.to_string()}")

# %% 6. Save
OUT.write_text(json.dumps(dict(
    labelled=n_lab, support=n_sup, edge_kinds=kinds,
    labelled_neighbours_per_support=dist.astype(int).to_dict(), bridges=int(bridge.sum()), largest_hub=int(k.max()),
    isolated_gain=gain.reset_index().to_dict("records"),
    homophily=dict(random_baseline=round(baseline, 3), our_graph=round(h_now, 3), our_graph_edges=e_now),
    collapsed=coll.reset_index().to_dict("records")), indent=2, default=str))
print(f"\nsaved {OUT.relative_to(ROOT)}")

# %% THINK
# 1. Are there any S-S edges? The TwiBot-20 FAQ says support users only link to labelled ones.
# 2. What fraction of support users are dead ends (exactly 1 labelled neighbour)? Why does that
#    make "a random sample of support users" a poor choice?
# 3. Compare homophily of the bridge edges with our graph and the random baseline. If bridge
#    edges are MORE homophilous, what does that say about finding 2 ("GCN hurts under heterophily")?
# 4. How does the hub cap trade edges against homophily? Which cap would you use for step 4, and why?
