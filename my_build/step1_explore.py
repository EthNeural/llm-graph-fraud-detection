"""
Step 1: Get to know the Instagram dataset, then measure how heterophilous it is.

Run:  python my_build/step1_explore.py
"""
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "instagram"
import matplotlib.pyplot as plt


# ---------------- Part A: explore ----------------
def load():
    x_text = np.load(DATA_DIR/"x_text.npy")
    y = np.load(DATA_DIR/"y.npy")
    edge_index = np.load(DATA_DIR/"edge_index.npy")

    # TODO 2: edge_index is stored as (E, 2). Transpose it to (2, E), the PyG convention.
    # Return x_text, edge_index, y
    edge_index = edge_index.T   
    return x_text,edge_index,y



def explore(x_text, edge_index, y):
    n = len(y)                                   # number of users (nodes)
    E = edge_index.shape[1]                      # number of edge columns
    print(f"Nodes: {n:,}   Edge rows: {E:,}")

    # --- Labels ---
    labels, counts = np.unique(y, return_counts=True)
    for lab, c in zip(labels, counts):
        print(f"Label {lab}: {c:,} users ({c/n:.1%})")

    # --- Degree: how many connections each user has ---
    deg = np.bincount(edge_index[0], minlength=n)    # count appearances as source
    print(f"Median degree: {np.median(deg):.0f}   Max: {deg.max()}   "
          f"Isolated (degree 0): {(deg == 0).sum():,}")

    # --- Text stats ---
    texts = x_text.astype(str)                   # make sure numpy treats them as strings
    lens = np.char.str_len(texts)                # length of each text in characters
    print(f"Text length: mean {lens.mean():.0f}, median {np.median(lens):.0f}, max {lens.max()}")
    print(f"Empty texts: {(np.char.strip(texts) == '').sum():,}")

    # --- 3 random examples from each class ---
    rng = np.random.default_rng(42)              # fixed seed = same examples every run
    for lab in labels:
        idx = rng.choice(np.where(y == lab)[0], 3, replace=False)
        print(f"\n--- Label {lab} examples ---")
        for i in idx:
            print(f"[user {i}] {texts[i][:200]}")   # first 200 chars

    # --- TODO 4: is the graph undirected? ---
    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    missing = sum((b, a) not in pairs for a, b in pairs)
    print(f"\nUnique edges: {len(pairs):,}   Missing reverse edges: {missing:,}")
    print("Undirected ✅" if missing == 0 else "Not fully undirected ❌ (some edges only go one way)")

# ---------------- Part B: homophily ----------------
def edge_homophily(edge_index, y):
    # TODO 5: fraction of edges whose two ends have the same label.
    src, dst = edge_index
    same = y[src] == y[dst]                      # True where both ends share a label
    return same.mean()                           # mean of booleans = fraction True


def class_homophily(edge_index, y):
    # TODO 6: h_k for each class k: of all edges leaving a class-k node,
    # what fraction land on another class-k node?
    # (The edges are already stored both ways, so no need to stack them.)
    src, dst = edge_index
    h_k = {}
    for k in np.unique(y):
        out = y[src] == k                        # edges leaving a class-k node
        h_k[int(k)] = (y[dst][out] == k).mean()  # ...that land on class k
    return h_k


def class_insensitive_homophily(h_k, y):
    # TODO 7: Lim et al. (2021):  h = 1/(C-1) * sum_k max(0, h_k - p_k)
    # p_k = fraction of ALL nodes in class k. A random graph gives h_k ≈ p_k, so this
    # only counts homophily ABOVE chance. 0 = no better than random, 1 = perfectly homophilous.
    C = len(h_k)
    total = 0.0
    for k, h in h_k.items():
        p_k = (y == k).mean()                    # chance level for class k
        total += max(0.0, h - p_k)
    return total / (C - 1)


def node_homophily(edge_index, y):
    # TODO 8: for each node, the fraction of its neighbours with the same label.
    # Hint: np.bincount(src, weights=same.astype(float)) / degree. Skip isolated nodes.
    # Print the mean and the share with h == 0 for each class.
    n = len(y)
    src, dst = edge_index
    same = y[src] == y[dst]
    deg = np.bincount(src, minlength=n)
    n_same = np.bincount(src, weights=same.astype(float), minlength=n)
    has_edges = deg > 0                          # isolated nodes would divide by 0
    h = np.full(n, np.nan)
    h[has_edges] = n_same[has_edges] / deg[has_edges]

    # Store distributions for the plot
    
    h_classes = {}

    for k in np.unique(y):
        h_cls = h[(y == k) & has_edges]
        h_classes[k] = h_cls

        print(f"  class {k}: mean {h_cls.mean():.3f}, h=0 share {(h_cls == 0).mean():.3f}")
    # TODO 10 (bonus): save overlapping histograms of node homophily per class to
    # my_build/results/node_homophily.png
    out_dir = Path("my_build/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    colors = {0: "#1f77b4", 1: "#ff7f0e"} # Distinct colors for class 0 and 1
    
    for k, h_val in h_classes.items():
        plt.hist(
            h_val, 
            bins=20, 
            alpha=0.5,             # Makes them overlapping/transparent
            label=f"Class {k}", 
            color=colors.get(k),
            density=True           # Normalized to account for class imbalance
        )
        
    plt.title("Distribution of Node Homophily per Class")
    plt.xlabel("Node Homophily Score ($h_i$)")
    plt.ylabel("Density")
    plt.legend(loc="upper left")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(out_dir / "node_homophily.png", dpi=300, bbox_inches="tight")
    plt.close()
        
    return h


def main():
    x_text, edge_index ,y = load()
    explore(x_text, edge_index, y)

    print(edge_index.shape)        # should print (2, 144010)
    print(edge_index[:, :5])       # first 5 edges, now as 2 rows
    
    print("\n=== Homophily ===")
    print(f"Edge homophily ............. {edge_homophily(edge_index, y):.3f}")
    h_k = class_homophily(edge_index, y)
    for k, h in h_k.items():
        print(f"h_k class {k}: {h:.3f}")
    print(f"Class-insensitive .......... {class_insensitive_homophily(h_k, y):.3f}")
    print("Node homophily:")
    node_homophily(edge_index, y)
   

if __name__ == "__main__":
    main()
    

# ---------------------------------------------------------------------------
# CHECK YOUR ANSWERS (to 3 decimals):
#   nodes 11339 | edge rows 144010 | undirected: yes (100% reversed) | self-loops 0
#   labels  0: 7224 (63.7%)   1: 4115 (36.3%)
#   isolated nodes 1838 | median degree 12
#   text: median 119 chars, max 523, empty 99
#   edge homophily .............. 0.586
#   h_k   class 0: 0.686   class 1: 0.392
#   class-insensitive ........... 0.078   <-- almost no better than random!
#   node homophily  class 0: mean 0.692, h=0 share 0.048
#                   class 1: mean 0.389, h=0 share 0.155
#
# THINK ABOUT (write answers in your study guide):
#   - Class 1 is 36.3% of nodes and its h_k is 0.392. What does that say about whether
#     the graph "knows" anything about class 1?
#   - Why is 0.586 edge homophily misleading here?
#   - 1,838 users have no edges. What will a GNN predict for them, and what signal is left?
#   - Given all this, what do you predict: will GCN beat a text-only model? Write it down
#     NOW, before step 4, and check yourself later.
# ---------------------------------------------------------------------------
