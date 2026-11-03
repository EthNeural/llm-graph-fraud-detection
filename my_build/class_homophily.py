"""
Do minority nodes "camouflage" among majority neighbours? FLAG's Figure 5 test, on our graphs.

FLAG (Yang et al., KDD 2025, Sec. 4.1.1, Fig. 5) checks that its fraud datasets show
neighbourhood camouflage: per-node homophily (share of a node's neighbours with the same
label) is LOW for the minority class and HIGH for the majority class. We compute the same
histogram (10 bins, share of each class's connected nodes per bin) for TwiBot-20 A/B and
Instagram A/B. Isolated nodes have no homophily and are left out (their count is reported).
Uses all labels: a description of the data, never a model input.

Run:  python my_build/class_homophily.py      (CPU, seconds)
Writes results/class_homophily.json and figures/class_homophily.png
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BINS = np.linspace(0, 1, 11)
GRAPHS = [("twibot20", "A", "TwiBot-20 A (natural)"), ("twibot20", "B", "TwiBot-20 B (1:10)"),
          ("instagram", "A", "Instagram A (natural)"), ("instagram", "B", "Instagram B (1:10)")]


def one_setting():
    sys.path.insert(0, str(ROOT))
    from common import load
    _, e, y = load()
    n = len(y)
    src, dst = e
    deg = np.bincount(src, minlength=n)
    same = np.bincount(src, weights=(y[src] == y[dst]).astype(float), minlength=n)
    conn = deg > 0
    h = same[conn] / deg[conn]
    yc = y[conn]
    # Chance level: with random edges, a class-k node's neighbour is class k with probability
    # equal to class k's share of all edge ends.
    ends = np.bincount(np.concatenate([y[src], y[dst]]), minlength=2) / (2 * len(src))
    out = {}
    for k, name in [(1, "minority"), (0, "majority")]:
        hk = h[yc == k]
        counts = np.histogram(hk, bins=BINS)[0]
        out[name] = dict(share_per_bin=(counts / len(hk)).round(4).tolist(), connected=int(len(hk)),
                         isolated=int(((~conn) & (y == k)).sum()), mean=round(float(hk.mean()), 4),
                         median=round(float(np.median(hk)), 4), chance=round(float(ends[k]), 4))
    out["base_rate_minority"] = round(float(y.mean()), 4)
    print(json.dumps(out))


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    maj, mino = "#2a78d6", "#eb6834"                      # validated slots 1-2 (light mode)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": grid,
                         "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2})
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True, facecolor=surface)
    centers = (BINS[:-1] + BINS[1:]) / 2
    w = 0.042                                             # two thin bars per 0.1 bin, a gap between
    for ax, (key, title) in zip(axes.ravel(), [(f"{d} {s}", t) for d, s, t in GRAPHS]):
        r = results[key]
        minority_name = "bots" if key.startswith("twibot") else "commercial"
        majority_name = "humans" if key.startswith("twibot") else "personal"
        ax.set_facecolor(surface)
        ax.bar(centers - w / 2 - 0.002, r["majority"]["share_per_bin"], w, color=maj, label=majority_name)
        ax.bar(centers + w / 2 + 0.002, r["minority"]["share_per_bin"], w, color=mino, label=minority_name)
        for k, c, nm in [("majority", maj, majority_name), ("minority", mino, minority_name)]:
            ax.axvline(r[k]["mean"], ymax=0.70, color=c, lw=1.2, ls=(0, (3, 2)))   # stops below the labels
        top = max(max(r["majority"]["share_per_bin"]), max(r["minority"]["share_per_bin"]))
        ax.set_ylim(0, top * 1.42)
        ax.set_title(title, loc="left", fontsize=10, color=ink, fontweight="600")
        ax.text(0.01, 0.97, f"mean   {majority_name} {r['majority']['mean']:.2f}   {minority_name} {r['minority']['mean']:.2f}\n"
                f"chance {majority_name} {r['majority']['chance']:.2f}   {minority_name} {r['minority']['chance']:.2f}",
                transform=ax.transAxes, va="top", fontsize=8, color=ink2, family="monospace")
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.set_xlim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("per-node homophily (share of neighbours with the same label)")
    for ax in axes[:, 0]:
        ax.set_ylabel("share of the class's connected nodes")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=maj, label="majority class (TwiBot-20: humans · Instagram: personal)"),
                        Patch(color=mino, label="minority class (TwiBot-20: bots · Instagram: commercial)")],
               loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=2, frameon=False, fontsize=8.5, labelcolor=ink2)
    fig.suptitle("The minority's low homophily in the 1:10 settings is what chance predicts",
                 x=0.01, ha="left", fontsize=11, color=ink, fontweight="600")
    fig.text(0.01, 0.005, "Dashed lines: class means. Chance: the class's share of all edge ends (random edges). "
             "Isolated nodes excluded. Same test as FLAG (KDD 2025), Fig. 5.", fontsize=7.5, color=ink2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(path, dpi=200, facecolor=surface)


def main():
    if os.environ.get("_CH_CHILD"):
        return one_setting()
    results = {}
    for d, s, _ in GRAPHS:
        env = dict(os.environ, DATASET=d, SETTING=s, _CH_CHILD="1")
        res = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=True)
        results[f"{d} {s}"] = json.loads(res.stdout.strip().splitlines()[-1])
    for k, r in results.items():
        print(f"{k:12s} minority mean {r['minority']['mean']:.3f} (median {r['minority']['median']:.3f}), "
              f"majority mean {r['majority']['mean']:.3f}; base rate {r['base_rate_minority']:.3f}; "
              f"isolated minority/majority {r['minority']['isolated']}/{r['majority']['isolated']}")
    (ROOT / "results" / "class_homophily.json").write_text(json.dumps(results, indent=2))
    (ROOT / "figures").mkdir(exist_ok=True)
    plot(results, ROOT / "figures" / "class_homophily.png")
    print("saved results/class_homophily.json and figures/class_homophily.png")


if __name__ == "__main__":
    main()
