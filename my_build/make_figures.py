"""
Report figures (Sat 7 Nov). Every number is read from a result file or copied from the report table named beside it.
Output: artifacts/figures/fig1_pipeline.png ... fig5_split_reconciliation.png (300 dpi, Arial, white background).
Run: python my_build/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "my_build" / "results" / "twibot22"
OUT = ROOT / "artifacts" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# validated categorical slots (dataviz default palette, light): blue, orange, aqua, yellow
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 300})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("saved", name)


def grid_x(ax):
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ---------- Figure 1: pipeline ----------
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 100), ax.set_ylim(0, 50), ax.axis("off")

    def box(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=col + "22", ec=col, lw=1.2))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.5, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=7.2, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0, shrinkA=0, shrinkB=0))

    box(1, 14, 16, 22, "Data", "TwiBot-22\n1M users,\nfollow + interaction\nlinks\n(TwiBot-20 to check)", INK2)
    box(23, 34, 22, 14, "Own features", "15 profile fields\nmultilingual-E5 text", BLUE)
    box(23, 18, 22, 13, "Neighbour features", "per-relation averages\nof neighbours", AQUA)
    box(23, 1, 22, 14, "LLM (Qwen3-1.7B)", "zero-shot · LoRA\nDGP: + neighbour\nsummaries", YELLOW)
    box(51, 24, 20, 22, "Models", "trees (own)\nXGB-Graph (own +\nneighbours)\nGNNs vs MLP control", ORANGE)
    box(51, 2, 20, 16, "Fusion", "stack LLM score\nwith tree score", YELLOW)
    box(77, 8, 22, 36, "Evaluation", "development test set\n(used while building)\n\nfinal test set\n(scored once)\n\n986 expert-labelled\nusers + annotation\nstudy", INK2)
    for y in (41, 24.5, 8):
        arrow(17.8, 25, 22.2, y)
    arrow(45.8, 41, 50.2, 37), arrow(45.8, 24.5, 50.2, 30), arrow(45.8, 8, 50.2, 10)
    arrow(71.8, 35, 76.2, 30), arrow(71.8, 10, 76.2, 16)
    save(fig, "fig1_pipeline.png")


# ---------- Figure 2: main results (final test set, final_run.json; text rows from Table 1) ----------
def fig2():
    m = json.load(open(RES / "final_run.json"))["models"]
    a = lambda k: m[k]["target"]["auroc"]
    rows = [  # label, auroc, colour
        ("XGB-Graph + profile", a("full/xgb_graph_prof"), AQUA),
        ("Profile + text (tree)", a("full/tree_self_prof"), BLUE),
        ("Profile only (tree)", a("full/profile_only"), BLUE),
        ("FAGCN (best GNN)", a("full_gnn/FAGCN"), ORANGE),
        ("LINKX", a("full_gnn/LINKX"), ORANGE),
        ("SAGE", a("full_gnn/SAGE"), ORANGE),
        ("GCN", a("full_gnn/GCN"), ORANGE),
        ("R-GCN", a("full_gnn/RGCN"), ORANGE),
        ("MLP, no edges (GNN control)", a("full_gnn/MLP"), INK2),
        ("Links only (neighbour counts)", a("full/links_only"), AQUA),
    ]
    text_rows = [("TF-IDF + LR (70,000 train)", 0.727, BLUE), ("LoRA Qwen3-1.7B (1,300 train)", 0.710, YELLOW),
                 ("Zero-shot Qwen3-1.7B", 0.461, YELLOW)]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.6), sharex=True, gridspec_kw={"height_ratios": [len(rows), len(text_rows)], "hspace": 0.35})
    for ax, rs, title in ((axes[0], rows, "All 1M users, final test set"), (axes[1], text_rows, "Text-only models, 5,000-user sample, development test set")):
        y = list(range(len(rs)))[::-1]
        ax.barh(y, [r[1] - 0.4 for r in rs], left=0.4, height=0.62, color=[r[2] for r in rs])
        for yi, r in zip(y, rs):
            ax.text(r[1] + 0.005, yi, f"{r[1]:.3f}", va="center", fontsize=7.5, color=INK)
        ax.set_yticks(y, [r[0] for r in rs], fontsize=8)
        ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.set_title(title, loc="left", fontsize=8.5, color=INK, weight="bold")
        grid_x(ax)
        ax.tick_params(axis="y", length=0)
    axes[0].text(0.503, len(rows) - 0.4, "random = 0.5", fontsize=7, color=INK2, va="bottom")
    axes[1].set_xlim(0.4, 0.92), axes[1].set_xlabel("AUROC (higher is better)")
    save(fig, "fig2_main_results.png")


# ---------- Figure 3: camouflage (relation_homophily_by_class.json, local_homophily.json) ----------
def fig3():
    rel = json.load(open(RES / "relation_homophily_by_class.json"))
    base = rel["base_rate_bot"]
    names = list(rel["relations"])
    pb = [rel["relations"][r]["out"]["p_nbr_bot_given_bot"] * 100 for r in names]
    ph = [rel["relations"][r]["out"]["p_nbr_bot_given_human"] * 100 for r in names]
    lh = json.load(open(RES / "local_homophily.json"))["b_gain_by_bin_test_work"]
    bins = ["[0, 0.25)", "[0.25, 0.5)", "[0.5, 0.75)", "[0.75, 1]", "no train neighbour"]
    blabel = ["0–25%", "25–50%", "50–75%", "75–100%", "none\nlabelled"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.6), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.34})
    x = range(len(names))
    w = 0.38
    a1.bar([i - w / 2 - 0.01 for i in x], pb, w, color=ORANGE, label="bots' neighbours")
    a1.bar([i + w / 2 + 0.01 for i in x], ph, w, color=BLUE, label="humans' neighbours")
    a1.axhline(base * 100, color=INK, lw=0.9, ls=(0, (3, 2)))
    a1.text(2.0, base * 100 + 0.4, f"base rate {base*100:.0f}%", ha="left", fontsize=7, color=INK)
    a1.set_xticks(list(x), names, rotation=35, ha="right", fontsize=7.5)
    a1.set_ylabel("% of neighbours that are bots")
    a1.set_title("(a) Bots hide among humans on every relation", loc="left", fontsize=8.5, weight="bold")
    a1.legend(frameon=False, fontsize=7, loc="upper right")
    a1.set_ylim(0, 22)
    a1.grid(axis="y", color=GRID, lw=0.6), a1.set_axisbelow(True)
    series = [("XGBGraph_vs_ownTree (profile)", "neighbour features (XGB-Graph − tree)", AQUA, "o"),
              ("FAGCN_vs_MLP", "FAGCN − MLP", ORANGE, "s"), ("RGCN_vs_MLP", "R-GCN − MLP", INK2, "D")]
    for k, (key, lab, col, mk) in enumerate(series):
        xs = [i + (k - 1) * 0.22 for i in range(len(bins))]
        d = [lh[key][b]["auroc_diff_ci"] for b in bins]
        a2.errorbar(xs, [v[0] for v in d], yerr=[[v[0] - v[1] for v in d], [v[2] - v[0] for v in d]], fmt=mk, ms=5,
                    color=col, ecolor=col, elinewidth=1, capsize=0, label=lab, markeredgecolor=SURF, markeredgewidth=0.8)
    a2.axhline(0, color=INK, lw=0.8)
    a2.set_xticks(range(len(bins)), [f"{l}\n{lh[series[0][0]][b]['users']:,} users\n{lh[series[0][0]][b]['bot_share']*100:.0f}% bots" for l, b in zip(blabel, bins)], fontsize=6.5)
    a2.set_xlabel("share of a user's labelled neighbours with the same label", fontsize=7.5)
    a2.set_ylabel("AUROC gain [95% CI]")
    a2.set_title("(b) Neighbours help some users, hurt others", loc="left", fontsize=8.5, weight="bold")
    a2.legend(frameon=False, fontsize=6.8, loc="upper left", ncol=1)
    a2.set_ylim(-0.085, 0.15)
    a2.grid(axis="y", color=GRID, lw=0.6), a2.set_axisbelow(True)
    save(fig, "fig3_camouflage.png")


# ---------- Figure 4: empty-bio lift (Table 3) ----------
def fig4():
    rows = [("986 experts", "dataset labels", 8.7, 6.4, 12.3, ORANGE), ("986 experts", "expert labels", 7.2, 5.3, 10.3, BLUE),
            ("458 annotation-study users", "dataset labels", 5.0, 3.3, 7.7, ORANGE),
            ("458 annotation-study users", "independent annotators", 1.6, 0.7, 3.1, AQUA)]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    y = [4.2, 3.4, 1.4, 0.6]
    for yi, (grp, lab, v, lo, hi, col) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round")
        ax.plot(v, yi, "o", ms=8, color=col, markeredgecolor=SURF, markeredgewidth=1.2)
        ax.text(hi + 0.25, yi, f"{v:.1f}×  [{lo}, {hi}]", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [r[1] for r in rows], fontsize=8)
    for yy, g in ((4.75, rows[0][0]), (1.95, rows[2][0])):
        ax.text(0, yy, g, fontsize=8, weight="bold", color=INK, transform=ax.get_yaxis_transform(), ha="right")
    ax.axvline(1, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(1.08, 2.4, "1× = no link", fontsize=7, color=INK2)
    ax.set_xlim(0, 14.5), ax.set_ylim(0, 5)
    ax.set_xlabel("lift: bot rate among empty-bio users ÷ bot rate among users with a bio")
    grid_x(ax), ax.tick_params(axis="y", length=0)
    save(fig, "fig4_empty_bio_lift.png")


# ---------- Figure 5: split reconciliation (Table 4, seed 0, verified out) ----------
def fig5():
    rows = [("Own-feature tree", 0.764, 0.860), ("XGB-Graph + profile", 0.758, 0.873), ("FAGCN", 0.745, 0.828),
            ("R-GCN", 0.726, 0.800), ("MLP, no edges", 0.734, 0.799)]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    y = list(range(len(rows)))[::-1]
    for yi, (n, off, ours) in zip(y, rows):
        ax.plot([off, ours], [yi, yi], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.plot(off, yi, "o", ms=8, color=ORANGE, markeredgecolor=SURF, zorder=2)
        ax.plot(ours, yi, "o", ms=8, color=BLUE, markeredgecolor=SURF, zorder=2)
        ax.text(off - 0.004, yi, f"{off:.3f}", ha="right", va="center", fontsize=7.5)
        ax.text(ours + 0.004, yi, f"{ours:.3f}", ha="left", va="center", fontsize=7.5)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.plot([], [], "o", color=ORANGE, label="official released split"), ax.plot([], [], "o", color=BLUE, label="our stratified split")
    ax.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_xlim(0.69, 0.9), ax.set_xlabel("AUROC, `verified` excluded (seed 0)")
    grid_x(ax), ax.tick_params(axis="y", length=0)
    save(fig, "fig5_split_reconciliation.png")


if __name__ == "__main__":
    fig1(), fig2(), fig3(), fig4(), fig5()
