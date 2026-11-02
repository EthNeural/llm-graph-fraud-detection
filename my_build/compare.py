"""
Final comparison: every model, both settings, one table per setting.

Run:  python my_build/compare.py

Also prints how the graph itself changes between settings (homophily), using your step 1
functions, because FLAG-style subsampling removes edges as well as users.

FLAG baseline: once it has been run on OUR splits, save its test metrics (same keys as
evaluate() returns) as a list of dicts in results/A/flag.json and results/B/flag.json.
They are picked up automatically and shown at the bottom of each table.

Note: LLM-reasoner rows (step 5) are scored on a stratified test SAMPLE, so compare them
with the "(same sample)" rows next to them, not with the full-test-set rows above.
"""
import json
import os
import subprocess
import sys

import numpy as np

from common import SHARED_DIR

FILES = [
    ("metadata_shortcut.json", "metadata shortcut (reference only, NOT a valid model)"),
    ("step2_baselines.json", "baselines"),
    ("step3_text_only.json", "text only (LLM encoder + LR)"),
    ("step4_gnn.json", "graph (LLM encoder + GNN), mean of 3 seeds"),
    ("step5_llm.json", "LLM reasoner, test SAMPLE"),
    ("step6_lora.json", "LLM fine-tuned with LoRA (DGP replication), 1 seed; same sample unless 'full test'"),
    ("flag.json", "FLAG baseline"),
]
DISPLAY = {"e5": "multilingual-E5", "e5en": "E5-base (English)"}       # short code names -> names used in tables and the report


def pretty(model):
    # "e5 + GCN" -> "multilingual-E5 + GCN"
    head, sep, tail = model.partition(" + ")
    return DISPLAY.get(head, head) + sep + tail


COLS = [("macro_f1", "macroF1"), ("f1_class1", "F1_1"), ("auroc", "AUROC"),
        ("pr_auc", "PR-AUC"), ("macro_f1_connected", "conn"), ("macro_f1_isolated", "iso")]


def graph_stats(setting):
    # Run in a subprocess so common.py picks up the right SETTING.
    code = (
        "import numpy as np\n"
        "from common import load, isolated_mask\n"
        "from step1_explore import edge_homophily, class_homophily, class_insensitive_homophily\n"
        "x, e, y = load()\n"
        "hk = class_homophily(e, y)\n"
        "print(f'nodes {len(y):,}  class-1 share {y.mean():.3f}  edges {e.shape[1]:,}  '\n"
        "      f'isolated {isolated_mask(e, len(y)).sum():,}')\n"
        "print(f'edge homophily {edge_homophily(e, y):.3f}  '\n"
        "      f'h_k ' + '  '.join(f'{k}:{v:.3f}' for k, v in hk.items()) +\n"
        "      f'  class-insensitive {class_insensitive_homophily(hk, y):.3f}')\n"
    )
    env = dict(os.environ, SETTING=setting)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    return out.stdout.strip() or out.stderr.strip()[-300:]


def table(setting):
    d = SHARED_DIR / setting
    print(f"\n{'=' * 30} SETTING {setting} "
          f"({'natural ratio' if setting == 'A' else 'FLAG-style 1:10'}) {'=' * 30}")
    print(graph_stats(setting))
    print(f"\n{'model':24s}" + "".join(f"{h:>9s}" for _, h in COLS))
    for fname, title in FILES:
        if not (d / fname).exists():
            if fname == "flag.json":
                print(f"\n-- {title}: not run yet (add results/{setting}/flag.json)")
            continue
        print(f"-- {title}")
        for r in json.load(open(d / fname)):
            vals = "".join(f"{r[k]:9.3f}" if k in r else f"{'-':>9s}" for k, _ in COLS)
            std = f"  ±{r['macro_f1_std']:.3f}" if "macro_f1_std" in r else ""
            print(f"{pretty(r['model']):24s}{vals}{std}")


def ladder(setting):
    # Model ladder: one row per encoder, the same three ways (text only / +GCN / +SAGE).
    d = SHARED_DIR / setting
    if not (d / "step4_gnn.json").exists():
        return
    rows = {r["model"]: r for r in json.load(open(d / "step4_gnn.json"))}
    encoders = [m.split(" + ")[0] for m in rows if m.endswith(" + MLP")]
    if not encoders:
        return
    cell = lambda m: f"{rows[m]['macro_f1']:.3f} / {rows[m]['auroc']:.3f}" if m in rows else "-"
    print(f"\nLADDER, setting {setting}  (macro-F1 / ROC-AUC, mean of 3 seeds)")
    print(f"{'encoder':16s}{'text only (MLP)':>18s}{'hybrid GCN':>18s}{'hybrid SAGE':>18s}")
    for e in encoders:
        print(f"{DISPLAY.get(e, e):16s}{cell(e + ' + MLP'):>18s}{cell(e + ' + GCN'):>18s}{cell(e + ' + SAGE'):>18s}")
    print(f"{'graph only':16s}{'':>18s}{cell('graph-only + GCN'):>18s}{cell('graph-only + SAGE'):>18s}")
    meta = d / "metadata_shortcut.json"
    if meta.exists():
        for r in json.load(open(meta)):
            if r["model"] == "meta: all except verified":
                print(f"reference: metadata without 'verified' = {r['macro_f1']:.3f} / {r['auroc']:.3f}")


if __name__ == "__main__":
    for s in ["A", "B"]:
        table(s)
        ladder(s)
