"""
Independent study, stage 2 (GPU; rules in reports/independent_study/DECISIONS.md, Sat 7 Nov ~15:45).
Zero-shot machine-text scores per comment: Binoculars, analytic Fast-DetectGPT, log-perplexity (Qwen2.5-1.5B base /
Instruct pair). Sealed users skipped. Output: results/independent/botsim_stage2_zeroshot.json (+ per-comment npz).
Run: HF_HUB_OFFLINE=1 python -I my_build/independent/botsim_stage2_zeroshot.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MODEL", "xgb_graph")
from twibot22_metrics import metrics, val_cutoff

DATA = ROOT.parent / "data" / "botsim24"
OUTD = ROOT / "results" / "independent"
OUT = OUTD / "botsim_stage2_zeroshot.json"
NPZ = OUTD / "botsim_stage2_comment_scores.npz"
OBS, PERF = "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"
MAXTOK, MINTOK, TOKBUDGET = 128, 5, 1024
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
FOCAL = {"delve", "delves", "delved", "delving", "showcasing", "showcases", "boasts", "underscore", "underscores", "underscoring",
         "comprehending", "intricacies", "intricate", "surpassing", "surpasses", "garnered", "emphasizing", "realm", "groundbreaking",
         "advancements", "aligns", "potential", "findings", "crucial", "notably", "across", "additionally", "comprehensive",
         "enhancing", "exhibited", "insights", "particularly", "within"}


@torch.no_grad()
def score_batch(texts, tok, mo, mp):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAXTOK).to(DEV)
    lo = mo(**enc).logits[:, :-1].float()
    lp = mp(**enc).logits[:, :-1].float()
    tgt = enc.input_ids[:, 1:]
    m = enc.attention_mask[:, 1:].float()
    logp_p = torch.log_softmax(lp, -1); del lp
    p_o = torch.softmax(lo, -1); del lo
    nll = -logp_p.gather(-1, tgt[..., None]).squeeze(-1)                     # performer NLL of true tokens
    xent = -(p_o * logp_p).sum(-1)                                           # cross-entropy observer -> performer
    mu = (p_o * logp_p).sum(-1)                                              # Fast-DetectGPT: E_q[log p]
    var = (p_o * logp_p.pow(2)).sum(-1) - mu.pow(2)
    n = m.sum(1)
    log_ppl = (nll * m).sum(1) / n
    bino = log_ppl / ((xent * m).sum(1) / n)
    fdg = ((-nll - mu) * m).sum(1) / torch.sqrt((var * m).sum(1))
    return bino.cpu().numpy(), fdg.cpu().numpy(), log_ppl.cpu().numpy()


def main():
    assert not OUT.exists(), "never overwritten"
    u, sp = np.load(DATA / "users.npz"), np.load(DATA / "split.npz")
    y, uid = u["y"], list(u["user_id"])
    sealed = set(sp["test_sealed"].tolist())
    texts = json.load(open(DATA / "texts.json"))
    tok = AutoTokenizer.from_pretrained(PERF)
    tok.padding_side = "right"
    rows = []                                                                # (user index, comment)
    for i, k in enumerate(uid):
        if i in sealed:
            continue
        for c in texts[k]["comments"]:
            if len(tok(c, add_special_tokens=False).input_ids) >= MINTOK:
                rows.append((i, c))
    print(f"{len(rows):,} comments from {len({r[0] for r in rows}):,} non-sealed users", flush=True)
    mo = AutoModelForCausalLM.from_pretrained(OBS, dtype=torch.float16).to(DEV).eval()
    mp = AutoModelForCausalLM.from_pretrained(PERF, dtype=torch.float16).to(DEV).eval()
    lens = np.array([min(len(tok(c, add_special_tokens=False).input_ids), MAXTOK) for _, c in rows])
    order = np.argsort(lens, kind="stable")                                  # length-sorted batches = less padding
    batches, cur = [], []
    for j in order:                                                          # token budget keeps vocab tensors < ~0.6 GB
        if cur and (len(cur) + 1) * lens[j] > TOKBUDGET:
            batches.append(cur); cur = []
        cur.append(j)
    batches.append(cur)
    B, F, P = (np.zeros(len(rows), np.float32) for _ in range(3))
    done = 0
    for bi, idx in enumerate(batches):
        b, f, p = score_batch([rows[j][1] for j in idx], tok, mo, mp)
        B[idx], F[idx], P[idx] = b, f, p
        done += len(idx)
        if bi % 200 == 0:
            print(f"  {done:,}/{len(rows):,} (batch {bi:,}/{len(batches):,})", flush=True)
    owner = np.array([r[0] for r in rows])
    np.savez(NPZ, owner=owner, binoculars=B, fast_detectgpt=F, log_ppl=P)
    # user scores (machine-HIGH orientation for AUROC: -B, +F, -P)
    n = len(y)
    feats = np.full((n, 4), np.nan)
    for j, (sc, sign) in enumerate(((B, -1), (F, 1), (P, -1))):
        tot = np.bincount(owner, sign * sc, minlength=n)
        cnt = np.bincount(owner, minlength=n)
        feats[cnt > 0, j] = tot[cnt > 0] / cnt[cnt > 0]
    for i, k in enumerate(uid):
        if i in sealed:
            continue
        toks = " ".join(texts[k]["comments"]).lower().split()
        feats[i, 3] = sum(t.strip(".,;:!?()\"'") in FOCAL for t in toks) / max(len(toks), 1)
    has = ~np.isnan(feats[:, :3]).any(1)
    tr, va, te = (sp[k][has[sp[k]]] for k in ("train", "val", "test_work"))
    out = {"comments": len(rows), "users_scored": int(has.sum()), "test_users": int(len(te)), "test_bots": int(y[te].sum()),
           "untrained_auroc": {}, "comment_level_auroc_test": {}}
    names = ["binoculars", "fast_detectgpt", "log_ppl", "focal_word_rate"]
    for j, nm in enumerate(names):
        out["untrained_auroc"][nm] = {"working_test": round(float(roc_auc_score(y[te], feats[te, j])), 4),
                                      "train_val": round(float(roc_auc_score(y[np.r_[tr, va]], feats[np.r_[tr, va], j])), 4)}
    te_set = set(te.tolist())
    cm = np.array([o in te_set for o in owner])
    for nm, sc, sign in (("binoculars", B, -1), ("fast_detectgpt", F, 1), ("log_ppl", P, -1)):
        out["comment_level_auroc_test"][nm] = round(float(roc_auc_score(y[owner[cm]], sign * sc[cm])), 4)
    mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-9
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit((feats[tr] - mu) / sd, y[tr])
    s_all = lr.predict_proba((np.nan_to_num(feats) - mu) / sd)[:, 1]
    c = val_cutoff(y[va], s_all[va])
    out["trained_lr_4_scores"] = {k: round(v, 4) for k, v in metrics(y[te], s_all[te], c).items() if k in ("macro_f1", "auroc", "pr_auc")}
    out["mean_score_by_class_working_test"] = {nm: {"bot": round(float(np.nanmean(feats[te][y[te] == 1, j])), 4),
                                                    "human": round(float(np.nanmean(feats[te][y[te] == 0, j])), 4)} for j, nm in enumerate(names)}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
