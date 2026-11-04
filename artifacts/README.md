# Artifacts: the documents behind the thinking

Everything in this folder was written to be **read**, not run. These are the reading guides, report drafts and handoff notes from the project's working sessions. Read this page first: it tells you what each file is and how the thinking got from "Elliptic" to "TwiBot-22".

**Project:** COMP8851 (Macquarie University). LLM-based approaches for graph fraud detection, evaluated under **heterophily**, as the supervisor asked.

**Builder:** Prajan Bhattarai. Code in `my_build/`, results in `my_build/results/`.

## How the thinking moved

| When (2026) | What changed | Why |
|---|---|---|
| Sep–Oct | Team pipeline on **Elliptic** (Bitcoin transactions) | Original Assessment 2 plan |
| 27 Oct | Dropped Elliptic, rebuilt on **Instagram** (GLBench text graph) | Elliptic has no real text for an LLM to read; supervisor asked for heterophily |
| 30 Oct | First report draft (file 01) | Instagram results: a good text encoder wins; GNNs add nothing |
| 1–3 Nov | Added **TwiBot-20** (real bot labels, bio + tweets) as the main dataset | A real bot/fraud-like task with a benchmark split |
| 3–4 Nov | TwiBot-20 findings | Text does the work. Edges carry almost no label signal (adjusted homophily ≈ 0). The link signal that exists is a crawl artefact. `verified` is a shortcut |
| 4 Nov | Started **TwiBot-22** | 1M users; text is weak there, so it tests whether the graph helps when text cannot (file 02, `reports/twibot22_plan.md`) |

## Files

| File | What it is | Read it for |
|---|---|---|
| [00_handoff_for_claude.md](00_handoff_for_claude.md) | Handoff note written on 4 Nov to brief a new Claude session. Part 1 is a summary; Part 2 is the full working report as of that morning; Part 3 is the DGP replication audit | The fastest way to get the whole picture, including how the builder likes to work |
| [01_final_report_draft_instagram_era_oct30.md](01_final_report_draft_instagram_era_oct30.md) | First draft of the final report (30 Oct), written in the university template, **Instagram era**. Outdated: TwiBot-20 and TwiBot-22 came later. One figure was dropped in export | The original framing and the template structure |
| [04_decisions_and_reasons.md](04_decisions_and_reasons.md) | **Every decision and its reason**, for Instagram, TwiBot-20 and TwiBot-22, in date order, plus the rules that now apply to everything | Start here to understand why the project looks the way it does |
| [03_context_colab_tweets.md](03_context_colab_tweets.md) | Briefing written for a separate Claude chat on extracting TwiBot-22 tweets in Colab (4 Nov). Not used in the end: the Chrome downloads finished, and the tweets were extracted locally in 7 minutes | How the tweet files are laid out |
| [02_reading_guide_industry_big_graphs.md](02_reading_guide_industry_big_graphs.md) | Plain-language reading guide (4 Nov): how Facebook (DEC), Uber, GADBench and Pinterest (PinSAGE) handle big graphs. A glossary sits under every section. Corrected after a fact-check (see `reports/gadbench_deep_dive.md`) | Why we keep the whole TwiBot-22 graph for cheap models and use a sample only for the LLM |

## Where everything else is

- **`REPORT.md`**: the full working report. Every TwiBot-20 and Instagram result, caveat, decision log and changelog. The most complete single document.
- **`reports/twibot22_plan.md`**: the TwiBot-22 plan and its fixed decisions.
- **`reports/twibot20_lessons_for_twibot22.md`**: what went wrong or was found late on TwiBot-20, plus a pre-registration checklist.
- **`reports/graph_slicing_research.md`**: research on big-graph practice (slicing vs. sampling the computation).
- **`reports/gadbench_deep_dive.md`**: GADBench fact-check and the exact XGB-Graph recipe.
- **`reports/dgp_replication_audit.md`**: our step 6 (DGP) against the paper, component by component.

## Ground rules this project follows

1. **Question every claim.** New findings go through a sceptic check before anything is built on them. Several of our own claims were corrected this way: these files record the corrections, not only the conclusions.
2. **Decisions before test scores.** Rules (split, checkpoint choice, metrics) are fixed before any test result is seen.
3. **Honest metrics.** Macro-F1, AUROC and PR-AUC at the natural class share, with bootstrap CIs. Accuracy alone hides an "always human" model.
4. **No shortcuts in the inputs.** `verified` (and other fields that fail a pre-set rule) is excluded.
5. **No dataset files in the repository.** TwiBot-22 is CC BY-NC-ND.
