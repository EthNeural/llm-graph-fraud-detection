# How AI agents should run empirical ML research reliably (notes)

Purpose: a checklist for an unattended, laptop-scale overnight study (bot detection, graph + text).
Method: web search / fetch of arXiv and venue pages on 2026-11-06. Search results are mostly abstracts and
summaries, not full-text reads. "Source" = the paper that documents the failure; the checklist action itself is
my inference from it unless stated. Items tagged (GENERAL PRACTICE) are standard ML hygiene with no specific 2024-26 source.

## Sources (existence verified via arXiv/venue pages in search results)

| Tag | Title | Authors | ID | Status |
|---|---|---|---|---|
| [AIS1] | The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery | Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, David Ha | arXiv 2408.06292 | verified |
| [AIS2] | The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search | Yutaro Yamada, Robert Tjarko Lange, Cong Lu, et al. (Sakana AI) | arXiv 2504.08066 | verified (author list partial) |
| [BEEL] | Evaluating Sakana's AI Scientist: Bold Claims, Mixed Results, and a Promising Future? | Joeran Beel, Min-Yen Kan, Moritz Baumgart | arXiv 2502.14297 (SIGIR Forum 2025) | verified |
| [AGLAB] | Agent Laboratory: Using LLM Agents as Research Assistants | Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Zicheng Liu, Emad Barsoum | arXiv 2501.04227 | verified (authors/title/date; contents not read in full) |
| [COSCI] | Towards an AI co-scientist | Juraj Gottweis et al. (Google; 34 authors) | arXiv 2502.18864 | verified (title/first author; arXiv title may now read "Accelerating scientific discovery with Co-Scientist") |
| [PITFALLS] | The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems | Ziming Luo, Atoosa Kasirzadeh, Nihar B. Shah | arXiv 2509.08713 (NeurIPS 2025 AI4Science; PNAS version also listed) | verified |
| [MLRB] | MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research | Hui Chen, Miao Xiong, et al. | arXiv 2505.19955 (NeurIPS 2025 D&B) | verified (first two authors only) |
| [PAPERB] | PaperBench: Evaluating AI's Ability to Replicate AI Research | Starace et al. (OpenAI) | arXiv 2504.01848 (ICML 2025) | verified (first author surname only; scores differ between versions) |
| [EXPB] | EXP-Bench: Can AI Conduct AI Research Experiments? | Patrick Tser Jern Kon et al. | arXiv 2505.24785 (ICLR 2026) | verified |
| [GPTZ] | GPTZero investigation: hallucinated citations in NeurIPS 2025 accepted papers | GPTZero (vendor, commercial interest) | web report, no arXiv | verified to exist; vendor claim, counts vary by outlet (~100 citations / ~51-53 papers) |
| [PHANTOM] | "Phantom References" (RefChecker on ICLR/ICML/NeurIPS/USENIX papers) | Russinovich et al. | arXiv 2607.00738 | UNVERIFIED: only a truncated search snippet seen |

Not verified / not found: "ResearchBench" (no paper located; do not cite). MLE-bench exists as a benchmark per EXP-Bench's
related work but I did not locate its own paper: UNVERIFIED. Other 2026-numbered arXiv items seen in results
(2606.04220 "Dead Science Walking: Publication Bias and the AI Scientist Pipeline"; 2606.27687 "Mitigating LLM-based
p-Hacking by Preregistering for the Next LLM"; 2605.08956; 2607.26064) were seen as titles only: UNVERIFIED, not relied on.

## Key findings used
- [PITFALLS]: four failure modes in AI-scientist systems: inappropriate benchmark selection, data leakage, metric misuse, post-hoc selection bias. They are far easier to detect from trace logs + code than from the final paper. Recommends requiring logs/code with submissions.
- [MLRB]: coding agents often fabricate or leave unverified results after execution failures (abstract reports ~80% of cases; measurement method not checked).
- [BEEL]: 5 of 12 proposed experiments failed from coding errors; those that ran were often logically flawed or misleading; literature search was keyword-level, so novelty judgements were poor; communication overstated autonomy.
- [EXPB]: complete executable experiments succeed only 0.5% of the time even though sub-steps score 20-35%.
- [PAPERB]: best agent ~21% (v1) / 14.1% (ICML summary) replication vs humans ~22% on a subset; grading needs rubrics, not vibes.
- [AIS2]: authors themselves note verifying AI outputs is time-intensive and hypotheses are rarely novel.
- [GPTZ]/[PHANTOM]: fabricated citations pass even peer review, often as partly-real references.

## Checklist (22 items)

### Planning
1. Write a dated pre-registration file before any run: hypotheses, primary metric, datasets, splits, baselines, and the exact comparison that decides each hypothesis; never edit it afterwards, only append dated amendments. Source: [PITFALLS] (post-hoc selection bias, metric misuse); preregistration idea echoed in a 2026 title, UNVERIFIED (2606.27687).
2. Fix one primary metric and the benchmark/dataset choice up front; list secondary metrics as exploratory. Source: [PITFALLS] (inappropriate benchmark selection, metric misuse).
3. Set stop rules in advance: maximum number of configs, maximum runtime, and "stop and write up if the baseline already matches the proposed method". Source: [EXPB], [BEEL] (agents burn effort on flawed experiments); otherwise GENERAL PRACTICE.
4. Count every configuration tried (the forking-paths log) so selection can be reported and corrected for. Source: [PITFALLS] (selection step inside AI-scientist loops behaves like automated p-hacking).
5. Define in advance what counts as a failed run and what the agent does then (log, retry once, report), never "fill in plausible numbers". Source: [MLRB].

### Data
6. Split by the unit that can leak (users, connected components, or time), with graph edges and text from test nodes excluded from training and from any feature or embedding fitting. Source: [PITFALLS] (data leakage); graph-specific detail GENERAL PRACTICE.
7. Fit all normalisation, vocabularies, label-derived statistics and hyperparameter selection on train/validation only. Source: [PITFALLS]; GENERAL PRACTICE.
8. Run a shortcut audit: a text-only model, a degree/metadata-only model, and a label-permuted model; if a trivial feature nearly matches the full system, report it as the finding. Source: [PITFALLS] (benchmark/metric issues); GENERAL PRACTICE for the specific probes.
9. Check the construction process of the dataset (how bots and humans were sampled or crawled) for artefacts before interpreting any graph gain. Source: GENERAL PRACTICE (user's own BotSim-24 notes in this repo make the same point).
10. Save split indices and a data checksum so every result can be tied to the exact split. Source: GENERAL PRACTICE.

### Experiments
11. Include strong, tuned simple baselines (logistic regression, MLP on features, text-only, graph-only) with the same tuning budget as the proposed model. Source: [PITFALLS] (inappropriate benchmark/baseline selection); [BEEL].
12. Use at least 5 seeds for every reported number and give mean plus spread; one seed is anecdote. Source: GENERAL PRACTICE.
13. Run ablations that remove each claimed component (text, edges, each relation type) and a shuffled-edge / random-graph control. Source: GENERAL PRACTICE; motivated by [BEEL] (misleading conclusions from flawed experiments).
14. Smoke-test the pipeline on a tiny subset and assert output shapes, label alignment, and that train and test node sets are disjoint, before launching long jobs. Source: [BEEL] (42% of experiments failed on coding errors); [EXPB].
15. Log every run to disk automatically (config, seed, git hash, raw predictions, timestamps), so the report is generated from logs, not from the agent's memory. Source: [PITFALLS] (logs and code reveal failures the paper hides).
16. Report results from the pre-registered primary metric even when another metric looks better; choose checkpoints by validation, never test. Source: [PITFALLS] (metric misuse, post-hoc selection).

### Verification
17. Before trusting any positive result, run an independent sceptic pass (a separate agent or fresh context with only the code, logs and pre-registration, told to find leakage, bugs and overclaims). Source: [PITFALLS] (trace-log audit detects what paper review misses); [BEEL].
18. Recompute every number in the report from raw prediction files with a separate script and diff against the tables. Source: [MLRB] (fabricated/unverified results); GENERAL PRACTICE.
19. Treat any suspiciously large gain as a bug until reproduced from a clean run with a different seed and split. Source: [BEEL], [PITFALLS]; GENERAL PRACTICE.
20. Verify every citation against the arXiv/venue page (title, authors, id) and read at least the abstract; mark anything unchecked as UNVERIFIED; never cite from memory. Source: [GPTZ] (vendor report), [PHANTOM] (UNVERIFIED); fabricated references often mix real and fake fields.
21. Do not let the agent assert novelty from a keyword search; claim "to our knowledge" at most, with the search terms listed. Source: [BEEL] (keyword literature search gave poor novelty judgements).

### Reporting
22. Report negative and null results, all configurations tried, failures and deviations from the pre-registration; state uncertainty (spread over seeds, confidence intervals) and claim only what the controls support (no causal or "state-of-the-art" claims from one dataset, no generalisation beyond this dataset, no mechanism claims without an ablation). Source: [PITFALLS], [BEEL], [AIS2] (authors admit hypotheses are rarely novel and verification is costly); [PAPERB] for the point that agent output needs rubric-based grading.

## Caveats on the evidence
- Sakana's [AIS1]/[AIS2] and Agent Laboratory [AGLAB] and [COSCI] are system papers by the builders; I used them for context only, not as evidence of reliability. I did not read them in full.
- Numbers from [MLRB] (80%), [PAPERB] (21.0 vs 14.1%), [GPTZ] come from abstracts or secondary summaries and differ between versions or outlets.
- Items 3, 9, 10, 12, 13 and parts of 6-8 are standard practice I am adding, not findings from these papers.
