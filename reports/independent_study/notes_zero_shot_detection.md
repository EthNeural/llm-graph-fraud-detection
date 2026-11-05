# Zero-shot machine-text scores: Binoculars and Fast-DetectGPT (implementation notes)

Sources (read via pdftotext): Hans et al., "Spotting LLMs with Binoculars", ICML 2024 (arXiv 2401.12070v3) = **[B]**; Bao et al., "Fast-DetectGPT", ICLR 2024 (arXiv 2310.05130v3) = **[F]**.
Convention: "NOT IN PAPER" = not stated in the PDF; where I add a recollection of the authors' released code or my own suggestion it is labelled as such and must be verified before use.
Target: per-account scores, offline, with Qwen/Qwen2.5-1.5B (base), Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen3-1.7B, on MPS.

---

## 1. Exact score formulas

### 1.1 Binoculars [B]

Notation (Sec 3.1): tokenizer T, tokens x = T(s), L tokens, model output Y = M(x), Y_ij = P(v_j | x_{0:i-1}) (Eq. 1).

- Eq. 2, log-perplexity of model M on s: `log PPL_M(s) = -(1/L) * sum_i log Y_{i,x_i}` (mean negative log-likelihood of the observed tokens, nats; it is a log, not exp'ed).
- Eq. 3, log cross-perplexity: `log X-PPL_{M1,M2}(s) = -(1/L) * sum_i M1(s)_i . log(M2(s)_i)`, where "." is the dot product over the vocabulary. So per position: expected NLL under M2's log-probs, with the expectation taken under **M1's** next-token distribution. Footnote 2: "This requires that M1 and M2 share a tokenizer."
- Eq. 4 (Sec 3.3), the score: `B_{M1,M2}(s) = log PPL_{M1}(s) / log X-PPL_{M1,M2}(s)`.
- Decision: LOW score = machine, HIGH = human (Table 1: score 0.73 for LLM text, "well below the global threshold of 0.901"; random tokens score ~1.35, humans ~1; Sec 5.5/App A.7).

Which model is observer vs performer (this is muddled in the paper):
- Sec 1 prose: "log perplexity ... using an observer LLM ... next-token predictions using a performer LLM and compute their perplexity according to the observer".
- Sec 3.3 (concrete): "numerator i.e. PPL using the Falcon-7B-Instruct model ... denominator i.e. X-PPL with Falcon-7B and Falcon-7B-Instruct respectively".
- Table 3 header: "PPL Scorer (M1)" = Falcon-7B-Instruct; "X-Cross PPL Scorers (M1', M2)" = (Falcon-7B, Falcon-7B-Instruct). Default row = Falcon-7B-Instruct / (Falcon-7B, Falcon-7B-Instruct).
- Reading Table 3 with Eq. 3: PPL numerator uses the **instruct** model's log-probs of the true tokens; X-PPL denominator takes expectation under the **base** model's distribution of the **instruct** model's log-probs. A.8 calls the Instruct model "performer (M2)"; the base is the "observer".
- So the implementable recipe is: `ppl = mean NLL of true tokens under INSTRUCT(performer)`; `xppl = mean_i sum_v softmax(BASE logits)_i,v * (-log_softmax(INSTRUCT logits)_i,v)`; `B = ppl / xppl`.
- Note the Sec 1 prose ("perplexity by the observer") is inconsistent with Table 3 / Sec 3.3. I follow Table 3. (From memory of the authors' released code, not in the paper: observer = Falcon-7B, performer = Falcon-7B-Instruct, `ppl` from performer logits, `x_ppl` = cross-entropy of performer logits against softmax(observer logits). This matches Table 3. VERIFY if ever possible; do not rely on it.)
- Note the paper does not state whether the Table 3 default row's mapping is robust to swapping; there is no swap ablation. NOT IN PAPER.

### 1.2 Fast-DetectGPT [F]

Conditional probability function (Eq. 2): `p_theta(x~|x) = prod_j p_theta(x~_j | x_<j)`, tokens x~_j predicted independently given the fixed real prefix x. Special case p(x|x) = p(x).

- Eq. 3 (curvature / "sampling discrepancy"): `d(x, p_theta, q_phi) = (log p_theta(x|x) - mu~) / sigma~`.
- Eq. 4: `mu~ = E_{x~ ~ q_phi(.|x)}[log p_theta(x~|x)]`, `sigma~^2 = E[(log p_theta(x~|x) - mu~)^2]`. q_phi = sampling model, p_theta = scoring model.
- Algorithm 1: sample N alternatives (default N = 10,000, Sec 2.3), mu~ = mean of their log p, sigma~^2 = sample variance (1/(N-1)), `d = (log p(x) - mu~)/sigma~`, return d > epsilon (HIGH = machine).
- **Analytic version (Appendix B, Eq. 6-7)**: because tokens are conditionally independent, enumerate the vocabulary at each position j:
  - `mu~_j = sum_v q(v | x_<j) * log p(v | x_<j)`, `mu~ = sum_j mu~_j` (Eq. 6)
  - `sigma~^2 = sum_j [ sum_v q(v|x_<j) * (log p(v|x_<j))^2 - mu~_j^2 ]` (Eq. 7)
  - numerator `log p(x) = sum_j log p(x_j | x_<j)`
  - so `d = ( sum_j log p(x_j|x_<j) - sum_j mu~_j ) / sqrt( sum_j var_j )`. Appendix B: matches 10k-sample version "almost identical" and is ~10% faster.
- Sums over j (not means) appear in the equations (log p(x) is a joint log-prob). Whether to divide by L: NOT IN PAPER (paper's formulas are sums; sum-based d grows ~sqrt(L), so thresholds are length-dependent. See Sec 4/5.)
- Same-model special case (Eq. 5, q = p): numerator = Likelihood (log p(x)) + Entropy (-mu~); i.e. `d = (log p(x) + H) / sigma~` with `H = -mu~`. Entropy ablation (App E): subtracting the entropy-like baseline is where 65% of the 75% relative gain comes from; sigma-normalisation gives 10%.
- Settings used: white-box `*/*` (source model samples and scores); black-box `GPT-J / Neo-2.7` (sampling / scoring) (Table 2/3, App C). The released-code details (e.g. sum vs mean over tokens, which model is "reference") are NOT IN PAPER.

---

## 2. Model pairs and the shared-tokenizer requirement

Binoculars:
- Footnote 2 (Sec 3.2, Eq. 3): M1 and M2 must share a tokenizer, because the cross-term `M1(s)_i . log(M2(s)_i)` is a dot product over the same vocabulary index; positions i must align token for token, and so must vocabulary entries.
- Paper says the method "works best for two models that are very close to each other in performance" (Sec 3.3), contrast with contrastive/speculative decoding which pair strong+very weak.
- Pairs tested (Table 3): Falcon-7B + Falcon-7B-Instruct (best: TPR 100% at 0.01% FPR in the top row; a range of 23-100% across the Falcon-Instruct rows (rows are different datasets, the table does not label them), i.e. strongly dataset-dependent); Llama-2-7B with Llama-7B/Llama-2-7B; Llama-2-13B with Llama-13B/Llama-2-13B; Llama-2-13B-Chat with Llama-2-13B (weaker rows, 20-71% TPR at 0.01% FPR). Note the LLaMA-1 base is combined with a LLaMA-2 model there, so "same tokenizer" is "same vocabulary", not "same model family/version".
- A.5 / Fig 13: identical models (Falcon-7B for both roles) is "competitive" but worse than base+instruct on all three domains; Falcon-7B-Instruct alone as both is worse still (Student Essay AUC 0.795 vs Binoculars 0.998).
- A.8 / Fig 15: performer quality improves nearly monotonically with instruction tuning (base -> Alpaca-tuned steps -> Falcon-7B-Instruct).
- Implication for Qwen2.5-1.5B (observer) + Qwen2.5-1.5B-Instruct (performer): matches the paper's recommended structure (base + its instruct tune). Qwen3-1.7B: NOT IN PAPER; whether its tokenizer/vocab ids are identical to Qwen2.5's must be checked by comparing `tok.get_vocab()` (or at least vocab size and a hash of the vocab), not assumed. Use Qwen3 only as a third, independent scorer or as an identical-model-pair (A.5) variant.

Fast-DetectGPT:
- Sampling model q and scoring model p must index the same vocabulary for the vocabulary-enumeration sum (Eq. 6-7). The paper does not state this in words ("same tokenizer" appears nowhere); it is implied by the formulas and by its pairs (GPT-J / GPT-Neo-2.7B, both Pile/GPT-2-family BPE). NOT STATED IN PAPER.
- Ablation (Table 8, App E): a stronger sampling model (GPT-J) improves white-box AUROC by about +0.002 (27% relative); a weak sampling model (GPT-2, Neo-2.7) can hurt when the scoring model is much larger (e.g. GPT-J scoring: 0.9534-0.9153 vs 0.9771).
- Black-box surrogate pair "GPT-J/Neo-2.7" is chosen empirically (Neo-2.7 best among GPT-2, Neo-2.7, GPT-J as scorer). Mismatch of scoring-model training corpus with source model hurts (App D.1: avg relative gain over DetectGPT drops from 74.5% to 53.4% for BLOOM/OPT-13/Llama/Llama2, "potential mismatch ... training corpus"). Discussion: "no single model can seamlessly span all linguistic territories and domains".
- For Qwen: Qwen2.5-1.5B-Instruct as sampler and Qwen2.5-1.5B as scorer (or either model for both) are the natural options. Which is better: NOT IN PAPER; validate on labelled data.

---

## 3. Dependence on text length and domain

Binoculars [B]:
- Metric: TPR at 0.01% FPR (not AUC), Sec 4.2. Evaluation uses a prefix of 512 tokens per document unless stated (A.1.2).
- Fig 2 (Sec 4.3): TPR at 0.01% FPR vs document size 128/256/512 tokens on News / Creative Writing / Student Essay: Binoculars rises with length; "advantages of Binoculars are even clearer in the few-token regime"; **only plotted, no numeric values in the text for 128 tokens**. Nothing below 128 tokens was evaluated. Numbers for < 128 tokens: NOT IN PAPER.
- Fig 10 (A.3): score vs sequence length is nearly uncorrelated with class (length alone does not bias the score).
- Domains: Ghostbuster News/Creative Writing/Student Essay (ChatGPT), CCNews/PubMed/CNN (LLaMA-2-13B, Falcon-7B), Open-Orca (GPT-3 92%, GPT-4 89.57% detected at global threshold; later A.10: GPT-4 only 41.86% correct (58% FNR) and Gemini-Pro 96.89% on 129 samples). **M4 Reddit** (Fig 4, ChatGPT text vs human): shown only as a precision-recall plot (Binoculars "more precise over 4 domains" Arxiv/Reddit/Wikihow/Wikipedia vs RoBERTa, LR-GLTR, Stylistic, NELA); no numeric Reddit value in the text. Tweet/social-media-length text: NOT IN PAPER.
- Other languages (Sec 5.2, Fig 6, M4: Urdu, Russian, Bulgarian, Arabic): precision stays high (~60-90%), recall very low (<2%): machine text in low-resource languages is mostly called human. Authors hypothesise a stronger multilingual model pair would help.
- Non-native English: EssayForum original vs grammar-corrected, both 99.67% accuracy, score distributions overlap (Sec 5.2, Fig 7).

Fast-DetectGPT [F]:
- Fig 4 (Sec 3.4): AUROC on WritingPrompts truncated to target number of words (for ChatGPT and GPT-4 text); "consistent, monotonic increase in accuracy as passage length grows" for Fast-DetectGPT; numerical values only in the figure, none in text. Axis range of word counts is not stated in the text; short-text (< 128 token) numbers: NOT IN PAPER. Related-work sentence: zero-shot methods "generally exhibit higher detection accuracy on longer passages".
- Domains: XSum (news), SQuAD (Wikipedia), WritingPrompts (stories), WMT16 en/de, PubMedQA. Table 3 (ChatGPT, black-box GPT-J/Neo-2.7): XSum 0.991, Writing 0.992, PubMed 0.902 (avg 0.9615); GPT-4: 0.907 / 0.961 / 0.850 (avg 0.906). PubMedQA (short ~50-word answers) is the weakest domain, consistent with a length effect, but the paper does not attribute it.
- Fig 3: for ChatGPT stories, recall 87% at 1% FPR, 98% at 10% FPR; GPT-4: 89% recall at < 10% FPR.
- Fig 5: AUROC vs supervised detectors, competitive in-distribution and better out-of-distribution (PubMedQA, WMT16-German). Only English and German tested. Reddit/social text: NOT IN PAPER.
- Generation lengths used: "about 150 words" (news, stories), "about 50 words" (PubMedQA) (App C.2); human prompts 30 tokens (Sec 3.1).

---

## 4. Thresholds, calibration, failure modes

Binoculars:
- Single global threshold chosen by maximising accuracy on the combined **training splits** of Ghostbuster News/Creative/Essay (ChatGPT) plus CCNews/CNN/PubMed generated by LLaMA-2-13B and Falcon-7B, balanced human/machine (A.1.2); the three Ghostbuster test sets are excluded to keep it "out-of-domain". Reported global threshold for the Falcon-7B/Instruct pair: **0.901** (Table 1 caption). This number is specific to the Falcon pair and does not transfer to Qwen: recalibrate per model pair. Calibration advice in the paper is only this; no recommendation beyond choosing an FPR target (they use 0.01% FPR and say it is the smallest they can evaluate significantly).
- The paper stresses TPR at low FPR over AUC/F1 (Sec 4.2, Table 4: AUC 1.00 while TPR@0.01% FPR varies 0.04-0.99 across PPL, X-PPL and the ratio). Perplexity alone and cross-perplexity alone are poor (Table 4, Fig 11): the ratio is what works.
- Failure modes (Sec 5): memorised text is flagged as machine (US Constitution 0.76, "I have a dream" 0.81, Cosmos snippet 0.85, all below 0.901; the other 8 famous texts human-side; Table 5/A.4); low-resource / non-English machine text goes undetected (low recall); random tokens/hashcodes are scored human (mean ~1.35, Fig 14); GPT-4 and similar strong APIs have high FNR (A.10, 58% for GPT-4 on Open-Orca); explicit evasion / adversarial paraphrase not studied ("we do not consider explicit efforts to bypass detection"); prompting styles (Carl Sagan, no-robotic, pirate) cost at most ~1% sensitivity (Sec 5.4, Fig 8); code not tested. Paraphrase attack numbers: NOT IN PAPER for Binoculars (only the F paper has them). Text < 128 tokens: NOT IN PAPER.
- Also: the score detects "text a similar LM could generate", so it also fires on human text that is highly predictable (memorised, boilerplate, templated spam, repetitive tweets).

Fast-DetectGPT:
- Threshold epsilon: "should be chosen to balance the false and true positive rates in practice" (Fig 2 caption). Paper reports AUROC only plus operating points read from ROC curves (Fig 3). Typical values: curvature of machine text "hovers around 3, human text close to 0" (Sec 1, Fig 1, 30-token prefixes on XSum, sampling and scoring GPT-family models). Do not reuse these values across model pairs/lengths; calibrate on labelled data. Any further calibration procedure: NOT IN PAPER.
- Normalising by sigma~ makes curvature distributions "more directly comparable" across models/datasets (App E), but with the sum formulation the scale still depends on L (my observation, not in paper).
- Failure modes: paraphrasing attack (T5 paraphraser per sentence) drops average AUROC 0.9641 -> 0.8715 on ChatGPT text (App F.2, Table 11); the paper attributes it to lost cross-sentence coherence and confirms with a "decoherence attack" (swap adjacent words): comparable drop (e.g. Fast(GPT-J/GPT-J) 0.8663). Implication for us: concatenating unrelated posts into one string is also a coherence break; the paper does not test it. Black-box detection of GPT-4 is much harder than ChatGPT (Table 3). Non-native-speaker false positives: acknowledged as a risk (Ethical Considerations, citing Liang et al.), not measured. Memorised text: NOT IN PAPER for this paper. Non-English: only German tested (OK out-of-distribution); low-resource languages NOT IN PAPER ("no single model can seamlessly span all linguistic territories", Sec 4). Short text: see Sec 3 above.
- Decoding: top-p 0.96, top-k 40, T=0.8 all fine (AUROC ~0.99+ white-box, Table 9); lower p/k/T makes detection easier.

---

## 5. Compute cost

- Fast-DetectGPT [F] Sec 3.2: on a Tesla A100, XSum generations from the five models of Table 2, excluding model loading: 233 s total for Fast-DetectGPT vs 79,113 s (~22 h) for DetectGPT with 100 perturbations, i.e. ~340x. Cost is **one forward pass** of the scoring (and sampling) model per passage; with the analytic solution (App B) no sampling is needed (about 10% faster than 10k samples); same model for q and p = a single call (Sec 2.3). Per-1,000-token cost: NOT IN PAPER (Number of passages/tokens behind the 233 s is not given in the text; 5 models x 500 samples per dataset x passage length is an inference, not stated for the timing run).
- Binoculars [B]: needs two forward passes per text (observer and performer; 7B models in the paper). No timing, latency or per-token cost reported; only "limited GPU memory" prevented 30B+ models (Sec 6). Cost per 1,000 tokens: NOT IN PAPER. Estimate (mine): roughly 2 x (2 x params x tokens) FLOPs ~ 2 x 3 GFLOP/token for 1.5B models, trivial; on MPS the cost is dominated by the vocabulary softmax over ~152k entries at every position, not by the transformer.

---

## 6. PyTorch pseudocode (offline, MPS)

Design decisions (mine, flag as NOT IN PAPER): score each post independently (no cross-post attention), then pool token-level sums per account; pool numerator and denominator separately (ratio of sums) instead of averaging per-post ratios, because short posts give very noisy ratios. This equals Binoculars on the concatenation but without the coherence break. A fallback "concatenate posts to <= 512 tokens" is shown after. Qwen tokenizers have no BOS by default, so prepend a start token to score the first real token (otherwise the first token has no prediction and is just dropped; either is defensible but be consistent).

```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"          # set BEFORE importing transformers
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

dev = "mps" if torch.backends.mps.is_available() else "cpu"
BASE, INST = "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"

tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
tok_i = AutoTokenizer.from_pretrained(INST, local_files_only=True)
assert tok.get_vocab() == tok_i.get_vocab()      # shared-tokenizer check (B footnote 2). Also check
                                                  # model.config.vocab_size equal (logit dims!). Same check for Qwen3-1.7B.
def load(name):  # fp16 on MPS is OK for scoring; compute softmax in fp32 below
    return AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, local_files_only=True).to(dev).eval()
observer, performer = load(BASE), load(INST)      # B: observer=base, performer=instruct (Table 3)

BOS = tok.bos_token_id if tok.bos_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
tok.padding_side = "right"                         # right padding: simplest alignment
if tok.pad_token_id is None: tok.pad_token = "<|endoftext|>"

def encode(texts, max_len=512):
    enc = tok(texts, add_special_tokens=False, truncation=True, max_length=max_len - 1,
              padding=True, return_tensors="pt")   # padded to longest in batch
    ids = torch.cat([torch.full((len(texts), 1), BOS), enc.input_ids], 1)
    att = torch.cat([torch.ones(len(texts), 1, dtype=torch.long), enc.attention_mask], 1)
    return ids.to(dev), att.to(dev)

@torch.no_grad()
def token_stats(texts, mode):
    """Returns per-text (sum_num, sum_den, n_tokens) -> poolable across posts of one account.
       mode='bino':  num = sum_i -log p_perf(x_i),  den = sum_i sum_v softmax(obs)_v * -log_softmax(perf)_v
       mode='fdg':   num = sum_i log p_score(x_i) - mu_i ;  den = sum_i var_i   (analytic, App B Eq. 6-7)"""
    ids, att = encode(texts)
    tgt  = ids[:, 1:]                                   # token i is predicted from positions < i
    mask = att[:, 1:].float()                           # 1 for real tokens, 0 for padding
    lg_o = observer(input_ids=ids, attention_mask=att).logits[:, :-1].float()   # (B, T-1, V)
    lg_p = performer(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    lp_p = F.log_softmax(lg_p, -1)
    lp_o = F.log_softmax(lg_o, -1)
    if mode == "bino":
        nll = -lp_p.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)           # (B, T-1)  Eq. 2 terms
        xent = -(lp_o.exp() * lp_p).sum(-1)                              # (B, T-1)  Eq. 3 terms: M1=obs probs, M2=perf logprobs
        return (nll*mask).sum(1), (xent*mask).sum(1), mask.sum(1)
    else:  # Fast-DetectGPT analytic. sampling q = performer(Instruct), scoring p = observer(Base) (choice is NOT IN PAPER)
        q = lp_p.exp();  lp = lp_o                                       # q(v|x_<j), log p(v|x_<j)
        mu_j  = (q * lp).sum(-1)
        var_j = (q * lp.pow(2)).sum(-1) - mu_j.pow(2)
        ll_j  = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)             # log p(x_j | x_<j)
        return ((ll_j - mu_j)*mask).sum(1), (var_j*mask).sum(1), mask.sum(1)

def score_texts(texts, mode, bs=4):
    out = []
    for i in range(0, len(texts), bs):                 # bs=2-4: 152k vocab x 511 x fp32 = ~0.3 GB per
        out += zip(*[t.tolist() for t in token_stats(texts[i:i+bs], mode)])   # tensor per sequence, several tensors alive
    return out                                         # [(num, den, n), ...]

def account_score(post_stats, mode, min_tokens=...):
    num = sum(s[0] for s in post_stats); den = sum(s[1] for s in post_stats); n = sum(s[2] for s in post_stats)
    if n < min_tokens: return None                      # abstain / NaN; zero-shot scores are unreliable on little text
    if mode == "bino": return num / den                 # B_{M1,M2} (Eq. 4); low = machine. num/n and den/n cancel the 1/L
    else:              return num / den**0.5            # d (Eq. 3); high = machine
    # optional per-token variant for Fast-DetectGPT: return (num / n) / (den**0.5 / n)  -- same thing; if you want a
    # length-free scale use num/ n^0.5 / sqrt(den/n)  -- NOT IN PAPER, calibrate on labelled data

# Per account: stats = score_texts(account_posts, "bino"); b = account_score(stats, "bino")
```

Notes on the pseudocode:
- Padding: right padding + `attention_mask` for the forward pass, and the same mask multiplies every per-token term, so padded positions contribute nothing; the pad id is never a target. With right padding, positions of real tokens are unchanged (causal model), so no `position_ids` fix is needed (with left padding you would need them).
- Truncation: 512 tokens per post (the length the Binoculars paper used for evaluation, A.1.2). Posts are usually far shorter, so batch similar-length posts together (sort by length) to waste less compute.
- Memory/MPS: do `log_softmax` in fp32 (done by `.float()`); for the 151,936-entry vocab free `lg_*`/`lp_*` promptly, use `torch.mps.empty_cache()` between batches; if memory is tight, chunk over the time axis. fp16 logits on MPS can overflow in softmax if not upcast first (done above).
- Deterministic and offline: `local_files_only=True` plus HF_HUB_OFFLINE=1; no sampling randomness since the analytic version is used.
- Identical-model variant (A.5): set `performer = observer` (or both = Qwen3-1.7B) to get the "competitive but worse" single-model Binoculars; if q = p in Fast-DetectGPT, one model call suffices (Eq. 5).
- Fallback concatenation per account: join posts with "\n" up to 512 tokens and call `token_stats` once; simpler but introduces unrelated-context transitions (see decoherence result, [F] App F.2), so prefer per-post scoring with pooled sums.
- Sanity checks before trusting scores: (1) token-count equality between observer and performer inputs; (2) Binoculars sign: texts you generate yourself with Qwen2.5-1.5B-Instruct should score LOW (< typical human); (3) calibrate the threshold on labelled data from your own domain (the 0.901 is Falcon-specific); (4) report TPR at a fixed low FPR rather than AUC (B Sec 4.2); (5) record per-account token count and treat low-count accounts as missing, because both papers show accuracy rising with length and neither covers < 128 tokens.

---

## 7. Cautions specific to our use (social-media accounts)

- Both papers test paragraphs/essays (>= ~50-150 words, mostly 128-512 tokens). Tweets/short Reddit comments are outside the evaluated range: expect noisy scores; account-level pooling over many posts is the main remedy, but that is a design choice not validated in either paper.
- Spam/templated/bot-copy text, quotes, lyrics and boilerplate (memorisation) will look "machine" under both scores even if human or rule-based; URLs, hashtags, emoji and non-English text are out-of-distribution (Binoculars: low recall in non-English; random-like strings score human).
- Instruction-tuned LLM text from unknown generators: Binoculars detected ~90% of GPT-3/GPT-4 on Open-Orca at the global threshold but later API tests show GPT-4 FNR of 58% (A.10); Fast-DetectGPT black-box GPT-4 AUROC 0.906. Expect partial coverage for strong modern generators.
- A zero-shot score tells "is this text unusually predictable to a small LM", which is a feature to feed the graph model, not a verdict. Use as a feature and evaluate it by AUC / TPR at low FPR on labelled accounts.
