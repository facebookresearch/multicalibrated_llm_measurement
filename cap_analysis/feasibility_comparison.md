# Verbalized Confidence: Feasibility Method Comparison

Comparison of 7 methods for extracting classification confidence across 3 models on the CAP law/crime binary classification task. All methods score each document on a [0, 1] scale representing P(law/crime = Yes).

Models:
- **Llama 3.3 70B Instruct** — 4-bit quantized (NF4), A100 80GB. Standard instruction-tuned LLM.
- **DeepSeek-R1-Distill-Qwen-32B** — fp16 across 2x A100 40GB. Reasoning model (produces `<think>` chains before answering).
- **Claude (Sonnet 4)** — API-based. Used via Claude Code sub-agents (each agent has fresh context, no prior conversation).

## Methods

1. **Logprob** — Extract next-token log-probabilities for "Yes" and "No", compute P(Yes)/(P(Yes)+P(No)). No generation needed; reads the model's internal probability directly. Standard approach, not from Tian et al.

2. **1S 0-100 k=10** — Ask the model to rate its confidence on a 0-100 integer scale in a single prompt, then repeat k=10 times at temperature 0.7 and average the responses. Designed to combat the bimodal overconfidence problem by smoothing across stochastic samples. Our design.

3. **1S top-1** *(Tian et al. 2023, Table 6)* — Single-stage prompt that asks the model to provide its answer (Yes/No) AND the probability that the answer is correct (0.0-1.0) in one response. Score = P(correct) if answer is Yes, 1-P(correct) if answer is No.

4. **1S top-2** *(Tian et al. 2023, Table 6)* — Single-stage prompt that asks the model to provide both possible answers (Yes and No) along with P(correct) for each. Score = P(correct) for the "Yes" guess directly. Elicits a full probability distribution in one shot.

5. **1S P(Y/N)** — Single-stage prompt that asks the model to directly state P(Yes) and P(No) as probabilities summing to 1.0, without first committing to an answer. Our design.

6. **2S top-1** *(Tian et al. 2023, Table 6)* — Two-stage dialogue. Stage 1: ask the classification question, get a Yes/No answer. Stage 2: ask the model "What is the probability that your answer is correct?" Score = P(correct) if answer is Yes, 1-P(correct) if answer is No.

7. **2S top-1+nudge** — Same as 2S top-1, but the stage-2 prompt includes an anti-certainty instruction ("Note: very few things are 0% or 100% certain") to discourage degenerate exact-0/1 outputs. Tian et al. base method + our modification.

## Results

All evaluated on the 70B feasibility sample (~500 docs per method, ~10.4% law/crime prevalence; logprob uses the full 7K feasibility sample).

| Metric | Logprob | 1S 0-100 k=10 | 1S top-1 | 1S top-2 | 1S P(Y/N) | 2S top-1 | 2S+nudge |
|---|---|---|---|---|---|---|---|
| N | 7,000 | 500 | 500 | 500 | 466 | 500 | 500 |
| Prevalence | 11.5% | 10.4% | 10.4% | 10.4% | 10.7% | 10.4% | 10.4% |
| **AUC** | 0.904 | **0.907** | 0.647 | 0.888 | 0.884 | 0.881 | 0.886 |
| **Log loss** | 1.707 | 1.202 | 3.947 | 0.990 | 0.924 | 0.552 | **0.525** |
| ECE | 0.195 | 0.208 | 0.449 | 0.234 | 0.186 | 0.230 | 0.241 |
| Acc (t=.5) | 0.795 | 0.772 | 0.574 | 0.790 | 0.794 | 0.772 | 0.772 |
| Precision | 0.348 | 0.306 | 0.184 | 0.322 | 0.336 | 0.304 | 0.304 |
| Recall | 0.890 | 0.942 | 0.904 | 0.923 | 0.940 | 0.923 | 0.923 |
| Pos mean | 0.889 | 0.898 | 0.813 | 0.854 | 0.860 | 0.825 | 0.817 |
| Neg mean | 0.217 | 0.244 | 0.523 | 0.274 | 0.211 | 0.276 | 0.290 |
| **Exact 0.0%** | 15.5% | 49.0% | 0.0% | 20.4% | 63.3% | 6.0% | **0.0%** |
| **Exact 1.0%** | 7.6% | 9.8% | 21.4% | 7.4% | 6.2% | 0.2% | **0.0%** |
| **In [.1,.9]** | 5.9% | 23.0% | 78.0% | 71.8% | 30.5% | 80.2% | **82.2%** |
| Unique scores | 636 | 83 | 7 | 9 | 5 | 8 | 7 |

## Interpretation

**Discriminative power (AUC).** All methods except 1S top-1 achieve AUCs in the 0.88–0.91 range, indicating the underlying classification is strong regardless of how confidence is elicited. The logprob and 1S 0-100 k=10 methods are marginally best (~0.905), but the difference is small.

**Score quality (log loss).** This is where methods diverge sharply. Log loss penalizes confident wrong predictions, making it the right metric for evaluating scores destined for post-hoc calibration. The ranking is clear: 2S top-1+nudge (0.525) > 2S top-1 (0.552) > 1S P(Y/N) (0.924) ≈ 1S top-2 (0.990) > 1S 0-100 k=10 (1.202) > Logprob (1.707) > 1S top-1 (3.947). The two-stage methods produce much better-calibrated scores out of the box.

**Score distribution degeneracy.** For downstream calibration (isotonic regression, multicalibration), we need scores that aren't piled up at 0 and 1:
- *Logprob* is heavily bimodal: 23% exact 0/1, only 6% in the mid-range, but 636 unique values.
- *1S 0-100 k=10* is worse: 49% exact zeros, though k=10 averaging creates 83 unique values.
- *1S P(Y/N)* nearly collapses to binary: 63% exact zeros, only 5 unique values.
- *1S top-1* has the opposite problem: 21% exact 1.0 from a parsing pathology where the model reports P(correct)=0.0 for "No" answers, which inverts to P(Yes)=1.0. This destroys discriminative power (AUC 0.647).
- *1S top-2* is the best single-stage option: 72% mid-range, but still 20% exact zeros.
- *2S top-1+nudge* is the cleanest: 0% exact 0 or 1, 82% mid-range. The nudge eliminates the degenerate boundary mass entirely.

**ECE is misleading here.** 1S P(Y/N) achieves the lowest ECE (0.186) despite having the most degenerate distribution (63% exact zeros). This is because ECE rewards sharpness — a near-binary predictor that's mostly right will bin well even if the scores are useless for calibration. Log loss is the better metric for our purpose.

**Bottom line.** For the paper's goal — a "modal researcher" approach that produces scores amenable to post-hoc multicalibration — the two-stage methods dominate. 2S top-1+nudge has the best log loss, zero degenerate boundary mass, and the highest mid-range density. Among single-stage methods, 1S top-2 is the only viable option, but it still has 20% exact zeros and nearly double the log loss of 2S+nudge.

---

## Cross-Model Comparison

### Claude (Sonnet 4) Results

Evaluated on 500-doc subsample of the same feasibility sample (seed=42). Each document classified by an independent sub-agent with a fresh context window (simulating a "modal researcher" workflow). Note: logprob and 1S 0-100 k=10 cannot be replicated via API (no access to token probabilities or temperature sampling).

| Metric | 1S top-1 | 1S top-2 | 1S P(Y/N) | 2S top-1 | 2S+nudge |
|---|---|---|---|---|---|
| N | 500 | 500 | 500 | 500 | 500 |
| **AUC** | 0.967 | **0.978** | 0.973 | 0.973 | 0.974 |
| **Log loss** | 0.185 | **0.149** | **0.148** | 0.206 | 0.180 |
| ECE | 0.078 | 0.051 | 0.051 | 0.098 | 0.078 |
| Acc (t=.5) | 0.938 | 0.942 | 0.942 | 0.924 | 0.938 |
| **Exact 0.0%** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Exact 1.0%** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **In [.1,.9]** | 30.8% | 17.6% | 18.6% | 34.8% | 35.0% |
| Unique scores | 31 | 32 | 31 | 37 | 34 |

### DeepSeek-R1-Distill-Qwen-32B Results

Logprob evaluated on full 7K feasibility sample. Verbalized methods on 100-doc subsample (reasoning chains make inference ~50s/doc). 1S top-2 and 2S+nudge pending.

| Metric | Logprob |
|---|---|
| N | 7,000 |
| **AUC** | 0.715 |
| **Log loss** | 3.513 |
| **Exact 0.0%** | 0.0% |
| **Exact 1.0%** | 0.0% |
| **In [.1,.9]** | 0.0% |
| Unique scores | 445 |

### Cross-Model Summary (key metrics)

| Method | Llama 70B AUC | Llama 70B LL | Claude AUC | Claude LL | DS-R1 AUC | DS-R1 LL |
|---|---|---|---|---|---|---|
| Logprob | 0.904 | 1.707 | — | — | 0.715 | 3.513 |
| 1S 0-100 k=10 | 0.907 | 1.202 | — | — | — | — |
| 1S top-1 | 0.647 | 3.947 | 0.967 | 0.185 | — | — |
| 1S top-2 | 0.888 | 0.990 | 0.978 | 0.149 | *pending* | *pending* |
| 1S P(Y/N) | 0.884 | 0.924 | 0.973 | 0.148 | — | — |
| 2S top-1 | 0.881 | 0.552 | 0.973 | 0.206 | — | — |
| 2S+nudge | 0.886 | 0.525 | 0.974 | 0.180 | *pending* | *pending* |

### Cross-Model Interpretation

**Claude is dramatically better-calibrated.** Claude's log loss (0.148–0.206) is 3–4x better than Llama's best (0.525). Claude also achieves AUCs of 0.967–0.978, roughly 8 points above Llama's 0.88–0.91 range. Every Claude method produces zero exact-0 or exact-1 scores.

**Elicitation method matters for Llama, not for Claude.** Llama's log loss spans a 7.5x range across methods (0.525 to 3.947). Claude's spans only 1.4x (0.148 to 0.206). Claude produces well-calibrated scores regardless of how you ask — the prompt engineering that is critical for open-weight models becomes nearly irrelevant with a frontier API model.

**The nudge helps Llama but is unnecessary for Claude.** For Llama, the anti-certainty nudge eliminates all boundary mass (6% exact-0 → 0%) and slightly improves log loss (0.552 → 0.525). Claude already has zero boundary mass without the nudge, and adding it makes negligible difference.

**DeepSeek-R1 logprob is fundamentally broken.** AUC of 0.715 with all scores clustered above 0.9 (pos mean 0.986, neg mean 0.978). Reasoning models produce `<think>` chains before answering, so the next-token logit distribution at the prompt boundary is not informative about classification confidence. The logprob approach assumes the model's first-token prediction directly reflects its belief — this assumption breaks for reasoning models. Verbalized results (pending) will test whether the model's explicit confidence statements are better-calibrated.
