# Verbalized Confidence: Feasibility Method Comparison

Comparison of 7 methods for extracting classification confidence from Llama 3.3 70B Instruct (4-bit quantized, A100 80GB) on the CAP law/crime binary classification task. All methods score each document on a [0, 1] scale representing P(law/crime = Yes).

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
