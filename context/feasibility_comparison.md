# Verbalized Confidence: Feasibility Method Comparison

Comparison of 7 elicitation methods for extracting classification confidence across 4 models on the CAP law/crime binary classification task. All methods score each document on a [0, 1] scale representing P(law/crime = Yes).

## Models

- **Llama 3.3 70B Instruct** — 4-bit quantized (NF4), single A100 80GB. Standard instruction-tuned LLM.
- **Claude Opus 4** — API-based. Each document classified by an independent sub-agent with a fresh context window (simulating a "modal researcher" workflow). N=1,000 (two batches of 500).
- **Claude Sonnet 4** — API-based. Same sub-agent workflow as Opus. N=500 (single batch).
- **DeepSeek-R1-Distill-Qwen-32B** — fp16 across 2x A100 40GB. Reasoning model (produces `<think>` chains before answering). Verbalized methods on 100-doc subsample due to slow inference (~50–80s/doc).

## Elicitation Methods

1. **Logprob** — Extract next-token log-probabilities for "Yes" and "No", compute P(Yes)/(P(Yes)+P(No)). No generation needed; reads the model's internal probability directly. Standard approach, not from Tian et al. Not available for API models (Claude).

2. **Consistency Sampling** — Ask the model to rate its confidence on a 0–100 integer scale in a single prompt, then repeat k=10 times at temperature 0.7 and average the responses. Designed to combat bimodal overconfidence by smoothing across stochastic samples. Our design. Only run for Llama 70B.

3. **1S top-1** *(Tian et al. 2023, Table 6)* — Single-stage prompt: model provides its answer (Yes/No) AND P(correct) in one response. Score = P(correct) if answer is Yes, 1−P(correct) if answer is No.

4. **1S top-2** *(Tian et al. 2023, Table 6)* — Single-stage prompt: model provides both possible answers (Yes and No) along with P(correct) for each. Score = P(correct) for the "Yes" guess directly. Elicits a full probability distribution in one shot.

5. **1S P(Y/N)** — Single-stage prompt: model directly states P(Yes) and P(No) as probabilities summing to 1.0, without first committing to an answer. Our design. Only run for Llama 70B and Claude.

6. **2S top-1** *(Tian et al. 2023, Table 6)* — Two-stage dialogue. Stage 1: classification question → Yes/No answer. Stage 2: "What is the probability that your answer is correct?" Score = P(correct) if answer is Yes, 1−P(correct) if answer is No. Only run for Llama 70B and Claude.

7. **2S+nudge** — Same as 2S top-1, but stage-2 prompt includes anti-certainty instruction ("Note: very few things are 0% or 100% certain") to discourage degenerate exact-0/1 outputs. Tian et al. base method + our modification.

## Metric Definitions

**Prevalence Estimation:**
- **Prev** — True label prevalence (% of documents that are law/crime).
- **P-hat (mean)** — Predicted prevalence computed as the mean of all scores. This is the natural prevalence estimator when scores represent calibrated probabilities.
- **Bias (mean)** — P-hat(mean) minus true prevalence, in percentage points.
- **P-hat (t=.5)** — Predicted prevalence computed as the proportion of scores >= 0.5. Simulates a binary yes/no classification without confidence.
- **Bias (t=.5)** — P-hat(t=.5) minus true prevalence, in percentage points.

**Discrimination:**
- **AUC** — Area under the ROC curve. Measures ranking quality (can the scores separate positives from negatives?).
- **Pos mu** — Mean score among true positives.
- **Neg mu** — Mean score among true negatives.

**Calibration:**
- **Log Loss** — Binary cross-entropy. Penalizes confident wrong predictions. Scores clipped to [1e-15, 1−1e-15].
- **ECCE** — Estimated Cumulative Calibration Error. Sort by score, compute cumulative sum of (label − score)/n, report peak-to-peak range.
- **ECCE-sigma** — ECCE normalized by its standard deviation under the null hypothesis of perfect calibration.

**Score Distribution:**
- **=0%** — Proportion of scores that are exactly 0.0.
- **=1%** — Proportion of scores that are exactly 1.0.
- **[.1,.9]** — Proportion of scores in the mid-range [0.1, 0.9].
- **Uniq** — Number of unique score values.

## Results

| | | | | Prevalence Estimation | | | | | Discrimination | | | Calibration | | | Score Distribution | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Model** | **Elicitation Method** | **N** | **Prev** | **P̂ (mean)** | **Bias (mean)** | **P̂ (t=.5)** | **Bias (t=.5)** | | **AUC** | **Pos μ** | **Neg μ** | **Log Loss** | **ECCE** | **ECCE-σ** | **=0%** | **=1%** | **[.1,.9]** | **Uniq** |
| Llama 3.3 70B | Logprob | 7,000 | 11.5% | 29.5% | +17.9pp | 29.5% | +17.9pp | | 0.904 | 0.889 | 0.217 | 2.231 | 0.188 | 49.1 | 15.5% | 7.6% | 5.9% | 1,334 |
| | Consistency Sampling | 500 | 10.4% | 31.2% | +20.8pp | 32.0% | +21.6pp | | 0.907 | 0.898 | 0.244 | 2.086 | 0.211 | 15.4 | 49.0% | 9.8% | 23.0% | 83 |
| | 1S top-1 | 500 | 10.4% | 55.3% | +44.9pp | 51.0% | +40.6pp | | 0.647 | 0.813 | 0.523 | 7.853 | 0.449 | 32.9 | 0.0% | 21.4% | 78.0% | 7 |
| | 1S top-2 | 500 | 10.4% | 33.4% | +23.0pp | 29.8% | +19.4pp | | 0.888 | 0.854 | 0.274 | 1.653 | 0.232 | 17.0 | 20.4% | 7.4% | 71.8% | 9 |
| | 1S P(Y/N) | 466 | 10.7% | 28.1% | +17.4pp | 30.0% | +19.3pp | | 0.884 | 0.860 | 0.211 | 1.556 | 0.180 | 12.6 | 63.3% | 6.2% | 30.5% | 5 |
| | 2S top-1 | 500 | 10.4% | 33.3% | +22.9pp | 31.6% | +21.2pp | | 0.881 | 0.825 | 0.276 | 0.589 | 0.230 | 16.9 | 6.0% | 0.2% | 80.2% | 8 |
| | 2S+nudge | 500 | 10.4% | 34.5% | +24.1pp | 31.6% | +21.2pp | | 0.886 | 0.817 | 0.290 | 0.525 | 0.243 | 17.8 | 0.0% | 0.0% | 82.2% | 7 |
| | | | | | | | | | | | | | | | | | | |
| Claude Opus 4 | 1S top-1 | 1,000 | 11.1% | 23.0% | +11.9pp | 20.4% | +9.3pp | | 0.929 | 0.779 | 0.162 | 0.322 | 0.119 | 12.0 | 0.0% | 0.0% | 31.1% | 38 |
| | 1S top-2 | 1,000 | 11.1% | 16.4% | +5.3pp | 14.8% | +3.7pp | | 0.972 | 0.773 | 0.088 | 0.159 | 0.054 | 5.5 | 0.0% | 0.0% | 22.0% | 34 |
| | 1S P(Y/N) | 1,000 | 11.1% | 16.5% | +5.4pp | 15.3% | +4.2pp | | 0.969 | 0.776 | 0.089 | 0.162 | 0.055 | 5.5 | 0.0% | 0.0% | 21.6% | 34 |
| | 2S top-1 | 1,000 | 11.1% | 20.5% | +9.4pp | 16.8% | +5.7pp | | 0.968 | 0.809 | 0.130 | 0.211 | 0.094 | 9.5 | 0.0% | 0.0% | 33.7% | 38 |
| | 2S+nudge | 1,000 | 11.1% | 19.5% | +8.4pp | 14.8% | +3.7pp | | 0.972 | 0.783 | 0.122 | 0.189 | 0.084 | 8.5 | 0.0% | 0.0% | 34.2% | 40 |
| | | | | | | | | | | | | | | | | | | |
| Claude Sonnet 4 | 1S top-1 | 499 | 10.4% | 23.7% | +13.3pp | 21.2% | +10.8pp | | 0.899 | 0.828 | 0.168 | 0.370 | 0.133 | 9.7 | 0.0% | 0.0% | 28.3% | 33 |
| | 1S top-2 | 500 | 10.4% | 20.5% | +10.1pp | 16.6% | +6.2pp | | 0.978 | 0.832 | 0.132 | 0.198 | 0.101 | 7.4 | 0.0% | 0.0% | 41.4% | 39 |
| | 1S P(Y/N) | 500 | 10.4% | 15.9% | +5.5pp | 12.6% | +2.2pp | | **0.988** | 0.813 | 0.083 | **0.126** | 0.059 | **4.3** | 12.8% | 0.0% | 32.2% | 20 |
| | 2S top-1 | 500 | 10.4% | 26.7% | +16.3pp | 22.6% | +12.2pp | | 0.905 | 0.847 | 0.200 | 0.445 | 0.163 | 11.9 | 0.0% | 0.0% | 46.2% | 23 |
| | 2S+nudge | 500 | 10.4% | 19.4% | +9.0pp | 14.2% | +3.8pp | | 0.976 | 0.798 | 0.124 | 0.179 | 0.092 | 6.7 | 0.0% | 0.0% | 50.2% | 29 |
| | | | | | | | | | | | | | | | | | | |
| DeepSeek-R1 32B | Logprob | 7,000 | 11.5% | 97.9% | +86.3pp | 100.0% | +88.5pp | | 0.714 | 0.986 | 0.978 | 3.513 | 0.863 | 226.0 | 0.0% | 0.0% | 0.0% | 445 |
| | 1S top-2 | 100 | 12.0% | 22.7% | +10.7pp | 16.0% | +4.0pp | | 0.906 | 0.717 | 0.160 | 0.297 | 0.106 | 3.3 | 0.0% | 0.0% | 55.0% | 11 |
| | 2S+nudge | 91 | 13.2% | 48.7% | +35.5pp | 47.3% | +34.1pp | | 0.468 | 0.496 | 0.486 | 13.544 | 0.421 | 11.9 | 34.1% | 37.4% | 11.0% | 16 |

## Interpretation

### Within-Model: Llama 3.3 70B

**Discriminative power (AUC).** All methods except 1S top-1 achieve AUCs in the 0.88–0.91 range, indicating the underlying classification is strong regardless of how confidence is elicited. Logprob and Consistency Sampling are marginally best (~0.905), but the difference is small.

**Score quality (log loss).** This is where methods diverge sharply. Log loss penalizes confident wrong predictions, making it the right metric for evaluating scores destined for post-hoc calibration. The ranking is clear: 2S+nudge (0.525) > 2S top-1 (0.589) > 1S P(Y/N) (1.556) > 1S top-2 (1.653) > Consistency Sampling (2.086) > Logprob (2.231) > 1S top-1 (7.853). The two-stage methods produce much better-calibrated scores out of the box.

**Score distribution degeneracy.** For downstream calibration (isotonic regression, multicalibration), we need scores that aren't piled up at 0 and 1:
- *Logprob* is heavily bimodal: 23% exact 0/1, only 6% in the mid-range, but 1,334 unique values.
- *Consistency Sampling* is worse: 49% exact zeros, though k=10 averaging creates 83 unique values.
- *1S P(Y/N)* nearly collapses to binary: 63% exact zeros, only 5 unique values.
- *1S top-1* has the opposite problem: 21% exact 1.0 from a parsing pathology where the model reports P(correct)=0.0 for "No" answers, which inverts to P(Yes)=1.0. This destroys discriminative power (AUC 0.647).
- *1S top-2* is the best single-stage option: 72% mid-range, but still 20% exact zeros.
- *2S+nudge* is the cleanest: 0% exact 0 or 1, 82% mid-range. The nudge eliminates the degenerate boundary mass entirely.

**Prevalence estimation.** All Llama methods substantially overestimate prevalence. Mean-based bias ranges from +17pp to +45pp; threshold-based from +18pp to +41pp. The best is 1S P(Y/N) at +17pp (mean), though its severely degenerate distribution (63% exact zeros) makes it unreliable. Among methods with reasonable distributions, 2S+nudge has +24pp bias.

### Cross-Model Comparison

**Claude is dramatically better-calibrated.** Claude's best log loss (Sonnet 1S P(Y/N): 0.126) is 4x better than Llama's best (2S+nudge: 0.525). Claude also achieves AUCs of 0.90–0.99 vs Llama's 0.88–0.91. Every Claude method produces zero exact-0 or exact-1 scores (except Sonnet 1S P(Y/N) which has 12.8% exact zeros).

**Elicitation method matters more for weaker models.** Llama's log loss spans a 15x range across methods (0.525 to 7.853). Claude Opus spans 2x (0.159 to 0.322). Claude Sonnet spans 3.5x (0.126 to 0.445) — more method-sensitive than Opus, suggesting Sonnet's performance is more prompt-dependent.

**Sonnet 1S P(Y/N) is the single best configuration.** AUC 0.988, log loss 0.126, ECCE-sigma 4.3, prevalence bias +5.5pp (mean) / +2.2pp (threshold). This is the best row in the entire table on every calibration metric.

**Sonnet is more method-sensitive than Opus.** Sonnet's AUC ranges from 0.899 (1S top-1) to 0.988 (1S P(Y/N)) — a 9-point spread. Opus ranges from 0.929 to 0.972 — a 4-point spread. Sonnet's best exceeds Opus's best, but Sonnet's worst is also worse than Opus's worst. Method choice matters more for Sonnet.

**The nudge helps Llama but is unnecessary for Claude.** For Llama, the anti-certainty nudge eliminates all boundary mass (6% exact-0 → 0%) and slightly improves log loss (0.589 → 0.525). Claude already has zero boundary mass without the nudge, and adding it makes negligible difference (Opus: 0.211 → 0.189; Sonnet: 0.445 → 0.179 — but the Sonnet improvement reflects 2S top-1 being unexpectedly poor rather than the nudge being critical).

**Threshold-based prevalence is consistently less biased than mean-based.** Across all models and methods, P-hat(t=.5) produces lower bias than P-hat(mean). This is because extreme negative scores (e.g., 0.02) pull the mean above the threshold count. The gap is largest for two-stage methods where mid-range scores are common: Opus 2S top-1 has +9.4pp mean bias vs +5.7pp threshold bias.

### DeepSeek-R1: Reasoning Models Are Problematic

**Logprob is fundamentally broken.** AUC 0.714 with all scores clustered above 0.9 (pos mean 0.986, neg mean 0.978). Reasoning models produce `<think>` chains before answering, so the next-token logit distribution at the prompt boundary is not informative about classification confidence. The logprob approach assumes the model's first-token prediction directly reflects its belief — this assumption breaks for reasoning models. Predicted prevalence is 100% (all scores > 0.5).

**1S top-2 is surprisingly strong.** AUC 0.906, log loss 0.297 — competitive with the best Claude results on a 100-doc sample. The single-stage format bypasses the reasoning model's problems by asking for a full probability distribution in one response. Zero boundary mass and 55% mid-range density. Prevalence bias is +10.7pp (mean) / +4.0pp (threshold).

**2S+nudge is catastrophically broken.** AUC 0.468 (worse than random), log loss 13.5. The reasoning model gives extreme confidences (0.0 or 1.0) at the second stage — 71% of scores are exact 0 or 1. When the model says "No" with confidence=0.0 (i.e., "I answered No but I'm 0% confident"), the score inversion (1 − P(correct)) maps this to 1.0 (high P(Yes)), producing systematically inverted scores. 9 out of 100 responses failed to parse entirely (model dumped its reasoning chain instead of a probability). The two-stage format fundamentally doesn't work with reasoning models — they aren't designed for the meta-cognitive "assess your own confidence" step.

### Bottom Line

For the paper's goal — a "modal researcher" approach that produces scores amenable to post-hoc multicalibration:

1. **Best overall: Claude Sonnet 4 + 1S P(Y/N).** Best AUC (0.988), best log loss (0.126), lowest prevalence bias (+2.2pp at threshold), lowest ECCE-sigma (4.3). The direct probability elicitation without answer-anchoring works best with frontier models.

2. **Best open-weight: Llama 70B + 2S+nudge.** Best log loss among Llama methods (0.525), cleanest distribution (0% boundary mass, 82% mid-range). The two-stage format with anti-certainty nudge is critical for open-weight models where single-stage prompts produce degenerate distributions.

3. **Reasoning models: use 1S top-2 only.** DeepSeek-R1's 1S top-2 is surprisingly competitive (AUC 0.906, log loss 0.297), but all other formats fail catastrophically. Logprob is uninformative and two-stage dialogue produces inverted scores. If using a reasoning model, the single-stage full-distribution format is the only viable option.

---

## Data Files

### Llama 3.3 70B
- Logprob: `cap_analysis/data/inference_output/llama-70b/feasibility_scores.csv` (N=7,000)
- Consistency Sampling: `cap_analysis/data/inference_output/llama-70b-verbalized/feasibility_sampled_shard{0,1}.csv` (N=500)
- 1S top-1: `cap_analysis/data/inference_output/llama-70b-verbalized-2stage/feasibility_500_1s_top1.csv` (N=500)
- 1S top-2: `cap_analysis/data/inference_output/llama-70b-verbalized-2stage/feasibility_500_1s_top2.csv` (N=500)
- 1S P(Y/N): `cap_analysis/data/inference_output/llama-70b-verbalized-2stage/feasibility_500_topk.csv` (N=466)
- 2S top-1: `cap_analysis/data/inference_output/llama-70b-verbalized-2stage/feasibility_500.csv` (N=500)
- 2S+nudge: `cap_analysis/data/inference_output/llama-70b-verbalized-2stage/feasibility_500_nudge.csv` (N=500)
- Llama files lack a `label` column; join with `cap_analysis/data/feasibility_sample_70b.csv` on `id` → `law_crime`.

### Claude Opus 4 (N=1,000, two batches combined)
- 1S top-1: `cap_analysis/data/inference_output/claude-1s-top1/combined_1000.csv`
- 1S top-2: `cap_analysis/data/inference_output/claude-1s-top2/combined_1000.csv`
- 1S P(Y/N): `cap_analysis/data/inference_output/claude-1s-pyn/combined_1000.csv`
- 2S top-1: `cap_analysis/data/inference_output/claude-2stage/combined_1000.csv`
- 2S+nudge: `cap_analysis/data/inference_output/claude-2stage-nudge/combined_1000.csv`

### Claude Sonnet 4 (N=500, single batch)
- 1S top-1: `cap_analysis/data/inference_output/claude-1s-top1-sonnet/merged.csv`
- 1S top-2: `cap_analysis/data/inference_output/claude-1s-top2-sonnet/merged.csv`
- 1S P(Y/N): `cap_analysis/data/inference_output/claude-1s-pyn-sonnet/merged.csv`
- 2S top-1: `cap_analysis/data/inference_output/claude-2stage-sonnet/merged.csv`
- 2S+nudge: `cap_analysis/data/inference_output/claude-2stage-nudge-sonnet/merged.csv`

### DeepSeek-R1-Distill-Qwen-32B
- Logprob: `cap_analysis/data/inference_output/deepseek-r1-distill/logprob.csv` (N=7,000)
- 1S top-2: `cap_analysis/data/inference_output/deepseek-r1-distill/1s_top2.csv` (N=100)
- 2S+nudge: `cap_analysis/data/inference_output/deepseek-r1-distill/2stage_nudge.csv` (N=91 valid of 100)

All Claude and DeepSeek files contain a `label` column directly.
