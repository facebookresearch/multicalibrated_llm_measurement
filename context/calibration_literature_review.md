# Extracting Well-Calibrated Probabilities from LLMs for Text Classification: A Literature Review

## Context

This review targets the specific problem encountered in your paper: using Llama 3.1 8B Instruct (and Llama 3.3 70B) for binary topic classification of parliamentary texts, extracting P(Yes)/(P(Yes)+P(No)) from token logprobs, and obtaining extremely bimodal score distributions (79--94% of positives above 0.99, 53--78% of negatives below 0.01) that are poorly calibrated for prevalence estimation.

---

## 1. Token Logprob Extraction (Current Approach)

### How it works
Prompt the model with a binary question ("Is this about Law & Crime? Respond Yes or No"), extract the log-probabilities assigned to the first output token for "Yes" and "No", then compute:

$$P(\text{Yes}) = \frac{\exp(\log p_{\text{Yes}})}{\exp(\log p_{\text{Yes}}) + \exp(\log p_{\text{No}})}$$

### Why it produces bimodal scores

The bimodality is **not inherent to all LLMs** but is a direct consequence of **instruction tuning (SFT + RLHF/DPO)**. The mechanism involves several reinforcing factors:

1. **RLHF/DPO training pushes toward decisive outputs.** During preference optimization, the model is rewarded for giving clear, committed answers rather than hedging. This amplifies the probability mass on the "correct" token and suppresses alternatives, pushing logit gaps to extreme values. Tian et al. (2023) explicitly document that "RLHF-LMs produce conditional probabilities that are very poorly calibrated" compared to base (pre-trained only) models, which maintain much better calibration.

2. **SFT on instruction-following data.** The supervised fine-tuning stage trains the model on examples where human annotators gave definitive Yes/No answers (not "75% Yes"). This teaches the model that the "correct" behavior is to commit fully to one answer.

3. **Softmax sharpening at scale.** In larger models with wider logit distributions, the softmax function naturally produces more extreme probabilities. The logit gap between the top token and alternatives grows with model capability, making the resulting probabilities more peaked.

4. **Temperature during training vs. inference mismatch.** Some RLHF/DPO training procedures effectively sharpen the policy's output distribution relative to the reference model, creating an implicit low-temperature effect at standard inference temperature (T=1.0).

5. **Quantization effects.** Your use of 4-bit quantized models (NF4) may further affect the logit distribution, though this effect is generally smaller than the RLHF effect.

**Key empirical finding from your own paper:** Despite strong AUC (0.878--0.955), raw averaged scores overestimate Law & Crime prevalence by +15 to +28 percentage points. This is the core disconnect: rank-ordering quality (discrimination) is excellent, but the probability values are systematically wrong.

### Key references
- **Tian et al. (2023)** "Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback." *EMNLP 2023.* -- Documents that RLHF destroys logprob calibration.
- **Plaut, Khanh & Trinh (2024)** "Probabilities of Chat LLMs Are Miscalibrated but Still Predict Correctness on Multiple-Choice Q&A." -- Studies 15 chat LLMs; finds MSPs are consistently miscalibrated but still useful for ranking. Key finding: "a strong directional correlation between Q&A accuracy and MSP correctness prediction, while finding no correlation between Q&A accuracy and calibration error." This means better models discriminate better but are NOT better calibrated.
- **Kadavath et al. (2022)** "Language Models (Mostly) Know What They Know." *Anthropic.* -- Shows base models have reasonable calibration on MC/TF questions that degrades after RLHF.
- **OpenAI (2023)** GPT-4 Technical Report. -- Notes calibration degradation post-RLHF.

### Calibration quality
- ECE for instruction-tuned models on binary tasks is typically 0.15--0.35 (very poor).
- Base (non-instruction-tuned) models typically achieve ECE 0.05--0.15 on the same tasks.

### Produces continuous scores?
**No.** This is the fundamental problem. Instruction-tuned models produce scores piled at 0 and 1.

### Requires labeled data?
No (zero-shot), but the scores need post-hoc calibration which does require labels.

### Practical considerations
- Single forward pass per document (cheapest possible).
- Requires model that exposes logprobs (rules out some API-only models like GPT-4o unless using the logprobs API parameter).
- Score quality depends heavily on prompt wording.

---

## 2. Verbalized / Stated Confidence

### How it works
Instead of (or in addition to) extracting logprobs, ask the LLM to explicitly state a confidence level as part of its text output. Typical prompts:

- "On a scale of 0 to 100, how confident are you that this text is about Law & Crime? Give only a number."
- "Classify this text and state your confidence as a percentage."
- "Rate your certainty: 0% (definitely no), 25% (probably no), 50% (unsure), 75% (probably yes), 100% (definitely yes)."

The model generates a number as text tokens, which is parsed.

### Key references

- **Tian et al. (2023)** "Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback." *EMNLP 2023.*
  - Tested ChatGPT, GPT-4, Claude on TriviaQA, SciQ, TruthfulQA.
  - **Central finding:** Verbalized confidences are "typically better-calibrated than the model's conditional probabilities," reducing ECE by ~50% relative.
  - Verbalized scores use a wider range of the [0,1] interval (less bimodal) because the model has learned from training data that humans express uncertainty in graded terms.
  - Works best for factual QA; effectiveness for binary classification specifically is less studied.

- **Xiong et al. (2024)** "Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs." *ICLR 2024.*
  - Systematic framework with three components: prompting strategies, sampling methods, aggregation techniques.
  - **Key finding:** LLMs are systematically overconfident when verbalizing, potentially imitating human patterns.
  - Calibration improves with model scale.
  - White-box (logprob) methods still slightly outperform black-box (verbalized) methods (AUROC gap: 0.522 to 0.605), but the gap is narrow.
  - Human-inspired prompts ("If you were a human expert, how confident would you be?") reduce overconfidence.

- **Lin, Hilton & Evans (2022)** "Teaching Models to Express Their Uncertainty in Words." *EMNLP 2022 (Findings).*
  - First paper showing GPT-3 can produce calibrated verbalized uncertainty.
  - Model generates answers paired with confidence statements (e.g., "90% confidence").
  - Calibration generalizes moderately under distribution shift.
  - Key insight: verbalized and logit-based uncertainty show comparable generalization.

- **Zhang, Huang et al. (2024)** "Calibrating the Confidence of Large Language Models by Eliciting Fidelity." *EMNLP 2024.*
  - Decomposes confidence into Uncertainty (about the question) and Fidelity (to the answer).
  - Tests on 6 RLHF-optimized models across 4 MC QA datasets.
  - Proposes plug-and-play method that doesn't require model internals.

### Calibration quality
- Reported ECE improvements of ~50% relative over raw logprobs (Tian et al.).
- Still systematically overconfident (Xiong et al.).
- Typical ECE: 0.08--0.20, depending on model and task.

### Produces continuous scores?
**Partially.** Models tend to cluster on "round numbers" (50%, 70%, 80%, 90%, 95%, 100%), so the distribution is not truly continuous but is substantially less bimodal than logprobs. The clustering on round numbers is a well-documented artifact of how humans express confidence, which the model has learned.

### Requires labeled data?
No (zero-shot), though calibration can be improved with post-hoc scaling if labels are available.

### Practical considerations
- Requires parsing model text output (may fail if model doesn't follow instructions).
- Single forward pass per document.
- Prompt engineering matters greatly -- "state your confidence as a percentage from 0 to 100" works better than "how sure are you?"
- Less reliable for smaller models (8B) -- Xiong et al. show calibration improves with scale, so Llama 3.1 8B may not benefit as much as 70B.
- **For your specific use case (binary classification for quantification):** Verbalized confidence might produce a less bimodal distribution, which could help with prevalence estimation. However, the overconfidence bias means it may still overestimate prevalence. Worth testing empirically.

---

## 3. Temperature Scaling / Softmax Temperature

### How it works
Two distinct uses of "temperature" in this context:

**A. Inference-time generation temperature (standard):**
The sampling temperature T applied during generation: $p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$. At T=1 (default), you get the model's trained distribution. Higher T flattens the distribution; lower T sharpens it.

**B. Post-hoc temperature scaling (Guo et al. 2017):**
Learn a single scalar T on a validation set to minimize negative log-likelihood: $\hat{p}_i = \sigma(z_i / T)$. This is a post-processing step applied to logits after the forward pass.

### Post-hoc temperature scaling for logprobs

For your setting, the relevant approach is: extract the raw logits for Yes/No tokens, then apply a learned temperature:

$$P_T(\text{Yes}) = \frac{\exp(\log p_{\text{Yes}} / T)}{\exp(\log p_{\text{Yes}} / T) + \exp(\log p_{\text{No}} / T)}$$

Equivalently, if $s = \log p_{\text{Yes}} - \log p_{\text{No}}$ is the log-odds ratio, then $P_T(\text{Yes}) = \sigma(s/T)$. With $T > 1$, the sigmoid curve is flattened, spreading scores away from 0 and 1.

### Key references
- **Guo, Pleiss, Sun & Weinberger (2017)** "On Calibration of Modern Neural Networks." *ICML 2017.*
  - Foundational paper showing modern deep networks are poorly calibrated (overconfident).
  - Temperature scaling (single parameter T) is "surprisingly effective" -- often matches or beats more complex methods (Platt scaling, isotonic regression, histogram binning).
  - Learned on validation set by minimizing NLL.
  - ECE reduction from 0.15 to 0.01--0.03 on CIFAR/ImageNet for standard classifiers.

- **Liu, Khalifa & Wang (2023)** "LitCab: Lightweight Language Model Calibration over Short- and Long-form Responses." *arXiv 2023.*
  - Adds a single linear layer (input-dependent bias) to LM logits -- a learned, input-conditional generalization of temperature scaling.
  - Reduces ECE by up to 30% on Llama2-7B.
  - Adds < 2% parameters.

### Calibration quality
- For standard classifiers: excellent (ECE 0.01--0.03 after scaling).
- For LLM logprobs: **the situation is fundamentally different**. Standard temperature scaling assumes the logits have a monotone relationship with the true probability, just at the wrong scale. For bimodal LLM scores where 10--24% of negatives score above 0.99, the problem is not just scale but **rank violations** (confidently wrong predictions). Temperature scaling can spread the scores but cannot fix rank violations.
- **Critical point for your use case:** Temperature scaling with T>1 will de-bimodalize the score distribution (scores will move away from 0 and 1), but it cannot achieve calibration if the model is confidently wrong on a substantial fraction of cases. It will reduce, but not eliminate, the prevalence estimation bias.

### Produces continuous scores?
**Yes** -- T>1 directly spreads scores toward 0.5, breaking bimodality. However, the resulting scores are not calibrated, just less extreme.

### Requires labeled data?
- Standard temperature scaling: Yes, a held-out validation set to learn T (typically a few hundred examples suffice).
- Ad hoc T>1: No labels needed, but no calibration guarantee.

### Practical considerations
- Extremely cheap: single scalar parameter.
- For your binary Yes/No setup: apply $P_T = \sigma((s - b)/T)$ where s is the log-odds from the model, T and b are learned on validation data. This is equivalent to Platt scaling.
- **Key limitation for quantification:** Temperature scaling is a form of global calibration. As your paper demonstrates, global calibration is insufficient under covariate shift. Temperature scaling will reduce bias at the calibration distribution but not provide the multicalibration guarantees needed for cross-population prevalence estimation.
- A reasonable practical workflow: temperature-scale first (to de-bimodalize), then apply MCGrad for multicalibration. This two-stage approach may be more numerically stable than applying MCGrad directly to bimodal scores (which is why you currently use the squashing transformation).

---

## 4. Multi-Token / Linguistic Scale Approaches

### How it works
Instead of forcing a binary Yes/No response, provide a multi-point ordinal scale and extract logprobs for each option:

**Example prompt:** "Classify this text on the following scale: (A) Definitely about Law & Crime, (B) Probably about Law & Crime, (C) Unsure, (D) Probably NOT about Law & Crime, (E) Definitely NOT about Law & Crime. Respond with only the letter."

Then extract logprobs for tokens A--E and compute a weighted score:
$$\hat{p} = \frac{4 \cdot p_A + 3 \cdot p_B + 2 \cdot p_C + 1 \cdot p_D + 0 \cdot p_E}{4}$$

Or use a more principled mapping where the weights are learned from calibration data.

### Key references

This is a less well-studied approach with no single canonical paper, but several lines of work are relevant:

- **Zheng et al. (2023)** "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS 2023.* -- Uses 1--10 rating scales for LLM evaluation; documents that LLMs can use fine-grained scales meaningfully, though with position bias and other artifacts.

- **Bai et al. (2023)** "Benchmarking Foundation Models with Language-Model-as-an-Examiner." *NeurIPS 2023.* -- Uses multi-level scales for LLM evaluation.

- **Anthropic / Constitutional AI literature** -- Uses multi-point scales for harmlessness evaluation.

- The **ordinal regression / label smoothing** literature in NLP is tangentially relevant: breaking a binary decision into ordinal categories can produce better-distributed scores.

### Calibration quality
- No systematic studies comparing ordinal-scale logprobs to binary logprobs for calibration in classification tasks specifically.
- The approach may produce less bimodal distributions because the model distributes probability mass across 5 categories instead of 2, but the RLHF-induced overconfidence will still push mass toward the extreme categories ("Definitely Yes" or "Definitely No").
- The weighted combination introduces a modeling assumption (the numeric mapping of categories to probabilities) that may or may not match the true conditional probability.

### Produces continuous scores?
**Somewhat.** The score is a weighted average of 5 category probabilities, so it has more granularity than binary Yes/No. However, if most mass is on the extreme categories (likely with instruction-tuned models), the effective distribution may still be bimodal.

### Requires labeled data?
- Zero-shot for extraction; but the mapping from category weights to calibrated probabilities requires labeled data.

### Practical considerations
- Single forward pass per document.
- Requires careful prompt engineering (the exact wording of scale points matters).
- The model may not use all scale points equally -- instruction-tuned models may still prefer extreme categories.
- Parsing is simpler than verbalized confidence (just extract logprobs for 5 tokens).
- **For your use case:** Worth trying empirically, but the theoretical advantage over binary logprobs + temperature scaling is unclear. The fundamental issue (RLHF overconfidence) affects all logprob-based approaches regardless of the number of output categories.

---

## 5. Chain-of-Thought then Judge

### How it works
Two-stage approach:
1. **Reasoning stage:** Prompt the model to reason step-by-step about whether the text matches the classification criteria.
2. **Judgment stage:** After reasoning, give a final verdict (Yes/No) and/or confidence level.

Then extract the score either from the final verdict's logprobs or from a verbalized confidence.

### Key references

- **Wei et al. (2022)** "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS 2022.*
  - Foundational CoT paper. Shows +17.9% on GSM8K arithmetic.
  - CoT improves accuracy but the paper does not study calibration directly.

- **Kadavath et al. (2022)** "Language Models (Mostly) Know What They Know." *Anthropic.*
  - Introduces P(True) method: generate a candidate answer, then ask "Is this answer True or False?"
  - **Key calibration finding:** P(True) is "well-calibrated" for larger models on diverse MC and TF questions.
  - Calibration improves when the model evaluates multiple sampled answers.
  - Models partially generalize P(IK) ("probability of knowing") across tasks.
  - Calibration degrades on novel tasks and with RLHF fine-tuning.

- **Wang et al. (2023)** "Self-Consistency Improves Chain of Thought Reasoning in Language Models." *ICLR 2023.*
  - Sample multiple CoT reasoning paths at T>0, take majority vote.
  - Improves accuracy significantly (+11--18% on reasoning tasks).
  - The fraction of samples reaching each conclusion provides a natural confidence score.

- **Lanham et al. (2023)** "Measuring Faithfulness in Chain-of-Thought Reasoning." *Anthropic.*
  - Finds "large variation across tasks in how strongly [models] condition on the CoT."
  - Larger models sometimes produce CoT that is *less* faithful to their actual decision process.
  - Implication: CoT may improve expressed calibration without the reasoning actually driving the decision.

### Calibration quality
- CoT generally improves accuracy, which can indirectly improve calibration (fewer confidently wrong predictions).
- The effect on calibration per se is mixed:
  - CoT may **improve** calibration by surfacing relevant considerations that make the model more uncertain when appropriate.
  - CoT may **worsen** calibration by producing post-hoc rationalizations that increase the model's (unjustified) confidence.
- **Lyu et al. (2024, AAAI)** find that "explanatory reasoning... boost[s] calibration," suggesting CoT helps in consistency-based approaches.
- No strong evidence that CoT substantially reduces bimodality of logprobs in binary classification settings.

### Produces continuous scores?
- If using logprobs of the final Yes/No token after CoT: still likely bimodal.
- If using verbalized confidence after CoT: somewhat less bimodal (see section 2).
- If using sampling consistency across multiple CoT paths (section 6): yes, naturally continuous.

### Requires labeled data?
No (zero-shot), but few-shot CoT examples improve quality.

### Practical considerations
- **Compute cost:** Much higher -- each document requires generating a multi-token reasoning chain (50--200 tokens) instead of a single token.
- For 105,000 documents at 8B parameters, this could increase inference time by 10--50x.
- May improve accuracy, which helps calibration indirectly, but won't fix the bimodality problem on its own.
- **For your use case:** CoT is probably not worth the compute cost for the marginal calibration benefit in a large-scale classification pipeline. More promising as a component of the ensemble/consistency approach (section 6).

---

## 6. Ensemble / Consistency Methods

### How it works
Sample multiple completions from the model at temperature T>0 (or use diverse prompts), then compute the fraction of responses that say "Yes" as the confidence score:

$$\hat{p} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{response}_i = \text{Yes}]$$

Variants:
- **Simple sampling:** Same prompt, T>0, sample N times.
- **Self-consistency (Wang et al.):** Sample N CoT reasoning paths, take majority vote; the vote fraction is the confidence.
- **Prompt perturbation:** Rephrase the question N different ways, take the consensus.
- **Multi-model ensemble:** Query multiple models, aggregate.

### Key references

- **Wang et al. (2023)** "Self-Consistency Improves Chain of Thought Reasoning in Language Models." *ICLR 2023.* -- Sample diverse reasoning paths, majority vote. The vote fraction is a natural confidence score.

- **Lyu et al. (2024)** "Calibrating Large Language Models with Sample Consistency." *AAAI 2024.*
  - Systematically tests three consistency metrics across 9 reasoning datasets and multiple models.
  - **Key finding:** "Consistency-based calibration methods outperform existing post-hoc approaches."
  - Larger sample sizes enhance calibration (diminishing returns after ~10--20 samples).
  - Instruction-tuning "creates obstacles" -- instruction-tuned models are harder to calibrate via consistency because they are more deterministic.
  - Recommends specific consistency metrics based on model characteristics.

- **Xiong et al. (2024)** "Can LLMs Express Their Uncertainty?" *ICLR 2024.* -- Tests sampling + aggregation as one component of their framework. Finds consistency-based measures are competitive with verbalized confidence.

- **Kuhn, Gal & Farquhar (2023)** "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation." *ICLR 2023 Spotlight.*
  - Proposes semantic entropy: cluster sampled responses by meaning, compute entropy over semantic clusters.
  - Outperforms lexical-level consistency for uncertainty estimation.
  - Requires multiple samples but accounts for paraphrasing.

- **Chen & Mueller (2023)** "Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness."
  - BSDetector: black-box confidence scoring via multiple samples.
  - Works with API-only models.
  - Selects highest-confidence response, improving overall quality.

### Calibration quality
- Consistency-based methods are among the best-calibrated approaches (Lyu et al. 2024).
- The key advantage: the score is a **frequency** (fraction of Yes responses), which is inherently well-behaved (continuous, bounded in [0,1], and related to the model's predictive distribution in a principled way).
- ECE improvement over raw logprobs: typically 30--60% relative improvement.
- **Critical limitation for instruction-tuned models:** If the model almost always says "Yes" or "No" deterministically (which happens at T=1 with highly confident instruction-tuned models), the consistency score will also be bimodal. You need to use T sufficiently high (e.g., T=0.7--1.5) to get variability, but this may introduce incoherent responses.

### Produces continuous scores?
**Yes, in principle.** With N=20 samples, you get scores in {0/20, 1/20, ..., 20/20}. But if the model is very confident, most documents will score 0/20 or 20/20 even at elevated temperature.

### Requires labeled data?
No -- fully zero-shot. But the raw consistency scores may still benefit from post-hoc calibration.

### Practical considerations
- **Compute cost: N times more expensive.** With N=20 samples, classification of 105K documents requires 2.1M forward passes. At 8B parameters this is substantial but feasible.
- **How many samples?** Lyu et al. suggest diminishing returns after 10--20. For binary classification, N=10--20 is a practical sweet spot.
- **Temperature choice matters:** Too low and you get degenerate consistency (all same answer); too high and responses become random.
- **For your specific problem:** Consistency-based scoring is promising because it naturally produces a frequency-based score. However, with Llama 3.1 8B Instruct, the model may be too deterministic on most documents. Test with T=0.7--1.0 and N=20. If you still see >80% of scores at 0 or 1, the approach won't help with bimodality.
- Consistency scores still need MCGrad/post-hoc calibration for the prevalence estimation use case, but they may provide better input scores (less extreme, more continuous) for the calibration algorithm.

---

## 7. Post-Hoc Calibration

### How it works
Take the raw model scores and apply a learned transformation to calibrate them. Standard methods:

- **Platt scaling:** Fit a logistic regression $\hat{p} = \sigma(a \cdot s + b)$ where s is the raw logit/score. Two parameters (a, b).
- **Temperature scaling:** Special case of Platt scaling with b=0 and a=1/T. One parameter.
- **Isotonic regression:** Non-parametric monotone function fit via the pool-adjacent-violators algorithm. No functional form assumption.
- **Histogram binning:** Bin scores and assign each bin the observed positive rate.
- **Beta calibration:** Fit a Beta distribution to map scores (good for bimodal distributions).
- **Multicalibration (MCGrad, etc.):** Calibration conditional on features -- see your paper.

### Key references

- **Guo et al. (2017)** "On Calibration of Modern Neural Networks." *ICML 2017.*
  - Temperature scaling outperforms more complex methods for standard neural classifiers.
  - ECE reduction from ~0.15 to 0.01--0.03.

- **Niculescu-Mizil & Caruana (2005)** "Predicting Good Probabilities with Supervised Learning." *ICML 2005.*
  - Foundational comparison of calibration methods (Platt scaling, isotonic regression) on traditional ML classifiers.

- **Kull, Silva Filho & Flach (2017)** "Beta Calibration: A Well-Founded and Easily Implemented Improvement on Logistic Calibration for Binary Classifiers." *AISTATS 2017.*
  - Beta calibration handles bimodal/U-shaped score distributions better than Platt scaling.
  - Directly relevant to LLM logprob distributions.
  - Uses three parameters (a, b, c) in a Beta-family transformation.

- **Hébert-Johnson et al. (2018)** "Multicalibration: Calibration for the (Computationally-Identifiable) Masses." *ICML 2018.* -- Foundation for MCGrad.

- **Tax et al. (2026)** "MCGrad: Multicalibration at Web Scale." *KDD 2026.* -- Your group's algorithm.

- **Detommaso et al. (2024)** "Multicalibration for Confidence Scoring in LLMs." *ICML 2024.* -- Applies multicalibration to LLM confidence scores using embedding clustering and self-annotation for group construction.

### How much labeled data is needed?

| Method | Labeled data needed | Notes |
|---|---|---|
| Temperature scaling | 500--2,000 | Single parameter; very sample-efficient |
| Platt scaling | 500--2,000 | Two parameters; similarly efficient |
| Isotonic regression | 2,000--10,000 | Non-parametric; needs enough data per bin |
| Beta calibration | 1,000--5,000 | Three parameters; good for bimodal scores |
| Histogram binning | 5,000--20,000 | Needs many observations per bin |
| MCGrad | 10,000--50,000+ | Needs enough data per (feature-group, score-bin) cell |

### Calibration quality
- Global methods (Platt, isotonic, temperature): excellent for global calibration (ECE < 0.03) but insufficient under covariate shift (as your paper demonstrates).
- MCGrad: near-zero bias within calibration distribution, modest degradation on OOD populations.
- Beta calibration: specifically designed for U-shaped score distributions and may be useful as a preprocessing step before MCGrad.

### Produces continuous scores?
**Yes** -- all calibration methods map to [0,1] and can produce continuous outputs. However, if the input is bimodal with almost all mass at 0 and 1, the calibrated output may still have limited effective resolution. Isotonic regression on bimodal inputs may produce a step function with only 2--3 effective levels.

### Practical considerations
- All methods require a held-out labeled calibration set.
- **For your specific problem:** The key insight from your paper is that global calibration is insufficient; MCGrad is needed. But MCGrad has numerical issues with the extreme bimodal scores (the logit transform maps 0.99 to +4.6 and 0.999 to +6.9). Your current squashing solution ($\epsilon = 0.05$) is practical but ad hoc.
- **Recommended pipeline:** Raw logprobs → Platt scaling or Beta calibration (to de-bimodalize) → MCGrad (for feature-conditional calibration). The first stage provides stable inputs to MCGrad without the need for arbitrary squashing.

---

## 8. Embedding-Based Approaches

### How it works
Instead of using the model's generated text or logprobs, extract the hidden-state embeddings (typically the last layer's representation of the [CLS] or final token) and train a separate classifier head (logistic regression, small MLP) on top.

$$h = \text{LLM}_{\text{encoder}}(x), \quad \hat{p} = \sigma(w^\top h + b)$$

### Key references

- **General approach:** This is the standard "probing" or "linear probe" paradigm from representation learning (Belinkov 2022, "Probing Classifiers: Promises, Shortcomings, and Advances").

- **Detommaso et al. (2024)** -- Uses LLM embedding clustering as features for multicalibration but does not train a classifier head directly.

- **Liu et al. (2024)** "Uncertainty Estimation and Quantification for LLMs: A Simple Supervised Approach."
  - Shows that hidden neural activations contain uncertainty signals.
  - Supervised approach using labeled data.
  - Transfers across OOD scenarios.
  - Works in black-box, grey-box, and white-box settings.

- **Your own paper (SI Appendix):** "We also tested a richer feature set by adding 500 PCA components of the LLM's last-layer hidden states (8192-dim) as additional numerical segment features [for MCGrad]. The embedding features produced negligible changes in bias and RMSE across all scenarios."

### Calibration quality
- Embedding + logistic regression typically produces well-calibrated probabilities (logistic regression is inherently calibrated under correct specification).
- The classifier head can be calibrated post-hoc with standard methods.
- Calibration quality depends on the quality of the embeddings and the amount of labeled training data.

### Produces continuous scores?
**Yes** -- logistic regression on embeddings produces smooth, well-distributed scores (no bimodality). This is a key advantage.

### Requires labeled data?
**Yes** -- this is fundamentally a supervised approach. You need labeled examples to train the classifier head. Typical requirements: 500--5,000 labeled examples depending on embedding quality.

### Practical considerations
- Requires model access to extract embeddings (not possible with all APIs).
- Training the classifier head is cheap (logistic regression on 8192-dim embeddings).
- **Forward pass cost:** Same as logprob extraction (single pass), plus minimal overhead for the classifier.
- **The fundamental trade-off:** Embedding approaches give up the zero-shot capability (you need labeled data for each new task) but gain well-distributed scores. For your setting where you already have 40K labeled calibration examples, this is viable.
- **Your finding that embeddings "produced negligible changes"** when used as MCGrad features suggests the embeddings may be redundant with the classification score for the calibration task. However, using embeddings to *replace* the logprob score entirely (training a separate classifier) is different from using them as calibration features. A logistic regression on embeddings may produce better-distributed scores than the LLM's own logprobs.
- **Caveat:** The embedding classifier is a standard ML classifier and is subject to the same calibration-under-shift issues as any other classifier. You would still need MCGrad for the prevalence estimation use case.

---

## 9. "Calibrate Before Use" (Contextual Calibration)

### How it works
Zhao et al. (2021) identify that few-shot LLM predictions are biased by:
- **Majority label bias:** predicting the most common label.
- **Recency bias:** predicting the label of the most recent example.
- **Common token bias:** predicting tokens common in pre-training.

**Contextual calibration** corrects this by:
1. Run the model on a "content-free" input (e.g., "N/A", empty string, or "Input: [MASK]") with the same prompt template and few-shot examples.
2. Record the output distribution over label tokens -- this captures the model's intrinsic bias.
3. Apply an affine transformation to all predictions to remove this bias: $\hat{p}_{\text{calibrated}} = W^{-1} \hat{p}_{\text{raw}}$, where W is a diagonal matrix derived from the content-free output.

### Key references

- **Zhao, Wallace, Feng, Klein & Singh (2021)** "Calibrate Before Use: Improving Few-Shot Performance of Language Models." *ICML 2021.*
  - Up to 30% absolute accuracy improvement on few-shot tasks.
  - Reduces variance across different prompt orderings and formats.
  - Tested on GPT-3, GPT-2 family.

- **Zhou et al. (2024)** "Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering." *ICLR 2024.*
  - Extends contextual calibration to a batch-level correction.
  - Controls "contextual bias from the batched input" and "unifies various prior approaches."
  - Tested on PaLM 2 and CLIP.
  - Works in both zero-shot and few-shot settings.

### Relevance for zero-shot
- Contextual calibration was designed for few-shot prompting where example ordering introduces bias. In zero-shot, there are no examples to order, so the recency bias is absent.
- However, the **common token bias** and **majority label bias** still apply in zero-shot. The model may have an intrinsic preference for "Yes" over "No" (or vice versa) due to pre-training statistics.
- In your case, the LLM overestimates prevalence by +15--28pp even at baseline, suggesting a systematic bias toward "Yes." Contextual calibration could correct this marginal bias.
- **Key limitation:** Contextual calibration is a global (marginal) correction -- it shifts all predictions by the same amount. It will not fix feature-conditional miscalibration and will not survive covariate shift. It's analogous to bias correction in Platt scaling, which your paper shows is insufficient.

### Calibration quality
- Improves accuracy substantially in few-shot settings.
- Calibration per se is not the primary focus of the paper -- the correction is about removing label bias, not producing calibrated probabilities.
- For zero-shot binary classification: can remove the marginal Yes/No bias but will not produce calibrated probabilities.

### Produces continuous scores?
**No** -- it shifts the raw logprob distribution but doesn't change its shape. If the raw scores are bimodal, the contextually calibrated scores will still be bimodal (just shifted).

### Requires labeled data?
**No** -- uses content-free inputs, no labels needed. But the content-free input must be chosen carefully.

### Practical considerations
- Trivially cheap: one additional forward pass on the content-free input.
- Good as a preprocessing step to remove marginal label bias before applying other calibration methods.
- **For your use case:** Could help with the +15--28pp positive bias if part of it is due to an intrinsic "Yes" bias in the model. But it won't address bimodality or feature-conditional miscalibration.

---

## 10. Additional Approaches

### A. Base model (non-instruction-tuned) logprobs

Use the base Llama 3.1 8B model (not Instruct) for classification. Base models have been shown to have substantially better-calibrated logprobs because they haven't undergone RLHF/DPO.

**Trade-off:** Base models are worse at following instructions, so prompt engineering is harder. You may need to frame the task as text completion rather than question-answering (e.g., "This parliamentary question is about Law & Crime: [Yes/No]" and extract the completion probability).

**References:** Tian et al. (2023) document the base-vs-instruct calibration gap. Kadavath et al. (2022) show base model calibration.

**Practical note for your setting:** This is worth testing. If the base Llama 3.1 8B produces less bimodal scores while maintaining reasonable discrimination (AUC), it could be a better starting point for MCGrad calibration.

### B. Label smoothing during prompting

Instead of binary Yes/No, use soft targets in few-shot examples:
- "Text: [example]. Classification: Probably Yes (80% confidence)"

This may "teach" the model to produce less extreme responses, but the effect is uncontrolled and model-dependent.

### C. Conformal prediction

- **Angelopoulos et al. (2023)** "Prediction-Powered Inference." -- Already cited in your paper.
- Provides valid prediction intervals/sets without calibration assumptions.
- Doesn't directly produce point probabilities but can be used for prevalence estimation with validity guarantees.

### D. Multiple prompt variants + aggregation

Use several differently-worded prompts for the same classification task and average the scores. This is a form of ensemble that doesn't require sampling at T>0:

1. "Is this text about Law & Crime? Yes or No."
2. "Does this text address criminal justice or legal matters? Yes or No."
3. "Would you classify this as related to law, crime, or justice? Yes or No."

Average the logprob-based scores across prompts. This diversifies the model's decision making and can reduce prompt-specific biases.

**Reference:** Sclar et al. (2024) "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design" documents that prompt formatting causes up to 76% accuracy variation -- suggesting substantial gains from prompt diversification.

### E. Hybrid scoring

Combine multiple score extraction methods:
$$\hat{p}_{\text{hybrid}} = \alpha \cdot p_{\text{logprob}} + \beta \cdot p_{\text{verbalized}} + \gamma \cdot p_{\text{consistency}}$$

Learn weights ($\alpha, \beta, \gamma$) on calibration data. This could capture complementary information from each scoring method.

### F. Retrieval-augmented scoring

Provide the model with the classification codebook, examples of positive and negative cases, and relevant context before asking for classification. This may improve accuracy and could affect calibration by anchoring the model's decision to specific criteria.

---

## Summary Comparison Table

| Approach | Requires labeled data? | Produces continuous scores? | Addresses bimodality? | Survives covariate shift? | Compute cost (per doc) | Best ECE reported |
|---|---|---|---|---|---|---|
| 1. Token logprobs (baseline) | No | No (bimodal) | No | No | 1 forward pass | 0.15--0.35 |
| 2. Verbalized confidence | No | Partially (round numbers) | Partially | No | 1 forward pass | 0.08--0.20 |
| 3. Temperature/Platt scaling | Yes (500--2K) | Yes | Yes (T>1) | No (global only) | 1 pass + trivial | 0.01--0.05 |
| 4. Multi-token Likert scale | No | Somewhat | Uncertain | No | 1 forward pass | Not well-studied |
| 5. CoT then judge | No | Depends on scoring | Marginally | No | 10--50x passes | Varies |
| 6. Consistency/sampling | No | Yes (frequency) | Yes (if T high enough) | No | N passes (10--20x) | 0.05--0.15 |
| 7a. Post-hoc global calib. | Yes (1K--10K) | Yes | Yes | No | Trivial | 0.01--0.05 |
| 7b. Multicalibration (MCGrad) | Yes (10K--50K) | Yes | Yes | **Yes** (within calibrated features) | Trivial | ~0 (within distribution) |
| 8. Embedding classifier | Yes (500--5K) | Yes | Yes | No (needs MCGrad too) | 1 pass + trivial | 0.03--0.08 |
| 9. Contextual calibration | No | No (shifts only) | No | No | 1 extra pass | Improves accuracy, not calibration per se |

---

## Recommended Pipeline for Your Use Case

Given your specific goals (prevalence estimation under covariate shift using LLama 3.1/3.3 for parliamentary text classification across countries), the literature suggests the following pipeline:

### Optimal approach:
1. **Score extraction:** Use token logprobs (current approach) OR test verbalized confidence on a subset to see if scores are less bimodal with your specific model/task.
2. **De-bimodalization:** Apply Platt scaling (logistic regression on raw logits) or Beta calibration on a calibration set. This replaces the ad hoc squashing and produces scores that are better-distributed inputs for MCGrad.
3. **Multicalibration:** Apply MCGrad with country, document type, and other features (as you already do).
4. **Prevalence estimation:** Average MCGrad-calibrated scores over the target population.

### Worth testing:
- **Base model logprobs:** Try Llama 3.1 8B (base, not Instruct) to see if logprobs are less bimodal while maintaining discrimination.
- **Consistency scoring with N=10--20 samples at T=0.7:** Produces frequency-based scores that may be better inputs for MCGrad than logprobs.
- **Verbalized confidence:** Quick to test; may provide less bimodal scores especially with the 70B model.

### Not recommended for your setting:
- **CoT:** Too expensive for 105K documents with marginal calibration benefit.
- **Contextual calibration alone:** Insufficient under covariate shift.
- **Multi-token Likert scale:** Insufficient evidence of benefit and still affected by RLHF overconfidence.

---

## Key Papers Reference List

### Calibration of LLMs
1. Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., Manning, C.D. (2023). "Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback." EMNLP 2023.
2. Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., Hooi, B. (2024). "Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs." ICLR 2024.
3. Lin, S., Hilton, J., Evans, O. (2022). "Teaching Models to Express Their Uncertainty in Words." EMNLP 2022 (Findings).
4. Kadavath, S. et al. (2022). "Language Models (Mostly) Know What They Know." arXiv:2207.05221.
5. Plaut, B., Khanh, N.X., Trinh, T. (2024). "Probabilities of Chat LLMs Are Miscalibrated but Still Predict Correctness on Multiple-Choice Q&A."
6. Zhang, M., Huang, M., Shi, R., Guo, L., Peng, C., Yan, P., Zhou, Y., Qiu, X. (2024). "Calibrating the Confidence of Large Language Models by Eliciting Fidelity." EMNLP 2024.
7. Geng, J., Cai, F., Wang, Y., Koeppl, H., Nakov, P., Gurevych, I. (2024). "A Survey of Confidence Estimation and Calibration in Large Language Models." arXiv:2311.08298.

### Calibration Methods (General)
8. Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. (2017). "On Calibration of Modern Neural Networks." ICML 2017.
9. Niculescu-Mizil, A., Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning." ICML 2005.
10. Kull, M., Silva Filho, T., Flach, P. (2017). "Beta Calibration: A Well-Founded and Easily Implemented Improvement on Logistic Calibration for Binary Classifiers." AISTATS 2017.

### Consistency / Sampling
11. Wang, X. et al. (2023). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." ICLR 2023.
12. Lyu, Q. et al. (2024). "Calibrating Large Language Models with Sample Consistency." AAAI 2024.
13. Kuhn, L., Gal, Y., Farquhar, S. (2023). "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation." ICLR 2023 (Spotlight).
14. Chen, J., Mueller, J. (2023). "Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness."

### Contextual / Batch Calibration
15. Zhao, Z., Wallace, E., Feng, S., Klein, D., Singh, S. (2021). "Calibrate Before Use: Improving Few-Shot Performance of Language Models." ICML 2021.
16. Zhou, H. et al. (2024). "Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering." ICLR 2024.

### Chain-of-Thought
17. Wei, J. et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022.
18. Lanham, T. et al. (2023). "Measuring Faithfulness in Chain-of-Thought Reasoning." Anthropic.

### Multicalibration
19. Hébert-Johnson, U., Kim, M.P., Reingold, O., Rothblum, G.N. (2018). "Multicalibration: Calibration for the (Computationally-Identifiable) Masses." ICML 2018.
20. Detommaso, G., Bertran, M., Fogliato, R., Roth, A. (2024). "Multicalibration for Confidence Scoring in LLMs." ICML 2024.

### Other
21. Liu, X., Khalifa, M., Wang, L. (2023). "LitCab: Lightweight Language Model Calibration over Short- and Long-form Responses."
22. Liu, L., Pan, Y., Li, X., Chen, G. (2024). "Uncertainty Estimation and Quantification for LLMs: A Simple Supervised Approach."
23. Sclar, M., Choi, Y., Tsvetkov, Y., Suhr, A. (2024). "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design." (FormatSpread paper.)
24. Lin, Z., Trivedi, S., Sun, J. (2023). "Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models."
