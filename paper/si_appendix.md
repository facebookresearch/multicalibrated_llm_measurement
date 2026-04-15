---
title: "SI Appendix: Multicalibration for Unbiased Model-Based Prevalence Estimation"
bibliography: references.bib
geometry: margin=1in
fontsize: 11pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{amsmath}
  - \usepackage{amssymb}
---

# SI Appendix

## S1. Formal Definitions of Prevalence Estimation Methods

This section provides full mathematical definitions of the seven prevalence estimation methods compared in the simulation study.

**Setup.** Let $h(X) \in [0,1]$ denote the device's probabilistic prediction for input $X$, with true label $Y \in \{0,1\}$. The goal is to estimate the target prevalence $\pi^* = P^*(Y=1)$ using only unlabeled target data $\{X_i^*\}_{i=1}^n$ and calibration parameters estimated from a labeled source dataset.

### S1.1 Uncalibrated Averaging (Raw Scores)

$$\hat{\pi}_{\text{raw}} = \frac{1}{n} \sum_{i=1}^n h(X_i^*)$$

### S1.2 Classify & Count

Given a threshold $\tau$ chosen on calibration data:
$$\hat{\pi}_{\text{CC}} = \frac{1}{n} \sum_{i=1}^n \mathbf{1}[h(X_i^*) \geq \tau]$$

In the simulation, $\tau$ is chosen so that $\hat{\pi}_{\text{CC}}$ matches the true prevalence on the calibration set.

### S1.3 Rogan-Gladen (Adjusted Count)

$$\hat{\pi}_{\text{RG}} = \frac{\hat{\pi}_{\text{CC}} - \widehat{\text{FPR}}}{\widehat{\text{TPR}} - \widehat{\text{FPR}}}$$

where $\widehat{\text{TPR}}$ and $\widehat{\text{FPR}}$ are estimated from calibration data at threshold $\tau$ [@rogan1978estimating].

### S1.4 Probabilistic Adjusted Classify & Count (PACC)

$$\hat{\pi}_{\text{PACC}} = \frac{\bar{h} - \hat{\mu}_0}{\hat{\mu}_1 - \hat{\mu}_0}$$

where $\bar{h} = \frac{1}{n}\sum_i h(X_i^*)$, $\hat{\mu}_1 = \hat{\mathbb{E}}[h(X)|Y=1]$, and $\hat{\mu}_0 = \hat{\mathbb{E}}[h(X)|Y=0]$ are estimated from calibration data [@gonzalez2017review].

### S1.5 SLD (EMQ)

The Saerens-Latinne-Decaestecker algorithm iterates:

1. Initialize $\hat{\pi}^{(0)}$ from the source prevalence.
2. E-step: Adjust posteriors for the new prior:
$$\tilde{h}_i^{(t)} = \frac{(\hat{\pi}^{(t)} / \pi_s) \cdot h(X_i^*)}{(\hat{\pi}^{(t)} / \pi_s) \cdot h(X_i^*) + ((1-\hat{\pi}^{(t)}) / (1-\pi_s)) \cdot (1 - h(X_i^*))}$$
3. M-step: $\hat{\pi}^{(t+1)} = \frac{1}{n}\sum_i \tilde{h}_i^{(t)}$
4. Repeat until convergence [@saerens2002adjusting].

### S1.6 Global Calibration

In the simulation, global calibration applies a multiplicative correction:
$$h_{\text{cal}}(X) = c \cdot h(X)$$
where $c = \bar{Y}_{\text{cal}} / \bar{h}_{\text{cal}}$ is estimated on calibration data. In the empirical applications, global calibration uses isotonic regression.

### S1.7 Multicalibration

In the simulation, which has a single binary covariate, multicalibration reduces to stratum-specific additive corrections:
$$h_{\text{mc}}(X) = h(X) + \hat{\epsilon}_g \quad \text{for } X \in \text{stratum } g$$
where $\hat{\epsilon}_g = \bar{Y}_g - \bar{h}_g$ is estimated on calibration data within each stratum.

In the empirical applications, we use MCGrad [@tax2026mcgrad], a multicalibration algorithm based on gradient boosting. MCGrad operates in logit space: given a base predictor $f_0(X)$ with logit $F_0(X) = \text{logit}(f_0(X))$, it iteratively fits gradient boosted decision trees (GBDTs) on the residuals between labels and current predictions. At each round $t$, a GBDT $g_t$ is trained with the current logit predictions as `init_score` and with the feature matrix consisting of the segment features (categorical and numerical) augmented by the current logit prediction as an additional input feature. The logit predictor is then updated as $F_{t+1}(X) = \alpha_t \cdot (F_t(X) + g_t(X))$, where $\alpha_t$ is an unshrinkage factor estimated by logistic regression to counteract the GBDT's learning rate. By including the prediction as a feature, GBDT splits naturally discover miscalibrated regions in the joint space of features and score levels, thereby approximating multicalibration without requiring explicit group specification. Early stopping on a validation set prevents overfitting. MCGrad uses LightGBM as the GBDT implementation. See @tax2026mcgrad for convergence results and deployment details.

## S2. Robustness: Replication with Open-Weight LLM (Llama 3.3 70B)

The main text reports results using Claude Opus 4.6 as the LLM measurement device. To verify that the findings are not specific to a particular model, we replicate the CAP analysis using Llama 3.3 70B Instruct [@llama2024herd] (4-bit NF4 quantized, run on a single A100 80GB GPU). This section reports results using two score extraction methods: token log-probabilities and verbalized confidence elicitation.

### S2.1 Score Extraction Methods

**Log-probabilities.** For each document, the model is prompted with the CAP codebook definition of Law & Crime and asked to respond Yes or No. The score is extracted from next-token log-probabilities: $h(X) = P(\text{Yes}) / (P(\text{Yes}) + P(\text{No}))$. This produces highly bimodal scores: 23% of the 105,000 documents score at exactly 0.0 or 1.0, and only 6% fall in the mid-range [0.1, 0.9]. Because MCGrad's internal logit transform maps values near 0 and 1 to $\pm\infty$, a linear squashing transformation $h'(X) = \epsilon + (1 - 2\epsilon) \cdot h(X)$ with $\epsilon = 0.05$ is applied before fitting MCGrad.

**Verbalized confidence (2-stage).** A two-stage dialogue first asks the model to classify the document (Yes/No), then asks it to estimate the probability that its answer is correct, with an anti-certainty instruction ("Note: very few things are 0% or 100% certain") to discourage degenerate outputs [@tian2023verbalized]. The score is $P(\text{correct})$ if the answer is Yes and $1 - P(\text{correct})$ if No. This produces scores in [0.01, 0.99] with negligible boundary mass and 11 unique score values. No squashing is required.

### S2.2 Data and Calibration

The Llama analysis uses the full 105,000-document sample (15,000 per sub-population for Denmark questions, Spain questions, U.S. bills, and Belgium newspaper; 30,000 for Spanish media; 15,000 for Belgian TV). The calibration set ($n \approx 40{,}000$) is drawn equally from the four in-distribution sub-populations. MCGrad is calibrated with categorical features (country, document type, party) and one numerical feature (decade).

### S2.3 Results: Verbalized Confidence Scores

Table S3 shows prevalence estimation bias using Llama 3.3 70B with verbalized confidence scores.

| Scenario | Shift Type | True Prev. | CC | RG | IPW | Iso. | MCGrad |
|---|---|---|---|---|---|---|---|
| Baseline | None | 8.1% | +14.7 | +0.6 | +0.1 | +0.2 | +0.2 |
| Country shift | Within-cal. | 8.7% | +15.6 | +2.2 | -0.3 | +1.3 | +0.1 |
| Doc-type shift | Within-cal. | 6.5% | +16.0 | +1.7 | +0.0 | +0.8 | +0.1 |
| Party shift | Within-cal. | 9.0% | +14.4 | +0.6 | -0.3 | -0.4 | +0.2 |
| Spain media | OOD doc type | 19.3% | +15.4 | +6.6 | -9.8 | -7.2 | -4.9 |
| Belgium TV | OOD doc type | 11.1% | +13.3 | -0.0 | -3.5 | -1.6 | -3.4 |

*Table S3: Prevalence estimation bias (pp) for Law & Crime topic using Llama 3.3 70B with verbalized confidence scores. CC = Classify & Count, RG = Rogan-Gladen, IPW = importance-weighted estimation, Iso. = isotonic regression.*

The pattern is consistent with the main text's Claude Opus results: MCGrad achieves near-zero bias within the calibration distribution ($\leq 0.2$pp) and degrades on OOD populations (-3.4 to -4.9pp). Several differences are notable:

- **Higher raw CC bias** (+14-16pp vs. +2-5pp with Opus), reflecting the Llama model's poorer calibration out of the box.
- **Comparable MCGrad within-calibration performance** ($\leq 0.2$pp for both models), confirming that multicalibration corrects for model-specific calibration errors.
- **Larger OOD bias** on Spanish media (-4.9pp vs. -2.5pp with Opus binary labels), reflecting the combination of a weaker base model with the coarser verbalized score distribution (11 unique values vs. Opus's 43).

### S2.4 Score Distribution: Log-Probabilities vs. Verbalized Confidence

The bimodal distribution of Llama's log-probability scores illustrates a broader challenge for LLM-based measurement. RLHF-tuned instruction-following models tend to produce highly confident outputs, pushing token probabilities toward 0 or 1. This creates two problems for prevalence estimation: (1) the scores carry little information about uncertainty, producing large raw bias even at baseline (+18pp), and (2) post-hoc calibration methods that operate in logit space (including MCGrad) require score preprocessing to avoid numerical instability.

Verbalized confidence elicitation partially addresses both problems by producing scores that are better distributed (75% in [0.1, 0.9]) and better calibrated out of the box (log loss 0.525 vs. 1.707 for log-probabilities). However, the scores remain coarsely discretized (11 unique values), and as shown in both the Llama and Opus analyses, the quality of the input scores matters less than the metadata features for MCGrad's prevalence estimation performance under shift.

### S2.5 Additional Baselines: SLD and PACC on Llama Scores

The SLD (EMQ) algorithm, designed for label shift rather than covariate shift, diverges catastrophically on Llama's verbalized confidence scores, producing prevalence estimates biased by +33 to +60pp. This occurs because the verbalized scores are not calibrated posteriors, violating SLD's core assumption. PACC shows moderate bias (+0.6 to +5.9pp within calibration, +2.4 to +5.9pp OOD). Full results including SLD and PACC are available in the replication code.

## S3. Detailed Results Tables

### Table S1: ACS Employment Prevalence Estimation Bias

| Setting | Age Dist.    | True Prev. | Raw   | CC    | RG      | PACC    | SLD     | IPW   | Iso.  | MCGrad |
|---------|--------------|------------|-------|-------|---------|---------|---------|-------|-------|--------|
| In-Dist | Original     | 46.0%      | -0.31 | -0.07 | +0.26   | -0.07   | +0.01   | -0.3  | -0.30 | -0.27  |
| In-Dist | Young-skewed | 12.8%      | +1.93 | +2.47 | -12.82  | -12.82  | -11.95  | -0.3  | +2.04 | -0.11  |
| In-Dist | Old-skewed   | 16.8%      | +7.23 | -6.62 | -16.77  | -16.77  | -16.76  | -1.2  | +6.65 | +0.22  |
| In-Dist | Bimodal      | 21.1%      | +4.57 | -1.50 | -18.62  | -19.97  | -16.14  | +4.7  | +4.33 | +0.12  |
| OOD     | Original     | 45.1%      | +1.15 | +1.40 | +2.12   | +2.13   | +2.25   | +1.7  | +1.17 | +1.35  |
| OOD     | Young-skewed | 13.0%      | +2.93 | +3.73 | -12.96  | -12.96  | -11.38  | +0.8  | +3.08 | +0.88  |
| OOD     | Old-skewed   | 16.0%      | +8.47 | -5.64 | -15.97  | -15.97  | -15.97  | +0.1  | +7.91 | +1.01  |
| OOD     | Bimodal      | 20.8%      | +5.91 | +0.09 | -16.14  | -17.27  | -15.20  | +6.3  | +5.69 | +1.13  |

*Prevalence estimation bias in percentage points (pp) under synthetic age distribution shift. Raw = uncalibrated score average, CC = Classify & Count, RG = Rogan-Gladen, IPW = importance-weighted prevalence estimation, Iso. = Isotonic regression. Bootstrap RMSE (200 iterations) closely tracks absolute bias in all scenarios.*

### Table S2: CAP Law & Crime Prevalence Estimation Bias (Claude Opus 4.6)

| Scenario | Shift Type | True Prev. | CC | RG | SLD | IPW | Iso. | MC (binary) | MC (scores) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | None | 8.1% | +1.9 | +0.5 | +6.6 | -0.0 | -0.3 | -0.2 | -0.2 |
| Country shift | Within-cal. | 8.9% | +2.5 | +1.7 | +8.8 | -0.4 | +0.3 | -0.5 | -0.2 |
| Doc-type shift | Within-cal. | 6.8% | +1.6 | -0.3 | +5.1 | -0.4 | -0.2 | -0.2 | +0.0 |
| Spain media | OOD doc type | 19.5% | +3.6 | +3.1 | +21.5 | -12.1 | -2.7 | -2.5 | -4.4 |
| Belgium TV | OOD doc type | 11.1% | +4.8 | +3.7 | +13.7 | -4.4 | +2.4 | +0.4 | +1.6 |

*CC = Classify & Count (fraction of Yes labels); RG = Rogan-Gladen adjustment on binary labels; SLD = Saerens-Latinne-Decaestecker (label shift, applied to probability scores); IPW = importance-weighted estimation (target-specific density ratio); Iso. = isotonic regression on probability scores; MC (binary) = MCGrad on binary labels with base-rate initialization; MC (scores) = MCGrad on probability scores.*

## S4. Simulation: RMSE

![](images/figure_sim_rmse.png){width=100%}

*Figure S2: Root mean squared error (RMSE) under covariate shift for the same four methods shown in Figure 1, averaged over 50 simulation runs. RMSE closely tracks absolute bias for all methods, confirming that variance is small relative to bias at this sample size. MCGrad maintains the lowest RMSE across all shift levels.*

## S5. Simulation: All Methods

![](images/figure_sim_lineplot_all.png){width=100%}

*Figure S3: Simulation bias curves for all seven methods. Rogan-Gladen and PACC exhibit catastrophic failure (bias exceeding -200% at extreme shifts). SLD shows large bias under covariate shift because it assumes label shift. The uncalibrated baseline shows moderate bias. MCGrad maintains near-zero bias throughout.*

## S6. Claude Opus Score Distribution

![](images/figure_cap_score_distribution.png){width=100%}

*Figure S4: Claude Opus 4.6 P(Yes) score distribution by label across six CAP sub-populations. Scores are well-separated (mean 0.75 for positives vs. 0.07 for negatives) with 43 unique values and no boundary mass.*

# References
