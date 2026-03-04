---
title: "SI Appendix: Multicalibration Is Necessary for Unbiased Model-Based Prevalence Estimation"
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

In the simulation, multicalibration applies stratum-specific additive corrections:
$$h_{\text{mc}}(X) = h(X) + \hat{\epsilon}_g \quad \text{for } X \in \text{stratum } g$$
where $\hat{\epsilon}_g = \bar{Y}_g - \bar{h}_g$ is estimated on calibration data within each stratum. In the empirical applications, multicalibration uses MCGrad [@tax2025mcgrad].

## S2. Score Squashing Sensitivity Analysis

[TODO: Add results showing robustness of MCGrad estimates to $\epsilon \in \{0.01, 0.05, 0.10\}$ in the CAP application. Run the analysis with different epsilon values and report the bias across scenarios.]

## S3. Additional Figures

[TODO: Add any supplementary figures, e.g., calibration curves, detailed bootstrap distributions.]

# References
