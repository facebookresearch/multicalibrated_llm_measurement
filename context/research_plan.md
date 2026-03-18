# Research Plan: Multicalibration for Unbiased Model-Based Prevalence Estimation

## 1. Research Question

When researchers use AI systems — including large language models (LLMs) — as measurement devices to estimate how common a phenomenon is in a population, what calibration properties are required to ensure unbiased prevalence estimates when the target population differs from the validation population?

## 2. Motivation and Background

Researchers across the social sciences, public health, and platform governance increasingly deploy AI classifiers and LLMs as zero-shot measurement devices to estimate the prevalence of phenomena (e.g., disease rates, topic frequencies in text corpora, content policy violations) in populations where manual annotation is infeasible. A critical but overlooked problem arises: standard methods for correcting classifier errors assume that error rates remain stable across populations. When the composition of a target population differs from the validation population — a scenario known as covariate shift — these assumptions break down, and prevalence estimates become biased. Because prevalence estimates inform policy decisions, scientific conclusions, and resource allocation, such bias can have serious downstream consequences.

Existing quantification methods (Classify & Count, Rogan-Gladen adjustment, PACC, SLD/EMQ) attempt to correct for imperfect classifiers but rely on the assumption that device error properties remain constant. The multicalibration literature, developed in the context of algorithmic fairness, provides theoretical guarantees about feature-conditional prediction accuracy but has not been connected to the prevalence estimation problem.

## 3. Research Objectives

1. **Theoretical contribution:** Show that multicalibration — calibration conditional on input features, not just on average — is both necessary and sufficient for unbiased prevalence estimation under covariate shift, connecting the multicalibration literature to the longstanding quantification problem.

2. **Simulation study:** Demonstrate in a controlled setting that standard quantification methods produce bias that grows with the magnitude of distributional shift, while a multicalibrated estimator maintains near-zero bias.

3. **Empirical validation (ACS):** Apply the framework to estimating employment prevalence across U.S. states using the American Community Survey, with synthetic age distribution shifts to create controlled covariate shift scenarios.

4. **Empirical validation (CAP/LLM):** Apply the framework to LLM-based topic classification across four countries using the Comparative Agendas Project, demonstrating that the results extend to the increasingly common setting of LLM-based measurement with non-standard score distributions.

## 4. Data Sources

### 4.1 American Community Survey (ACS)
- **Source:** U.S. Census Bureau, accessed via the publicly available *folktables* Python package
- **Task:** Binary employment status prediction (employed vs. not employed)
- **Features:** 16 sociodemographic variables (age, education, marital status, disability, citizenship, etc.)
- **Scale:** ~2.5 million observations across 14 U.S. states, 2016–2018
- **Labels:** Self-reported employment status (survey ground truth)
- **Privacy:** Public-use microdata; no individual identifiers

### 4.2 Comparative Agendas Project (CAP)
- **Source:** Comparative Agendas Project (https://www.comparativeagendas.net), a publicly available dataset of expert-coded political texts
- **Task:** Binary classification — whether a document addresses Law & Crime (CAP major topic code 12)
- **Documents:** 105,000 political texts from six sub-populations across four countries (Denmark, Spain, United States, Belgium) and four languages (Danish, Spanish, English, Dutch)
- **Document types:** Parliamentary questions, congressional bills, newspaper articles, TV news transcripts, media articles
- **Labels:** Expert-coded policy topic labels provided by the CAP project
- **Privacy:** All texts are from public political proceedings and published media; no private or personal data

### 4.3 Simulation Data
- Synthetically generated; no real-world data involved

## 5. Models and Methods

### 5.1 Classifiers / Measurement Devices
- **ACS application:** Logistic regression trained on ACS features (standard scikit-learn implementation). No LLM involvement.
- **CAP application:** Llama 3.3 70B Instruct (open-weight model by Meta, 4-bit NF4 quantization) used as a zero-shot classifier. The model is prompted with a topic description and asked to classify documents. Scores are extracted via token log-probabilities.

### 5.2 Calibration Methods Compared
- **Uncalibrated raw scores** (baseline)
- **Classify & Count** with prevalence-matched or Youden's J threshold
- **Rogan-Gladen** (Adjusted Count) estimator
- **PACC** (Probabilistic Adjusted Classify & Count)
- **SLD/EMQ** (Saerens-Latinne-Decaestecker EM algorithm)
- **Isotonic regression** (global calibration baseline)
- **MCGrad** (multicalibration algorithm; open-source at https://mcgrad.dev)

### 5.3 Evaluation
- **Primary metric:** Prevalence estimation bias (percentage points)
- **Secondary metric:** Root mean squared error (RMSE) via bootstrap resampling (200 iterations)
- **Shift construction:** Synthetic covariate shift via importance-weighted resampling (ACS) or held-out sub-populations with novel feature values (CAP)

## 6. Computational Resources

- **ACS analysis:** Standard CPU computation (scikit-learn, MCGrad). No GPU required.
- **Simulation:** Standard CPU computation. No GPU required.
- **CAP LLM inference:** Single A100 GPU (40GB or 80GB) for running Llama 3.3 70B (4-bit) on 105,000 documents. Estimated runtime: 1–5 hours per scoring configuration.
- **CAP analysis notebook:** Standard CPU computation (MCGrad, scikit-learn, pandas).

## 7. Risks and Mitigations

### 7.1 Data Privacy
- **ACS:** Public-use microdata released by the U.S. Census Bureau for research purposes. No individual identifiers.
- **CAP:** Publicly available political texts from government proceedings and published media. No private or personal data.
- **No user data:** This research does not involve any Meta user data, internal platform data, or proprietary datasets.

### 7.2 Model Risks
- **Llama 3.3 70B** is used strictly as a text classifier (binary topic detection), not for content generation. The model receives political texts and produces a Yes/No classification regarding topical relevance.
- The model is not fine-tuned, modified, or retrained. It is used via standard inference with publicly available weights.
- No model outputs are published or released — only aggregate statistics (prevalence estimates, bias measurements, AUC scores).

### 7.3 Potential for Misuse
- The research identifies a vulnerability in existing measurement practices (biased prevalence estimates under shift) and proposes a mitigation (multicalibration). The findings are defensive in nature: they help researchers produce more accurate measurements.
- No new attack vectors, exploits, or dual-use capabilities are developed.

### 7.4 Societal Impact
- **Positive:** The research improves the reliability of AI-based measurement, which is increasingly used for policy-relevant decisions (disease prevalence tracking, content moderation metrics, social science measurement). Reducing measurement bias directly benefits the quality of downstream decisions.
- **Negative risks are minimal:** The research does not introduce new measurement capabilities, only improves the accuracy of existing ones.

## 8. Outputs and Deliverables

- **Paper:** Submitted to a peer-reviewed venue (target: PNAS or similar)
- **Code:** Simulation and analysis code will be made publicly available
- **Software:** MCGrad is already open-source at https://mcgrad.dev (separate from this paper)
- **No model release:** No new models are trained or released as part of this research

## 9. Team

- Thomas Leeper (Meta)
- Fridolin Linder (Meta)
- Daniel Haimovich (Meta)
- Niek Tax (Meta)
- Lorenzo Perini (Meta)
- Milan Vojnovic (Meta & London School of Economics)

## 10. Timeline

- Research and analysis: completed
- Internal review: in progress
- Submission: upon review approval
