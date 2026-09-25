---
title: "Multicalibration for Unbiased Model-Based Prevalence Estimation"
author:
  - Fridolin Linder^1^
  - Thomas Leeper^1^
  - Daniel Haimovich^1^
  - Niek Tax^1^
  - Lorenzo Perini^1^
  - Milan Vojnovic^1,2^
bibliography: references.bib
csl: chicago-author-date.csl
geometry: margin=1in
fontsize: 11pt
numbersections: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{graphicx}
  - \usepackage{float}
  - \usepackage{amsmath}
  - \usepackage{amssymb}
---

^1^Meta Platforms Inc., ^2^The London School of Economics and Political Science

Corresponding author: Fridolin Linder (flinder@meta.com)

## Abstract {.unnumbered}

Political scientists increasingly use classifiers, including large language models (LLMs), to estimate how prevalent a concept is in a corpus. Standard validation reports discriminative metrics such as accuracy or the area under the ROC curve (AUC) on a labeled sample. This certifies neither unbiased prevalence estimates in the validated population nor their transport to new ones, because discriminative metrics are blind to the feature-dependent errors that bias an aggregate. Existing corrections either rely on label-shift-type invariances or must be re-estimated for every target. Building on @kim2022universal, we show that a *multicalibrated* classifier, one calibrated conditional on input features rather than only on average, yields unbiased prevalence estimates under any covariate shift along the calibrated features within their support, and needs to be fitted only once. In a simulation and two applications, an LLM coding policy topics across four countries and languages and a logistic regression predicting employment from survey data, multicalibration keeps bias below half a percentage point for shifts within the calibrated features, where standard corrections miss by up to 19 points, and degrades only modestly beyond them.

**Keywords:** prevalence estimation, multicalibration, covariate shift, quantification, large language models, measurement validity

# Introduction

Machine learning classifiers, and increasingly large language models (LLMs), have become standard tools to estimate the prevalence of a concept in political text corpora [@baumgartner2006cap; @benoit2026llm; @overos2024protest; @kinglowe2003automated; @weidmann2026democracy; @mellon2024survey; @gilardi2023chatgpt; @ornstein2025parrot]. The quantity of interest is usually not the label on any single document but an *aggregate* count or prevalence [@gonzalez2017review; @hopkinsking2010nonparametric].

The standard way to validate such a classifier is to measure how well it discriminates: its agreement with human codes, reported as accuracy, $F_1$, or AUC and judged against intercoder reliability [@halterman2026codebook; @hayeskrippendorff2007answering]. For prevalence estimation this is not sufficient. First, a classifier can rank documents almost perfectly yet misstate the proportion that are positive, because its false positives and false negatives need not cancel. The quantification literature corrects this with error rates estimated on the labeled sample [@gonzalez2017review]. Second, that correction is tied to the population it was estimated on. When the instrument is applied to a new target, for example to compare sub-corpora or track a quantity over time, the mix of documents changes (languages, venues, eras) while the codebook rule linking content to category stays fixed: covariate shift. If the target holds more of the cases the classifier misjudges, errors that canceled on the validation sample no longer do. A high AUC does not protect against this, and standard validation will not reveal it: validity established in one context need not hold in another [@adcockcollier2001measurement].

Existing tools each address part of the problem, but none yields a *reusable* instrument. Classify \& Count takes the classifier's labels at face value. Its bias-corrected relatives [@rogan1978estimating; @saerens2002adjusting] and the classifier-free ReadMe family [@hopkinsking2010nonparametric; @jerzakkingstrezhnev2023improved] instead carry class-conditional quantities, such as error rates or word profiles within each category, from the labeled sample to the target. That is justified under label shift, where only category frequencies change, but covariate shift generally changes those quantities [@tasche2023invariance]. Design-based supervised learning [@egami2023dsl] and prediction-powered inference [@angelopoulos2023prediction] require gold labels drawn from the population being analyzed. Importance weighting [@shimodaira2000improving; @sugiyama2007direct] shares the covariate-shift assumption but must be re-estimated for each target, and it sets the classifier aside: its estimate is a reweighted average of hand codes, with nothing to reweight where the target leaves the source's support. Related work on learned proxies and text measures [@knoxlucascho2022proxies; @rodriguezspirling2022embeddings] and on LLMs as annotators or substitutes for human data [@ziems2024llmcss; @argyle2023outofone; @bisbee2024synthetic] raises the same worry that imperfect measures distort inference; @weidmann2026democracy find that LLM coders deviate from experts in systematic directions.

Averaging a model's outputs gives an unbiased prevalence estimate if the classifier is *calibrated*, its outputs matching empirical frequencies in the population at hand. But calibration is a property of one population. It degrades under covariate shift even when accuracy is preserved [@yang2023calibration], and confidence elicitation [@tian2023verbalized; @kadavath2022language; @wang2023selfconsistency] targets only this global property. The closest precedent, @calibrateextrapolate2024, calibrates classifier outputs on a labeled sample and extrapolates to the target, assuming a stable calibration curve or class-conditional score distribution. A globally calibrated score keeps neither under covariate shift when its errors vary with the features, because reducing $X$ to a score can break covariate shift [@tasche2022covariate]. Building on @kim2022universal, we argue that *multicalibration*, calibration conditional on the input features [@hebertjohnson2018multicalibration; @detommaso2024mcllm], fills this gap: calibrated once on a labeled source sample, it returns unbiased prevalence estimates on any target whose shift the features capture and that lies within their support [@kim2022universal; @wu2024bridging].

# Prevalence Bias Reduction through Multicalibration

Consider a binary outcome $Y \in \{0,1\}$, features $X$, and a predictor $h(X)\in[0,1]$. The estimand is the target prevalence $\pi^*=P^*(Y=1)$, estimated by averaging predictions over unlabeled target observations. We assume *covariate shift*: $P(X)$ may differ between source and target while $P(Y\mid X)$ is stable, in contrast to *label shift*, which holds $P(X\mid Y)$ fixed.\footnote{The causal heuristic behind these invariance assumptions traces to @scholkopf2012causal; we rely on the invariance, not the causal direction.} We also require overlap: any feature region that occurs in the target also occurs in the source. Under concept drift, where $P(Y\mid X)$ changes, target prevalence cannot generally be recovered without target labels.

Global calibration is sufficient for unbiased prevalence estimation in a given population: if $\mathbb{E}[Y\mid h(X)=p]=p$ for every prediction value $p$, then $\mathbb{E}[h(X)]=\mathbb{E}[Y]$. But it need not survive a change in the distribution of $X$. Partition the population into feature-defined groups $G$ with weights $w_G$, and let $\epsilon_G=\mathbb{E}[h(X)-Y\mid X\in G]$ be the mean signed error in group $G$. If the target changes only the group weights, its prevalence error is $\mathbb{E}^*[h(X)] - \pi^* = \sum_G w_G^*\epsilon_G$. Global calibration in the source requires only that the source-weighted errors cancel; under the target weights they need not.

Discrimination does not prevent this. AUC is invariant to monotone transformations of the scores, which can change prevalence estimates substantially, and accuracy and $F_1$ measure unsigned rather than signed errors. A classifier can therefore perform well overall while erring systematically in a group that is rare in the source and common in the target.

Multicalibration removes these group-level errors rather than relying on their cancellation. Let $\mathcal{G}$ contain all groups that can be formed from the calibrated features, including their interactions and intersections. A predictor $f$ is multicalibrated with respect to $\mathcal{G}$ if $\mathbb{E}[Y\mid f(X)=v,\,X\in G]=v$ for every $G\in\mathcal{G}$ and prediction value $v$ [@hebertjohnson2018multicalibration]. Averaging over prediction values shows that $f$ has zero mean signed error within every group in $\mathcal{G}$. A target shift is captured by $\mathcal{G}$ if it can be expressed as a reweighting of these groups. Because each group error is zero, changing their weights cannot create aggregate error. Under covariate shift and overlap,
$$\mathbb{E}^*[f(X)]=\mathbb{E}^*[Y]=\pi^*.$$
This is the "universal adaptability" guarantee of @kim2022universal, who show that such an estimate competes with inverse propensity weighting under any propensity model the group class can express; multi-accuracy, zero mean error within each group, already suffices (SI Section S1.8). The same predictor serves every target shift the calibrated features capture, without a new correction per target and without seeing the target. Where importance weighting corrects the *sample* toward one target, multicalibration corrects the *classifier* once: the estimate averages bounded predictions over every target document, so no small set of observations can dominate it.

The guarantee is exact only when the target-to-source density ratio lies in the span of the group indicators. Otherwise it is approximate, with error governed by how well the calibrated groups approximate that ratio; with finite calibration data, multicalibration itself also holds only approximately (SI Section S1.8). Whether covariate shift and this representation condition are plausible requires substantive judgment. Overlap can be diagnosed, though not generally certified, using unlabeled target data. We obtain the corrected predictor using MCGrad [@tax2026mcgrad], a post-hoc gradient-boosting multicalibrator.

# Simulation

We first illustrate the mechanism with a binary feature $X$ (for example, language) and a content signal $U\sim N(0,1)$. The outcome model $P(Y=1\mid X,U)=\sigma(a_X+1.5\,U)$ is the same in every population; only $P(X)$ shifts. The classifier's score adds a logit offset for $X=0$ documents only, so it overstates their probability while still discriminating well (AUC $\approx0.89$), and scores overlap across the groups. Each estimator is fit on 10,000 labeled documents at $P(X=0)=0.5$ and applied, with its parameters fixed, to unlabeled targets with $P(X=0)$ from 0.01 to 0.99, over 50 repetitions. Global recalibration is isotonic regression on the score; multicalibration is MCGrad on the score and $X$. Parameters are in SI Section S4.

![](images/figure_sim_lineplot.png){width=88%}

*Figure 1. Relative prevalence bias under covariate shift (50 runs), against the change in $P(X=0)$ from its calibration value of 0.5. Classify \& Count and Rogan-Gladen diverge, isotonic recalibration drifts, and MCGrad stays near zero. Cropped at $\pm40\%$; full range, all methods, and RMSE in SI Sections S4--S5.*

At the calibration distribution every method is approximately unbiased (Figure 1). Under shift, Classify \& Count reaches about $-19\%$, and Rogan-Gladen, whose ratio form amplifies the error, exceeds $\pm30\%$. Isotonic recalibration is the instructive case. It maps a given score to the same probability whatever the document's $X$, averaging over the two groups at their calibration mix: too high for $X=0$ documents and too low for $X=1$. The errors cancel only at that mix, so the bias grows to about $+19\%$ when nearly all documents have $X=0$. MCGrad, which calibrates within each value of $X$, stays within about 2\% throughout, although all four estimators start from the same scores.

# Applications

We apply the method in two settings with known ground truth. First, Claude Opus 4.6\footnote{An open-weights replication with Llama 3.3 70B (SI Section S2) reproduces the qualitative pattern and permits full reproduction.} classifies 30,000 Comparative Agendas Project texts [@baumgartner2006cap] from six sub-populations in four countries and languages for whether they concern Law \& Crime (CAP major topic 12), scored against the expert codes. The topic is rare: 7.9\% at baseline and 6--20\% across scenarios. Second, a logistic regression predicts employment from 16 sociodemographic features in American Community Survey (ACS) microdata. In both, the targets' covariate shift is known by design: within-support reweightings (of countries and document types; of age groups) and out-of-support targets (unseen document types; held-out states). The LLM returns a Yes/No label and an elicited probability [@tian2023verbalized; @kadavath2022language]. We multicalibrate on the bare label, the output nearly all applied studies use, and give the probability to the methods that require a score. Details are in SI Sections S2--S3.

![](images/figure_combined_v1.png){width=100%}

*Figure 2. Absolute prevalence bias (percentage points) by method and scenario. Top: CAP Law \& Crime coding with Claude Opus 4.6, multicalibrated on the Yes/No label. Bottom: ACS employment with logistic regression. Left: within-support targets; right: out-of-distribution targets. Markers denote scenarios and bars their means. Full numbers in SI Tables S1--S2.*

The LLM clears any discriminative bar (per-language AUC 0.98--0.99), yet Classify \& Count, the applied default, is biased by +1.6 to +4.8pp (Figure 2, top), overstating prevalence by nearly 30\% at baseline. Rogan-Gladen reduces this within support but misses by +3.1 to +3.7pp out of it. SLD, run on isotonic-recalibrated scores, is accurate at baseline (+0.1pp) but drifts to +3.8 to +4.7pp out of support, worse than isotonic recalibration alone, because it attributes the covariate shift to a change in the class prior.\footnote{ReadMe [@hopkinsking2010nonparametric], the canonical political-science quantifier, is classifier-free and fails as the label-shift diagnosis predicts, exceeding +60pp under some shifts (SI Section S7).} Isotonic recalibration is accurate within support ($\le 0.9$pp) but drifts out of it (up to 2.6pp). IPW, with a gradient-boosted propensity model on the multicalibration features, is near zero within support but misses by $-11.9$pp on Spanish media, whose feature values are absent from the source. Multicalibration on the bare labels stays within 0.4pp within support and 1.9pp out of it. Feeding it the elicited probabilities performs comparably within support and somewhat worse out of it (SI Table S2): the correction comes from the features and the calibration sample, not from the model's confidences.

The survey application reproduces this ordering with a wider spread (Figure 2, bottom). Rogan-Gladen and SLD fail by 12--19pp under the age shifts, and Classify \& Count and isotonic regression reach 6--8pp at the largest. IPW, using the same learner as MCGrad without tuning, is accurate on most targets but collapses on two ($-10.1$ and $-15.5$pp): with a calibration sample thirty times larger than the target, a few source observations receive extreme weights (SI Section S3). Multicalibration stays within 0.27pp on every within-support target and 0.88--1.35pp on held-out states. Global recalibration thus suffices only when the predictor's errors are roughly homogeneous across the groups a shift reweights, as approximately in CAP. The ACS shifts move age, which drives employment (76\% employed at ages 25--54, 17\% at 65+), so group errors are large and heterogeneous, and only feature-conditional calibration removes them.

# Discussion

Discriminative validation shows only that a classifier separates classes. It is silent on whether the resulting prevalence estimates are unbiased, in the validated population or any other: in our LLM application a classifier with per-language AUC above 0.98 misstated prevalence by roughly 30--40\% of the true rate. Multicalibration removes this bias where the classifier was validated and keeps it removed as the target shifts. Unlike importance weighting, it is fitted once, before any target is seen, and yields a single instrument for a time series or a set of sub-corpora without re-estimation or per-target diagnostics.

However, the guarantee has limits. It holds only along calibrated dimensions, so calibration data must span the anticipated variation. It also requires labeled calibration data, which in zero-shot settings reintroduces some hand-coding, though far less than per-document labeling and only once. Under mixtures the covariate-shift condition holds only approximately. Our applications have ground truth, so we can verify the estimates directly; a practitioner usually cannot, since the target's labels are what is missing.

# Funding {.unnumbered}

This research was conducted as part of the authors' employment at Meta Platforms, Inc. and received no specific external funding.

# Acknowledgements {.unnumbered}

The authors used Claude Opus 4.6, Claude Opus 4.7, and Gemini 3 Pro to assist with literature search, code, data analysis, visualization, and drafting and editing of the manuscript; the full declaration is in SI Section S8. The authors are entirely responsible for the scientific content of the paper, which adheres to the journal's authorship policy.

# Competing Interests {.unnumbered}

All authors are employees of Meta Platforms, Inc.; Milan Vojnovic is also affiliated with the London School of Economics and Political Science. Niek Tax, Lorenzo Perini, Fridolin Linder, Daniel Haimovich, and Milan Vojnovic are authors of MCGrad [@tax2026mcgrad], the open-source multicalibration method evaluated here. The authors declare no other competing interests.

# Data Availability Statement {.unnumbered}

Replication code and data will be deposited in the Political Analysis Dataverse and are available at <https://github.com/facebookresearch/multicalibrated_llm_measurement>. The analyses use the Comparative Agendas Project datasets [@capdata] and American Community Survey microdata [@census2018acspums], accessed through the folktables package [@ding2021retiring].

# References {.unnumbered}
