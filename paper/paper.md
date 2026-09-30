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
fontsize: 12pt
//linestretch: 2
numbersections: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{graphicx}
  - \usepackage{float}
  - \usepackage{amsmath}
  - \usepackage{amssymb}
---

^1^Meta Platforms Inc., ^2^The London School of Economics and Political Science

Corresponding author: Fridolin Linder (fridolin.linder@gmail.com)

## Abstract {.unnumbered}

Political scientists increasingly use classifiers, including large language models (LLMs), to estimate how prevalent a concept is in a population. Standard validation reports discriminative metrics such as accuracy or the area under the ROC curve (AUC) on a labeled sample, which certifies neither unbiased prevalence estimates in the validated population nor their transport to new ones. Existing corrections, rarely used in practice, either assume label shift or must be re-estimated for every target. Building on @kim2022universal, we show that a *multicalibrated* classifier, one calibrated within feature-defined subgroups rather than only on average, needs to be fitted only once and yields unbiased prevalence estimates under any covariate shift that the calibrated groups can express, within their support. In a simulation and two applications, an LLM coding policy topics across four countries and languages and a logistic regression predicting employment from survey data, multicalibration keeps prevalence error below half a percentage point for shifts within the calibrated features, where standard corrections miss by up to 19 points, and degrades only modestly beyond them.

**Keywords:** prevalence estimation, multicalibration, covariate shift, quantification, large language models, measurement validity

# Introduction

Machine learning classifiers, and increasingly large language models (LLMs), have become standard tools for measuring political concepts from text, images and administrative records [@benoit2026llm; @overos2024protest; @mellon2024survey; @ornstein2025parrot; @torres2022learning; @imai2016improving]. The quantity of interest is often not the label on any single unit but an *aggregate* count or prevalence [@hopkinsking2010nonparametric; @gonzalez2017review; @esuli2023learning].

Such classifiers are typically validated by how well they classify units or agree with human coders, using accuracy, $F_1$, AUC or intercoder reliability [e.g., @birkenmaier2024search; @tornberg2024bestpractices; @knoxlucascho2022proxies; @halterman2026codebook; @grimmer2013text; @barbera2021automated; @gilardi2023chatgpt; @burnham2025stance]. For prevalence estimation this is insufficient, for two reasons. First, a classifier can rank units almost perfectly yet misstate the share of positives, because its false positives and false negatives need not cancel. Second, even a correction estimated on the labeled sample is tied to that sample's population. When the instrument is applied to a new target, such as another sub-corpus or a later period, the mix of units can change (covariate shift). If the target holds more of the cases the classifier misjudges, errors that canceled in the validation sample no longer do: validity established in one context need not hold in another [@adcockcollier2001measurement; @torres2022learning; @decterfrain2022proxy].

In practice, most applied work simply counts the classifier's labels, known as Classify \& Count [@esuli2023learning]. Corrections exist, but each addresses only part of the problem, and none yields a *reusable* instrument. Adjusted Classify \& Count [@rogan1978estimating; @saerens2002adjusting] and the classifier-free ReadMe family [@hopkinsking2010nonparametric; @jerzakkingstrezhnev2023improved] carry class-conditional quantities, such as error rates or word profiles within each category, from the labeled sample to the target. This is justified under label shift, where only category frequencies change [@tasche2023invariance], but covariate shift generally changes those quantities [@tasche2022covariate]. Design-based supervised learning [@egami2023dsl] and prediction-powered inference [@angelopoulos2023prediction] require gold labels drawn from the population being analyzed or, for the latter, a label shift or known covariate-shift weights linking the two. Importance weighting [@shimodaira2000improving; @sugiyama2007direct; @tasche2023invariance] shares the covariate-shift assumption but must be re-estimated for each target, and it sets the classifier aside: its estimate is a reweighted average of hand codes, with nothing to reweight where the target lies outside the source's support. Work on learned proxies [@knoxlucascho2022proxies], LLM annotators [@ziems2024llmcss] and LLMs as substitutes for survey respondents [@argyle2023outofone; @bisbee2024synthetic] raises the same worry that imperfect measures distort inference; @weidmann2026democracy find that LLM coders deviate from experts in systematic directions.

Averaging a classifier's scores gives an unbiased prevalence estimate if the classifier is *calibrated*, that is, if its scores match empirical frequencies in the population at hand. @calibrateextrapolate2024, for example, calibrate scores on a labeled sample and extrapolate to the target by assuming a stable calibration curve. But when the classifier's errors vary with the features, a feature shift that the score does not capture changes what a given score means, and the calibration curve can move [@tasche2022covariate]. Building on @kim2022universal, we argue that *multicalibration*, calibration within each of a rich class of feature-defined subgroups [@hebertjohnson2018multicalibration], addresses this problem: a model multicalibrated once on a labeled source sample returns unbiased prevalence estimates on any target whose shift the features capture and that lies within their support.

# Prevalence Bias Reduction through Multicalibration

Consider a binary outcome $Y \in \{0,1\}$, features $X$, and a predictor $h(X)\in[0,1]$. The estimand is the target prevalence $\pi^*=P^*(Y=1)$, estimated by averaging predictions over unlabeled target observations. We assume *covariate shift*: $P(X)$ may differ between source and target while $P(Y\mid X)$ is stable, in contrast to *label shift*, which holds $P(X\mid Y)$ fixed [@morenotorres2012unifying]. We also require overlap: any feature region that occurs in the target also occurs in the source. Under concept drift, where $P(Y\mid X)$ changes, target prevalence cannot generally be recovered without target labels.

The prevalence error of a predictor is its mean signed error, $\mathbb{E}[h(X)-Y]$. Discriminative metrics do not constrain this error. For example, AUC depends only on how the scores rank units, so it is unchanged by any monotone transformation, yet such transformations can move the average score almost anywhere in $(0,1)$. Accuracy and $F_1$ count errors without their sign, whereas the prevalence error of Classify \& Count is the difference between its false-positive and false-negative shares. For instance, a classifier with 90\% accuracy can misstate prevalence by anything from zero, if its errors balance, to ten percentage points, if they all go one way.

Calibration can close this gap. If $\mathbb{E}[Y\mid h(X)=p]=p$ for every prediction value $p$, then $\mathbb{E}[h(X)]=\mathbb{E}[Y]$. But calibration holds in a specific population. Partition that population into feature-defined groups $G$ with weights $w_G$, and let $\epsilon_G=\mathbb{E}[h(X)-Y\mid X\in G]$ be the mean signed error in group $G$. Calibration ensures only that these errors cancel under the source weights, $\sum_G w_G\epsilon_G=0$. If covariate shift reweights the groups, the errors stay fixed but the weights change, and the prevalence error becomes $\mathbb{E}^*[h(X)]-\pi^*=\sum_G (w_G^*-w_G)\,\epsilon_G$. A classifier that discriminates well and is calibrated overall can thus still err systematically within groups, and these errors become prevalence bias once the group mix changes.

Multicalibration removes these group errors rather than relying on their cancellation. Let $\mathcal{G}$ contain the groups that can be formed from the calibrated features, including their intersections. A predictor $f$ is multicalibrated with respect to $\mathcal{G}$ if $\mathbb{E}[Y\mid f(X)=v,\,X\in G]=v$ for every $G\in\mathcal{G}$ and prediction value $v$ [@hebertjohnson2018multicalibration]. Averaging over $v$ shows that $f$ has zero mean signed error within every group in $\mathcal{G}$. A target shift is captured by $\mathcal{G}$ if it can be expressed as a reweighting of these groups. Setting $\epsilon_G=0$ in the decomposition above then gives $\mathbb{E}^*[f(X)]=\pi^*$ for every captured target within the source's support. This is the "universal adaptability" guarantee of @kim2022universal, who show that such an estimate is as accurate as inverse propensity weighting (IPW), up to the multicalibration error, under any propensity model whose odds the group class can express.\footnote{The weaker property of \emph{multi-accuracy}, zero mean error within each group, already suffices. SI Section S1.8 explains why we nonetheless recommend multicalibration.} The same predictor serves every such target, without per-target correction or target labels.

The guarantee is exact only when the target-to-source density ratio lies in the span of the group indicators; otherwise it is approximate, with error governed by how well the groups approximate that ratio. With finite calibration data, multicalibration itself also holds only approximately. Whether covariate shift and this representation condition are plausible is a substantive judgment about the application. We obtain the multicalibrated predictor with MCGrad [@tax2026mcgrad], a post-hoc gradient-boosting multicalibrator.

# Simulation

We illustrate the mechanism with a simulation. Each observation has a binary feature $X$ and a continuous signal $U\sim N(0,1)$, and $P(Y=1\mid X,U)=\sigma(a_X+1.5\,U)$ in the calibration sample and all targets, so only $P(X)$ shifts. The classifier's score adds a logit offset for $X=0$ observations only, so it overstates their probability while still discriminating well (AUC $\approx0.89$). Each estimator is fit on 10,000 labeled observations at $P(X=0)=0.5$ and applied to unlabeled targets with $P(X=0)$ from 0.01 to 0.99, over 50 repetitions. We compare MCGrad with Classify \& Count, the Rogan-Gladen adjustment, and global recalibration by isotonic regression (details in SI Section S4).

![](images/figure_sim_lineplot.png){width=88%}

*Figure 1. Relative prevalence bias under covariate shift, against the change in $P(X=0)$ from its calibration value of 0.5. Faint lines are individual runs; thick lines average over 50 runs. The figure is cropped at $\pm40\%$; SI Sections S4--S5 show the full range, all methods, and RMSE.*

At the calibration distribution, Classify \& Count with the default 0.5 threshold overstates prevalence by about 12\%: good discrimination does not make counts accurate. The labeled sample can remove this bias in three common ways: choosing the threshold so that the count reproduces the calibration prevalence, adjusting the count for the estimated error rates (Rogan-Gladen), or recalibrating the scores (isotonic regression). All three are unbiased at the calibration distribution, but each fits a single correction to the calibration mix of $X$ and fails when that mix changes. The matched-threshold count drifts to about $-19\%$, and Rogan-Gladen, whose ratio form amplifies changes in the error rates, exceeds $\pm30\%$. Isotonic recalibration shows the mechanism most clearly: it maps a score to the same probability regardless of $X$, which is too high for $X=0$ observations and too low for $X=1$. These errors cancel only at the calibration mix, so the bias grows to about $+19\%$ when nearly all observations have $X=0$. MCGrad uses the same labeled sample and scores but calibrates within each value of $X$, so its average bias stays within about 2\% throughout. The 0.5-threshold count is wrong everywhere; its bias barely moves only because its relative error happens to be similar in the two groups.

# Applications

We apply the method in two settings with known ground truth. In the first, Claude Opus 4.6\footnote{An open-weights replication with Llama 3.3 70B (SI Section S2) reproduces the qualitative pattern and permits full reproduction.} codes 29,900 Comparative Agendas Project texts [@baumgartner2006cap] from six sub-populations in four countries and languages for whether they concern Law \& Crime (CAP major topic 12), scored against the expert codes. The topic's prevalence is 7.9\% at baseline and 6--20\% across scenarios. The LLM returns a Yes/No label and an elicited probability [@tian2023verbalized]; we multicalibrate the label and give the probability to methods that require a score. In the second, a logistic regression predicts employment from 16 sociodemographic features in American Community Survey (ACS) microdata (details in SI Sections S2--S3). In both, we construct targets with known covariate shift: within-support reweightings (of countries and document types; of age groups) and out-of-support targets (unseen document types; held-out states). Uncertainty comes from a bootstrap of the calibration and target samples that holds the classifier fixed (SI Section S1.9).

![](images/figure_combined_v1.png){width=100%}

*Figure 2. Absolute prevalence error (percentage points) by method and scenario. Top: CAP Law \& Crime coding of 29,900 texts (1947--2018) with Claude Opus 4.6, multicalibrated on the Yes/No label. Bottom: ACS employment with logistic regression (2016--2018 microdata; eight source and six held-out states). Left: within-support targets; right: out-of-distribution targets. Markers are scenarios, horizontal bars their means, and vertical lines 95\% bootstrap intervals, with the upper bound printed where an interval exceeds the axis. Full results in SI Tables S1--S2.*

Although the LLM discriminates well (per-language AUC about 0.98), Classify \& Count, the applied default, errs by +1.6 to +4.8pp (Figure 2, top), overstating prevalence by nearly 30\% at baseline. Rogan-Gladen reduces this error within support but misses by +3.1 to +3.7pp outside it.\footnote{SI Tables S1--S2 report additional methods, including SLD [@saerens2002adjusting] and PACC. ReadMe [@hopkinsking2010nonparametric], the best-known political-science quantifier, is classifier-free and fails as the label-shift diagnosis predicts, exceeding +60pp under some shifts (SI Section S7).} Isotonic recalibration is accurate within support ($\le 0.9$pp) but drifts outside it (up to 2.6pp). IPW, with a gradient-boosted propensity model on the multicalibration features, is near zero within support but misses by $-11.9$pp on Spanish media, whose feature values are absent from the source. Multicalibration errs by at most 0.4pp within support and by 0.7 and 1.9pp outside it, with 95\% bootstrap intervals within 5pp (SI Table S2).

The survey application reproduces this ordering with a wider spread (Figure 2, bottom). Rogan-Gladen fails by 12--19pp under the age shifts, and Classify \& Count and isotonic regression reach 6--8pp under the largest. IPW, whose propensity model uses MCGrad's untuned learner, is accurate on some targets but unstable on all four age-skewed ones: its point errors reach $-10.1$ and $-15.5$pp, and it errs by more than 5pp in a quarter to three quarters of bootstrap draws (SI Table S1). The calibration sample is thirty times larger than each target, so target propensities are small and a few source observations receive extreme weights (SI Section S3). Multicalibration errs by at most 0.27pp on within-support targets and by 0.88--1.35pp on held-out states. The contrast with CAP shows when global recalibration suffices: when the predictor's errors are roughly homogeneous across the groups a shift reweights, as they approximately are in CAP. The ACS shifts move age, which drives employment (76\% employed at ages 25--54, 17\% at 65+), so group errors are large and heterogeneous, and only feature-conditional calibration removes them.

# Discussion

Discriminative validation shows that a classifier separates classes, not that its prevalence estimates are unbiased. Counting the labels of an LLM with a per-language AUC of about 0.98 overstated prevalence by 18--44\% of the true rate. Multicalibration removes this bias in the validation population and keeps it removed as the target shifts along the calibrated features. Unlike importance weighting, it is fitted once, before any target is seen, and yields a single instrument for a time series or a set of sub-corpora without target labels or per-target re-estimation. Because it removes error within every calibrated group, not only on average, it also makes group-level estimates comparable: differences in estimated prevalence across countries, languages or sources are not just artifacts of a classifier that errs differently in each. It also leaves the usual workflow intact: prevalence is still the average of the classifier's outputs, after a one-time adjustment on the labeled sample already used for validation. Label-shift corrections remain appropriate when only the frequency of the category changes, not what its instances look like; when composition changes along observed features, as across countries, sources or periods, multicalibration is the reusable choice.

The guarantee has limits. It holds only along calibrated features, so the calibration data must span the anticipated variation; beyond them error grows, if only modestly in our out-of-support targets. Calibrating on more features widens the guarantee but requires enough labels in each group; in zero-shot settings this reintroduces some hand-coding, though far less than per-document labeling and only once. Finally, where the target combines covariate shift with changes in $P(Y\mid X)$, the guarantee holds only approximately.

# Funding {.unnumbered}

This research was conducted as part of the authors' employment at Meta Platforms, Inc. and received no specific external funding.

# Acknowledgements {.unnumbered}

The authors used Claude Opus 4.6, 4.7, 5, 5.5, and Gemini 3 Pro to assist with literature search, code, data analysis, visualization, and drafting and editing of the manuscript. The authors are entirely responsible for the scientific content of the paper, which adheres to the journal's authorship policy. 

# Data Availability Statement {.unnumbered}

Replication code and data will be deposited in the Political Analysis Dataverse and are available at <https://github.com/facebookresearch/multicalibrated_llm_measurement>. The analyses use Comparative Agendas Project data from Denmark [@capdkquestions], Spain [@capesquestions; @capesmedia], the United States [@wilkerson2025bills] and Belgium [@capbemedia], coded under the CAP master codebook [@jones2025codebook], and American Community Survey microdata [@census2018acspums], accessed through the folktables package [@ding2021retiring].

# Competing Interests {.unnumbered}

All authors are employees of Meta Platforms, Inc.; Milan Vojnovic is also affiliated with the London School of Economics and Political Science. Llama 3.3, used in the replication in SI Section S2, is developed by Meta. Niek Tax, Lorenzo Perini, Fridolin Linder, Daniel Haimovich, and Milan Vojnovic are authors of MCGrad [@tax2026mcgrad], the open-source multicalibration method evaluated here. The authors declare no other competing interests.

# References {.unnumbered}
