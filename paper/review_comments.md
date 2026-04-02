# Review Comments on paper.md

Collected from three reviewers: Daniel (danielha1, GitHub PR), Niek (TaXxER, GitHub PR), Milan (annotated PDF).

---

## Reviewer: Daniel (GitHub PR #3)

### D1. Calibration error terminology (line 55)
> Calling epsilon "calibration error" is inconsistent with the generally accepted definition of calibration in the previous paragraph because you're not conditioning on the score.

### D2. Label shift inclusion (line 49)
> The way this is phrased, it includes label shift in it. So maybe worth rephrasing as "neither".

Refers to the concept drift bullet: "The conditional P(Y|X) itself changes". Daniel suggests phrasing should clarify it's distinct from label shift too.

### D3. Alpha-multicalibration vs multicalibration (line 67)
> Based on this formula, you want the preceding paragraph to introduce multicalibration rather than alpha-multicalibration, there's no room for alpha here.

The paragraph at line 65 introduces alpha-multicalibration, but the formula at line 67 drops the alpha tolerance.

### D4. Logistic regression features (line 93)
> The results are very favourable for MCGrad - is this logistic regression trained reasonably well and includes age as a feature?

Wants confirmation that the baseline logistic regression is competitive (includes age, is well-tuned), so MCGrad's advantage isn't just from a weak baseline.

### D5. Squashing fairness & broader method comparison (line 130)
> 1. Squashing for MCGrad seems slightly unfair - other methods don't get the opportunity for tweaks to their input. Is it totally broken without it? Is there a subtle way to extract slightly less extreme probabilities from the LLM?
>
> 2. Relatedly and more broadly, I wonder if there's an opportunity here to be more thorough. First, with methods to extract probabilities from LLMs based on recent paper (e.g. token probability, stated confidence, CoT etc), with/without MCGrad. We might end up seeing that with/without MCGrad makes more of a difference. Secondly - this is a bit more complicated so I'm not sure we want it - by using the LLM to extract a richer feature set either through embeddings or with additional questions (if there's something that makes sense for this specific use case, maybe the answers for the other topics, as in a multi-task scenario?).

Two sub-points: (1) squashing is MCGrad-specific preprocessing that other methods don't get; (2) opportunity to compare across different probability extraction methods for LLMs.

---

## Reviewer: Niek (TaXxER, GitHub PR #3)

### N1. Remedies introduced before being defined (line 33)
> Here the logical flow of the text feels a little off. Previous sentence only says that there are risks of false positives/negatives, but does not make any reference to **remedies** for such errors. In this sentence we highlight a limitation of such a remedy before we introduce the concept of a remedy itself. Maybe there is just one sentence missing here in between.

Follow-up comment: "Actually the introduction of 'remedies' / corrections is what the next paragraph is about. So maybe this 'However, commonly used remedies...' sentence is just in the wrong paragraph?"

### N2. "Canceling out" unclear for unfamiliar readers (line 35)
> I understand what you mean with "canceling out" because I already know about multicalibration. But a reader who doesn't have prior knowledge probably doesn't follow this.

### N3. Terminology: "prevalence estimation methods" vs "measurement methods" (line 37)
> Terminology: "prevalence estimation methods" and "measurement methods" seem to be used interchangeably in this introduction. I don't mind either term. Just wonder what is the commonly accepted terminology in this particular quantification literature, and whether readers with backgrounds in those fields will understand that these are the same.

### N4. "Covariate shift" comes out of nowhere (line 37)
> The term "covariate shift" comes a bit out of nowhere here, and doesn't occur anywhere earlier in the introduction.

### N5. Storkey reference for dataset shift types (line 45)
> Storkey's work might be a relevant reference here: https://homepages.inf.ed.ac.uk/amos/publications/Storkey2009TrainingTestDifferent.pdf

### N6. Spacing: `\mid` instead of `|` (line 47)
> Very nit comment: The spacing looks more natural when using `\mid` instead of `|`. I.e., `$P(Y \mid X)$` instead of `$P(Y|X)$`.

### N7. X->Y assumption too strong? (line 47)
> For the main argument of the paper, it is quite important that applying a device to a new population leads **only** to covariate shift. This sentence here seems to explicitly make the claim that this is the case. But this is not so clear.
>
> If I have a P(scam=1) prevalence quantity on a marketplace (e.g., eBay) that we are estimating, then depending on what is in X, one could argue that Y -> X instead of X -> Y is very reasonable in many cases too. If X contains covariates about how the user uses the website (e.g., representations of contents of listings that they create), then it is likely that **because the user wants to conduct a scam, these features have certain values**, rather than **because these features have certain values, the user is a scammer**.

### N8. "even under" phrasing (line 53)
> Why "even"?

Referring to: "even under covariate shift with stable P(Y|X)".

### N9. Bias can be zero even when epsilon_g are not all zero (line 55)
> Strictly speaking it is possible for the bias to be zero even when ε_g are not all 0 as examples can be constructed where they cancel out even under covariate shift. Typically they don't of course. Maybe still worth brief discussion in footnote or appendix.

### N10. Why is threshold set to prevalence, not 0.5? (line 57)
> I don't understand why τ is set to prevalence, and not to 0.5? If prevalence is 0.001, and we have an unbiased model, then if the score distribution is not skewed, ~50% of the predictions will be >0.001 threshold. But if Classify & Count gives an estimate of 50%, that seems very wrong?

### N11. MCGrad paper year should be 2026 (line 77)
> nit: MCGrad paper's year should be 2026

---

## Reviewer: Milan (annotated PDF)

### M1. "Necessary and sufficient" — mathematical or empirical? (page 1, Significance section)
> Maths language, shown by a theorem and confirmed by experiments, or only confirmed by experiments?

Underlined "necessary and sufficient" in the Significance section. Asking whether this claim is backed by a formal theorem or just empirical evidence.

### M2. Emphasize importance of unbiased prevalence estimation (page 1)
> Emphasising the importance of unbiased prevalence estimation?

Seems to suggest adding more motivation for why unbiased prevalence estimation matters.

### M3. "Want to say this?" (page 1)
> Want to say this?

Context unclear — appears near the intro text. May refer to a specific claim being too strong or unnecessary.

### M4. "Referring to covariate shift?" (page 1)
> Referring to covariate shift?

Asking for clarity on what type of shift is being discussed at a point in the intro.

### M5. Compress label shift / concept drift section (page 3, lines 48-49)
> Needed? Distractive to the reader? Compress? Perhaps focus only on explaining covariate shift, as this is only what is being used, and mention label shift and concept shift in passing.

Highlighted the full label shift and concept drift bullet points. Suggests compressing these since only covariate shift is used in the paper.

### M6. Define notation: dP*(x), E*, π* (page 4, line 73)
> Def?

Underlined `dP*(x)`, `E*`, and `π*` in the equation at line 73. Wants these symbols defined before use.

### M7. Define ΔP(X=0) (page 4, around simulation section)
> Def ΔP(X = 0) here

Wants the x-axis variable in the simulation plot defined explicitly.

### M8. Define "Average bias (%)" (page 4, Figure 1)
> Def?

Underlined "Average bias (%)" — wants a definition of this metric, likely in the figure caption or surrounding text.

### M9. "Necessary and sufficient" repeated emphasis (page 10)
Underlined "is necessary and sufficient" again in the discussion/conclusion. Consistent with M1 — wants to ensure this claim is properly backed.
