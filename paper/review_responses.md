# Review Responses

Tracking how each comment was addressed.

---

## Daniel (GitHub PR)

| # | Summary | Response |
|---|---------|----------|
| D1 | `ε_g` called "calibration error" but not conditioned on score | **Addressed.** Reframed ε_g as score-conditional calibration error within each group: E[Y \| h(X), G=g] - h(X). This keeps the illustration in the calibration (not multi-accuracy) framing. |
| D2 | Concept drift bullet inadvertently includes label shift | **Addressed.** Clarified that concept drift refers to the mechanism itself changing, not as a consequence of shifting P(X) or P(Y). |
| D3 | Intro uses α-multicalibration but formula drops α | **Addressed.** Dropped α-multicalibration; now introduces multicalibration directly with exact equality. |
| D4 | Is logistic regression well-tuned / includes age? | **No change needed.** Confirmed: the logistic regression uses all 16 ACSEmployment features from folktables, including AGEP (age) as a numerical feature. This is a standard, competitive baseline — MCGrad's advantage is not due to a weak base model. Will note this in response to Daniel. |
| D5a | Squashing fairness | **No change needed.** Response to Daniel: Squashing is a technical necessity, not a performance-enhancing tweak — MCGrad's internal logit transform maps scores near 0/1 to ±∞, causing numerical overflow. Without squashing MCGrad literally cannot run. The paper documents this and shows that applying squashing to SLD makes it substantially worse, so other methods aren't being denied something helpful. |
| D5b | Broader LLM probability extraction comparison | **Deferred.** May investigate further for this version. |

## Niek (GitHub PR)

| # | Summary | Response |
|---|---------|----------|
| N1 | "Remedies" referenced before introduced | **Addressed.** Moved "However, commonly used remedies..." sentence to the next paragraph. |
| N2 | "Canceling out" unclear without prior knowledge | **Addressed.** Changed to "errors that happened to balance in the original population no longer cancel out." |
| N3 | "Prevalence estimation" vs "quantification" terminology | **Addressed.** Added parenthetical clarifying that "quantification" and "prevalence estimation" are used interchangeably. |
| N4 | "Covariate shift" comes out of nowhere in intro | **Addressed.** Replaced "under covariate shift" with plain language "when the target population differs in composition from the source population"; formal definition deferred to Results. |
| N5 | Storkey reference for dataset shift types | **Addressed.** Added `@storkey2009dataset` citation where the three shift types are introduced. |
| N6 | `\mid` spacing in conditionals | **Addressed.** Replaced all bare `\|` with `\mid` in conditional probability/expectation expressions throughout. |
| N7 | X→Y assumption too strong for some domains | **Addressed.** Added sentence in covariate shift bullet acknowledging Y→X settings (e.g., fraud). Added corresponding note in Limitations. |
| N8 | "even under" phrasing | **No change.** The "even" is intentional — it highlights that stable P(Y\|X) is still not enough to preserve calibration, which is the key insight of the paragraph. |
| N9 | Bias can be zero even when ε_g not all zero | **Addressed.** Reworded: "generally nonzero unless the device is calibrated within every subgroup (ε_g = 0 for all g)" — avoids the overly strong "unless every ε_g = 0" while keeping the point that group-level calibration is the fix. |
| N10 | Why is CC threshold set to prevalence, not 0.5? | **No change to main text.** Response to Niek: A 0.5 threshold is an arbitrary default that can be trivially improved. If the score is calibrated, setting the threshold to the known prevalence ensures CC is unbiased on the calibration distribution. For uncalibrated scores, one can search for the optimal threshold. In either case, the threshold loses validity under shift. Additionally, added footnote in CAP Methods explaining why simulation/ACS use prevalence-matching but CAP uses Youden's J (bimodal scores cause degenerate prevalence-matching threshold of 1.0). |
| N11 | MCGrad paper year should be 2026 | **Addressed.** Updated year to 2026 and conference to 32nd KDD in references.bib; updated citation key throughout. |

## Milan (annotated PDF)

| # | Summary | Response |
|---|---------|----------|
| M1 | "Necessary and sufficient" — theorem or empirical? | **Addressed.** Clarified in Significance: "established theoretically and confirmed by simulation and empirical applications." |
| M2 | Emphasize importance of unbiased prevalence estimation | **Addressed.** Added sentence on downstream consequences (policy, science, resource allocation). |
| M3 | "Want to say this?" (unclear context) | **Needs clarification.** Could not determine which phrase this refers to from the PDF annotation. Will ask Milan. |
| M4 | "Referring to covariate shift?" | **Needs clarification.** Could not determine exact referent from the PDF annotation. Will ask Milan. |
| M5 | Compress label shift / concept drift section | **No change.** Both bullets are now referenced elsewhere: the label shift bullet supports the Y→X acknowledgment (N7), and concept drift is tied back to in Limitations. Compressing further would lose useful context. |
| M6 | Define notation: dP*(x), E*, π* | **Addressed.** Added parenthetical defining E*, P*, and π* before the equation. |
| M7 | Define ΔP(X=0) | **Addressed.** Added definition of x-axis to Figure 1 caption. |
| M8 | Define "Average bias (%)" | **Addressed.** Added formula for bias metric to Figure 1 caption. |
| M9 | "Necessary and sufficient" repeated (same as M1) | **Addressed.** Same fix as M1 — the Significance section now clarifies "established theoretically and confirmed by simulation and empirical applications." |
