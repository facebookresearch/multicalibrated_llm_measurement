# LLM Context: Multicalibration for Prevalence Measurement Paper

## Project Overview

**Paper title:** The Crucial Role of Multicalibration in Model-based Prevalence Measurement

**Authors:** Fridolin Linder, Thomas Leeper (both Meta)

**Target venue:** Science/Nature/PNAS style general audience journal.

**Core argument:** Multicalibration is necessary and sufficient for accurate out-of-domain prevalence estimation. Standard calibration methods fail under mix shift (covariate shift that changes subgroup composition). The paper connects multicalibration theory (from algorithmic fairness) to the quantification/prevalence estimation problem.

**Key resources:**
- Google Doc brainstorm: https://docs.google.com/document/d/1IfTYVHQHkt9EiWoQZlFWyFkvjq7pJ_WaSLkeeiL21zU/edit
- Simulation notebook: [N8677564](https://www.internalfb.com/intern/anp/view/?id=8677564) — binary outcome, single binary feature illustration

## Current Status

**Written (draft quality):**
- Introduction (mostly complete, needs LLM hook)
- Theory section: calibration approaches, AC/Rogan-Gladen, mix-shift failure
- Simulation study: DGP, covariate shift protocol, 5 estimation methods with math, results discussion
- Multicalibration formal definition
- Appendix: full simulation procedure (S1–S6)

**Not yet written:**
- Abstract
- Connection between multicalibration definition and measurement problem (transition paragraph)
- Empirical applications (dataset not selected)
- Conclusion
- Multicalibration simulation figure and discussion

## Open Questions and TODOs

### Structural / strategic
- Whether to lead with population mix shift / concept drift framing (Thomas's open question)
- How to handle true concept drift — how do we set it aside vs. mix shift?
- Whether there's a separate paper connecting IPSW and multicalibration (Thomas's suggestion)

### Content gaps
- Add stronger LLM hook in the introduction to convey importance/novelty
- Footnote about float output from LLMs (how LLMs produce probability-like outputs)
- Footnote about other non-classification quantification methods (King; Resnick; etc.)
- Be explicit that the theory section is in an X→Y scenario
- Add analogy to conditioning on post-treatment variables
- Discussion of why we should care about bias vs. variance
- Note that Grimmer/Roberts/Stewart review hardly mentions calibration
- Feature selection / heterogeneity / data size for achievable multicalibration
- Results across varying levels of model quality
- Sensitivity analysis under imperfect multicalibration (shifting distribution, correlation with outcome, numbers/selection of features)

### Figures
- Make y-axes the same scale across all simulation panels (Thomas)
- Add a classify-and-count panel that might look the same as another panel (Thomas)
- Create multicalibration simulation figure

### Mathematical
- Verify the AC reformulation as calibration: Fridolin's comment questions whether assigning TPR to positives and 1-FPR to negatives actually recovers the Rogan-Gladen formula. The algebra needs checking.

## Co-author Comments (from Google Doc)

### Thomas Leeper
1. "maybe add a stronger llm hook here to convey importance/novelty" — on the intro paragraph about LLMs
2. "footnote something about float output from LLMs" — on device/instrument definition
3. "can we say 'device' instead of 'instrument'?" — terminology (resolved: using "device")
4. "be explicit that we're in an X->Y scenario" — on the theory section setup
5. "analogy here to conditioning on post-treatment variables" — on the mix-shift discussion
6. "maybe: add a classify and count example, might look the same as this" — on simulation figure
7. "ditto" / "make the y-axis the same scale as the other plots" — figure formatting
8. "there might actually be a separate paper here connecting IPSW and multicalibration" — strategic note
9. "open question whether to lead with something about population mix shift / concept drift" — framing question

### Fridolin Linder
1. Questions whether the AC-as-calibration reformulation is algebraically correct: "If I assign TPR to positively classified and 1-FPR to negatives I get: E[score] = TPR*P(y_hat=1) + (1-FPR)*P(y_hat=0) = (TPR + FPR−1)p_hat + (1−FPR) != (p_hat-FPR) / (TPR - FPR)"

## DAG Scenario Working Notes

Three causal scenarios under consideration:

**Scenario 1: X → Y** (strictly X causes Y, constant P(Y|X))
- Y = a*X1 + b*X2 + c*X1*X2
- "Y can't change without a change in X"

**Scenario 2: Y → X** (strictly Y causes X, constant P(X|Y))
- Multiple sub-DAGs: X1 ← Y → X2, Y → X1 → X2, etc.
- Need to distinguish True X (actual DAG) from Observed X* (what you have)

**Scenario 3: Unknown DAG** (mix of X → Y and Y → X)

## LLM Zero-Shot Measurement Literature

| Paper | Setting | Output |
|-------|---------|--------|
| Weidmann et al. 2025 (PS) | GPT-4o/Llama-3.1 code V-Dem democracy indicators zero-shot | Democracy ratings by country |
| Overøs et al. 2024 (PNAS Nexus) | GPT-4 codes protest events in news; sensitive to temperature/prompt framing in zero-shot | Protest-event article rates |
| Mellon et al. 2024 (Research & Politics) | 6 LLMs vs supervised models on BES "most important issue" responses; Claude-1.3 achieves 93.9% accuracy (human: 94.7%, supervised@1k: 93.5%) | Issue category prevalence |
| mRNA vaccine tweet study (PMC) | GPT-3.5-Turbo labels 740k+ tweets; extracts safety (73.4%) and trust (64.0%) prevalence | Attitude/topic prevalence by region/time |
| Burstein et al. 2025 (PMC) | Zero-shot prompting on caregiver survey free-text; up to 96% agreement with curated benchmarks | Reason category prevalence |
| Sushil et al. 2024 (JAMIA) | GPT-4 zero-shot on breast cancer pathology reports; avg κ = 0.85 rivaling supervised models | Clinical attribute prevalence |
| Sibley et al. 2025 (medRxiv) | Llama 3.3 zero-shot matches task-specific BERT (F1 = 0.83) for Goals-of-Care discussions in 617-patient test set | Clinical metric identification |
| Halterman 2025 (Political Analysis) | "Codebook LLMs" — operationalize polisci concepts | Scalable concept measurement |
| Karjus 2025 (HSSC) | GPT-3.5/4 as coders in mixed-methods pipelines; 12+ case studies across 9 languages including theme extraction, stance detection, genre analysis | Quantitative variables from qualitative corpora |
| Tojima & Yoshida 2025 (IEEE Access) | 4-bit quantized Llama-3 70B zero-shot on art auction data; accuracy >0.90 for art-form classification, slightly outperforms GPT-4o | Art-form prevalence labels |
| Gur-Arieh et al. 2025 (SSRN) | "Ambiguity collapse" in LLM interpretation | Framework for evaluating LLM coding distortions |
| Gilardi et al. 2023 (PNAS) | ChatGPT zero-shot vs. crowdworkers on polisci tasks | Document-level labels for social constructs |
| Wu et al. 2023 (arXiv) | LLM pairwise comparisons + Bradley-Terry scaling | Latent ideological position estimates |
| Ziems et al. 2024 (Comp Ling) | Systematic evaluation of LLMs as zero-shot annotators for CSS | Benchmarked zero-shot concept coding |

## Related Methodological Work

- **Zhao et al. 2021 (ICML):** "Calibrate before use" — estimates label bias from content-free inputs, reweights outputs. Relevant as a calibration baseline.
- **Angelopoulos et al. 2023 (PNAS):** Prediction-Powered Inference (PPI) — statistical framework for combining ML predictions with small labeled datasets. Potential follow-up connection.
- **Bareinboim & Pearl 2016:** Causal inference and data-fusion — transportability theory.
- **Rajadesingan et al. 2021 (ICWSM):** Community-specific learning for toxicity classification.
- **Calibrate-Extrapolate (ICWSM 2024):** Same argument as Wu/Resnick from a different angle.

## Possible Extensions

- Calibrating pre-trained models for classification (using LLMs, web services, etc. for measurement)
- Connecting PPI framework with multicalibration guarantees
- Cost savings analysis: how many labels are needed under multicalibration vs. standard calibration
