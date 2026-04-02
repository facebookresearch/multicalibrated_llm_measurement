# Multi-accuracy vs. multicalibration for prevalence estimation

## Question

Our paper claims that multicalibration is necessary for unbiased prevalence estimation under covariate shift. But is full multicalibration actually necessary, or does the weaker condition of multi-accuracy suffice?

## Setup

Consider a measurement device $f(X, Z)$ that produces scores based on features $X$ (whose distribution shifts between source and target populations) and $Z$ (whose distribution is stable). The covariate shift assumption is that $P(Y \mid X, Z)$ is stable and only $P(X)$ changes.

Recall the definitions:

- **Multi-accuracy** wrt groups $\mathcal{G}$: $\mathbb{E}[f(X,Z) - Y \mid G] = 0$ for all $G \in \mathcal{G}$. The mean prediction equals the mean outcome within each group.

- **Multicalibration** wrt groups $\mathcal{G}$: $\mathbb{E}[Y \mid f(X,Z) = v, G] = v$ for all $G \in \mathcal{G}$ and all score values $v$. The prediction is calibrated at every score level within each group.

Multicalibration implies multi-accuracy, but not vice versa.

## Argument: multi-accuracy suffices for prevalence

The bias under covariate shift in $X$ is:

$$\text{Bias} = \mathbb{E}^*[f(X,Z)] - \pi^* = \sum_x \mathbb{E}[f(X,Z) - Y \mid X = x] \cdot P^*(X = x)$$

For this to be zero for all target distributions $P^*(X)$, we need:

$$\mathbb{E}[f(X,Z) - Y \mid X = x] = 0 \quad \forall x$$

This is multi-accuracy with respect to $X$. We do **not** need $\mathbb{E}[Y \mid f(X,Z) = v, X = x] = v$ (multicalibration).

Intuitively: $Z$ creates within-group variation in scores (different documents with the same $X$ get different scores because of $Z$). The errors for individual $Z$ values can be nonzero, as long as they average to zero within each $X$-group. Since $P(Z \mid X)$ is stable across populations (the shift is only through $X$), whatever cancellation happens in the source also happens in the target. Only the group mean matters for prevalence, not the within-group calibration curve.

This step in the argument — $P(Z \mid X)$ being stable — relies on an "ignorability"-type assumption: we have correctly identified the features $X$ that fully capture the shift mechanism.

## Connection to Kim et al. (2022)

Kim et al.'s universal adaptability theorem states that a multicalibrated predictor yields imputation estimates competitive with propensity scoring across arbitrary target populations. However, inspecting the proof, the key step uses the law of iterated expectations to collapse the multicalibration guarantee $\mathbb{E}[c_\sigma(X) \cdot (f(X) - Y) \mid f(X) = v] \leq \alpha$ into the unconditional $\mathbb{E}[c_\sigma(X) \cdot (f(X) - Y)] \leq \alpha$, which is exactly multi-accuracy with respect to the likelihood ratio functions. The conditioning on score level $v$ is not used in the final bound.

So Kim et al. use multicalibration as a *sufficient condition* (because it implies multi-accuracy, and because MCBoost naturally produces it), but multi-accuracy is the operative property for mean estimation.

## Implications for our paper

1. **The "necessary" claim needs nuancing.** What is strictly necessary for unbiased prevalence estimation is multi-accuracy wrt the features that capture the covariate shift — not full multicalibration. At the pointwise level ($G = \{x\}$), the two concepts collapse, so "calibration conditional on $X$" is correct. But the formal definition of multicalibration (conditioning on both group AND score level) is stronger than what's required.

2. **Multicalibration is still the right practical prescription.** It implies multi-accuracy for any sub-partition of the feature space. This means you don't need to know *which* features capture the shift — if the predictor is multicalibrated wrt a rich class, multi-accuracy follows automatically. If you only target multi-accuracy wrt a specific partition $X$ but guess wrong (some "stable" feature $Z$ actually shifts too), you're exposed.

3. **MCGrad delivers multicalibration**, which is more than enough. The practical message doesn't change: apply MCGrad, get multicalibrated scores, prevalence estimation is unbiased. But the theoretical characterization could be sharpened.

## Possible paper adjustments

- Option A: Keep the current framing but add a remark noting that multi-accuracy is the minimal sufficient condition, and multicalibration provides robustness to misspecification of the shift structure.
- Option B: Reframe the theoretical contribution around multi-accuracy as the necessary and sufficient condition, with multicalibration as the practical solution (since it implies multi-accuracy and doesn't require knowing the shift features).
- Option C: Leave as is — the pointwise argument in the paper ("calibration conditional on $X$") is correct, and "multicalibration" in the title refers to the algorithmic framework rather than the strict formal definition.
