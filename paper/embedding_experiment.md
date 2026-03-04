# Embedding Features for Multicalibration: Negative Result

## Motivation

Standard MCGrad uses categorical (country, doc_type, party) and numerical
(decade) features to learn group-conditional calibration corrections. These
features are coarse. We hypothesized that using dense text representations
from the LLM itself might let MCGrad learn finer-grained corrections that
transfer better under distributional shift, particularly to out-of-distribution
(OOD) document types.

## Method

1. **Embedding extraction.** We ran Llama 3.3 70B Instruct (4-bit NF4
   quantized) in forward-pass mode with `output_hidden_states=True` on all
   105,000 CAP documents. For each document we extracted the last-token
   hidden state from the final transformer layer, yielding an 8192-dimensional
   float16 vector per document.

2. **Dimensionality reduction.** We fit PCA on the ~40,000 calibration-set
   embeddings and retained 500 components (86.1% of variance explained;
   top 10 components alone captured 39.0%).

3. **MCGrad + Embeddings.** We fit a second MCGrad model using the same
   categorical features (doc_type, country, party) plus 501 numerical
   features (decade + 500 PCA components). MCGrad passes these directly
   to LightGBM, which handles high-dimensional numerical features natively.

4. **Evaluation.** We compared MCGrad vs MCGrad + Emb. on the same
   6-scenario shift gradient (4 within-calibration, 2 OOD).

## Results

| Scenario | MCGrad Bias | MCGrad + Emb. Bias |
|---|---|---|
| Baseline (balanced test) | -0.33pp | -0.49pp |
| Country shift (overweight Belgium) | -0.30pp | -0.39pp |
| Doc-type shift (overweight bills) | -0.01pp | -0.16pp |
| Party shift (right-heavy) | -0.09pp | -0.28pp |
| Spain media (OOD doc type) | -9.73pp | -9.64pp |
| Belgium TV (OOD doc type) | -1.53pp | -1.31pp |

Adding 500 PCA embedding features produced negligible changes. OOD bias
improved by 0.09pp (Spain media) and 0.22pp (Belgium TV). Within-calibration
bias worsened slightly (0.1--0.2pp additional negative bias across all four
scenarios). RMSE was similarly unchanged.

## Interpretation

The embeddings encode rich information about document content, but this
information is largely redundant with what already drives the LLM's
classification score. The OOD bias arises from systematic differences in
how the model scores different document types (e.g., media vs. parliamentary
text), and the embeddings capture the same distributional structure that
produces those score differences. They cannot independently correct for a
shift they are themselves subject to.

This result is consistent with the broader finding that MCGrad's OOD
limitations stem from encountering genuinely new feature combinations
(unseen document types) rather than from insufficient feature richness
within the calibration domain.
