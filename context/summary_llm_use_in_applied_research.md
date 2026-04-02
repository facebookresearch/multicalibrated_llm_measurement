Here is the detailed breakdown of how each paper uses and validates the LLM, based on a direct review of the published methodologies.

As the authors of your PNAS manuscript point out, none of these papers use confidence elicitation or post-hoc calibration; they rely entirely on direct prompting and rank-based validation metrics.

1. Gilardi et al. (2023): ChatGPT Outperforms Crowd Workers for Text-Annotation Tasks
How the model is used: Zero-shot direct prompting. The researchers feed tweets and news articles to ChatGPT and ask it to assign discrete categorical labels for relevance, stance, topics, and frames (e.g., Yes/No or choosing from a list of predefined topics). They do not extract confidence probabilities or apply post-hoc calibration.

How it is validated: Ground truth validation against human coders. The researchers had trained research assistants and MTurk crowd-workers label a subset of texts. They validated the LLM by computing its zero-shot accuracy (percentage of exact matches with the human gold standard) and intercoder agreement (raw agreement percentage and Pearson correlation). They concluded the LLM was valid because its accuracy exceeded crowd-workers by roughly 25 percentage points.

2. Mellon et al. (2024): Scaling Open-Ended Survey Responses Using LLM-Paired Comparisons
How the model is used: Zero-shot pairwise prompting. Rather than asking the LLM for a direct 0–10 score, the researchers prompt the LLM to compare two open-ended survey responses and decide which one "wins" on a latent dimension (e.g., demonstrating more economic knowledge). The LLM's discrete binary decisions are then fed into a Bayesian Bradley-Terry model to create a continuous scale. There is no confidence elicitation or multicalibration of the LLM itself.

How it is validated: Cross-validation against benchmark scores. They validate the resulting LLM-generated ideological and knowledge scores by measuring their correlation against widely used human-expert benchmarks, such as the Chapel Hill Expert Survey (CHES) and established poll-based estimates.

3. Weidmann et al. (2025): Large Language Models Are Democracy Coders with Attitudes
How the model is used: Zero-shot direct prompting. The researchers prompt LLMs (like Llama-3.1 and GPT-4o) with V-Dem codebook definitions and ask them to act as expert coders by rating a country's democratic characteristics on an ordinal scale (e.g., from illiberal to democratic). No probability scores or post-hoc calibrations are used.

How it is validated: Ground truth validation against the actual V-Dem dataset. They compute the correlation coefficient between the LLM's zero-shot ratings and the final aggregated human expert ratings, as well as the average absolute deviation (error) per country. Interestingly, their validation actually reveals the calibration issue the PNAS paper discusses: while the LLMs correlate well with humans (up to 0.88), some models exhibit severe raw bias, consistently underestimating or overestimating democratic quality in absolute terms.

4. Benoit et al. (2026): Using Large Language Models to Analyze Political Texts
How the model is used: Zero-shot direct prompting. The researchers prompt LLMs to read entire political party manifestos holistically and assign them numerical scores on predefined policy scales (e.g., economic policy, European integration, environment).

How it is validated: Validation against benchmark expert surveys. They measure the Pearson correlation between the LLM-generated estimates of party positions and the equivalent mean ratings provided by human country specialists. They validate the method by reporting high correlations (0.87 to 0.92) across the six key issue dimensions.

5. Sibley et al. (2025): Assessment of a zero-shot LLM in measuring goals-of-care discussions
How the model is used: Zero-shot direct classification. The researchers use Llama 3.3 to screen unstructured Electronic Health Record (EHR) notes and classify whether a "goals-of-care" discussion took place (a binary Yes/No). No verbalized confidence scores or multicalibration techniques are applied to the LLM's output.

How it is validated: Ground truth validation against manually annotated clinical notes. They compare the LLM's performance against a supervised BERT model trained on over 4,600 manually labeled notes. The validation metrics used are strictly discriminative: Area Under the ROC Curve (AUC), Area Under the Precision-Recall Curve (AUPRC), and maximal F1 score on a held-out test set.

6. Tojima et al. (2025): Zero-Shot Classification of Art With Large Language Models
How the model is used: Zero-shot direct prompting. The researchers use LLMs to automatically annotate and categorize historical auction records into specific art forms (e.g., painting, print, sculpture).

How it is validated: Ground truth validation using standard classification metrics. They compare the LLM's discrete categorical predictions against a pre-labeled dataset of art categories, validating the model by reporting an overall accuracy score of over 0.90.
