# LLM Application Plan: CAP Policy Topic Classification

## Overview

Add a second empirical application to the paper using an LLM as the measurement device,
classifying political texts by policy topic using the Comparative Agendas Project (CAP)
codebook. This addresses the main weakness identified in the journal suitability review:
the current empirical application uses logistic regression, not an LLM, despite LLMs being
the primary motivation.

## Why CAP

1. **Task simplicity.** "What policy topic is this about?" is unambiguous, well-defined,
   and something LLMs do well zero-shot. The 20 major CAP topic codes (economy, health,
   defense, education, law & crime, etc.) are intuitive and standardized across countries
   and document types.

2. **Rich shift story.** Two orthogonal shift dimensions create a compelling design:
   - **Country**: Parliamentary language, institutional norms, and policy agendas differ
     across countries (covariate shift), but topic definitions are the same everywhere
     (stable P(Y|X)).
   - **Document type**: Bills, parliamentary questions, media articles, and TV news have
     very different text styles and topic prevalence distributions. Within-country
     doc-type variation (up to 15pp) is even larger than cross-country variation.

3. **Rich auxiliary features for multicalibration.** Genuine independent metadata:
   - **Country** (categorical) — primary shift dimension
   - **Document type** (question, bill, media article, TV news) — second shift dimension,
     creates massive prevalence variation and correlates with LLM error patterns
   - **Year/decade** (temporal shift)
   - **Party/party family** of the speaker/author (where available)
   - **Language** (Danish, Spanish, English, Dutch)
   These are recorded in metadata, not derived from the input text.

4. **Public data.** Freely downloadable from comparativeagendas.net.

5. **Disciplinary relevance.** Political science is a core audience. Several cited papers
   (Weidmann, Halterman, Overøs, Gilardi) use LLMs to classify political text across
   heterogeneous corpora (news, surveys, parliamentary records). Our design mirrors this
   real-world heterogeneity.

## Dataset Selection

**Maximum data approach:** Pool multiple document types across 4 countries and 4 languages.

| Country | Document Types | N | Language | LC12% | LC12 N | Party? |
|---------|---------------|---|----------|-------|--------|--------|
| Denmark | Parl. Questions | 110K | Danish | 7.4% | 8,132 | Yes (20) |
| Spain | Oral Questions + Media (El Pais + El Mundo) | 150K | Spanish | 10.4–20.2% | 23,818 | Partial |
| US | Congressional Bills | 468K | English | 4.6% | 21,394 | Yes (Rep/Dem) |
| Belgium | TV News + Newspaper | 157K | Dutch | 8.7–11.5% | 17,434 | No |
| **Total** | **7 sub-populations** | **~885K** | **4 languages** | **4.6–20.2%** | **70,778** | - |

**Why this combination:**
- **Denmark** (parl. questions): Largest single-type dataset, rich party metadata, Danish
- **Spain** (questions + 2 media outlets): Multiple doc types within one country, Spanish.
  Media has dramatically higher LC12 (16–20%) than questions (10.4%), creating within-
  country doc-type shift
- **US** (congressional bills): Non-European country, English, party data (Rep/Dem),
  lowest LC12 prevalence (4.6%), creates maximum downward shift. Dramatically different
  political system and document format
- **Belgium** (TV news + newspaper): Dutch, large media datasets, LC12 range 8.7–11.5%.
  Media-only design contrasts with question/bill-heavy other countries

**Key advantages over parliamentary-questions-only design:**
- 15.6pp LC12 prevalence range (vs. 7pp with questions only)
- Document type as a genuine multicalibration feature
- Matches real-world LLM measurement practice (heterogeneous corpora)
- 885K records provides massive statistical power

## Topic: Law & Crime (CAP Code 12)

Binary classification: "Is this text about law, crime, or criminal justice?"

Selected over immigration (original plan) based on data-driven analysis:
- **Universal prevalence**: >4% in all 7 sub-populations (vs. immigration <2% in Spain)
- **High cross-population variation**: 4.6% (US bills) to 20.2% (Spain media) = 15.6pp
- **Massive positive counts**: 70K+ law & crime cases total (vs. ~6K for immigration)
- **Substantively unambiguous**: Crime, policing, justice system topics are clear-cut for
  LLM classification
- **Interesting shift story**: Media covers law/crime much more than legislative texts;
  Southern European parliaments focus more on it than Nordic ones

## Experimental Design

### LLM Predictions

- **Model:** Llama 3.3 70B Instruct (4-bit quantized, ~38GB VRAM). Upgraded from
  8B (insufficient multilingual capability, Spanish AUC ~0.50) and switched from
  3.1 to 3.3 for improved multilingual performance. 3.3 matches 3.1 405B on many
  benchmarks. Weights downloaded from Manifold (asa bucket) for devserver use;
  HuggingFace path (meta-llama/Llama-3.3-70B-Instruct) for reproducibility.
- **Inference frameworks:** Two backends for reproducibility:
  - `llm_inference.py` — MLX (Apple Silicon, MacBook Pro M4 Max)
  - `llm_inference_cuda.py` — PyTorch/Transformers (NVIDIA GPU, devserver)
  Same prompt, same score extraction, same output format.
- **Prompt:** Zero-shot, multilingual: "The following text is in {language}. Does it
  primarily discuss law, crime, or criminal justice? Respond Yes or No."
  The LLM handles language comprehension and classification in a single pass.
- **Score extraction:** P(Yes) / (P(Yes) + P(No)) from log-probabilities → continuous
  score in [0,1].
- **Inference strategy:** Subsample to ~105K documents (15K per sub-population) for
  feasibility. Full run on A100 80G devserver (~1-3 hours for 105K).

### Calibration and Evaluation

Mirror the ACS design. Key insight from review: the calibration set must include
variation in all features used for multicalibration (doc_type, country, language).
Calibrating on a single country/doc_type makes those features constant, preventing
MCGrad from learning corrections for them.

1. **Calibration population:** Balanced sample from 3 sub-populations:
   - Denmark parliamentary questions (~15K sample)
   - Spain oral questions (~15K sample)
   - US congressional bills (~15K sample)
   Total calibration: ~45K with variation in doc_type (question vs. bill),
   country (3 countries), language (3 languages), party, and decade.
   This mirrors the ACS design where the calibration set spans multiple states
   with variation in age, education, etc.

2. **In-distribution test:** Remaining data from the 3 calibration sub-populations.

3. **Out-of-distribution targets** (completely unseen during calibration):
   - Spain media El Pais (same language as calibration, but media not questions)
   - Spain media El Mundo (same language, different media outlet)
   - Belgium TV news (new country, new language, new doc type)
   - Belgium newspaper (new country, new language, different doc type)

4. **Shift scenarios:**
   - **Baseline:** Balanced test split (same composition as calibration)
   - **Within-calibration doc-type shift:** Resample to overweight US bills
   - **Within-calibration party shift:** Resample to overweight one party family
   - **OOD same language:** Spain media (Spanish, but media not questions)
   - **OOD new country:** Belgium TV (Dutch, unseen country, new doc type)
   - **OOD new country:** Belgium newspaper (Dutch, unseen, different doc type)

5. **Methods compared** (same seven as ACS):
   - Raw LLM scores (uncalibrated)
   - Classify & Count
   - Rogan-Gladen
   - PACC
   - SLD (EMQ)
   - Isotonic regression (global calibration)
   - MCGrad (multicalibration)

6. **Multicalibration features:** doc_type, country, decade, party (all vary within
   the calibration set)

7. **Evaluation:** Bias and RMSE via bootstrap (200 iterations per scenario,
   sample size = min(N, 20000) per scenario)

## Language Handling

Data is in 4 languages: Danish, Spanish, English, Dutch. The LLM handles language
comprehension and classification in a single prompt. This:
- Reduces pipeline complexity (one model call, not two)
- Better reflects real-world LLM measurement practice
- Language-specific error patterns are themselves a form of covariate shift, which
  strengthens the multicalibration story

Llama 3.1 has strong multilingual support for all four target languages.

## Implementation Steps

1. **Data preparation:** Standardize all 7 sub-population datasets with consistent
   columns (id, text, country, doc_type, language, year, decade, party, majortopic,
   law_crime). Ensure all datasets are downloaded and cleaned.
2. **Feasibility test:** Classify ~500 documents (from 2-3 sub-populations) with
   Llama 3.1 8B. Check zero-shot accuracy. Iterate on prompt if needed.
3. **Full LLM inference:** Run classification on all documents. May need to subsample
   if 885K is too large for overnight. Extract log-probability scores.
4. **Analysis pipeline:** Adapt the ACS notebook structure:
   - Calibration/test split on Denmark questions
   - Fit calibration methods (isotonic, MCGrad)
   - Compute prevalence estimates under shift gradient
   - Bootstrap for RMSE
5. **Figures and tables:** Same format as ACS (Figure analogue + Table analogue)
6. **Paper integration:** Add as a new results section complementing ACS.

## Implementation Status

- [x] Data preparation script (cap_analysis/prepare_data.py) — downloads all 7 datasets
      from comparativeagendas.net and standardizes columns. Fully reproducible.
- [x] LLM inference script — MLX backend (cap_analysis/llm_inference.py)
- [x] LLM inference script — CUDA backend (cap_analysis/llm_inference_cuda.py)
- [x] Analysis notebook (cap_analysis/cap_analysis.ipynb) — mirrors ACS design
- [x] Devserver setup script (cap_analysis/setup_devserver.sh)
- [x] All 7 datasets downloaded, cleaned, standardized
- [x] Feasibility test with 8B on MacBook (510 docs):
      - Danish: AUC 0.747 (usable)
      - English: AUC 0.850 (good)
      - Spanish: AUC 0.891 on small sample, but ~0.50 on larger sample (no signal)
- [x] 8B full run started (105K sample) — confirmed Spanish AUC ~0.50 at 7K docs
- [ ] **IN PROGRESS:** 70B feasibility test on A100 devserver (7,000 docs, 1K per sub-pop)
- [ ] Full 70B inference run (105K docs on A100, ~1-3 hours)
- [ ] Analysis pipeline execution
- [ ] Paper integration

## 8B Feasibility Results (Llama 3.1 8B, 4-bit, MLX on M4 Max)

The 8B model showed good discriminative power for English and Danish but essentially
random performance for Spanish:

| Sub-population | N scored | AUC | Pos mean | Neg mean |
|---|---|---|---|---|
| Denmark / parliamentary_question | 5,040 | 0.793 | 0.396 | 0.182 |
| US / bill (English) | 170 | 0.850 | 0.465 | 0.111 |
| Spain / parliamentary_question | 627 | 0.468 | 0.142 | 0.171 |
| Spain / media | 1,320 | 0.505 | 0.191 | 0.179 |

Decision: upgrade to 70B for all languages. Running feasibility test on A100 devserver.

## Open Questions

All resolved:

- [x] Which countries? → Denmark, Spain, US, Belgium (4 countries, 4 languages)
- [x] Which document types? → Questions, media, bills, TV news (7 sub-populations)
- [x] Language handling? → LLM classifies in original language (multilingual prompt)
- [x] Which topic? → Law & Crime (CAP code 12)
- [x] Which LLM? → Llama 3.1 8B (4-bit) as default; 70B on GPU as fallback
- [x] Hardware? → MacBook Pro M4 Max via MLX
- [x] Does this replace the ACS application? → Complement
- [x] Budget? → N/A, local inference

## Remaining Contingencies

- If 70B Spanish AUC is still too low → consider language-specific prompts or a
  different model (e.g., Llama 3.3 70B)
- If party metadata is too sparse → use country + doc_type + decade as primary
  multicalibration features
