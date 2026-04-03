# Claude Opus 4.6 Inference on CAP 30K Sample

## Overview

Two separate inference campaigns were run using Claude Opus 4.6 (via Claude Code sub-agents) on 30,000 political documents from the Comparative Agendas Project (CAP). The task is binary classification: does the document primarily discuss "Law and Crime" (CAP major topic code 12)?

The two campaigns were run independently to avoid anchoring contamination:

1. **Campaign 1 (Binary)**: Simple Yes/No classification only
2. **Campaign 2 (P(Y/N))**: Direct probability elicitation without first committing to an answer

Each campaign classified all 30,000 documents. The model used is **Opus 4.6** (sub-agents inherited the parent model). All inference was done with Opus.

## Data

### Input
- **Source**: `cap_analysis/data/opus_30k_sample.csv` (30,009 rows, stratified sample from `full_sample.csv`)
- **Shards**: `cap_analysis/data/claude_shards_30k/shard_{0-299}.json` (300 files, 100 docs each)
  - Each shard is a JSON array of objects with fields: `id`, `text`, `language`, `label`
- **Sub-populations** (5K docs each, 6 total):

| Sub-population | Country | Language | Doc type | N |
|---|---|---|---|---|
| Denmark questions | Denmark | Danish | Parliamentary question | 5,000 |
| Spain questions | Spain | Spanish | Oral question | 5,000 |
| US bills | United States | English | Congressional bill | 5,000 |
| Belgium newspaper | Belgium | Dutch | Newspaper | 5,000 |
| Spain media (OOD) | Spain | Spanish | Media | 5,000 |
| Belgium TV (OOD) | Belgium | Dutch | TV news | 5,000 |

### Output — Campaign 1 (Binary)
- **Directory**: `cap_analysis/data/inference_output/claude-opus-30k-binary/`
- **Files**: `shard_{0-299}.csv` (300 files, 100 data rows each)
- **Columns**: `id,answer,language,label`
  - `answer`: "Yes" or "No"
  - `label`: ground-truth (1 = Law & Crime, 0 = not)

### Output — Campaign 2 (P(Y/N))
- **Directory**: `cap_analysis/data/inference_output/claude-opus-30k-pyn/`
- **Files**: `shard_{0-299}.csv` (300 files, 100 data rows each)
- **Columns**: `id,score,p_yes,p_no,language,label`
  - `score`: equals `p_yes` (probability document is Law & Crime)
  - `p_yes` + `p_no` sum to 1.0

## Prompts

### Campaign 1 (Binary)
Each sub-agent received a fresh context and was instructed to classify each document as Yes/No using the full CAP "Law and Crime" codebook definition (12+ subtopics). No probability elicitation.

### Campaign 2 (P(Y/N))
Each sub-agent was instructed to estimate P(Yes) and P(No) **without first committing to a Yes/No answer** — go directly to probability estimation. This avoids the anchoring effect where committing to an answer biases the subsequent probability toward 0 or 1.

The CAP codebook definition used (same for both campaigns):
> Law and Crime includes: general law, crime, and family issues; law enforcement agencies including border, customs, and specialized enforcement agencies; white collar crime, organized crime, counterfeiting, fraud, cyber-crime, and money laundering; illegal drug crime and enforcement, criminal penalties for drug crimes, and international efforts to combat drug trafficking; court administration, bail, pre-release, fines, and legal representation; prisons, jails, and parole systems; juvenile crime and justice; child abuse, child pornography, sexual exploitation of children, and parental kidnapping; family issues, domestic violence, child welfare, and family law; domestic criminal and civil codes; crime control, prevention, and impact of crime; and police and domestic security responses to terrorism.

## Performance Summary

### Campaign 1 — Binary
| Metric | Value |
|---|---|
| N | 30,000 |
| Prevalence | 10.4% |
| AUC | 0.940 |
| Accuracy | 0.956 |
| Precision | 0.731 |
| Recall | 0.920 |

### Campaign 2 — P(Y/N) Probabilities
| Metric | Value |
|---|---|
| N | 30,000 |
| Prevalence | 10.5% |
| **AUC** | **0.981** |
| **Log loss** | **0.122** |
| **ECE** | **0.033** |
| Acc (t=.5) | 0.955 |
| Precision | 0.747 |
| Recall | 0.862 |
| Pos mean | 0.746 |
| Neg mean | 0.066 |
| Exact 0.0% | 0.0% |
| Exact 1.0% | 0.0% |
| In [.1,.9] | 18.8% |
| Unique scores | 43 |

**Per-language AUC (Campaign 2)**:
| Language | AUC | N | Prevalence |
|---|---|---|---|
| Danish | 0.985 | 5,000 | 7.3% |
| Dutch | 0.984 | 10,000 | 10.0% |
| English | 0.978 | 5,000 | 4.8% |
| Spanish | 0.978 | 10,000 | 15.3% |

### Comparison to feasibility study (Claude Sonnet 4, 500-doc sample)

| Metric | Sonnet 4 (1S P(Y/N), N=500) | Opus 4.6 (P(Y/N), N=30K) |
|---|---|---|
| AUC | 0.973 | 0.981 |
| Log loss | 0.148 | 0.122 |
| ECE | 0.051 | 0.033 |
| Exact 0/1 | 0% | 0% |
| Unique scores | 31 | 43 |

Opus is slightly better calibrated and more discriminative than Sonnet on this task.

## Integration into the paper analysis

The main analysis notebook is `cap_analysis/cap_analysis.ipynb`. It currently uses Llama 3.3 70B verbalized 2-stage scores. To integrate Opus scores:

1. **Merge shards into a single file** aligned with `full_sample.csv` or `opus_30k_sample.csv`:
   - The `id` column in the shard CSVs maps to the `id` column in the sample CSVs
   - Campaign 2's `score` column is the primary input for calibration methods

2. **The notebook expects** a scores DataFrame with at least an `id` and `score` column, joined to `data_df` (loaded from `full_sample.csv`). The Opus 30K sample is a subset of the full 105K sample.

3. **Key differences from Llama scores**:
   - No squashing needed — Opus scores have zero boundary mass (no exact 0/1)
   - Score range: 0.01–0.97 (well within [0,1], no logit overflow risk)
   - Only 6 sub-populations (30K) vs Llama's 7 sub-populations (105K) — the Llama sample includes an additional Belgium newspaper OOD set that isn't in the Opus sample
   - Campaign 2 scores are the ones to use for calibration analysis (Campaign 1 is binary-only)

4. **Helper script**: `cap_analysis/opus_quick_analysis.py` loads all shards and computes the metrics above.

## Execution details

- **Model**: Claude Opus 4.6 (inherited from parent agent, not explicitly specified)
- **Architecture**: 300 shards processed via Claude Code sub-agents, 50 concurrent agents per round, 6 rounds per campaign
- **Date**: April 2026
- **Machine**: devvm28125.cco0.facebook.com
- **No API key needed**: Sub-agents ran within Claude Code sessions
