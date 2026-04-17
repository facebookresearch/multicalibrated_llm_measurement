# Claude Opus 4.6 Inference

## Overview

Two independent inference campaigns were run using Claude Opus 4.6 on 30,000 political documents from the Comparative Agendas Project (CAP). The task is binary classification: does the document primarily discuss "Law and Crime" (CAP major topic code 12)?

## Campaigns

### Campaign 1: Binary Classification
- **Prompt**: Classify each document as Yes/No using the full CAP "Law and Crime" codebook definition
- **Output**: `data/inference_output/claude-opus-30k-binary/shard_{0-299}.csv`
- **Columns**: id, answer, language, label

### Campaign 2: Direct Probability Elicitation (P(Y/N))
- **Prompt**: Estimate P(Yes) and P(No) without first committing to a Yes/No answer, to avoid anchoring
- **Output**: `data/inference_output/claude-opus-30k-pyn/shard_{0-299}.csv`
- **Columns**: id, score, p_yes, p_no, language, label

The two campaigns were run independently to avoid anchoring contamination.

## Execution Details

- **Model**: Claude Opus 4.6
- **Architecture**: 300 shards (100 docs each), processed via Claude Code sub-agents, 50 concurrent agents per round
- **Input data**: `data/opus_30k_sample.csv` (30,009 rows, stratified sample from full_sample.csv)
- **Date**: April 2026

## CAP Codebook Definition Used

> Law and Crime includes: general law, crime, and family issues; law enforcement agencies including border, customs, and specialized enforcement agencies; white collar crime, organized crime, counterfeiting, fraud, cyber-crime, and money laundering; illegal drug crime and enforcement, criminal penalties for drug crimes, and international efforts to combat drug trafficking; court administration, bail, pre-release, fines, and legal representation; prisons, jails, and parole systems; juvenile crime and justice; child abuse, child pornography, sexual exploitation of children, and parental kidnapping; family issues, domestic violence, child welfare, and family law; domestic criminal and civil codes; crime control, prevention, and impact of crime; and police and domestic security responses to terrorism.

## Prompts

**Campaign 1 (Binary):** "Does it primarily fall under the policy topic 'Law and Crime'...? Answer Yes or No."

**Campaign 2 (P(Y/N)):** "Without first deciding Yes or No, directly estimate the probability that this text is about Law and Crime. Provide P(Yes) and P(No) as your true belief probabilities, summing to 1.0."

The key design choice is that Campaign 2 asks for probabilities *without first committing to an answer*, to avoid the anchoring effect where a Yes/No decision biases the subsequent probability toward 0 or 1. See `claude_opus_inference.py` for the full prompt templates.

## Reproduction

The inference script `claude_opus_inference.py` contains the prompting logic and shard management utilities used for both campaigns. To reproduce, you need access to Claude Opus 4.6 via Claude Code sub-agents. The inference results (merged CSVs) are included in the repository. Raw shard files can be obtained from the authors upon request.
