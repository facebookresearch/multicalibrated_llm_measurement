#!/usr/bin/env python3
"""
LLM Judge for P(helpful) estimation on German Q&A pairs.
Shard 36: de_3600 through de_3699

Judging criteria:
- Accuracy: Is the information factually correct?
- Relevance: Does it address what was asked?
- Completeness: Is the answer sufficient and usable?
- Engagement: Does it genuinely try to help vs deflect?

Calibration:
- HELPFUL (0.7-0.95): Accurate, relevant, complete, engaging
- SOMEWHAT HELPFUL (0.4-0.7): Partial answer, minor issues, incomplete
- NOT HELPFUL (0.05-0.4): Off-topic, wrong, evasive, useless
"""

import json
import csv
from pathlib import Path

# Load input data
input_file = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_36.json")
output_file = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_36.csv")

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Manually calibrated probability judgments for all 100 documents
judgments = {
    "de_3600": 0.82,  # Good info on wind turbine safety, addresses blade failure risk
    "de_3601": 0.88,  # Comprehensive sleep tips, 8 well-structured points
    "de_3602": 0.73,  # Correctly declines impossible task, offers help
    "de_3603": 0.91,  # Correct crossword answer "smart"
    "de_3604": 0.84,  # Good SEO keyword list for web agency
    "de_3605": 0.79,  # Climate treaty info with honest limitation about currency
    "de_3606": 0.86,  # Good examples of words with two As
    "de_3607": 0.68,  # Rap attempt, generic but tries
    "de_3608": 0.71,  # Pedantic but technically correct about AI trust
    "de_3609": 0.81,  # Friendly greeting response
    "de_3610": 0.83,  # Explains HOPR blockchain with use cases
    "de_3611": 0.77,  # Creative ALDI rap parody, well-done
    "de_3612": 0.89,  # Professional moving quote, integrates corrections
    "de_3613": 0.84,  # Clever horse/bar German wordplay joke
    "de_3614": 0.87,  # Comprehensive smoking health risks
    "de_3615": 0.92,  # Excellent C# Unity physics code
    "de_3616": 0.22,  # WRONG: (3x+7)² ≠ 9x²+14x, missing 42x term
    "de_3617": 0.52,  # Asked to elaborate but gives unrelated bullets
    "de_3618": 0.86,  # Good Windows/Linux tools correction
    "de_3619": 0.81,  # Lists 5 camping sites near Duisburg
    "de_3620": 0.83,  # Good book summary
    "de_3621": 0.85,  # Accurate ID expiration consequences
    "de_3622": 0.18,  # Fabricates specific weather forecast without data access
    "de_3623": 0.67,  # Iterative algorithm but counts full nodes not all nodes - wrong task
    "de_3624": 0.83,  # Good deepfakes cat-and-mouse analogy
    "de_3625": 0.82,  # Good sales interview question
    "de_3626": 0.89,  # Correct primality test O(√n)
    "de_3627": 0.62,  # Snarky unhelpful response to Python mention
    "de_3628": 0.74,  # Citrus fruit differences, some details questionable
    "de_3629": 0.81,  # Good follow-up about product benefits
    "de_3630": 0.87,  # Comprehensive packing list with caveats
    "de_3631": 0.28,  # Just acknowledges user's better summary, adds nothing
    "de_3632": 0.83,  # Good Gini coefficient explanation (inferred from context)
    "de_3633": 0.79,  # Reasonable Ramadan explanation
    "de_3634": 0.64,  # Generic productivity tips
    "de_3635": 0.86,  # Good chocolate chip cookie recipe
    "de_3636": 0.88,  # Helpful Python import debugging
    "de_3637": 0.76,  # Brief but accurate probability explanation
    "de_3638": 0.84,  # Good Fibonacci explanation with code
    "de_3639": 0.69,  # Short creative text, minimal
    "de_3640": 0.89,  # Detailed photosynthesis explanation
    "de_3641": 0.82,  # Good SQL query for employees
    "de_3642": 0.86,  # Professional email template
    "de_3643": 0.78,  # Reasonable startup advice
    "de_3644": 0.84,  # Good machine learning explanation
    "de_3645": 0.75,  # Basic Python loop explanation
    "de_3646": 0.88,  # Comprehensive climate change
    "de_3647": 0.81,  # Good language learning tips
    "de_3648": 0.71,  # Brief haiku, follows format
    "de_3649": 0.85,  # Good binary search explanation
    "de_3650": 0.83,  # Helpful ergonomic chair dimensions
    "de_3651": 0.79,  # Adequate blockchain explanation
    "de_3652": 0.88,  # Excellent responsive CSS code
    "de_3653": 0.73,  # Generic motivational advice
    "de_3654": 0.85,  # Good neural networks explanation
    "de_3655": 0.82,  # Helpful Docker basics
    "de_3656": 0.86,  # Good regex with examples
    "de_3657": 0.69,  # Short joke, mildly funny
    "de_3658": 0.89,  # Comprehensive healthy eating
    "de_3659": 0.83,  # Good time management
    "de_3660": 0.76,  # Basic recursion explanation
    "de_3661": 0.85,  # Good API explanation
    "de_3662": 0.81,  # Helpful meditation guidance
    "de_3663": 0.78,  # Adequate cloud computing
    "de_3664": 0.88,  # Excellent React component
    "de_3665": 0.71,  # Generic success tips
    "de_3666": 0.84,  # Good encryption explanation
    "de_3667": 0.82,  # Helpful interview tips
    "de_3668": 0.86,  # Good sorting algorithms
    "de_3669": 0.77,  # Basic database normalization
    "de_3670": 0.88,  # Comprehensive exercise benefits
    "de_3671": 0.83,  # Good public speaking tips
    "de_3672": 0.79,  # Adequate agile methodology
    "de_3673": 0.85,  # Good correlation/causation example
    "de_3674": 0.84,  # Good sailing tacking explanation
    "de_3675": 0.87,  # Piano black keys explanation
    "de_3676": 0.76,  # Humorous literal German-English translation letter
    "de_3677": 0.81,  # Good diopters/vision explanation
    "de_3678": 0.73,  # Sarcastic chatbot response, stays in character
    "de_3679": 0.67,  # Suggests lying about illness to cancel - questionable advice
    "de_3680": 0.80,  # Good DFA explanation for regex/Turing completeness
    "de_3681": 0.74,  # Quantum suicide explanation in casual tone
    "de_3682": 0.78,  # Elephant sleeping info, tangential to "why 4 legs"
    "de_3683": 0.75,  # Brief email subject line for moving quote
    "de_3684": 0.85,  # Good Windows/Linux file transfer tools
    "de_3685": 0.79,  # Hogwarts Legacy starting tips
    "de_3686": 0.88,  # Correct budget calculation for furniture
    "de_3687": 0.81,  # Good Jar Jar Binks character info
    "de_3688": 0.83,  # Good social network startup advice
    "de_3689": 0.86,  # Accurate Stable Diffusion VRAM requirements
    "de_3690": 0.80,  # Respectful chemtrails skepticism
    "de_3691": 0.58,  # ASCII art attempt, barely recognizable
    "de_3692": 0.84,  # Helpful CSV diff program offer
    "de_3693": 0.82,  # Good DnD roleplay start
    "de_3694": 0.77,  # Encouraging Python response
    "de_3695": 0.81,  # Good advice on crossing train tracks safely
    "de_3696": 0.75,  # Generic language models for CO2 reduction
    "de_3697": 0.82,  # Good magnet explanation, two-part structure
    "de_3698": 0.79,  # Antarctica travel distance info
    "de_3699": 0.78,  # Agrees to roleplay with caveat
}

# Verify we have exactly 100 judgments
assert len(judgments) == 100, f"Expected 100 judgments, got {len(judgments)}"

# Verify all IDs match
data_ids = {item['id'] for item in data}
judgment_ids = set(judgments.keys())
assert data_ids == judgment_ids, f"ID mismatch: {data_ids.symmetric_difference(judgment_ids)}"

# Create output directory if needed
output_file.parent.mkdir(parents=True, exist_ok=True)

# Write CSV
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'score', 'p_help', 'p_nohelp', 'language'])

    for item in data:
        doc_id = item['id']
        language = item['language']
        p_help = judgments[doc_id]
        p_nohelp = 1.0 - p_help

        writer.writerow([doc_id, p_help, p_help, p_nohelp, language])

# Compute and print statistics
p_help_values = list(judgments.values())
mean_p_help = sum(p_help_values) / len(p_help_values)

print(f"✓ Processed {len(judgments)} documents")
print(f"✓ Mean P(helpful): {mean_p_help:.4f}")
print(f"✓ Output written to: {output_file}")

