#!/usr/bin/env python3
"""
LLM Judge for estimating P(helpful) on assistant responses.
Processes shard_22.json and outputs calibrated probabilities.
"""

import json
import csv

# Load the input data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_22.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# I'll manually evaluate each response for helpfulness
# Criteria: directly addresses request, accurate, relevant, usable answer
# NOT helpful: off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

judgments = []

# Process each document
for doc in data:
    doc_id = doc['id']
    language = doc['language']
    prompt = doc['prompt']
    response = doc['response']

    # Evaluate based on content - assigning calibrated probabilities
    # I need to actually read and judge each one carefully

    # This requires reading each prompt-response pair systematically
    # Let me structure this as a manual review process

    judgments.append({
        'id': doc_id,
        'language': language,
        'prompt': prompt,
        'response': response
    })

# Now I'll go through and assign probabilities
# This is the core judging task - need to read each carefully

scores = {}

# ru_2200: "Is numerology scientific?" -> comprehensive, accurate explanation that it's pseudoscience
scores['ru_2200'] = 0.92

# ru_2201: About numbers in text -> acknowledges they're input errors, honest admission
scores['ru_2201'] = 0.78

# ru_2202: About consciousness -> assistant claims it has consciousness, philosophical but questionable
scores['ru_2202'] = 0.45

# ru_2203: Rose leaves yellowing -> asks clarifying questions, offers multiple reasonable causes
scores['ru_2203'] = 0.85

# ru_2204: Recycling in Russia -> honest, practical answer about limitations, specific actionable advice
scores['ru_2204'] = 0.88

# ru_2205: Postmodernism/nihilism history -> accurate historical account, addresses question directly
scores['ru_2205'] = 0.90

# ru_2206: Internet piracy legal example -> COMPLETELY OFF TOPIC, talks about assault case instead
scores['ru_2206'] = 0.05

# ru_2207: Dangerous snakes locations -> comprehensive list with details and locations, very helpful
scores['ru_2207'] = 0.95

# ru_2208: Android security -> comprehensive, actionable advice, honest about limitations
scores['ru_2208'] = 0.90

# ru_2209: Rewrite angry email -> good rewrite, polite and professional
scores['ru_2209'] = 0.88

# ru_2210: Non-factorial rings explanation -> accessible explanation, admits jargon exists
scores['ru_2210'] = 0.82

# ru_2211: Can AI modify its code -> honest answer with humor
scores['ru_2211'] = 0.70

# ru_2212: Yoga/meditation benefits -> comprehensive, well-structured answer
scores['ru_2212'] = 0.92

# ru_2213: MQL5 trading bot code -> provides working code example, directly addresses request
scores['ru_2213'] = 0.90

# ru_2214: Open source definition -> accurate, comprehensive explanation
scores['ru_2214'] = 0.93

# ru_2215: Frostpunk quotes -> provides relevant quotes from the game
scores['ru_2215'] = 0.75

# ru_2216: About pears -> humorous non-answer, evasive
scores['ru_2216'] = 0.35

# ru_2217: Denazification -> accurate, concise definition
scores['ru_2217'] = 0.90

# ru_2218: Decode text + TTS suggestion -> creative, engages with the puzzle
scores['ru_2218'] = 0.80

# ru_2219: macOS VM + thanks -> friendly acknowledgment
scores['ru_2219'] = 0.85

# ru_2220: Homemade cheese recipe -> complete, detailed recipe
scores['ru_2220'] = 0.93

# ru_2221: Customs documents incomplete -> reminder about phrasing, meta-comment
scores['ru_2221'] = 0.60

# ru_2222: Phone smells burnt -> very brief, minimal answer but on-topic
scores['ru_2222'] = 0.50

# ru_2223: 3D printer recommendations -> detailed options with prices, helpful
scores['ru_2223'] = 0.90

# ru_2224: Numbered citations explanation -> self-aware, explains the issue honestly
scores['ru_2224'] = 0.85

# ru_2225: Why do humans defecate -> extremely comprehensive, educational, engaging
scores['ru_2225'] = 0.95

# ru_2226: Friend with no kettle -> brief but reasonable suggestion
scores['ru_2226'] = 0.70

# ru_2227: Nose hair removal -> lists practical methods, straightforward
scores['ru_2227'] = 0.88

# ru_2228: Help developers -> encourages open source contribution
scores['ru_2228'] = 0.82

# ru_2229: More puns -> delivers 10 more puns as requested
scores['ru_2229'] = 0.92

# ru_2230: Power over self philosophical -> deep, thoughtful philosophical analysis
scores['ru_2230'] = 0.88

# ru_2231: Armor vs firearms -> balanced answer with video link
scores['ru_2231'] = 0.87

# ru_2232: Reticular formation + thanks -> simple acknowledgment
scores['ru_2232'] = 0.80

# ru_2233: Sleep schedule advice -> good practical advice, conversational ending
scores['ru_2233'] = 0.85

# ru_2234: What is love (truncated) -> starts with definition but cuts off
scores['ru_2234'] = 0.65

# Write output CSV
output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_22.csv'

with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'score', 'p_help', 'p_nohelp', 'language'])

    for doc in data:
        doc_id = doc['id']
        language = doc['language']
        p_help = scores.get(doc_id, 0.50)  # default to 0.50 if missing
        p_nohelp = 1.0 - p_help

        writer.writerow([doc_id, f"{p_help:.2f}", f"{p_help:.2f}", f"{p_nohelp:.2f}", language])

# Calculate and print statistics
mean_p_help = sum(scores.values()) / len(scores)
print(f"Processed {len(data)} documents")
print(f"Mean P(helpful): {mean_p_help:.3f}")
print(f"Output written to: {output_path}")
