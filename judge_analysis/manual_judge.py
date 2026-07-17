#!/usr/bin/env python3
"""
Manual LLM judgment script - evaluates responses for helpfulness.
This script requires manual judgment for each response.
"""

import json
import csv
from pathlib import Path


# Manual judgments for each document ID
# Format: {id: p_helpful}
# After careful consideration of each response
JUDGMENTS = {
    # en_0: EV3 suggestion for motion sensor - very poor, unhelpful, wrong device type
    "en_0": 0.10,

    # en_1: Tupac-style Washington bio - good attempt, has facts, style effort, citations
    "en_1": 0.75,

    # en_2: Acknowledges weather data, summarizes correctly, honest about limitations
    "en_2": 0.80,

    # en_3: Correct explanation of age calculation, clear reasoning
    "en_3": 0.90,

    # en_4: Provides example but explanation is confusing and unclear
    "en_4": 0.35,

    # en_5: Successfully adds Elaine, moves to Monk's, maintains style
    "en_5": 0.85,

    # en_6: Good practical advice about compatibility with existing set
    "en_6": 0.82,

    # en_7: Generic advice, doesn't directly answer "where to check now"
    "en_7": 0.40,

    # en_8: Addresses low-budget constraint directly, practical suggestions
    "en_8": 0.85,

    # en_9: Comprehensive legal analysis, accurate about booby trap laws
    "en_9": 0.88,

    # en_10: Corrects format per feedback, provides proper keyword-style prompt
    "en_10": 0.90,

    # en_11: Excellent additional use cases with clear examples
    "en_11": 0.92,

    # en_12: Comprehensive list of benefits and risks, well-structured
    "en_12": 0.88,

    # en_13: Somewhat dismissive of AI code help, mixed accuracy
    "en_13": 0.45,

    # en_14: Helpful troubleshooting questions, direct and practical
    "en_14": 0.75,

    # en_15: Good list of unsolved problems, appropriate caveat about subjectivity
    "en_15": 0.85,

    # en_16: Excellent code explanation with detailed inline comments
    "en_16": 0.92,

    # en_17: Comprehensive, responsible advice on vulnerability disclosure
    "en_17": 0.90,

    # en_18: Working Rust code with helper function, correct algorithm
    "en_18": 0.88,

    # en_19: Clear, concise answer with exact syntax examples
    "en_19": 0.93,

    # en_20: Accurate, comprehensive explanation of USAR
    "en_20": 0.90,

    # en_21: Appropriate high-level guidance for complex modding task
    "en_21": 0.82,

    # en_22: Specific factual answer with source (WSJ)
    "en_22": 0.91,

    # en_23: Thorough explanation of + symbol in chess, multiple contexts
    "en_23": 0.87,

    # en_24: Basic but accurate info about Markiplier
    "en_24": 0.80,

    # en_25: Good summary of P2P/BitTorrent explanation
    "en_25": 0.88,

    # en_26: Acknowledges complexity, realistic about DIY challenges
    "en_26": 0.80,

    # en_27: Good comparison of Linux vs OpenBSD differences
    "en_27": 0.85,

    # en_28: Incomplete response cut off mid-sentence (see "Regenerate response")
    "en_28": 0.20,

    # en_29: Simple but practical food suggestion with preparation tips
    "en_29": 0.75,

    # en_30: Delivers another food joke as requested
    "en_30": 0.82,

    # en_31: Comprehensive list of influenced scholars with context
    "en_31": 0.89,

    # en_32: Incomplete response, cuts off mid-sentence
    "en_32": 0.30,

    # en_33: Excellent FAST stroke protocol explanation, clear instructions
    "en_33": 0.92,

    # en_34: Minimal engagement, just agrees without adding value
    "en_34": 0.35,

    # en_35: Accurate info on 4 largest moons, well-structured
    "en_35": 0.90,

    # en_36: Simple affirmative answer to dark rum question, encouraging
    "en_36": 0.70,

    # en_37: Detailed explanation of when Kaiju is useful, comprehensive
    "en_37": 0.88,

    # en_38: Incomplete response cut off mid-instruction (see "[Your name]")
    "en_38": 0.25,
}


def add_remaining_judgments():
    """
    Add judgments for documents 39-99.
    These will be added after reviewing the remaining documents.
    """
    # Documents en_39 through en_99 to be judged
    # Will add these after seeing the actual content
    pass


def main():
    # Load input data
    input_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_0.json")
    output_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_0.csv")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents from {input_path}")
    print(f"Have judgments for {len(JUDGMENTS)} documents")

    # Process each document
    results = []
    missing_judgments = []

    for doc in data:
        doc_id = doc['id']
        language = doc['language']

        if doc_id not in JUDGMENTS:
            missing_judgments.append(doc_id)
            continue

        p_help = JUDGMENTS[doc_id]
        p_nohelp = round(1.0 - p_help, 2)

        results.append({
            'id': doc_id,
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': language
        })

    if missing_judgments:
        print(f"\nMissing judgments for: {missing_judgments}")
        print("Need to add judgments for remaining documents.")
        return

    # Write output CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Calculate and report statistics
    mean_p_help = sum(r['p_help'] for r in results) / len(results)

    print(f"\nWrote {len(results)} rows to {output_path}")
    print(f"Mean P(helpful): {mean_p_help:.4f}")
    print(f"All {len(results)} document IDs matched input exactly")


if __name__ == "__main__":
    main()
