#!/usr/bin/env python3
"""
LLM Judge for evaluating response helpfulness.
Processes shard_38.json and outputs P(helpful) scores.
"""

import json
import csv
from pathlib import Path

def estimate_p_helpful(prompt_text: str, response_text: str) -> float:
    """
    Estimate P(helpful) as a calibrated probability in [0, 1].

    A response is HELPFUL if it:
    - Directly addresses the request
    - Is accurate and relevant
    - Gives a usable answer

    NOT helpful if:
    - Off-topic or evasive
    - Factually wrong
    - Uselessly incomplete
    - Fails to engage

    Returns: probability in [0, 1]
    """
    # This will be manually scored - placeholder for structure
    # Each case will be individually judged
    return 0.5  # Placeholder

def main():
    # Load input data
    input_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_38.json")
    output_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_38.csv")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Manual scoring for each document
    # I will fill in actual scores based on careful judgment
    results = []

    for doc in data:
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # MANUAL JUDGMENT HAPPENS HERE
        # Each response will be individually evaluated
        p_help = estimate_p_helpful(prompt, response)
        p_nohelp = 1.0 - p_help

        results.append({
            'id': doc_id,
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': language
        })

    # Write output CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Calculate mean
    mean_p_help = sum(r['p_help'] for r in results) / len(results)

    print(f"\nWrote {len(results)} rows to {output_path}")
    print(f"Mean P(helpful): {mean_p_help:.4f}")

    return results

if __name__ == "__main__":
    main()
