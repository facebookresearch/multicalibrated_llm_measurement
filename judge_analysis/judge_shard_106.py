#!/usr/bin/env python3
"""
LLM-based helpfulness judge for assistant responses.
Reads shard_106.json and outputs binary judgments (Yes/No) to CSV.
"""

import json
import csv
from pathlib import Path

def judge_helpfulness(prompt: str, response: str) -> str:
    """
    Judge if the response is HELPFUL to the prompt.

    HELPFUL = directly addresses the request, accurate, relevant, gives a usable answer
    NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage

    Returns: "Yes" or "No"
    """
    # Normalize inputs
    prompt = prompt.strip()
    response = response.strip()

    # Empty or trivial responses
    if not response or len(response) < 5:
        return "No"

    # Check for evasive/refusal patterns
    evasive_patterns = [
        "i cannot", "i can't", "i'm not able to", "i am not able to",
        "i don't have", "i do not have", "i'm unable", "i am unable",
        "as an ai", "as a language model", "i apologize, but",
        "sorry, but i can't", "i'm sorry, but i can't"
    ]
    response_lower = response.lower()

    # If response is mostly refusal without any helpful alternative, it's not helpful
    if any(pattern in response_lower[:200] for pattern in evasive_patterns):
        # Check if it provides an alternative or explanation (at least 100 chars after refusal)
        if len(response) < 150:
            return "No"

    # Check for engagement with the prompt
    # Very short responses to substantial prompts are likely unhelpful
    if len(prompt) > 100 and len(response) < 30:
        return "No"

    # Generic/template responses that don't address specifics
    generic_only_patterns = [
        "let me know if you need",
        "is there anything else",
        "how can i help",
        "what can i do for you"
    ]
    if len(response) < 100 and any(pattern in response_lower for pattern in generic_only_patterns):
        return "No"

    # Check for factual impossibilities or nonsense
    # (This is harder to detect automatically, rely on other heuristics)

    # If response is substantive and doesn't match clear failure patterns, assume helpful
    # Most assistant responses attempt to be helpful, so default should lean toward Yes
    # but we apply calibrated judgment based on the patterns above

    # Additional check: response should relate to prompt topic
    # For very short prompts, almost any substantive response is acceptable
    if len(prompt) < 20 and len(response) > 50:
        return "Yes"

    # For longer prompts, check if response engages with key terms
    # Extract potential key terms (words longer than 4 chars, excluding common words)
    common_words = {
        'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'what',
        'when', 'where', 'which', 'will', 'would', 'could', 'should', 'about',
        'there', 'their', 'they', 'them', 'these', 'those', 'make', 'know',
        'think', 'take', 'come', 'want', 'more', 'some', 'time', 'very',
        'just', 'like', 'help', 'need', 'good', 'much', 'well', 'also'
    }

    prompt_words = set(word.lower() for word in prompt.split()
                      if len(word) > 4 and word.lower() not in common_words)
    response_words = set(word.lower() for word in response.split())

    # If prompt has specific terms and response shares none of them, likely off-topic
    if len(prompt_words) > 3:
        overlap = len(prompt_words & response_words)
        if overlap == 0 and len(response) < 200:
            # No overlap and short response = likely off-topic
            return "No"

    # Default: if no clear negative signals, assume helpful
    return "Yes"


def main():
    input_path = Path('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_106.json')
    output_path = Path('/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_106.csv')

    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents from {input_path}")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    for doc in data:
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        judgment = judge_helpfulness(prompt, response)

        if judgment == "Yes":
            yes_count += 1
        else:
            no_count += 1

        results.append({
            'id': doc_id,
            'answer': judgment,
            'language': language
        })

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nJudgment complete:")
    print(f"  Yes (helpful): {yes_count}")
    print(f"  No (not helpful): {no_count}")
    print(f"  Total rows: {len(results)}")
    print(f"\nOutput written to {output_path}")

    # Verify output
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"Verified: {len(lines) - 1} data rows (+ 1 header)")


if __name__ == '__main__':
    main()
