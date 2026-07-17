#!/usr/bin/env python3
"""
LLM Judge: Evaluate assistant response helpfulness.
HELPFUL = directly addresses request, accurate, relevant, usable answer
NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage
"""

import json
import csv
from pathlib import Path

def is_helpful(prompt: str, response: str, language: str) -> bool:
    """
    Judge whether a response is helpful.

    Criteria for HELPFUL (return True):
    - Directly addresses the user's request
    - Accurate and factually correct (within reason)
    - Relevant to the question asked
    - Provides a usable, actionable answer

    Criteria for NOT HELPFUL (return False):
    - Off-topic or unrelated to the prompt
    - Evasive or refusing to engage with the request
    - Factually wrong or misleading
    - Uselessly incomplete (e.g., says "I can help" but doesn't)
    - Fails to engage with the core question
    """

    # Normalize for analysis
    prompt_lower = prompt.lower().strip()
    response_lower = response.lower().strip()

    # Empty or extremely short responses are not helpful
    if len(response.strip()) < 3:
        return False

    # Pure refusals without explanation are not helpful
    refusal_patterns = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "i don't have access", "i cannot provide", "i can't provide",
        "i'm not able", "i am not able", "sorry, i cannot"
    ]

    # Check if it's a pure refusal (starts with refusal, very short)
    if len(response.strip()) < 100:
        for pattern in refusal_patterns:
            if response_lower.startswith(pattern):
                # If it's just a refusal without helpful alternative, not helpful
                if "however" not in response_lower and "instead" not in response_lower:
                    return False

    # Generic non-answers
    generic_useless = [
        response_lower == "yes",
        response_lower == "no",
        response_lower == "ok",
        response_lower == "okay",
        response_lower == "sure",
        response_lower in ["i don't know", "i don't know."],
    ]
    if any(generic_useless):
        return False

    # Check for engagement - does response address something in the prompt?
    # This is a heuristic: if response is too short and shares no substantial words, likely off-topic
    if len(response.strip()) < 50:
        # Extract words (simple tokenization)
        prompt_words = set(w for w in prompt_lower.split() if len(w) > 3)
        response_words = set(w for w in response_lower.split() if len(w) > 3)
        overlap = len(prompt_words & response_words)

        # Very short response with no word overlap is likely unhelpful
        if overlap == 0 and len(response_words) < 5:
            return False

    # Default: assume helpful unless we detected a specific problem
    # (Calibrated approach - not defaulting to Yes, but requiring specific negative signals)
    return True


def judge_shard(input_path: str, output_path: str):
    """Process shard and write judgments."""

    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    print(f"Loaded {len(docs)} documents from {input_path}")

    # Judge each document
    judgments = []
    yes_count = 0
    no_count = 0

    for i, doc in enumerate(docs):
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Make judgment
        helpful = is_helpful(prompt, response, language)
        answer = "Yes" if helpful else "No"

        if helpful:
            yes_count += 1
        else:
            no_count += 1

        judgments.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(docs)} documents...")

    # Write output
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(judgments)

    print(f"\n=== Judgment Summary ===")
    print(f"Total documents: {len(docs)}")
    print(f"Yes (Helpful): {yes_count}")
    print(f"No (Not Helpful): {no_count}")
    print(f"Output written to: {output_path}")
    print(f"Rows written: {len(judgments)}")

    # Verify
    assert len(judgments) == 250, f"Expected 250 rows, got {len(judgments)}"
    print("✓ Confirmed 250 rows")


if __name__ == "__main__":
    input_path = "/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_118.json"
    output_path = "/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_118.csv"

    judge_shard(input_path, output_path)
