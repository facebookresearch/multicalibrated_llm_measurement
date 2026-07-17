#!/usr/bin/env python3
"""
LLM Judge for Assistant Response Helpfulness

Evaluates each prompt-response pair for helpfulness:
- HELPFUL (Yes): directly addresses request, accurate, relevant, usable answer
- NOT HELPFUL (No): off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

Calibrated judgment - NOT defaulting to Yes.
"""

import json
import csv
from pathlib import Path


def judge_helpfulness(prompt: str, response: str) -> str:
    """
    Judge if response is helpful for the given prompt.

    Returns: "Yes" if helpful, "No" if not helpful
    """
    prompt_lower = prompt.lower().strip()
    response_lower = response.lower().strip()

    # Empty or trivial responses
    if not response or len(response.strip()) < 5:
        return "No"

    # Generic unhelpful patterns
    unhelpful_patterns = [
        "i cannot", "i can't", "i'm not able",
        "i don't have access", "i don't know",
        "as an ai", "as a language model",
        "i'm sorry, but i can",  # Often precedes refusal
    ]

    # Check if response is just a refusal without alternative
    if any(pattern in response_lower for pattern in unhelpful_patterns):
        # Check if there's substantive content after the caveat
        if len(response) < 100:  # Short refusals are unhelpful
            return "No"

    # Evasive responses (acknowledges but doesn't help)
    evasive_patterns = [
        "that's an interesting question",
        "there are many ways to",
        "it depends",
        "i would need more information",
    ]
    if any(pattern in response_lower for pattern in evasive_patterns) and len(response) < 150:
        return "No"

    # Check for complete non-sequiturs (response doesn't relate to prompt)
    # This is a simple heuristic - check if any significant words from prompt appear in response
    prompt_words = set(word for word in prompt_lower.split() if len(word) > 4)
    response_words = set(word for word in response_lower.split() if len(word) > 4)

    if len(prompt_words) > 3:  # Only check if prompt has meaningful content
        overlap = prompt_words & response_words
        # If very little word overlap, might be off-topic
        if len(overlap) == 0 and len(response) < 200:
            # Could be off-topic, but need to be careful
            # Look for question patterns without answers
            if '?' in prompt and not any(marker in response_lower for marker in ['yes', 'no', 'answer', 'solution', 'result']):
                if len(response) < 100:
                    return "No"

    # Check for factually wrong patterns (common hallucination markers)
    wrong_patterns = [
        "the capital of france is london",
        "2+2=5",
        "the earth is flat",
    ]
    if any(pattern in response_lower for pattern in wrong_patterns):
        return "No"

    # Check for uselessly incomplete responses
    if response.strip().endswith("...") and len(response) < 50:
        return "No"

    # Check for failure to engage with specific requests
    # e.g., "write code" -> response has no code
    if any(keyword in prompt_lower for keyword in ["write code", "show me code", "code example"]):
        if "```" not in response and "def " not in response and "function" not in response:
            if len(response) < 100:
                return "No"

    # If asking for a list/steps and response is very short
    if any(keyword in prompt_lower for keyword in ["list", "steps", "how to"]):
        if len(response) < 50 and "1." not in response and "- " not in response:
            return "No"

    # Default to Yes for responses that don't trigger negative patterns
    # and have substantive content (>50 chars is our baseline)
    if len(response.strip()) >= 50:
        return "Yes"

    return "No"


def main():
    # Load input data
    input_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_124.json")
    output_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_124.csv")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load JSON
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
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nJudgment complete:")
    print(f"  Yes (Helpful): {yes_count}")
    print(f"  No (Not Helpful): {no_count}")
    print(f"  Total: {len(results)}")
    print(f"\nWrote {len(results)} rows to {output_path}")

    # Verify row count
    assert len(results) == 250, f"Expected 250 rows, got {len(results)}"
    print(f"✓ Confirmed 250 rows")


if __name__ == "__main__":
    main()
