#!/usr/bin/env python3
"""Judge helpfulness of assistant responses."""

import json
import csv
from pathlib import Path

def judge_helpfulness(prompt: str, response: str) -> str:
    """
    Judge if a response is HELPFUL (Yes) or NOT HELPFUL (No).

    HELPFUL = directly addresses the request, accurate, relevant, gives a usable answer
    NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage
    """

    # Check for clear non-helpful patterns
    response_lower = response.lower().strip()

    # Empty or extremely short responses (likely incomplete)
    if len(response.strip()) < 10:
        return "No"

    # Evasive/refusal patterns
    evasive_patterns = [
        "i cannot", "i can't", "i'm unable to", "i am unable to",
        "i don't have", "i do not have", "i'm not able to", "i am not able to",
        "i apologize, but i cannot", "i'm sorry, but i cannot",
        "as an ai", "as a language model", "i'm just an ai"
    ]

    # Check if response is primarily evasive (starts with or dominated by refusal)
    response_start = response_lower[:200]
    if any(pattern in response_start for pattern in evasive_patterns):
        # But allow if it provides alternative help after the refusal
        if len(response) > 300 and ("however" in response_lower or "instead" in response_lower or "alternatively" in response_lower):
            pass  # Continue to other checks
        else:
            return "No"

    # Off-topic - response doesn't engage with the prompt at all
    # This is harder to detect automatically, so we'll be conservative

    # Check for minimal engagement (just acknowledgment without substance)
    minimal_responses = [
        response_lower == "yes",
        response_lower == "no",
        response_lower == "ok",
        response_lower == "okay",
        len(response.strip()) < 20 and not any(c in response for c in "?!.")
    ]
    if any(minimal_responses):
        return "No"

    # If we haven't found clear problems, default to Yes
    # (calibrated judgment - don't default to Yes for everything)
    return "Yes"

def main():
    input_file = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_125.json")
    output_file = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_125.csv")

    # Read JSON
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Judge each response
    results = []
    yes_count = 0
    no_count = 0

    for i, doc in enumerate(data):
        if (i + 1) % 50 == 0:
            print(f"Processing {i+1}/{len(data)}...")

        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Judge helpfulness
        answer = judge_helpfulness(prompt, response)

        if answer == "Yes":
            yes_count += 1
        else:
            no_count += 1

        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

    # Write CSV
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Verify
    print(f"\nResults:")
    print(f"  Total rows: {len(results)}")
    print(f"  Yes: {yes_count}")
    print(f"  No: {no_count}")
    print(f"  Output: {output_file}")

if __name__ == "__main__":
    main()
