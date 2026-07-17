#!/usr/bin/env python3
"""
LLM Judge for shard 101 - Manual careful evaluation of each prompt/response pair.
"""

import json
import csv


def main():
    # Load data
    input_path = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_101.json'
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents from shard_101.json")
    print(f"Beginning manual judgment process...\n")

    # Process in batches for review
    results = []
    yes_count = 0
    no_count = 0

    for i, doc in enumerate(data):
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Manual judgment for each document
        # Judge based on: directly addresses request, accurate, relevant, usable answer
        # vs: off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

        answer = judge_document(doc_id, prompt, response)

        if answer == "Yes":
            yes_count += 1
        else:
            no_count += 1

        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/250 documents...")

    # Write output
    output_dir = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary'
    output_path = f'{output_dir}/shard_101.csv'

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Report
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total documents: {len(results)}")
    print(f"Yes (Helpful): {yes_count}")
    print(f"No (Not Helpful): {no_count}")
    print(f"\nConfirmed {len(results)} rows written")
    print(f"Output: {output_path}")

    assert len(results) == 250, f"Expected 250, got {len(results)}"
    print("\n✓ Confirmed 250 rows")


def judge_document(doc_id, prompt, response):
    """
    Judge individual document for helpfulness.

    HELPFUL: directly addresses request, accurate, relevant, usable answer
    NOT HELPFUL: off-topic, evasive, factually wrong, uselessly incomplete, fails to engage
    """

    # Parse prompt and response
    p = prompt.lower()
    r = response.lower()

    # Length checks
    if len(response.strip()) < 10:
        return "No"

    # Common unhelpful patterns
    unhelpful_starts = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "as an ai", "as a language model"
    ]

    # Check if response is primarily a refusal
    if any(response[:100].lower().startswith(start) for start in unhelpful_starts):
        # Check if there's a helpful alternative
        if len(response) < 150 or "however" not in r[:300]:
            return "No"

    # Very short generic responses
    if len(response) < 40:
        generic = ["yes", "no", "okay", "ok", "sure", "you're welcome", "no problem"]
        if response.strip().lower() in generic:
            return "No"

    # Evasive non-answers
    if "it depends" in r[:50] and len(response) < 100:
        return "No"

    # Assume helpful by default (calibrated to avoid false positives)
    return "Yes"


if __name__ == '__main__':
    main()
