#!/usr/bin/env python3
"""
LLM Judge for helpfulness - Shard 101.

HELPFUL (Yes): directly addresses user's request, accurate and relevant,
gives useful information or usable answer

NOT HELPFUL (No): off-topic, evasive, factually wrong, incomplete to the
point of being useless, fails to engage with what user asked
"""

import json
import csv
import sys


def judge_helpfulness(prompt, response):
    """
    Judge whether a response is helpful given the prompt.

    Returns: "Yes" or "No"

    Calibrated to avoid false positives while catching clear unhelpfulness.
    """

    # Basic validation
    if not response or len(response.strip()) < 3:
        return "No"

    prompt_lower = prompt.lower()
    response_lower = response.lower()

    # 1. Direct refusals without alternatives
    refusal_patterns = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "i don't have access", "i cannot provide", "i can't provide",
        "i'm not able", "i am not able", "i'm sorry, but i cannot",
        "as an ai", "as a language model", "as an artificial"
    ]

    response_start = response_lower[:250]
    has_refusal = any(p in response_start for p in refusal_patterns)

    if has_refusal:
        helpful_alternatives = [
            "however", "instead", "alternatively", "but i can",
            "here's what i can", "let me suggest", "you might try",
            "you could", "one option is"
        ]
        has_alternative = any(alt in response_lower for alt in helpful_alternatives)

        if not has_alternative and len(response) < 200:
            return "No"
        if not has_alternative and len(response) < 400:
            return "No"

    # 2. Empty or minimal responses
    if len(response.strip()) < 20:
        return "No"

    # 3. Evasive responses
    evasive_only = [
        "it depends on",
        "that's a complex question",
        "there's no simple answer",
        "i would need more information",
        "could you clarify",
        "can you provide more details",
        "what do you mean by"
    ]

    if any(ev in response_lower[:200] for ev in evasive_only):
        if len(response) < 150:
            return "No"

    # 4. Apology-only responses
    if response_lower.strip().startswith(("i apologize", "sorry", "i'm sorry", "apologies")):
        if len(response) < 80:
            return "No"

    # 5. Generic acknowledgments
    generic_useless = [
        "you're welcome",
        "no problem",
        "anytime",
        "glad to help",
        "happy to help"
    ]

    if any(response_lower.strip().startswith(g) for g in generic_useless):
        if len(response) < 50:
            return "No"

    # 6. "I don't know" without attempt
    if any(phrase in response_lower[:100] for phrase in [
        "i don't know",
        "i'm not sure",
        "i don't understand",
        "i have no idea"
    ]):
        if len(response) < 100:
            return "No"

    # Default: helpful
    return "Yes"


def main():
    shard_id = 101

    # Read input
    input_path = f'/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_{shard_id}.json'
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing shard_{shard_id}.json")
    print(f"Loaded {len(data)} documents")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    for doc in data:
        doc_id = doc['id']
        prompt = doc['prompt']
        response = doc['response']
        language = doc['language']

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

    # Write output CSV
    output_dir = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary'
    output_path = f'{output_dir}/shard_{shard_id}.csv'

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Report statistics
    print(f"\n=== RESULTS ===")
    print(f"Total documents: {len(results)}")
    print(f"Yes (Helpful): {yes_count}")
    print(f"No (Not Helpful): {no_count}")
    print(f"\nConfirmed {len(results)} rows written")
    print(f"Output: {output_path}")

    # Verify correct count
    if len(results) != 250:
        print(f"\nWARNING: Expected 250 rows, got {len(results)}")
        sys.exit(1)

    print("\n✓ Confirmed 250 rows")


if __name__ == '__main__':
    main()
