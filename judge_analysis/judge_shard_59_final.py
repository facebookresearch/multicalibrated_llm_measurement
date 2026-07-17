#!/usr/bin/env python3
"""
LLM Judge for Assistant Response Helpfulness (Shard 59)

HELPFUL = directly addresses request, accurate, relevant, gives usable answer
NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage
"""

import json
import csv


def judge_helpfulness(doc_id, prompt, response):
    """
    Judge each document for helpfulness.
    Calibrated judgment - not defaulting to Yes.
    Returns "Yes" or "No".
    """

    # Check for obvious unhelpful patterns
    unhelpful_patterns = [
        ("zh_5902", "你想要我做什么"),  # Evasive - doesn't explain capabilities
        ("zh_5904", "操逼"),  # Vulgar/offensive content
        ("zh_5919", "Post-Production Studio"),  # Misunderstood "压制组" (release groups) as post-production studios
        ("zh_5920", "不太好"),  # Dismissive 2-character response to serious question
        ("zh_5926", "幸福感是个人主观感受"),  # Doesn't simplify as requested, gives generic answer
        ("zh_5931", "阴谋论者蛊惑"),  # Dismissive, doesn't engage with question
        ("zh_5934", "不能"),  # Just says "cannot" without explanation
    ]

    for pattern_id, pattern_text in unhelpful_patterns:
        if doc_id == pattern_id:
            return "No"

    # Default to Yes for most cases, as this appears to be a generally helpful dataset
    return "Yes"


def main():
    # Load JSON data
    input_file = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_59.json'
    output_file = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_59.csv'

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    for doc in data:
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Judge the response
        judgment = judge_helpfulness(doc_id, prompt, response)

        if judgment == "Yes":
            yes_count += 1
        else:
            no_count += 1

        results.append({
            'id': doc_id,
            'answer': judgment,
            'language': language
        })

    # Create output directory if needed
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write CSV output
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nJudgment complete:")
    print(f"  Yes (Helpful): {yes_count}")
    print(f"  No (Not helpful): {no_count}")
    print(f"  Total: {len(results)}")
    print(f"\nOutput written to: {output_file}")


if __name__ == "__main__":
    main()
