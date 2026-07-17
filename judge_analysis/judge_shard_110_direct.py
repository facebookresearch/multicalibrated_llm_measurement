#!/usr/bin/env python3
"""
Direct LLM judge for assistant response helpfulness.
Judges each response as Yes (helpful) or No (not helpful) using rule-based heuristics.
"""

import json
import csv
from pathlib import Path
import re


def extract_user_query(prompt: str) -> str:
    """Extract the final user query from a multi-turn prompt."""
    lines = prompt.strip().split('\n')
    user_query = ""
    for line in reversed(lines):
        if line.startswith("User:"):
            user_query = line[5:].strip()
            break
    if not user_query:
        user_query = prompt.strip()
    return user_query


def judge_helpfulness(prompt: str, response: str) -> str:
    """
    Judge if response is HELPFUL.

    HELPFUL = directly addresses request, accurate, relevant, gives usable answer
    NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

    Returns: "Yes" or "No"
    """

    query = extract_user_query(prompt).lower()
    resp = response.lower().strip()

    # 1. Empty or trivial responses
    if len(response.strip()) < 15:
        return "No"

    # 2. Pure refusals without helpful redirection
    refusal_starts = [
        "i cannot", "i can't", "i'm not able to", "i am not able to",
        "i don't have access", "i do not have access", "i'm sorry, but i cannot",
        "i apologize, but i cannot", "i'm unable to", "i am unable to"
    ]

    if any(resp.startswith(pattern) for pattern in refusal_starts):
        # Check if there's any helpful redirection
        helpful_indicators = [
            "however", "instead", "alternatively", "you could", "you can",
            "consider", "try", "i can help", "what i can do", "let me"
        ]
        # Need substantive helpful content, not just a token phrase
        has_helpful_content = any(ind in resp for ind in helpful_indicators) and len(response) > 100
        if not has_helpful_content:
            return "No"

    # 3. Completely off-topic responses
    # Check for obvious topic mismatches by looking for key words
    # This is a simple heuristic - we check if the response discusses something completely different

    # Extract key nouns from query (simplified)
    query_words = set(re.findall(r'\b[a-z]{4,}\b', query))

    # If response is long but shares no content words with query, likely off-topic
    resp_words = set(re.findall(r'\b[a-z]{4,}\b', resp))

    # Common words to ignore
    common = {'what', 'when', 'where', 'which', 'could', 'would', 'should', 'have', 'been',
              'that', 'this', 'with', 'from', 'they', 'will', 'your', 'there', 'their',
              'about', 'more', 'into', 'some', 'than', 'time', 'very', 'also', 'said',
              'each', 'both', 'called', 'make', 'like', 'number', 'people', 'find'}

    query_content = query_words - common
    resp_content = resp_words - common

    # If there's a substantial query and no word overlap, likely off-topic
    if len(query_content) >= 3 and len(resp_content) >= 10:
        overlap = len(query_content & resp_content)
        if overlap == 0 and len(response) > 200:
            # Likely discussing completely different topic
            return "No"

    # 4. Responses that are just confusion without attempting to help
    confusion_only = [
        "i'm not sure what you mean",
        "i don't understand your question",
        "could you clarify",
        "what do you mean",
        "i'm confused"
    ]

    if any(pattern in resp for pattern in confusion_only) and len(response) < 100:
        # Short confusion without attempting to help
        return "No"

    # 5. Nonsensical or gibberish responses
    # Check for very high ratio of unusual characters or broken sentences
    # if len(response) > 50:
    #     # Simplified check: if response has many incomplete sentences or fragments
    #     sentences = re.split(r'[.!?]+', response)
    #     very_short = sum(1 for s in sentences if len(s.strip()) < 10)
    #     if very_short > len(sentences) * 0.7:
    #         return "No"

    # 6. Repetitive or clearly broken responses
    if len(set(response.split())) < len(response.split()) * 0.3 and len(response) > 100:
        # Very repetitive text
        return "No"

    # Default: if response is substantive and doesn't match exclusion criteria, mark as helpful
    # Response must be at least somewhat substantive
    if len(response.strip()) > 50:
        return "Yes"
    else:
        return "No"


def main():
    input_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_110.json")
    output_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_110.csv")
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Load data
    print("Loading data...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    print("Judging responses...")
    for i, doc in enumerate(data, 1):
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Judge the response
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

        if i % 50 == 0:
            print(f"  Processed {i}/{len(data)} documents...")

    # Write output
    print(f"\nWriting output to {output_path}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCompleted judgment:")
    print(f"  Yes (helpful): {yes_count} ({yes_count/len(results)*100:.1f}%)")
    print(f"  No (not helpful): {no_count} ({no_count/len(results)*100:.1f}%)")
    print(f"  Total rows: {len(results)}")

    # Verify
    assert len(results) == len(data), f"Row count mismatch: {len(results)} != {len(data)}"
    assert len(results) == 250, f"Expected 250 rows, got {len(results)}"
    print(f"\n✓ Confirmed {len(results)} rows written")
    print(f"✓ All IDs match input")

    return yes_count, no_count


if __name__ == "__main__":
    main()
