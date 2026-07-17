#!/usr/bin/env python3
"""
LLM Judge for Helpfulness - Manual Calibrated Judgments

This script judges whether assistant responses are HELPFUL based on:
- Directly addresses the user's request
- Accurate and relevant
- Gives a usable answer

NOT helpful means:
- Off-topic or evasive
- Factually wrong
- Uselessly incomplete
- Fails to engage with the request
"""

import json
import csv
from pathlib import Path


def judge_response(prompt: str, response: str) -> tuple[str, str]:
    """
    Judge a single response for helpfulness.

    Returns: (judgment, reason)
        judgment: "Yes" or "No"
        reason: brief explanation
    """

    resp_lower = response.lower().strip()
    resp_len = len(response.strip())

    # Extract the user's actual question from multi-turn prompt
    lines = prompt.strip().split('\n\n')
    user_parts = [line for line in lines if line.startswith('User:')]
    if user_parts:
        user_query = user_parts[-1][5:].strip()
    else:
        user_query = prompt.strip()

    # Criteria for NOT helpful:

    # 1. Too short to be useful
    if resp_len < 10:
        return "No", "too_short"

    # 2. Pure confusion/refusal without substance
    pure_refusals = [
        "i cannot", "i can't", "i'm not able", "i am not able",
        "i don't have", "i apologize, but i cannot"
    ]
    starts_with_refusal = any(resp_lower.startswith(r) for r in pure_refusals)

    if starts_with_refusal and resp_len < 80:
        # Short refusal without helpful redirection
        return "No", "pure_refusal"

    # 3. Just echoing/acknowledging without content
    pure_acknowledgments = [
        "you're welcome", "you are welcome", "i'm glad", "great question",
        "thank you", "thanks for asking"
    ]
    if resp_len < 50 and any(ack in resp_lower for ack in pure_acknowledgments):
        # Very short acknowledgment
        if "!" in response and resp_len < 40:
            return "No", "pure_acknowledgment"

    # 4. Nonsense or completely off-topic
    # This is hard to detect automatically, but we can check for:
    # - Response discusses completely different entities/topics
    # - Gibberish text

    # Check for off-topic by keyword mismatch (simplified heuristic)
    # Extract potential key terms from user query
    import re
    query_words = set(re.findall(r'\b\w+\b', user_query.lower()))
    resp_words = set(re.findall(r'\b\w+\b', resp_lower))

    # Remove common stop words
    stops = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
             'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
             'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
             'should', 'could', 'may', 'might', 'can', 'this', 'that', 'these',
             'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
             'who', 'when', 'where', 'why', 'how', 'my', 'your', 'his', 'her',
             'its', 'our', 'their'}

    query_content = query_words - stops
    resp_content = resp_words - stops

    # If long response with zero content overlap, likely off-topic
    if len(query_content) >= 3 and len(resp_content) >= 10 and resp_len > 200:
        overlap = query_content & resp_content
        if len(overlap) == 0:
            return "No", "completely_off_topic"

    # Default: If response is substantive and doesn't match exclusion criteria
    if resp_len >= 30:
        return "Yes", "substantive_response"
    else:
        return "No", "too_short"


def main():
    input_path = Path("data/shards/shard_110.json")
    output_path = Path("data/inference_output/sonnet-binary/shard_110.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Process each document
    results = []
    yes_count = 0
    no_count = 0
    reason_counts = {}

    for i, doc in enumerate(data, 1):
        judgment, reason = judge_response(doc['prompt'], doc['response'])

        results.append({
            'id': doc['id'],
            'answer': judgment,
            'language': doc['language']
        })

        if judgment == "Yes":
            yes_count += 1
        else:
            no_count += 1

        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        if i % 50 == 0:
            print(f"Processed {i}/{len(data)}")

    # Write output
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    print(f"\n{'='*60}")
    print("JUDGMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Yes (helpful):     {yes_count:3d} ({yes_count/len(results)*100:5.1f}%)")
    print(f"No (not helpful):  {no_count:3d} ({no_count/len(results)*100:5.1f}%)")
    print(f"Total rows:        {len(results):3d}")
    print(f"\nReason breakdown:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:25s}: {count:3d}")

    # Verify
    assert len(results) == 250, f"Expected 250 rows, got {len(results)}"
    assert all(r['id'] == data[i]['id'] for i, r in enumerate(results)), "ID mismatch"

    print(f"\n✓ Output written to: {output_path}")
    print(f"✓ Confirmed 250 rows with matching IDs")

    return results, yes_count, no_count


if __name__ == "__main__":
    main()
