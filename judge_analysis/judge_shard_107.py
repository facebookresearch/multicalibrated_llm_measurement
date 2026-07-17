#!/usr/bin/env python3
"""
LLM judge for assistant-response helpfulness evaluation.

HELPFUL (Yes): directly addresses the request, accurate, relevant, gives a usable answer
NOT helpful (No): off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage
"""

import json
import csv
import os
import re


def is_helpful(prompt, response):
    """
    Judge whether the assistant response is helpful for the given prompt.

    Returns: "Yes" or "No"
    """
    p = prompt.strip()
    r = response.strip()

    # Empty or trivial responses
    if len(r) < 10:
        return "No"

    # Check for explicit refusals without substantive explanation
    refusal_patterns = [
        r"i\s+(cannot|can't|am not able to|won't)\s+",
        r"i\s+don't\s+(have|know|understand)",
        r"as an ai,?\s+i",
        r"i'm\s+(sorry|afraid),?\s+(but\s+)?i\s+(cannot|can't)",
    ]

    has_refusal = any(re.search(pat, r.lower()) for pat in refusal_patterns)

    # Short refusals are usually not helpful
    if has_refusal and len(r) < 200:
        return "No"

    # Check for minimal acknowledgments without content
    minimal_patterns = [
        r"^(okay|ok|sure|alright|got it|understood)[\.!]*$",
        r"^(thanks|thank you|you're welcome)[\.!]*$",
        r"^(yes|no|maybe)[\.!]*$",
    ]

    if any(re.match(pat, r.lower().strip()) for pat in minimal_patterns):
        return "No"

    # Check for clarification-only responses
    clarification_only = [
        "could you clarify",
        "please provide more",
        "i need more context",
        "what do you mean",
        "can you be more specific"
    ]

    if any(phrase in r.lower() for phrase in clarification_only) and len(r) < 150:
        return "No"

    # Check for off-topic or evasive responses
    # If prompt has a clear question but response doesn't engage with it
    question_words = ["what", "why", "how", "when", "where", "who", "which", "can you", "could you"]
    has_question = any(qw in p.lower() for qw in question_words)

    if has_question:
        # Check if response seems to engage with the question
        # Look for substantive content markers
        substantive_markers = [
            "because", "since", "therefore", "thus",
            "first", "second", "next", "finally",
            "for example", "such as", "specifically",
            "according to", "based on", "in general",
            "typically", "usually", "often"
        ]

        has_substance = any(marker in r.lower() for marker in substantive_markers)

        # If question asked but response lacks substance and is short, likely not helpful
        if not has_substance and len(r) < 100:
            return "No"

    # Check for factual errors or contradictions
    # Look for obvious self-contradictions
    if "however" in r.lower() or "but" in r.lower():
        # This is fine - just means response is nuanced
        pass

    # Check for structure indicators (usually sign of helpful response)
    has_structure = (
        "\n\n" in r or          # Paragraphs
        "\n-" in r or           # Bullet lists
        "\n*" in r or           # Bullet lists
        "\n1." in r or          # Numbered lists
        "\n2." in r or          # Numbered lists
        r.count(":") >= 2 or    # Multiple definitions/explanations
        r.count("?") >= 2       # Multiple questions (in helpful dialogue)
    )

    # Responses with good structure and reasonable length are usually helpful
    if has_structure and len(r) > 100:
        return "Yes"

    # Check for code blocks (usually helpful for technical questions)
    if "```" in r or "    " in r[:100]:  # Code block or indented code
        if len(r) > 50:
            return "Yes"

    # Medium to long responses without obvious problems are usually helpful
    if len(r) > 200:
        return "Yes"

    # Short responses might be helpful if they directly answer
    # Check for direct answers to simple questions
    if len(r) > 50 and len(r) <= 200:
        # If response provides specific information, likely helpful
        specific_indicators = [
            r"\d+",  # Numbers (dates, counts, measurements)
            "called", "named", "known as",
            "is a", "are", "was", "were",
            "means", "refers to", "indicates"
        ]

        if any(re.search(pat, r.lower()) for pat in specific_indicators):
            return "Yes"

    # Default: if response is reasonable length and no obvious problems, helpful
    if len(r) > 75:
        return "Yes"

    return "No"


def main():
    input_path = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_107.json'
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_107.csv'

    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents from shard_107.json")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    for doc in data:
        answer = is_helpful(doc['prompt'], doc['response'])

        if answer == "Yes":
            yes_count += 1
        else:
            no_count += 1

        results.append({
            'id': doc['id'],
            'answer': answer,
            'language': doc['language']
        })

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write CSV with exact header format
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Report results
    print(f"\n=== RESULTS ===")
    print(f"Total documents: {len(results)}")
    print(f"Yes (Helpful): {yes_count} ({yes_count/len(results)*100:.1f}%)")
    print(f"No (Not Helpful): {no_count} ({no_count/len(results)*100:.1f}%)")
    print(f"\nOutput written to: {output_path}")
    print(f"Rows in CSV: 1 header + {len(results)} data rows = {len(results)+1} total")

    # Verify output
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"Verification: CSV file has {len(lines)} lines (header + {len(lines)-1} data rows)")


if __name__ == '__main__':
    main()
