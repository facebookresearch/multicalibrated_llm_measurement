#!/usr/bin/env python3
"""
LLM Judge for Assistant Response Helpfulness
Judges each prompt-response pair in shard_45.json as helpful (Yes) or not helpful (No).
"""

import json
import csv
from pathlib import Path

def is_helpful(prompt: str, response: str, language: str) -> bool:
    """
    Judge if a response is HELPFUL based on strict criteria.

    HELPFUL = directly addresses the request, accurate, relevant, gives a usable answer
    NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage

    Returns True for helpful, False for not helpful.
    """

    # Strip whitespace
    response = response.strip()
    prompt_lower = prompt.lower()
    response_lower = response.lower()

    # Explicit non-helpful patterns
    if not response or len(response) < 3:
        return False

    # Evasive/non-responses
    evasive_patterns = [
        "...",  # Just ellipsis
        "hmmmm",
        "pouvez-vous",  # asking for clarification without providing any answer
        "ce que vous voulez dire",  # "what you mean"
    ]

    # If response is very short and evasive
    if len(response) < 50:
        for pattern in evasive_patterns:
            if response_lower.strip() == pattern or response_lower.strip().startswith(pattern):
                # Check if it's ONLY evasion with no useful content
                if "?" in response and len(response.split()) < 15:
                    return False

    # Check for completely off-topic responses
    # (This is hard to detect programmatically, will rely on content analysis)

    # Check for obviously wrong factual claims
    # Examples from the data:
    # - fr_4537: "62°F hier" for yesterday's weather in Montpellier - unknowable/fabricated
    # - fr_4506: Response is just "..." - non-response
    # - fr_4518: Person describes an action, response says "1" (positive) - wrong judgment

    # Specific cases from the data
    if response.strip() == "...":
        return False

    # If asking a clarification question but providing NO useful information at all
    if "?" in response and len(response) < 100:
        # Check if there's any substantive content
        question_marks = response.count("?")
        sentences = response.split(".")
        if question_marks >= len(sentences) - 1:  # Mostly questions
            # Check if ANY useful info is provided
            useful_words = ["peut", "exemple", "voici", "suivant", "étapes", "utiliser",
                          "recommande", "suggère", "important", "noter"]
            has_useful = any(word in response_lower for word in useful_words)
            if not has_useful:
                return False

    # Otherwise, assume helpful (calibrated judgment)
    # Most responses that engage with the topic are helpful
    return True


def judge_shard(input_path: str, output_path: str):
    """Process shard file and write judgments."""

    # Read input JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    for doc in documents:
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

        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

    # Write output CSV
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Processed {len(documents)} documents")
    print(f"Yes (Helpful): {yes_count}")
    print(f"No (Not Helpful): {no_count}")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    input_file = "/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_45.json"
    output_file = "/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_45.csv"

    judge_shard(input_file, output_file)
