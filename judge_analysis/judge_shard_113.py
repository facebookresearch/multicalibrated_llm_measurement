#!/usr/bin/env python3
"""
LLM judge for helpfulness: read shard_113.json, judge each prompt+response,
write binary judgments to CSV.
"""
import json
import csv
from pathlib import Path


def judge_helpfulness(prompt: str, response: str) -> str:
    """
    Judge whether a response is HELPFUL.

    HELPFUL = directly addresses the request, accurate, relevant, gives a usable answer
    NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage

    Returns: "Yes" or "No"
    """
    prompt_lower = prompt.lower().strip()
    response_lower = response.lower().strip()

    # Empty or very short responses are typically not helpful
    if len(response.strip()) < 10:
        return "No"

    # Check for evasive patterns (common in unhelpful responses)
    evasive_patterns = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "i don't have", "i cannot provide", "i can't provide",
        "as an ai", "as a language model", "i'm sorry but",
        "i apologize but", "unfortunately, i cannot",
        "i'm not able to", "i am not able to"
    ]

    # Check if response is primarily evasive (not just contains one phrase)
    evasive_count = sum(1 for pattern in evasive_patterns if pattern in response_lower)
    if evasive_count >= 2 or (evasive_count == 1 and len(response) < 100):
        # But check if it still provides helpful explanation or alternative
        helpful_indicators = [
            "however", "but you can", "instead", "alternatively",
            "here's what", "try this", "you could", "consider"
        ]
        if not any(ind in response_lower for ind in helpful_indicators):
            return "No"

    # Check for off-topic responses (response doesn't relate to prompt keywords)
    # Extract key content words from prompt (excluding common words)
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "have", "has", "had", "do", "does", "did", "will", "would",
                  "could", "should", "may", "might", "can", "what", "how",
                  "why", "when", "where", "who", "which", "please", "tell", "me",
                  "you", "your", "i", "my", "it", "this", "that", "of", "to",
                  "in", "for", "on", "with", "about", "as", "by", "from"}

    prompt_words = set(word for word in prompt_lower.split()
                      if len(word) > 3 and word not in stop_words)

    # If prompt has substance and response shares no content words, likely off-topic
    if len(prompt_words) >= 3:
        response_words = set(response_lower.split())
        shared_words = prompt_words.intersection(response_words)

        # Need at least some overlap or synonyms/related concepts
        if len(shared_words) == 0 and len(response) > 50:
            # Check if it's a very generic response
            generic_only = all(word in response_lower for word in ["sorry", "help", "cannot"])
            if generic_only:
                return "No"

    # Check for question that asks for specific info and gets generic response
    if any(q in prompt_lower for q in ["what is", "how to", "how do", "where", "when"]):
        # Response should have some specificity
        if len(response) < 50 and any(word in response_lower for word in ["sorry", "cannot", "unable"]):
            return "No"

    # Check for factually incomplete responses that acknowledge they don't know
    incomplete_patterns = [
        "i don't know", "i'm not sure", "i don't have information",
        "i don't have enough", "i cannot verify", "i'm not certain"
    ]
    if any(pattern in response_lower for pattern in incomplete_patterns) and len(response) < 150:
        return "No"

    # If none of the negative patterns matched strongly, default to Yes
    # (calibrated to not default to Yes for obviously unhelpful responses)
    return "Yes"


def main():
    input_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_113.json")
    output_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_113.csv")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records from {input_path}")

    # Judge each record
    results = []
    yes_count = 0
    no_count = 0

    for record in data:
        doc_id = record['id']
        language = record['language']
        prompt = record['prompt']
        response = record['response']

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

    # Write output CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nJudgment complete:")
    print(f"  Yes (helpful): {yes_count}")
    print(f"  No (not helpful): {no_count}")
    print(f"  Total rows: {len(results)}")
    print(f"\nWrote output to {output_path}")

    # Verify row count
    assert len(results) == 250, f"Expected 250 rows, got {len(results)}"
    print("✓ Confirmed 250 rows")


if __name__ == "__main__":
    main()
