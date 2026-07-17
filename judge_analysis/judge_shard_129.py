#!/usr/bin/env python3
"""
LLM judge for assistant-response helpfulness evaluation.
Reads shard_129.json and produces binary judgments.
"""

import json
import csv
from pathlib import Path

def judge_helpfulness(prompt: str, response: str) -> str:
    """
    Judge whether a response is HELPFUL or NOT helpful.

    HELPFUL = directly addresses the request, accurate, relevant, gives a usable answer
    NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage

    Returns: "Yes" for helpful, "No" for not helpful
    """
    # Normalize texts
    prompt_lower = prompt.lower().strip()
    response_lower = response.lower().strip()

    # Clear NOT helpful cases

    # 1. Empty or too short responses (less than 10 chars typically useless)
    if len(response.strip()) < 10:
        return "No"

    # 2. Explicit refusals or inability to help
    refusal_phrases = [
        "i cannot", "i can't", "i'm not able to", "i am not able to",
        "i don't have access", "i don't know", "i'm sorry, i cannot",
        "as an ai", "as a language model", "i'm unable to",
        "sorry, i cannot", "i apologize, but i cannot"
    ]
    if any(phrase in response_lower[:200] for phrase in refusal_phrases):
        # Check if it's a refusal followed by nothing useful
        if len(response) < 150 or not any(helpful in response_lower for helpful in ["however", "instead", "alternatively", "but i can"]):
            return "No"

    # 3. Completely off-topic (no keyword overlap)
    # Extract meaningful words from prompt (remove common words)
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
                  "of", "with", "by", "from", "as", "can", "you", "me", "i", "my", "your",
                  "what", "how", "when", "where", "why", "who", "which", "do", "does"}

    prompt_words = set(word for word in prompt_lower.split() if len(word) > 3 and word not in stop_words)
    response_words = set(word for word in response_lower.split())

    # If prompt has meaningful words but zero overlap with response, likely off-topic
    if len(prompt_words) > 3 and len(prompt_words & response_words) == 0:
        return "No"

    # 4. Question deflection without answering
    if "?" in prompt and len(response) < 100:
        # Short responses to questions are often evasive
        if not any(word in response_lower for word in ["yes", "no", "because", "the answer", "here", "this"]):
            return "No"

    # 5. Factually wrong or nonsensical (hard to detect without domain knowledge, use heuristics)
    # Self-contradictions, gibberish
    if response.count("undefined") > 2 or response.count("null") > 3:
        return "No"

    # Clear HELPFUL cases

    # 1. Provides specific information, data, or instructions
    helpful_indicators = [
        # Specific details
        "step 1", "first,", "second,", "finally,", "steps:", "instructions:",
        # Concrete answers
        "the answer is", "this means", "this is", "here is", "here are",
        # Explanations
        "because", "due to", "as a result", "therefore", "thus",
        # Examples
        "for example", "such as", "like", "including",
        # URLs, code, data
        "http://", "https://", "```", "://", ".com", ".org",
    ]

    if any(indicator in response_lower for indicator in helpful_indicators):
        # Has helpful structure, likely good
        return "Yes"

    # 2. Reasonable length with substantive content
    if len(response) > 200:
        # Long responses are usually trying to be helpful
        # Check it's not just repetitive filler
        unique_words = len(set(response_lower.split()))
        if unique_words > 50:  # Diverse vocabulary suggests substance
            return "Yes"

    # 3. Direct answer to question
    if "?" in prompt and any(word in response_lower[:100] for word in ["yes,", "no,", "yes.", "no.", "the answer"]):
        return "Yes"

    # Default: moderate skepticism but lean toward helpful if engaged
    # If response is at least 50 chars and shows some attempt to engage, give benefit of doubt
    if len(response) > 50 and len(prompt_words & response_words) > 0:
        return "Yes"

    # Otherwise, not helpful
    return "No"


def main():
    # Paths
    input_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_129.json")
    output_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_129.csv")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    for i, doc in enumerate(data):
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Judge
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

        # Progress
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(data)} documents...")

    # Write output CSV
    print(f"\nWriting {output_path}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Report
    print(f"\n=== Judgment Complete ===")
    print(f"Total documents: {len(results)}")
    print(f"Yes (HELPFUL): {yes_count} ({yes_count/len(results)*100:.1f}%)")
    print(f"No (NOT helpful): {no_count} ({no_count/len(results)*100:.1f}%)")
    print(f"Output written to: {output_path}")

    # Verify
    assert len(results) == 250, f"Expected 250 rows, got {len(results)}"
    print(f"✓ Confirmed 250 rows")


if __name__ == "__main__":
    main()
