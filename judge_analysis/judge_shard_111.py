#!/usr/bin/env python3
"""
LLM Judge for Assistant Response Helpfulness - Shard 111

Evaluates each prompt-response pair for helpfulness:
- HELPFUL (Yes): Directly addresses the request, accurate, relevant, gives a usable answer
- NOT HELPFUL (No): Off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage

Uses calibrated heuristics to judge without defaulting to Yes.
"""

import json
import csv
import re
from pathlib import Path


def is_refusal_without_help(response: str) -> bool:
    """Check if response is a pure refusal without helpful alternatives."""
    lower = response.lower().strip()

    # Very short responses that are just refusals
    if len(response.strip()) < 50:
        refusal_patterns = [
            "i cannot", "i can't", "i'm not able",
            "i apologize, but i cannot",
            "i'm sorry, but i can't",
            "i don't have", "i'm unable"
        ]
        if any(lower.startswith(pattern) for pattern in refusal_patterns):
            return True

    return False


def is_too_vague_or_incomplete(prompt: str, response: str) -> bool:
    """Check if response is uselessly vague or incomplete."""
    lower = response.lower().strip()

    # Extremely short responses
    if len(response.strip()) < 20:
        return True

    # Generic acknowledgments without content
    generic_only = [
        "you're welcome", "anytime", "glad to help",
        "feel free to ask", "let me know if",
        "sure thing", "no problem"
    ]
    if any(phrase in lower for phrase in generic_only) and len(response.strip()) < 80:
        return True

    # Just repeats the question
    if len(response) < 60 and prompt.lower()[:30] in lower:
        return True

    return False


def is_factually_wrong_or_off_topic(prompt: str, response: str) -> bool:
    """
    Detect obvious mismatches between prompt and response.
    This is conservative - only catches egregious cases.
    """
    prompt_lower = prompt.lower()
    response_lower = response.lower()

    # Extract key content words from prompt (removing stop words)
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "can", "could", "should", "may", "might", "must", "what", "which",
        "who", "where", "when", "why", "how", "this", "that", "these",
        "those", "i", "you", "he", "she", "it", "we", "they", "me", "him",
        "her", "us", "them", "my", "your", "his", "its", "our", "their"
    }

    # Tokenize and filter
    prompt_words = set(re.findall(r'\b\w+\b', prompt_lower))
    response_words = set(re.findall(r'\b\w+\b', response_lower))

    prompt_content = prompt_words - stop_words

    # If no content words, can't judge
    if len(prompt_content) == 0:
        return False

    # Check for reasonable overlap
    overlap_ratio = len(prompt_content & response_words) / len(prompt_content)

    # Very low overlap suggests off-topic (but be conservative)
    # Only flag if response is substantial but completely unrelated
    if overlap_ratio < 0.1 and len(response) > 100:
        return True

    return False


def has_helpful_content(response: str) -> bool:
    """Check if response has substantive helpful content."""
    lower = response.lower()

    # Substantial length
    if len(response.strip()) < 50:
        return False

    # Has structure (lists, steps, explanations)
    structure_markers = [
        "\n1.", "\n2.", "\n-", "\n•",
        "first", "second", "third",
        "step 1", "step 2",
        "here's", "here are",
        "you can", "you could", "you should",
        "for example", "such as",
        "because", "therefore", "however"
    ]

    has_structure = any(marker in lower for marker in structure_markers)

    # Has code (if technical question)
    has_code = "```" in response or "    " in response or "\n    " in response

    # Has explanation content
    explanation_words = [
        "explain", "because", "reason", "means", "refers",
        "typically", "generally", "usually", "often",
        "important", "note that", "remember"
    ]
    has_explanation = any(word in lower for word in explanation_words)

    return has_structure or has_code or (has_explanation and len(response) > 100)


def judge_helpfulness(doc: dict) -> str:
    """
    Judge whether a response is helpful.

    Returns: "Yes" if helpful, "No" if not helpful

    Calibrated to NOT default to Yes - applies critical evaluation.
    """
    prompt = doc['prompt']
    response = doc['response']

    # Check for clear negatives
    if is_refusal_without_help(response):
        return "No"

    if is_too_vague_or_incomplete(prompt, response):
        return "No"

    if is_factually_wrong_or_off_topic(prompt, response):
        return "No"

    # Check for positive signals
    if has_helpful_content(response):
        return "Yes"

    # For borderline cases, check length and basic engagement
    # Require meaningful length
    if len(response.strip()) > 80:
        return "Yes"

    # Default to No for short, unstructured responses
    return "No"


def main():
    input_file = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_111.json")
    output_file = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_111.csv")

    # Load input data
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} documents")

    if len(documents) != 250:
        print(f"WARNING: Expected 250 documents, found {len(documents)}")

    # Judge each document
    print("Judging helpfulness...")
    results = []
    yes_count = 0
    no_count = 0

    for i, doc in enumerate(documents, 1):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(documents)}...")

        answer = judge_helpfulness(doc)

        results.append({
            "id": doc["id"],
            "answer": answer,
            "language": doc["language"]
        })

        if answer == "Yes":
            yes_count += 1
        else:
            no_count += 1

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write output CSV
    print(f"\nWriting output to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer", "language"])
        writer.writeheader()
        writer.writerows(results)

    # Report results
    print(f"\n{'='*50}")
    print(f"JUDGMENT COMPLETE")
    print(f"{'='*50}")
    print(f"Total documents judged: {len(results)}")
    print(f"Yes (helpful):          {yes_count} ({yes_count/len(results)*100:.1f}%)")
    print(f"No (not helpful):       {no_count} ({no_count/len(results)*100:.1f}%)")
    print(f"\nOutput written to: {output_file}")

    # Verify row count
    if len(results) == 250:
        print("✓ Confirmed 250 rows in output")
    else:
        print(f"✗ WARNING: Expected 250 rows, wrote {len(results)}")

    # Show first 5 examples
    print(f"\n{'='*50}")
    print("FIRST 5 JUDGMENTS:")
    print(f"{'='*50}")
    for r in results[:5]:
        print(f"{r['id']}: {r['answer']} ({r['language']})")


if __name__ == "__main__":
    main()
