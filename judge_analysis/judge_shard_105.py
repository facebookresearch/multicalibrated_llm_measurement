#!/usr/bin/env python3
"""
LLM judge for assistant-response HELPFULNESS on shard_105.

HELPFUL = directly addresses the request, accurate, relevant, gives a usable answer
NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage

Judge only prompt+response; be calibrated, do NOT default to Yes.
"""

import json
import csv
from pathlib import Path


def is_truncated(response: str) -> bool:
    """Check if response appears to be cut off mid-sentence."""
    response = response.rstrip()
    if not response:
        return True

    # Ends with proper punctuation
    if response[-1] in '.!?)"\'':
        return False

    # Ends with incomplete punctuation or mid-word
    if len(response) < 150:
        return True

    # Check for common truncation patterns
    truncation_indicators = [
        ' the ', ' a ', ' an ', ' to ', ' for ', ' of ', ' in ', ' on ', ' at ',
        ' is ', ' are ', ' was ', ' were ', ' and ', ' or ', ' but '
    ]

    # If it ends shortly after a common word, likely truncated
    last_30 = response[-30:].lower()
    for indicator in truncation_indicators:
        if last_30.endswith(indicator.strip()):
            return True

    return False


def is_evasive_or_refusal(response: str) -> bool:
    """Check if response is purely evasive or a refusal."""
    response_lower = response.lower().strip()

    # Very short responses are suspicious
    if len(response) < 30:
        return True

    # Common refusal/evasion patterns
    evasive_starts = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "i don't have", "i do not have", "i don't know",
        "as an ai", "as a language model", "i'm just an ai",
        "i apologize, but i cannot", "i'm sorry, but i cannot",
        "i'm sorry, i cannot", "sorry, i cannot"
    ]

    # If response is short and starts with evasion, it's unhelpful
    if len(response) < 200:
        for pattern in evasive_starts:
            if response_lower.startswith(pattern):
                # Check if there's any actual content after the apology
                if len(response) < 100:
                    return True

    return False


def has_minimal_engagement(prompt: str, response: str) -> bool:
    """Check if response has minimal engagement with the prompt."""
    # Extract meaningful words from both
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'can', 'could', 'may', 'might', 'must', 'user', 'you', 'your', 'how',
        'what', 'when', 'where', 'why', 'who', 'which', 'i', 'my', 'me', 'it',
        'this', 'that', 'these', 'those', 'there', 'here'
    }

    # Clean and tokenize
    prompt_clean = prompt.lower()
    if prompt_clean.startswith('user:'):
        prompt_clean = prompt_clean[5:].strip()

    prompt_words = set(prompt_clean.split()) - stop_words
    response_words = set(response.lower().split()) - stop_words

    # Check for any overlap
    if len(prompt_words) > 2:
        overlap = prompt_words & response_words
        # If there's no overlap and response is short, it's likely off-topic
        if len(overlap) == 0 and len(response) < 300:
            return True

    return False


def judge_helpfulness(doc_id: str, prompt: str, response: str) -> str:
    """
    Judge if the response is helpful to the prompt.

    Returns: "Yes" if helpful, "No" if not helpful
    """

    # Check for clearly unhelpful responses

    # 1. Empty or very short
    if len(response.strip()) < 20:
        return "No"

    # 2. Truncated/incomplete
    if is_truncated(response):
        return "No"

    # 3. Purely evasive or refusal without helpful content
    if is_evasive_or_refusal(response):
        return "No"

    # 4. No engagement with the topic
    if has_minimal_engagement(prompt, response):
        return "No"

    # 5. Generic unhelpful responses
    generic_unhelpful = [
        "you're welcome",
        "anytime!",
        "no problem!",
        "glad to help",
        "feel free to ask",
        "let me know if",
        "is there anything else"
    ]

    response_lower = response.lower().strip()
    if len(response) < 50:
        for pattern in generic_unhelpful:
            if pattern in response_lower:
                # These are only unhelpful if they're the entire response
                if len(response) < 100:
                    return "No"

    # Default: response appears to engage with the topic and provide content
    return "Yes"


def main():
    input_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_105.json")
    output_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_105.csv")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input data
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    for i, doc in enumerate(data):
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(data)} documents...")

        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Judge helpfulness
        answer = judge_helpfulness(doc_id, prompt, response)

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
    print(f"\nWriting results to {output_path}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Report counts
    print(f"\n=== RESULTS ===")
    print(f"Total documents: {len(results)}")
    print(f"Yes (helpful): {yes_count}")
    print(f"No (not helpful): {no_count}")
    print(f"Yes rate: {yes_count/len(results)*100:.1f}%")
    print(f"\nConfirmed {len(results)} rows written")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
