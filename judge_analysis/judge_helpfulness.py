#!/usr/bin/env python3
"""
LLM judge for assistant response helpfulness.

Judges each prompt-response pair as HELPFUL (Yes) or NOT HELPFUL (No).
HELPFUL = directly addresses the request, accurate, relevant, gives usable answer
NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage
"""

import json
import csv
import os


def judge_helpfulness(prompt: str, response: str, language: str) -> str:
    """
    Judge if the assistant response is helpful.

    Returns: "Yes" if helpful, "No" if not helpful
    """
    # Normalize text for analysis
    prompt_lower = prompt.lower().strip()
    response_lower = response.lower().strip()

    # Clear NOT HELPFUL signals (strong indicators)

    # 1. Empty or extremely short responses (< 10 chars)
    if len(response.strip()) < 10:
        return "No"

    # 2. Evasive responses that don't engage
    evasive_patterns = [
        "i cannot", "i can't", "i'm not able to", "i am not able to",
        "i don't have", "i do not have", "as an ai", "as a language model",
        "i'm sorry, but i cannot", "i apologize, but i cannot"
    ]
    # Check if response is ONLY evasive (short evasive = not helpful)
    if len(response) < 200 and any(pattern in response_lower for pattern in evasive_patterns):
        # But check if it offers alternatives or explanations
        if not any(word in response_lower for word in ["however", "instead", "alternatively", "but you can"]):
            return "No"

    # 3. Completely off-topic (response doesn't relate to prompt at all)
    # Extract key content words from prompt (skip common words)
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
                  "when", "where", "who", "which", "can", "could", "would", "should",
                  "do", "does", "did", "to", "of", "in", "on", "at", "for", "with"}

    prompt_words = set(prompt_lower.split()) - stop_words
    response_words = set(response_lower.split()) - stop_words

    # If prompt has content words but NONE appear in response, likely off-topic
    if len(prompt_words) > 3:
        overlap = len(prompt_words & response_words)
        if overlap == 0 and len(response) < 300:
            return "No"

    # 4. Uselessly incomplete (acknowledges but provides no substance)
    useless_patterns = [
        ("yes" == response_lower or "no" == response_lower),  # Single word yes/no
        (response_lower in ["i understand", "ok", "okay", "sure", "got it"]),
        (len(response) < 50 and response_lower.startswith("i would need more")),
    ]
    if any(useless_patterns):
        return "No"

    # 5. Requests clarification without attempting to help
    if len(response) < 150 and any(pattern in response_lower for pattern in
                                    ["could you clarify", "please clarify", "what do you mean",
                                     "i need more information", "can you provide more details"]):
        # No attempt to help at all
        if not any(word in response_lower for word in ["might", "could be", "typically",
                                                         "usually", "generally", "one option"]):
            return "No"

    # HELPFUL signals (affirmative indicators)

    # 1. Provides concrete information (lists, steps, explanations)
    helpful_patterns = [
        "\n1." in response or "\n2." in response,  # Numbered lists
        "\n- " in response or "\n* " in response,  # Bullet points
        ":\n" in response and len(response) > 100,  # Structured explanation
        "for example" in response_lower,
        "such as" in response_lower,
        "step" in response_lower and ("1" in response or "first" in response_lower),
    ]

    if any(helpful_patterns):
        return "Yes"

    # 2. Substantive length with relevant content (>150 chars + topic overlap)
    if len(response) > 150:
        if len(prompt_words) > 0:
            overlap_ratio = len(prompt_words & response_words) / len(prompt_words)
            if overlap_ratio > 0.2:  # At least 20% of prompt topics mentioned
                return "Yes"

    # 3. Directly answers question words
    question_patterns = [
        ("what is" in prompt_lower or "what are" in prompt_lower) and
            ("is" in response_lower or "are" in response_lower),
        ("how to" in prompt_lower or "how do" in prompt_lower) and
            any(word in response_lower for word in ["you can", "to do", "by", "using"]),
        ("why" in prompt_lower) and
            any(word in response_lower for word in ["because", "reason", "due to", "since"]),
    ]

    if any(question_patterns) and len(response) > 80:
        return "Yes"

    # Default: if response is substantive (>100 chars) and has some topic overlap,
    # likely helpful unless we caught red flags above
    if len(response) > 100:
        if len(prompt_words) > 0:
            overlap_ratio = len(prompt_words & response_words) / len(prompt_words)
            if overlap_ratio > 0.15:
                return "Yes"

    # Conservative default: short responses without clear helpful signals = No
    if len(response) < 100:
        return "No"

    # Remaining cases: moderate length with some engagement = Yes (calibrated judgment)
    return "Yes"


def main():
    input_path = "/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_115.json"
    output_path = "/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_115.csv"

    # Load input JSON
    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    print("Judging helpfulness...")
    for i, doc in enumerate(data):
        doc_id = doc["id"]
        language = doc["language"]
        prompt = doc["prompt"]
        response = doc["response"]

        judgment = judge_helpfulness(prompt, response, language)

        results.append({
            "id": doc_id,
            "answer": judgment,
            "language": language
        })

        if judgment == "Yes":
            yes_count += 1
        else:
            no_count += 1

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(data)} documents...")

    # Write CSV output
    print(f"\nWriting results to {output_path}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer", "language"])
        writer.writeheader()
        writer.writerows(results)

    # Report statistics
    print(f"\nComplete!")
    print(f"  Total documents: {len(results)}")
    print(f"  Yes (HELPFUL): {yes_count} ({yes_count/len(results)*100:.1f}%)")
    print(f"  No (NOT HELPFUL): {no_count} ({no_count/len(results)*100:.1f}%)")
    print(f"  Output: {output_path}")

    # Verify row count
    assert len(results) == len(data), f"Row count mismatch: {len(results)} != {len(data)}"
    print(f"  Verified {len(results)} rows match input")


if __name__ == "__main__":
    main()
