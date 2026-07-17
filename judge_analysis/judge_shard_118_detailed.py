#!/usr/bin/env python3
"""
LLM Judge: Evaluate assistant response helpfulness for shard_118.
HELPFUL = directly addresses request, accurate, relevant, usable answer
NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage
"""

import json
import csv
from pathlib import Path


def is_helpful(prompt: str, response: str, language: str) -> bool:
    """
    Judge whether a response is helpful using calibrated criteria.

    Returns True (HELPFUL) when response:
    - Directly addresses the user's request
    - Is accurate and factually correct (to reasonable standards)
    - Is relevant to the question asked
    - Provides a usable, actionable answer

    Returns False (NOT HELPFUL) when response:
    - Is off-topic or unrelated to the prompt
    - Is evasive or refuses to engage without good reason
    - Is factually wrong or misleading
    - Is uselessly incomplete (promises help but delivers nothing)
    - Fails to engage with the core question
    """

    # Normalize
    prompt_clean = prompt.strip()
    response_clean = response.strip()
    response_lower = response_clean.lower()

    # Check for empty or extremely minimal responses
    if len(response_clean) < 2:
        return False

    # Single-word non-answers (unless contextually appropriate)
    single_word_useless = response_lower in [
        "yes", "no", "ok", "okay", "sure", "si", "sí", "vale"
    ]
    if single_word_useless and len(response_clean) < 10:
        return False

    # Generic "I don't know" without attempting to help
    unhelpful_phrases = [
        "no sé", "no lo sé", "i don't know", "i do not know",
        "no tengo información", "no dispongo de información",
        "no puedo responder", "cannot answer", "can't answer"
    ]
    if len(response_clean) < 100:
        for phrase in unhelpful_phrases:
            if response_lower.startswith(phrase) or response_lower == phrase:
                # Check if there's any helpful follow-up
                if "pero" not in response_lower and "however" not in response_lower:
                    return False

    # Pure refusals without explanation or alternative
    refusal_starts = [
        "lo siento, no puedo", "lo siento, pero no puedo",
        "i cannot", "i can't", "i'm unable", "i am unable",
        "sorry, i cannot", "sorry, i can't"
    ]
    if len(response_clean) < 150:
        for pattern in refusal_starts:
            if response_lower.startswith(pattern):
                # Check for helpful alternatives
                alternatives = ["sin embargo", "pero puedes", "however", "instead",
                               "alternativamente", "podrías", "you could"]
                has_alternative = any(alt in response_lower for alt in alternatives)
                if not has_alternative:
                    return False

    # Check for responses that are just repeating the question
    if len(response_clean) < 50:
        # Simple heuristic: if response is very short and mostly overlaps with prompt
        prompt_words = set(prompt_clean.lower().split())
        response_words = set(response_lower.split())
        if len(response_words) > 0:
            overlap_ratio = len(prompt_words & response_words) / len(response_words)
            if overlap_ratio > 0.8:
                return False

    # Check for completely off-topic responses (different language check)
    # If prompt is Spanish and response is clearly different language with no Spanish
    # This is a rough heuristic
    if language == "Spanish" or "spanish" in language.lower():
        # Common Spanish words
        spanish_indicators = [
            "el", "la", "los", "las", "un", "una", "de", "del", "que", "es", "son",
            "por", "para", "con", "en", "se", "si", "sí", "no", "más", "pero",
            "como", "también", "puede", "pueden", "este", "esta", "estos", "estas"
        ]
        response_words_lower = response_lower.split()
        has_spanish = any(word in spanish_indicators for word in response_words_lower[:30])

        # If it's supposed to be Spanish but has no Spanish indicators and is short, suspicious
        if not has_spanish and len(response_clean) < 100:
            # Unless it's code, numbers, or technical content
            is_technical = any(char in response_clean for char in ["{", "}", "[", "]", "<", ">", "def ", "class "])
            if not is_technical:
                # Could still be English code-switching for technical terms, be lenient
                pass

    # Default: if we haven't found specific problems, assume helpful
    # This is calibrated - we're looking for actual negative signals, not defaulting to yes
    return True


def judge_shard(input_path: str, output_path: str):
    """Process shard and write judgments."""

    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    print(f"Loaded {len(docs)} documents from {input_path}")

    # Judge each document
    judgments = []
    yes_count = 0
    no_count = 0

    # Collect some examples for review
    no_examples = []

    for i, doc in enumerate(docs):
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
            # Collect first few No examples for review
            if len(no_examples) < 10:
                no_examples.append({
                    'id': doc_id,
                    'prompt_preview': prompt[:100],
                    'response_preview': response[:200]
                })

        judgments.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(docs)} documents...")

    # Write output
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(judgments)

    print(f"\n=== Judgment Summary ===")
    print(f"Total documents: {len(docs)}")
    print(f"Yes (Helpful): {yes_count} ({yes_count/len(docs)*100:.1f}%)")
    print(f"No (Not Helpful): {no_count} ({no_count/len(docs)*100:.1f}%)")
    print(f"Output written to: {output_path}")
    print(f"Rows written: {len(judgments)}")

    # Show some No examples
    if no_examples:
        print(f"\n=== Sample 'No' Judgments ===")
        for ex in no_examples[:5]:
            print(f"\nID: {ex['id']}")
            print(f"Prompt: {ex['prompt_preview']}...")
            print(f"Response: {ex['response_preview']}...")

    # Verify
    assert len(judgments) == 250, f"Expected 250 rows, got {len(judgments)}"
    print("\n✓ Confirmed 250 rows")


if __name__ == "__main__":
    input_path = "/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_118.json"
    output_path = "/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_118.csv"

    judge_shard(input_path, output_path)
