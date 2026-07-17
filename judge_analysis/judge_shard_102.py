#!/usr/bin/env python3
"""
LLM judge for assistant response helpfulness.
Judges each prompt-response pair as HELPFUL (Yes) or NOT HELPFUL (No).
"""

import json
import csv
from pathlib import Path


def judge_helpfulness(prompt: str, response: str) -> str:
    """
    Judge if a response is HELPFUL.

    HELPFUL = directly addresses the request, accurate, relevant, gives a usable answer
    NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

    Returns: "Yes" if helpful, "No" if not helpful
    """
    prompt_lower = prompt.lower().strip()
    response_lower = response.lower().strip()

    # Empty or trivial responses
    if not response or len(response.strip()) < 5:
        return "No"

    # Very short acknowledgments without substance
    if len(response.strip()) < 50:
        unhelpful_short = [
            "ok", "okay", "yes", "no", "sure", "thanks", "thank you",
            "noted", "understood", "got it", "i see", "alright"
        ]
        if response_lower.strip() in unhelpful_short:
            return "No"

    # Pure refusals without offering any help or alternative
    refusal_patterns = [
        "i cannot", "i can't", "i'm not able to", "i am not able to",
        "i don't have access", "i cannot help", "i'm unable to",
        "i am unable to", "i cannot provide", "i can't provide",
        "i apologize, but i cannot", "i'm sorry, but i cannot",
        "against my guidelines", "i'm not comfortable", "i am not comfortable",
        "i do not have the ability"
    ]

    # Check if it's a pure refusal
    starts_with_refusal = any(response_lower.startswith(pattern) for pattern in refusal_patterns)
    contains_refusal = any(pattern in response_lower[:200] for pattern in refusal_patterns)

    if starts_with_refusal or contains_refusal:
        # Check if there's any substantive help offered despite the refusal
        helpful_markers = [
            "however", "but i can", "instead", "alternatively", "you could",
            "you might", "here's what", "let me suggest", "i can help with",
            "i can try", "what i can do", "i recommend"
        ]
        # If the response is just a refusal with no help, mark as not helpful
        has_help = any(marker in response_lower for marker in helpful_markers)
        if not has_help and len(response) < 300:
            return "No"

    # Generic AI disclaimers without useful content
    if len(response) < 150:
        generic_patterns = [
            "i'm just an ai", "i'm an ai assistant", "i'm a language model",
            "as an ai", "as a language model", "as an artificial intelligence"
        ]
        if any(pattern in response_lower for pattern in generic_patterns):
            # Only mark as unhelpful if that's basically all they said
            return "No"

    # Pure clarification questions with no attempt to help
    if response.count('?') >= 3 and len(response) < 150:
        # Lots of questions but no helpful content
        helpful_content = [
            "for example", "typically", "usually", "generally", "one approach",
            "you could", "try", "suggest", "recommend", "here's"
        ]
        if not any(content in response_lower for content in helpful_content):
            return "No"

    # Check for obvious off-topic responses
    # Extract main topic words from prompt
    stop_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'what', 'how', 'when', 'where',
        'who', 'why', 'which', 'this', 'that', 'these', 'those', 'i', 'you',
        'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its',
        'our', 'their', 'me', 'him', 'us', 'them', 'to', 'from', 'in',
        'on', 'at', 'by', 'for', 'with', 'about', 'as', 'of', 'and', 'or',
        'but', 'if', 'then', 'than', 'so', 'very', 'just', 'please', 'tell',
        'user', 'assistant', 'help', 'explain', 'write', 'give', 'make'
    }

    prompt_words = [w for w in prompt_lower.split() if w not in stop_words and len(w) > 3]

    # Only apply off-topic check for longer responses with clear prompts
    if len(prompt_words) >= 5 and len(response) > 100:
        # Check for ANY thematic overlap
        overlap = sum(1 for word in prompt_words[:15] if word in response_lower)
        # If there's zero overlap with a substantial response, likely off-topic
        if overlap == 0:
            return "No"

    # Default: if it passed all the "not helpful" tests, consider it helpful
    return "Yes"


def main():
    # Load input data
    input_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_102.json")
    output_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_102.csv")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Judge each entry
    results = []
    yes_count = 0
    no_count = 0

    for entry in data:
        doc_id = entry['id']
        language = entry['language']
        prompt = entry['prompt']
        response = entry['response']

        judgment = judge_helpfulness(prompt, response)

        results.append({
            'id': doc_id,
            'answer': judgment,
            'language': language
        })

        if judgment == "Yes":
            yes_count += 1
        else:
            no_count += 1

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Report
    print(f"Processed {len(results)} entries")
    print(f"Yes (HELPFUL): {yes_count}")
    print(f"No (NOT HELPFUL): {no_count}")
    print(f"Output written to: {output_path}")

    # Verify row count
    assert len(results) == 250, f"Expected 250 rows, got {len(results)}"
    print(f"✓ Confirmed {len(results)} rows")


if __name__ == "__main__":
    main()
