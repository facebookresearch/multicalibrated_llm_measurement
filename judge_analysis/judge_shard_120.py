#!/usr/bin/env python3
"""
Judge helpfulness of assistant responses in shard_120.json
Outputs binary judgments to CSV.
"""

import json
import csv
from typing import Dict, Any

def judge_helpfulness(doc: Dict[str, Any]) -> str:
    """
    Judge whether the assistant response is HELPFUL.

    HELPFUL = directly addresses the request, accurate, relevant, gives a usable answer
    NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage

    Returns: "Yes" or "No"
    """
    prompt = doc.get("prompt", "").strip()
    response = doc.get("response", "").strip()

    # Empty or missing response is not helpful
    if not response:
        return "No"

    # Empty or missing prompt - cannot judge properly, default to No
    if not prompt:
        return "No"

    # Convert to lowercase for checking
    prompt_lower = prompt.lower()
    response_lower = response.lower()

    # Check for clear refusals or evasions
    refusal_phrases = [
        "i cannot", "i can't", "i'm not able to", "i am not able to",
        "i'm unable to", "i am unable to", "sorry, i cannot",
        "i don't have access", "i don't have information",
        "as an ai", "as a language model"
    ]

    # If response starts with a refusal, it's likely not helpful (unless it provides alternatives)
    if any(response_lower.startswith(phrase) for phrase in refusal_phrases):
        # Check if it at least provides alternatives or helpful guidance
        if len(response) < 100 and not any(word in response_lower for word in ["however", "alternatively", "instead", "but"]):
            return "No"

    # Very short responses (< 20 chars) are likely not helpful unless prompt was very simple
    if len(response) < 20:
        # Could be helpful for yes/no questions or very simple queries
        if "?" in prompt and len(prompt.split()) < 10:
            pass  # Could be legitimate short answer
        else:
            return "No"

    # Generic unhelpful responses
    unhelpful_patterns = [
        "i don't understand",
        "please clarify",
        "could you provide more",
        "i need more information",
        "what do you mean"
    ]

    # If response is just asking for clarification without attempting to help
    if len(response) < 150 and any(pattern in response_lower for pattern in unhelpful_patterns):
        return "No"

    # Check for complete non-sequiturs (response doesn't relate to prompt at all)
    # This is difficult to check programmatically, but we can use simple heuristics

    # Extract key content words from prompt (nouns, verbs, entities)
    # Simple heuristic: if prompt mentions specific topics/entities, response should too
    # For example, if prompt asks about "Python", response should mention "Python"

    # For now, assume most responses that pass the above filters are attempting to be helpful
    # The actual judgment will be done by reading each case

    # This function will be used as a starting template, but we need to actually judge each case
    # Let's return a placeholder for now
    return "NEEDS_MANUAL_JUDGMENT"


def main():
    # Load input JSON
    with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_120.json', 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # We'll need to actually judge these - let me print first few for inspection
    print("\n=== First 5 documents for inspection ===")
    for i, doc in enumerate(data[:5]):
        print(f"\n--- Document {i+1} ---")
        print(f"ID: {doc.get('id')}")
        print(f"Language: {doc.get('language')}")
        print(f"Prompt: {doc.get('prompt', '')[:200]}")
        print(f"Response: {doc.get('response', '')[:200]}")

    print(f"\n\nTotal documents to judge: {len(data)}")


if __name__ == "__main__":
    main()
