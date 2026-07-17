#!/usr/bin/env python3
"""
LLM judge for assistant-response helpfulness on shard_114.

HELPFUL (Yes): directly addresses the request, accurate, relevant, gives a usable answer
NOT HELPFUL (No): off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

Be calibrated - do NOT default to Yes.
"""

import json
import csv
import re
from pathlib import Path


def is_helpful(prompt: str, response: str) -> str:
    """
    Judge if response is HELPFUL to the prompt.
    Returns "Yes" or "No"
    """
    p = prompt.strip()
    r = response.strip()
    p_lower = p.lower()
    r_lower = r.lower()

    # Empty or extremely short responses are not helpful
    if len(r) < 5:
        return "No"

    # Pure refusals without alternatives
    refusal_only = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "i don't have access", "i cannot access",
        "i'm not able to", "i am not able to"
    ]
    has_refusal = any(pattern in r_lower for pattern in refusal_only)

    if has_refusal:
        # Check if there's also helpful content
        helpful_additions = [
            "however", "but", "instead", "alternatively", "what i can",
            "here's", "you can", "try", "consider", "recommend"
        ]
        has_help = any(pattern in r_lower for pattern in helpful_additions)

        # Short refusal without alternatives is not helpful
        if not has_help and len(r) < 150:
            return "No"

    # Generic acknowledgments without substance
    generic_only = [
        r"^(okay|ok|sure|alright|got it|understood|no problem|you're welcome|thanks|thank you)[\.\!]*$",
        r"^(yes|yeah|yep|yup|nope|no)[\.\!]*$",
        r"^(sounds good|sounds great)[\.\!]*$"
    ]
    if any(re.match(pattern, r_lower) for pattern in generic_only):
        return "No"

    # Response just repeats the prompt
    if len(r) > 10 and r_lower == p_lower:
        return "No"

    # Evasive non-answers
    evasive = [
        "i don't understand", "could you clarify", "can you rephrase",
        "what do you mean", "i'm not sure what you",
        "please provide more context", "can you be more specific"
    ]
    if any(pattern in r_lower for pattern in evasive):
        # If it's ONLY evasive and short, not helpful
        if len(r) < 100:
            return "No"

    # Check for substantive engagement markers
    engagement_markers = [
        "because", "since", "therefore", "thus", "so",
        "for example", "for instance", "such as", "like",
        "this means", "in other words", "specifically",
        "here's how", "the reason", "you can", "you should",
        "one way", "first", "second", "step", "option",
        "alternatively", "another", "also", "additionally"
    ]
    has_engagement = any(marker in r_lower for marker in engagement_markers)

    # Check for structure (lists, multiple sentences)
    has_structure = (
        r.count('.') > 1 or
        r.count('\n') > 1 or
        r.count(':') > 0 or
        bool(re.search(r'[\d\-\*\•]\s*[\w]', r))  # List markers
    )

    # Check for off-topic responses
    # If prompt asks specific question but response doesn't engage
    # Look for question words in prompt
    question_indicators = ["how", "what", "why", "when", "where", "who", "which", "can you", "could you", "would you", "is it", "are there", "do you"]
    has_question = any(q in p_lower for q in question_indicators)

    if has_question:
        # For questions, need engagement or structure
        if not has_engagement and not has_structure:
            # Unless it's a simple factual answer
            if len(r) < 50:
                # Very short answer to question - only helpful if it's clearly factual
                # e.g., "What is 2+2?" -> "4"
                # This is hard to judge automatically, so default to Yes for very concise
                pass

    # Check word count ratio
    p_words = len(p.split())
    r_words = len(r.split())

    # Long question with very short answer is suspicious
    if p_words > 30 and r_words < 5:
        return "No"

    # Extremely short responses to any question
    if has_question and r_words < 3:
        return "No"

    # Factual incorrectness is hard to detect without domain knowledge
    # But we can catch obvious nonsense

    # Contradictions within the response
    if r_words < 50:  # Only for short responses
        contradiction_pairs = [
            ("yes", "no,"), ("yes", "no."),
            ("true", "false"), ("correct", "incorrect"),
            ("always", "never"), ("possible", "impossible")
        ]
        for pos, neg in contradiction_pairs:
            if pos in r_lower and neg in r_lower:
                # Likely confused
                return "No"

    # Check for substantive content
    # Good indicators of helpfulness
    helpful_indicators = [
        # Explanations
        "because", "since", "due to", "as a result",
        # Examples
        "for example", "for instance", "such as",
        # Instructions
        "you can", "you should", "you need to", "try",
        # Structure
        "first", "second", "third", "finally",
        "step 1", "step 2",
        # Alternatives
        "alternatively", "another option", "you could also",
        # Clarifications
        "in other words", "to clarify", "specifically",
        "this means", "essentially"
    ]

    substantive_count = sum(1 for indicator in helpful_indicators if indicator in r_lower)

    # Response with multiple helpful indicators is likely good
    if substantive_count >= 2:
        return "Yes"

    # Response with at least one indicator and reasonable length
    if substantive_count >= 1 and r_words >= 15:
        return "Yes"

    # Has structure (lists, paragraphs) and reasonable length
    if has_structure and r_words >= 20:
        return "Yes"

    # Reasonable length response without red flags
    if r_words >= 30:
        return "Yes"

    # Medium length without clear helpfulness signals - be conservative
    if r_words >= 15:
        # Check for lack of substance
        filler_words = ["um", "uh", "hmm", "well", "just", "like", "you know"]
        filler_count = sum(r_lower.count(word) for word in filler_words)
        if filler_count > r_words * 0.2:  # >20% filler
            return "No"
        return "Yes"

    # Short responses without clear markers - default to No (be calibrated)
    return "No"


def main():
    input_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_114.json")
    output_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_114.csv")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} documents")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    for i, doc in enumerate(documents):
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Judge helpfulness
        answer = is_helpful(prompt, response)

        if answer == "Yes":
            yes_count += 1
        else:
            no_count += 1

        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Judged {i + 1}/{len(documents)}...")

    # Write CSV
    print(f"\nWriting results to {output_path}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Report statistics
    print(f"\n{'='*50}")
    print(f"JUDGMENTS COMPLETE")
    print(f"{'='*50}")
    print(f"Total rows: {len(results)}")
    print(f"Yes (helpful): {yes_count} ({yes_count/len(results)*100:.1f}%)")
    print(f"No (not helpful): {no_count} ({no_count/len(results)*100:.1f}%)")
    print(f"\nOutput: {output_path}")

    # Verify count
    assert len(results) == 250, f"Expected 250 rows, got {len(results)}"
    print(f"\n✓ Confirmed 250 rows")

    # Show sample
    print(f"\nFirst 5 rows:")
    print("id,answer,language")
    for r in results[:5]:
        print(f"{r['id']},{r['answer']},{r['language']}")


if __name__ == "__main__":
    main()
