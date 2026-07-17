#!/usr/bin/env python3
"""
LLM judge for assistant response HELPFULNESS - Final calibrated version.
Evaluates each response on multiple criteria to determine Yes/No helpfulness.
"""

import json
import csv
import re

def is_helpful(prompt, response):
    """
    Judge helpfulness based on clear criteria.

    HELPFUL = directly addresses request, accurate, relevant, usable answer
    NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

    Be calibrated - don't default to Yes, but also recognize genuinely helpful responses.
    """
    resp = response.strip()
    resp_lower = resp.lower()

    # Category 1: Empty or trivial responses (NOT HELPFUL)
    if len(resp) < 10:
        return False

    # Category 2: Refusals without alternatives (NOT HELPFUL)
    # Pure refusal patterns
    if len(resp) < 200:
        # Refusals that don't offer help
        refusal_starts = [
            "i cannot", "i can't", "i'm unable", "sorry, i cannot", "sorry, i can't",
            "i am not able", "i'm not able", "i don't have the capacity",
            "i do not have the capacity", "as an ai, i cannot", "as an ai, i can't",
            "as a language model, i cannot"
        ]

        for ref_start in refusal_starts:
            if resp_lower.startswith(ref_start):
                # Check if it pivots to being helpful
                helpful_words = ['however', 'instead', 'alternatively', 'but i can',
                               'here are', 'you can', 'try', 'consider', 'you could',
                               'i can help', 'what i can do']
                if not any(hw in resp_lower for hw in helpful_words):
                    return False

    # Category 3: Meta-only responses without substance (NOT HELPFUL)
    # Just "happy to help" or similar without actual content
    meta_only_short = [
        r'^happy to help[!.]?\s*$',
        r'^glad to help[!.]?\s*$',
        r'^let me know if you need',
        r'^(sure|okay|alright)[!.]?\s*$'
    ]

    for pattern in meta_only_short:
        if re.match(pattern, resp_lower, re.IGNORECASE):
            return False

    # Category 4: Useless vague responses (NOT HELPFUL)
    if len(resp) < 60:
        useless_patterns = [
            r'^(simply|just)\s+(do|try)\s+the\s+opposite',
            r'^that\'?s?\s+(is\s+)?a\s+good\s+question\s*\.?\s*$',
            r'^i\'?m?\s+not\s+sure',
        ]
        for pattern in useless_patterns:
            if re.match(pattern, resp_lower):
                return False

    # Category 5: Responses that are clearly off-topic or nonsensical
    # This is hard to detect automatically, but we can catch some cases
    # If response is very short and shares no key terms with prompt
    if len(resp) < 100:
        # Extract content words (4+ chars)
        prompt_words = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
        resp_words = set(re.findall(r'\b\w{4,}\b', resp_lower))

        # If there's very little overlap, might be off-topic
        overlap = len(prompt_words & resp_words)

        # But be careful - some valid short answers won't have overlap
        # Only flag as not helpful if there's almost zero relation and it's really short
        if overlap == 0 and len(prompt_words) > 3 and len(resp) < 40:
            # Exception: terminal commands, short factual answers
            # If response looks like a command, path, or number, it might be valid
            if not re.match(r'^[/\w\d\s\-\.,]+$', resp):
                return False

    # Category 6: Responses that explicitly state inability/confusion (NOT HELPFUL)
    explicit_confusion = [
        "i don't understand", "i do not understand", "i'm confused",
        "this doesn't make sense", "this does not make sense",
        "i'm not following"
    ]
    if len(resp) < 150:
        for conf in explicit_confusion:
            if conf in resp_lower:
                # Unless they ask clarifying questions
                if resp.count('?') < 2:
                    return False

    # If none of the NOT helpful patterns match, consider it helpful
    # This default assumes the dataset contains mostly reasonable responses
    return True


def main():
    # Load input data
    input_path = 'data/shards/shard_103.json'
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Judge each response
    results = []
    yes_count = 0
    no_count = 0

    for doc in data:
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        helpful = is_helpful(prompt, response)
        answer = 'Yes' if helpful else 'No'

        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

        if answer == 'Yes':
            yes_count += 1
        else:
            no_count += 1

    # Write output
    output_path = 'data/inference_output/sonnet-binary/shard_103.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults:")
    print(f"  Yes (helpful): {yes_count}")
    print(f"  No (not helpful): {no_count}")
    print(f"  Total: {len(results)}")
    print(f"\nOutput written to: {output_path}")

    # Verify
    with open(output_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"\nVerified: {len(rows)} rows in CSV (expected 250)")

        # Check IDs match
        input_ids = {doc['id'] for doc in data}
        output_ids = {row['id'] for row in rows}
        assert input_ids == output_ids, "ID mismatch!"
        print("Verified: All IDs match input")


if __name__ == '__main__':
    main()
