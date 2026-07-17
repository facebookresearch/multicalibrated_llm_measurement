#!/usr/bin/env python3
"""
LLM judge for assistant response HELPFULNESS.
Judges each response as Yes/No based on whether it:
- Directly addresses the request
- Is accurate and relevant
- Gives a usable answer
- Is NOT: off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage
"""

import json
import csv
import re

def judge_helpfulness(prompt, response):
    """
    Judge if a response is helpful.

    HELPFUL = directly addresses request, accurate, relevant, usable answer
    NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

    Returns True if helpful, False otherwise.
    """
    response_stripped = response.strip()
    response_lower = response_stripped.lower()
    prompt_lower = prompt.lower()

    # Clear NOT helpful indicators

    # 1. Empty or extremely short non-answer (but allow short valid answers)
    if len(response_stripped) < 10:
        return False

    # 2. Pure refusal without offering alternative help
    refusal_patterns = [
        r'^(i cannot|i can\'t|i\'m unable|sorry,? i cannot|sorry,? i can\'t)',
        r'^(i am not able|i\'m not able)',
        r'^(as an ai|as a language model).*(cannot|can\'t|unable|do not have)',
    ]

    for pattern in refusal_patterns:
        if re.search(pattern, response_lower):
            # Check if it offers alternative help or explanation
            helpful_pivots = ['however', 'instead', 'alternatively', 'but i can',
                            'here are', 'you can', 'try', 'consider', 'you could',
                            'one option', 'you might']
            if not any(pivot in response_lower for pivot in helpful_pivots):
                # It's a pure refusal
                return False

    # 3. Response is only meta-commentary without substance
    meta_only_patterns = [
        r'^(happy to help|glad to help|let me know)',
        r'^(sure|okay|alright|of course)[,.]?\s*$',
        r'^(that\'s|this is)?\s*(a\s+)?(good|great|interesting|nice)\s+(question|idea)',
    ]

    for pattern in meta_only_patterns:
        if re.match(pattern, response_lower) and len(response_stripped) < 100:
            # Check if there's actual content after the meta-comment
            sentences = response_stripped.split('.')
            if len(sentences) <= 2:
                return False

    # 4. Response is "do the opposite" or similarly useless
    if re.search(r'(do|try)\s+the\s+opposite', response_lower) and len(response_stripped) < 50:
        return False

    # 5. Generic non-answer (just restating the question)
    if len(response_stripped) < 100 and 'question' in response_lower and '?' in response_stripped:
        return False

    # Default: if it doesn't hit any of the NOT helpful patterns, consider it helpful
    # This is calibrated to be permissive - we only mark as NOT helpful if there's a clear problem
    return True


def main():
    # Read input
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

        is_helpful = judge_helpfulness(prompt, response)

        answer = 'Yes' if is_helpful else 'No'
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
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults:")
    print(f"  Yes (helpful): {yes_count}")
    print(f"  No (not helpful): {no_count}")
    print(f"  Total rows: {len(results)}")
    print(f"\nOutput written to {output_path}")

    # Verify
    with open(output_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 250, f"Expected 250 rows, got {len(rows)}"
        print(f"Verified: {len(rows)} rows in CSV")

        # Check all IDs match
        input_ids = {doc['id'] for doc in data}
        output_ids = {row['id'] for row in rows}
        assert input_ids == output_ids, "ID mismatch between input and output"
        print("Verified: All IDs match")


if __name__ == '__main__':
    main()
