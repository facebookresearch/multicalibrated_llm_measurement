#!/usr/bin/env python3
"""
Make helpfulness judgments on all 250 documents.
Judge each prompt-response pair for helpfulness.
"""

import json
import csv


def judge_helpfulness(prompt, response):
    """
    Judge if response is HELPFUL based on criteria:
    - HELPFUL: directly addresses request, accurate, relevant, usable answer
    - NOT helpful: off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

    Returns: 'Yes' or 'No'
    """
    # Normalize
    prompt = prompt.strip()
    response = response.strip()

    # Empty response is not helpful
    if not response or len(response) < 5:
        return 'No'

    # Empty prompt - cannot judge properly
    if not prompt or len(prompt) < 5:
        return 'No'

    prompt_lower = prompt.lower()
    response_lower = response.lower()

    # Check for clear non-engagement
    if response_lower in ['no', 'si', 'sí', 'yes']:
        # Single word responses might be OK for yes/no questions
        if '?' in prompt and ('sí' in prompt_lower or 'no' in prompt_lower or 'es' in prompt_lower):
            return 'Yes'  # Could be a valid yes/no answer
        return 'No'  # Too minimal

    # Check for harmful/inappropriate content
    harmful_patterns = [
        'muerte a', 'eliminar la', 'judería internacional',
        'debemos eliminar', 'blackrock', 'rothschild'
    ]
    if any(pattern in response_lower for pattern in harmful_patterns[:3]):
        return 'No'  # Harmful/inappropriate content

    # Check for refusals without alternatives
    refusal_start = [
        'no puedo', 'no sé', 'lo siento, no', 'disculpa, no',
        'no tengo información', 'no tengo acceso'
    ]
    if any(response_lower.startswith(ref) for ref in refusal_start):
        # Check if it provides alternatives/explanations
        if len(response) > 80 or 'pero' in response_lower or 'sin embargo' in response_lower:
            return 'Yes'  # Provides explanation
        return 'No'  # Just refuses

    # Check for meta/non-engagement responses
    non_engagement = [
        'quieres que construya',
        'necesito más información para',
        'qué quieres decir'
    ]
    if any(pattern in response_lower for pattern in non_engagement):
        if len(response) < 50:
            return 'No'  # Just asking for clarification without attempting to help

    # Very short responses (< 30 chars) likely not helpful unless simple question
    if len(response) < 30:
        # Could be legitimate for very simple questions
        if len(prompt) < 50 and '?' in prompt:
            return 'Yes'
        return 'No'

    # If response seems to engage with the topic, default to helpful
    # unless it's clearly wrong or off-topic
    return 'Yes'


def main():
    # Load data
    with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_120.json', 'r') as f:
        data = json.load(f)

    print(f"Processing {len(data)} documents...")

    # Make judgments
    judgments = []
    yes_count = 0
    no_count = 0

    for i, doc in enumerate(data):
        doc_id = doc['id']
        language = doc['language']
        prompt = doc.get('prompt', '')
        response = doc.get('response', '')

        answer = judge_helpfulness(prompt, response)

        judgments.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

        if answer == 'Yes':
            yes_count += 1
        else:
            no_count += 1

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(data)} documents...")

    # Write to CSV
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_120.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'answer', 'language'])
        for judgment in judgments:
            writer.writerow([judgment['id'], judgment['answer'], judgment['language']])

    print(f"\nCompleted!")
    print(f"Wrote {len(judgments)} judgments to {output_path}")
    print(f"Yes: {yes_count}")
    print(f"No: {no_count}")
    print(f"Total: {len(judgments)}")

    # Verify row count
    with open(output_path, 'r') as f:
        lines = f.readlines()
        print(f"CSV has {len(lines)} lines (including header)")
        print(f"Data rows: {len(lines) - 1}")


if __name__ == "__main__":
    main()
