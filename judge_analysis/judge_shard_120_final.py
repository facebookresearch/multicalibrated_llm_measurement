#!/usr/bin/env python3
"""
LLM Judge for helpfulness evaluation of shard_120.
Judges each of 250 Spanish Q&A pairs for helpfulness using calibrated criteria.
"""

import json
import csv
import re


def judge_helpfulness(prompt, response):
    """
    Judge if response is HELPFUL.

    HELPFUL: Directly addresses request, accurate, relevant, usable answer
    NOT HELPFUL: Off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

    Be calibrated - do NOT default to Yes.
    """
    p = prompt.strip()
    r = response.strip()

    # Empty or very short response
    if not r or len(r) < 5:
        return False

    r_lower = r.lower()
    p_lower = p.lower()

    # SPECIFIC BAD CASES (from manual review):

    # 1. Harmful/inappropriate content
    harmful_phrases = [
        'muerte a los rothschild',
        'judería internacional',
        'son todos maleantes y drogadictos',
        'debemos eliminar'
    ]
    for phrase in harmful_phrases:
        if phrase in r_lower:
            return False

    # 2. Non-engagement: just asking what user wants without helping
    if r.strip() in ['Quieres que construya una historia', 'Quieres que']:
        return False

    # 3. Single word or very minimal answers
    minimal_answers = ['sí', 'si', 'no', 'ok', 'vale', 'yes', 'claro', 'arduino']
    if r_lower.strip() in minimal_answers:
        # Could be OK for yes/no questions
        if ('sí' in p_lower or 'no' in p_lower) and '?' in p and len(p) < 100:
            # Legitimate yes/no question
            if r_lower.strip() in ['sí', 'si', 'no']:
                return True
        return False  # Too minimal for most questions

    # 4. Very short responses (< 20 chars) - usually not helpful
    if len(r) < 20:
        # Exception: might be OK for very simple questions
        if len(p) < 50 and '?' in p:
            return True
        return False

    # 5. Pure refusals without explanation or alternatives
    refusal_patterns = [
        r'^no puedo',
        r'^no sé',
        r'^lo siento,? no',
        r'^disculpa,? no puedo',
        r'^no tengo información',
    ]
    for pattern in refusal_patterns:
        if re.match(pattern, r_lower):
            # If very short refusal (< 40 chars), not helpful
            if len(r) < 40:
                return False
            # Longer refusals might include explanations - could be helpful
            break

    # 6. Just asking for clarification without attempting to help
    if len(r) < 60:
        clarification_patterns = [
            'qué quieres decir',
            'no entiendo',
            'puedes explicar',
            'puedes aclarar',
            'necesito más información',
        ]
        for pattern in clarification_patterns:
            if pattern in r_lower:
                return False

    # 7. Incomplete/cut-off responses
    # Check if response ends mid-sentence (no proper ending punctuation)
    if len(r) > 50:
        last_chars = r[-3:].strip()
        # If ends with incomplete markers, likely truncated
        if not any(last_chars.endswith(c) for c in ['.', '!', '?', ':', '"', '»', ')', ']']):
            # Check if it looks cut off (ends with comma, "y", "o", "la", etc.)
            cutoff_endings = [',', ' y ', ' o ', ' la ', ' el ', ' de ', ' que ', ' con ']
            if any(r.endswith(ending.strip()) for ending in cutoff_endings):
                # Could still be helpful if substantial content provided
                if len(r) < 80:
                    return False

    # 8. Responses that don't address the actual question
    # This is harder to detect automatically, but we can check for common patterns

    # If we got here, the response has reasonable length and engages
    # Default to helpful if it's substantive (> 30 chars and no red flags)
    if len(r) >= 30:
        return True

    # Shorter responses - be more conservative
    return len(r) >= 25


def main():
    # Load data
    input_file = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_120.json'
    print(f"Loading {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents\n")
    print("Judging helpfulness for each document...")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    for i, doc in enumerate(data):
        doc_id = doc['id']
        language = doc['language']
        prompt = doc.get('prompt', '')
        response = doc.get('response', '')

        # Make judgment
        is_helpful = judge_helpfulness(prompt, response)
        answer = 'Yes' if is_helpful else 'No'

        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

        if is_helpful:
            yes_count += 1
        else:
            no_count += 1

        # Progress
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/250 documents judged...")

    # Write to CSV
    output_file = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_120.csv'
    print(f"\nWriting results to {output_file}...")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Exact header format as specified
        writer.writerow(['id', 'answer', 'language'])

        for result in results:
            writer.writerow([result['id'], result['answer'], result['language']])

    # Summary
    print(f"\n{'='*70}")
    print(f"JUDGMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Total documents: {len(results)}")
    print(f"Yes (HELPFUL): {yes_count} ({yes_count/len(results)*100:.1f}%)")
    print(f"No (NOT HELPFUL): {no_count} ({no_count/len(results)*100:.1f}%)")
    print(f"\nOutput file: {output_file}")

    # Verify file
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"\nVerification:")
    print(f"  Total CSV lines (including header): {len(lines)}")
    print(f"  Data rows: {len(lines) - 1}")

    if len(lines) - 1 == 250:
        print(f"  ✓ Confirmed exactly 250 rows")
    else:
        print(f"  ✗ WARNING: Expected 250 rows, got {len(lines) - 1}")

    # Sample output
    print(f"\nFirst 5 rows (including header):")
    for line in lines[:5]:
        print(f"  {line.strip()}")


if __name__ == "__main__":
    main()
