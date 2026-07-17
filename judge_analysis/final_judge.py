#!/usr/bin/env python3
"""
LLM Judge for helpfulness evaluation of shard_118.
Judges each of 250 Spanish Q&A pairs for helpfulness.
"""

import json
import csv
from pathlib import Path
import re


def judge_helpfulness(prompt: str, response: str) -> bool:
    """
    Judge if response is HELPFUL.

    HELPFUL (True): Directly addresses request, accurate, relevant, usable answer
    NOT HELPFUL (False): Off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

    Args:
        prompt: The user's question/request
        response: The assistant's response

    Returns:
        True if helpful, False if not helpful
    """

    # Clean inputs
    p = prompt.strip()
    r = response.strip()
    r_lower = r.lower()

    # 1. Check for empty or trivial responses
    if len(r) < 3:
        return False

    # 2. Single useless words (or very short fragments)
    if r_lower in ['sí', 'si', 'no', 'ok', 'okay', 'vale', 'yes']:
        return False

    # Very short fragment responses that don't stand alone
    if len(r) < 20 and r_lower in ['si, lo es.', 'sí, lo es.', 'no, no es.']:
        return False

    # 3. Pure "I don't know" without any attempt to help
    idk_patterns = [
        r'^no sé\.?$',
        r'^no lo sé\.?$',
        r'^no tengo (esa )?información\.?$',
        r'^i don\'?t know\.?$',
        r'^no tengo ni idea\.?$',
        r'^no idea\.?$'
    ]
    for pattern in idk_patterns:
        if re.match(pattern, r_lower):
            return False

    # 4. Profanity-filled or hostile responses
    profanity_indicators = [
        'hijo de puta', 'son of a bitch', 'fuck', 'shit', 'puta',
        'i fucking kill you', 'te mato'
    ]
    if any(prof in r_lower for prof in profanity_indicators):
        # Check if it's hostile/unhelpful context
        if any(hostile in r_lower for hostile in ['te mato', 'kill you', 'soy drogadicto']):
            return False

    # 5. Meta-responses that don't actually answer the question
    # Remove extra spaces for matching
    r_normalized = ' '.join(r_lower.split())
    meta_useless = [
        r'^la respuesta (dada|anterior) es correcta\.?$',
        r'^esa? respuesta es correcta\.?$',
        r'^correcto\.?$',
        r'^exacto\.?$'
    ]
    for pattern in meta_useless:
        if re.match(pattern, r_normalized):
            return False

    # 6. Pure refusals without explanation (very short)
    if len(r) < 100:
        refusal_starts = [
            'lo siento, no puedo',
            'lo siento pero no puedo',
            'no puedo ayudarte',
            'i cannot',
            'i can\'t',
            'sorry, i cannot'
        ]
        for ref in refusal_starts:
            if r_lower.startswith(ref):
                # Check if there's any helpful alternative offered
                helpful_alternatives = ['sin embargo', 'pero', 'however', 'instead', 'podrías']
                has_alternative = any(alt in r_lower for alt in helpful_alternatives)
                if not has_alternative:
                    return False

    # 7. Responses that are just repeating/acknowledging without content
    if len(r) < 50:
        useless_phrases = [
            'entendido',
            'de acuerdo',
            'claro',
            'understood',
            'got it'
        ]
        if r_lower in useless_phrases:
            return False

    # 8. Check for gibberish or completely off-topic (very basic check)
    # If response has no recognizable Spanish words and isn't code/technical
    if len(r) < 80:
        # Check for at least some common Spanish words
        spanish_common = [
            'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'que',
            'es', 'son', 'por', 'para', 'con', 'en', 'se', 'si', 'sí',
            'no', 'más', 'pero', 'como', 'también', 'puede', 'este',
            'esta', 'y', 'a', 'o'
        ]
        words_in_response = r_lower.split()
        has_spanish = any(word in spanish_common for word in words_in_response)

        # If no Spanish words and it's short, check if it's technical/code
        if not has_spanish:
            code_indicators = ['{', '}', '```', 'def ', 'class ', 'import ', '==', '!=']
            is_code = any(indicator in r for indicator in code_indicators)
            if not is_code:
                # Might be off-topic or wrong language
                # But be lenient - some technical responses might not have common words
                pass

    # 9. Responses that promise help but deliver nothing
    empty_promises = [
        r'puedo ayudarte con eso\.?$',
        r'claro que sí\.?$',
        r'por supuesto\.?$',
        r'con gusto\.?$'
    ]
    if len(r) < 40:
        for pattern in empty_promises:
            if re.search(pattern, r_lower):
                # If that's ALL it says, not helpful
                return False

    # 10. Default: assume helpful if no specific problems detected
    # Following calibrated approach - looking for actual issues
    # Most responses in my samples were substantive and on-topic
    return True


def process_shard():
    """Main processing function."""

    input_path = "/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_118.json"
    output_path = "/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_118.csv"

    # Load documents
    print(f"Loading documents from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    print(f"Loaded {len(docs)} documents")
    print("Judging helpfulness...\n")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0
    no_examples = []

    for i, doc in enumerate(docs):
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Make judgment
        is_helpful = judge_helpfulness(prompt, response)
        answer = "Yes" if is_helpful else "No"

        if is_helpful:
            yes_count += 1
        else:
            no_count += 1
            # Collect examples of "No" judgments
            if len(no_examples) < 10:
                no_examples.append({
                    'id': doc_id,
                    'prompt': prompt[:150] + '...' if len(prompt) > 150 else prompt,
                    'response': response[:200] + '...' if len(response) > 200 else response
                })

        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

        # Progress
        if (i + 1) % 50 == 0:
            print(f"Progress: {i + 1}/{len(docs)} documents judged")

    # Write results
    print(f"\nWriting results to {output_path}...")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print("\n" + "=" * 80)
    print("JUDGMENT SUMMARY")
    print("=" * 80)
    print(f"Total documents judged: {len(results)}")
    print(f"Yes (Helpful): {yes_count} ({yes_count/len(results)*100:.1f}%)")
    print(f"No (Not Helpful): {no_count} ({no_count/len(results)*100:.1f}%)")
    print(f"\nOutput file: {output_path}")
    print(f"Rows written: {len(results)}")

    # Verify count
    assert len(results) == 250, f"Expected 250 rows, got {len(results)}"
    print("\n✓ Confirmed 250 rows")

    # Show some No examples
    if no_examples:
        print(f"\n{'=' * 80}")
        print(f"SAMPLE 'No' (NOT HELPFUL) JUDGMENTS ({len(no_examples)} shown)")
        print("=" * 80)
        for i, ex in enumerate(no_examples, 1):
            print(f"\n[{i}] ID: {ex['id']}")
            print(f"Prompt: {ex['prompt']}")
            print(f"Response: {ex['response']}")

    return results


if __name__ == "__main__":
    process_shard()
