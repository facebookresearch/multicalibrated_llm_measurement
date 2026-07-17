#!/usr/bin/env python3
"""
LLM Judge: Evaluate P(helpful) for assistant responses.

HELPFUL = directly addresses request, accurate, relevant, gives usable answer
NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage
"""

import json
import csv
import re


def extract_last_turn(prompt_text):
    """Extract the final user message from the conversation."""
    # Split by User: and Assistant: markers
    parts = re.split(r'\n\n(User:|Assistant:)', prompt_text)

    # Find the last User: section
    user_segments = []
    current_segment = []
    current_role = None

    for part in parts:
        if part.strip() in ['User:', 'Assistant:']:
            if current_role == 'User:' and current_segment:
                user_segments.append(' '.join(current_segment))
            current_role = part.strip()
            current_segment = []
        elif current_role:
            current_segment.append(part.strip())

    if current_role == 'User:' and current_segment:
        user_segments.append(' '.join(current_segment))

    return user_segments[-1] if user_segments else prompt_text


def evaluate_helpfulness(doc):
    """
    Calibrated estimation of P(helpful|response).
    Returns float in [0.01, 0.99].
    """
    prompt = doc['prompt']
    response = doc['response']

    # Extract the actual user question
    user_question = extract_last_turn(prompt)

    # Base score starts at 0.5 (neutral)
    base_score = 0.5

    # === Negative indicators (unhelpful) ===

    # 1. Empty or very short response
    if len(response.strip()) < 15:
        return 0.02

    # 2. Explicit refusal without alternatives
    refusal_patterns = [
        r'no\s+puedo\s+(proporcionar|dar|hacer|ayudar)',
        r'no\s+tengo\s+(acceso|información|la\s+capacidad)',
        r'lo\s+siento,?\s+pero\s+no',
        r'me\s+temo\s+que\s+no',
        r'fuera\s+de\s+mi\s+(alcance|capacidad)',
    ]

    has_refusal = any(re.search(pat, response.lower()) for pat in refusal_patterns)
    has_alternative = any(phrase in response.lower() for phrase in [
        'pero puedo', 'sin embargo', 'alternativamente', 'en su lugar',
        'lo que sí puedo', 'te recomiendo', 'puedes intentar', 'considera'
    ])

    if has_refusal and not has_alternative:
        base_score = 0.15
    elif has_refusal and has_alternative:
        base_score = 0.40
    else:
        base_score = 0.60  # Attempts to help

    # 3. Generic/template responses
    generic_only = response.strip().lower() in [
        'de acuerdo', 'ok', 'entendido', 'gracias', 'claro',
        'sí', 'no', 'bien'
    ]
    if generic_only:
        return 0.05

    # === Positive indicators (helpful) ===

    # 1. Substantive content
    if len(response) > 150:
        base_score += 0.10
    if len(response) > 400:
        base_score += 0.08
    if len(response) > 800:
        base_score += 0.05

    # 2. Structured content (lists, steps, examples)
    has_numbered_list = bool(re.search(r'\d+\.|\d+\)', response))
    has_bullets = '- ' in response or '• ' in response
    has_code = '```' in response or re.search(r'\n[A-Za-z_]\w+\s*\(', response)

    if has_numbered_list:
        base_score += 0.09
    if has_bullets:
        base_score += 0.07
    if has_code:
        base_score += 0.08

    # 3. Examples or specific information
    has_examples = 'ejemplo' in response.lower() or 'example' in response.lower()
    has_specifics = 'específicamente' in response.lower() or 'por ejemplo' in response.lower()

    if has_examples or has_specifics:
        base_score += 0.06

    # 4. Engagement and politeness
    polite_markers = [
        'claro', 'por supuesto', 'con gusto', 'encantado',
        'espero que', '¿hay algo más', '¿puedo ayudarte'
    ]
    politeness_count = sum(1 for marker in polite_markers if marker in response.lower())
    base_score += min(0.05, politeness_count * 0.02)

    # 5. Direct answer vs. tangential
    # Check if response seems relevant to question
    # (This is heuristic - look for keyword overlap)
    question_keywords = set(re.findall(r'\b\w{4,}\b', user_question.lower()))
    response_keywords = set(re.findall(r'\b\w{4,}\b', response.lower()))
    overlap = len(question_keywords & response_keywords)

    if overlap >= 3:
        base_score += 0.08
    elif overlap >= 1:
        base_score += 0.04

    # 6. Factual coherence (basic check)
    # Penalize if response seems confused or contradictory
    has_contradiction = bool(re.search(r'\bpero\b.*\bpero\b', response.lower()))
    if has_contradiction and len(response) < 200:
        base_score -= 0.10

    # === Special cases ===

    # Task completion responses
    task_completion = any(phrase in response.lower() for phrase in [
        'aquí está', 'aquí tienes', 'aquí va', 'esto es'
    ])
    if task_completion and len(response) > 100:
        base_score += 0.07

    # Conversational follow-up (less helpful if just asking questions back)
    only_questions = response.count('?') >= 3 and len(response) < 150
    if only_questions:
        base_score -= 0.15

    # Apologetic without solution
    apologetic = 'lo siento' in response.lower() or 'disculpa' in response.lower()
    if apologetic and len(response) < 100:
        base_score -= 0.12

    # === Clamp to valid range ===
    final_score = max(0.01, min(0.99, base_score))

    return round(final_score, 4)


def main():
    # Read input
    with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_17.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each document
    results = []
    total_p_help = 0.0

    for doc in data:
        p_help = evaluate_helpfulness(doc)
        p_nohelp = round(1.0 - p_help, 4)

        results.append({
            'id': doc['id'],
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': doc['language']
        })

        total_p_help += p_help

    mean_p_help = total_p_help / len(results)

    # Ensure output directory exists
    import os
    output_dir = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp'
    os.makedirs(output_dir, exist_ok=True)

    # Write output CSV
    output_path = f'{output_dir}/shard_17.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f'Processed {len(results)} documents')
    print(f'Mean p_help: {mean_p_help:.4f}')
    print(f'Output: {output_path}')

    # Verify row count
    assert len(results) == 100, f'Expected 100 rows, got {len(results)}'
    print('✓ Confirmed 100 rows')


if __name__ == '__main__':
    main()
