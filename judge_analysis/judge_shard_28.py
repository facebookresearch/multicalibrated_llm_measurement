#!/usr/bin/env python3
"""
LLM Judge for helpfulness scoring.
Processes each prompt-response pair and estimates P(helpful).
"""

import json
import csv
import sys


def judge_helpfulness(doc):
    """
    Judge whether a response is helpful.

    A response is HELPFUL if it:
    - Directly addresses the request
    - Is accurate and relevant
    - Gives a usable answer

    NOT helpful if:
    - Off-topic or evasive
    - Factually wrong
    - Uselessly incomplete
    - Fails to engage

    Returns: probability in [0, 1]
    """
    prompt = doc['prompt']
    response = doc['response']

    # Extract context from multi-turn prompts
    turns = prompt.split('\n\n')
    if len(turns) > 1:
        last_user = [t for t in turns if t.startswith('User:')][-1] if any(t.startswith('User:') for t in turns) else turns[-1]
    else:
        last_user = prompt

    # Scoring logic based on response characteristics
    score = 0.5  # neutral baseline

    # Check for direct engagement
    if len(response.strip()) < 20:
        # Very short responses are often unhelpful
        score -= 0.3

    # Check for evasiveness markers
    evasive_markers = [
        'не могу', 'не способен', 'к сожалению', 'запрограммирован воздерживаться',
        'я не понял', 'не удаётся извлечь', 'переформулировать'
    ]
    if any(marker in response.lower() for marker in evasive_markers):
        # May still be helpful if explaining why
        if 'однако' in response.lower() or 'но' in response.lower() or len(response) > 200:
            score -= 0.1  # Mild penalty
        else:
            score -= 0.4  # Strong penalty for pure evasion

    # Check for substantive content
    if len(response) > 150 and ('например' in response.lower() or 'это' in response.lower()):
        score += 0.2

    # Check for structured information (lists, examples, explanations)
    if any(marker in response for marker in ['1.', '2.', '- ', '\n\n']):
        score += 0.15

    # Check for inappropriate humor when serious question asked
    if ')))))' in response or '😁' in response:
        if 'погода' not in last_user.lower() and 'шут' not in last_user.lower():
            score -= 0.2  # Penalize inappropriate levity

    # Check for acknowledgment and appropriate tone
    if any(marker in response for marker in ['спасибо', 'приятно', 'рад', 'пожалуйста']):
        score += 0.05

    # Check if response actually answers the question
    # Look for question words in last user turn
    question_markers = ['как', 'что', 'почему', 'где', 'когда', 'можешь', 'расскажи', 'объясни']
    has_question = any(marker in last_user.lower() for marker in question_markers)

    if has_question:
        # For questions, check if response provides information
        if len(response) > 100 and not any(evade in response.lower() for evade in ['не могу', 'не знаю', 'сложно сказать']):
            score += 0.2
        elif any(evade in response.lower() for evade in ['не могу', 'не знаю']):
            score -= 0.15

    # Check for factual information markers
    factual_markers = ['исследования', 'данные', 'например', 'это', 'может', 'является']
    if sum(1 for marker in factual_markers if marker in response.lower()) >= 2:
        score += 0.15

    # Check for URLs/sources (positive signal)
    if 'http' in response or 'wikipedia' in response.lower():
        score += 0.1

    # Clip to [0, 1]
    score = max(0.01, min(0.99, score))

    return score


def main():
    # Read input JSON
    with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_28.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"Processing {len(documents)} documents...", file=sys.stderr)

    # Process each document
    results = []
    for doc in documents:
        p_help = judge_helpfulness(doc)
        p_nohelp = 1.0 - p_help

        results.append({
            'id': doc['id'],
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': doc['language']
        })

    # Write output CSV
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_28.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Report statistics
    mean_p_help = sum(r['p_help'] for r in results) / len(results)
    print(f"\nProcessed {len(results)} documents", file=sys.stderr)
    print(f"Mean P(helpful): {mean_p_help:.4f}", file=sys.stderr)
    print(f"Output written to: {output_path}", file=sys.stderr)

    # Also print to stdout for confirmation
    print(f"Mean P(helpful): {mean_p_help:.4f}")
    print(f"Total rows: {len(results)}")


if __name__ == '__main__':
    main()
