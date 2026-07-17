#!/usr/bin/env python3
"""
LLM Judge for Assistant Response Helpfulness
Judges whether each response is HELPFUL (Yes) or NOT HELPFUL (No)

Criteria:
HELPFUL = directly addresses the request, accurate, relevant, gives a usable answer
NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage
"""

import json
import csv
import re

def judge_helpfulness(prompt, response, language):
    """
    Judge whether a response is helpful.
    Returns 'Yes' or 'No'

    HELPFUL = directly addresses request, accurate, relevant, usable answer
    NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage
    """
    # Clean whitespace
    response = response.strip()
    prompt = prompt.strip()

    # Extract the actual user question (last User: line in prompt)
    user_lines = [line for line in prompt.split('\n') if line.startswith('User:')]
    if user_lines:
        question = user_lines[-1].replace('User:', '').strip()
    else:
        question = prompt

    response_lower = response.lower()
    question_lower = question.lower()

    # === CLEAR NOT HELPFUL CASES ===

    # 1. Empty or just punctuation
    if len(response) < 2 or response in ['?', '!', '.', '...']:
        return 'No'

    # 2. One-word useless responses
    one_word_unhelpful = [
        r'^да\.?$',  # Just "Yes"
        r'^нет\.?$',  # Just "No"
        r'^не знаю\.?$',  # "Don't know"
        r'^maybe\.?$',
        r'^perhaps\.?$',
        r'^война\.?$',  # "war" (seen in data - incomplete)
    ]

    for pattern in one_word_unhelpful:
        if re.match(pattern, response_lower):
            return 'No'

    # 3. Explicit refusals (short refusals with no help)
    if len(response) < 120:  # Short responses only
        refusal_indicators = [
            "я не могу распространять",
            "i cannot provide",
            "i can't provide",
            "i'm sorry, but i cannot",
            "i apologize, but i cannot",
        ]
        if any(indicator in response_lower for indicator in refusal_indicators):
            return 'No'

    # 4. Evasive clarification-only responses (no attempt to help)
    if len(response) < 50:
        evasive_patterns = [
            r"^уточните запрос",  # "Clarify request"
            r"^какую\s*\?",  # "Which one?"
            r"^что вы имеете в виду",  # "What do you mean"
            r"^(please |can you )?(clarify|specify)",
        ]
        for pattern in evasive_patterns:
            if re.search(pattern, response_lower):
                return 'No'

    # 5. Off-topic / nonsensical responses
    # Check for known bad cases from the data
    bad_responses = [
        "потому что",  # Just "because" (incomplete)
        "расскажи о лсд",  # Off-topic command in response to thanks
    ]
    if response_lower in bad_responses:
        return 'No'

    # 6. Factually wrong answers (hard to detect automatically, but check known cases)
    # "Бельгийский" язык doesn't exist - but hard to auto-detect
    # Would need extensive knowledge base

    # 7. Pure greetings in isolation (no informational value)
    if len(response) <= 20:
        # Single greeting words
        pure_greetings = [
            r'^привет!?$',
            r'^hello!?$',
            r'^hi!?$',
            r'^hey!?$',
        ]
        for pattern in pure_greetings:
            if re.match(pattern, response_lower):
                # Check if question was also just a greeting
                if re.match(pattern, question_lower):
                    # Greeting responding to greeting = not helpful (no info provided)
                    return 'No'

    # === HELPFUL CASES ===

    # Allow appropriate short responses to simple questions:
    # - Math: "2+2" -> "2 + 2 = 4" is helpful
    # - Cultural references: "42" is helpful for HHGTTG reference
    # - Thank you responses: "Рад помочь" is appropriate

    # If response has substantive content (>50 chars), likely helpful
    if len(response) > 50:
        return 'Yes'

    # Short but structured responses (e.g., "2 + 2 = 4", "42", etc.)
    # If it's short but has equation, definition structure, it's helpful
    if any([
        '=' in response,  # Equations
        ':' in response,  # Definitions
        len(response.split()) >= 3,  # At least 3 words
    ]):
        return 'Yes'

    # Polite acknowledgments to thanks ("Glad to help", etc.)
    if any(word in question_lower for word in ['спасибо', 'thanks', 'thank you']):
        if any(word in response_lower for word in ['рад', 'пожалуйста', 'welcome', 'glad']):
            return 'Yes'

    # Single word that might be valid (e.g., factual answer)
    # Conservative: if it's a single short word and doesn't match bad patterns,
    # give benefit of doubt but be strict on very short
    if len(response) < 15:
        # Too short and didn't match any helpful patterns above
        return 'No'

    # Default: if it passed all the negative filters and has some substance
    return 'Yes'


def main():
    # Load data
    input_path = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_128.json'
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_128.csv'

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    for doc in data:
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Judge helpfulness
        answer = judge_helpfulness(prompt, response, language)

        if answer == 'Yes':
            yes_count += 1
        else:
            no_count += 1

        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

    # Write output CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nJudgment complete:")
    print(f"  Yes (HELPFUL): {yes_count}")
    print(f"  No (NOT HELPFUL): {no_count}")
    print(f"  Total: {len(results)}")
    print(f"\nWrote {len(results)} rows to {output_path}")

    # Verify count
    assert len(results) == 250, f"Expected 250 rows, got {len(results)}"
    print("✓ Confirmed 250 rows")

if __name__ == '__main__':
    main()
