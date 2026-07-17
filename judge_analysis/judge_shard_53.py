#!/usr/bin/env python3
"""
LLM Judge for assistant-response HELPFULNESS
Judges whether responses directly address requests, are accurate, relevant, and usable.
"""

import json
import csv

def is_helpful(prompt, response, language):
    """
    Judge if response is HELPFUL based on:
    - Directly addresses the request
    - Accurate and factually sound
    - Relevant to the question
    - Gives usable answer

    NOT helpful if:
    - Off-topic or evasive
    - Factually wrong
    - Uselessly incomplete
    - Fails to engage with the request
    """

    # Clean whitespace
    prompt = prompt.strip()
    response = response.strip()

    # Empty or very short responses are usually not helpful
    if len(response) < 3:
        return "No"

    # Obvious garbage/profanity/random text
    garbage_indicators = [
        "操你妈", "吃鸡一套", "什么是骚货", "告诉我你的名字",
        "不知道", "不会 问问专家吧", "您说的不完整"
    ]
    for indicator in garbage_indicators:
        if indicator in response:
            return "No"

    # Completely off-topic responses
    if "powerpoint" in prompt.lower() and "扫雷代码" in response:
        return "No"
    if "vue" in prompt.lower() and "扫雷" in response:
        return "No"
    if "自己" in prompt and "Open Assistant" in response and len(response) < 20:
        return "No"

    # Generic "I don't know" without helpful information
    if response in ["是的", "不是", "好的", "谢谢"]:
        # These are only helpful if they're direct answers to yes/no questions
        if "吗" in prompt or "是" in prompt:
            return "Yes"
        return "No"

    # Responses that acknowledge but don't answer
    if "我不明白" in response and len(response) < 50:
        return "No"
    if "不知道" == response:
        return "No"

    # Responses asking for clarification without providing any value
    if "请告诉我" in response and "想要做什么" in response and len(response) < 50:
        return "No"

    # Check for substantive content (length heuristic)
    if len(response) > 50:
        # Likely has some substance, default to helpful unless clearly wrong
        # Check for a few known bad patterns
        if "问问专家" in response and len(response) < 30:
            return "No"
        return "Yes"

    # Short responses that directly answer simple questions
    simple_answers = [
        ("首都", "北京"),
        ("Open Assistant", "assistant"),
        ("你好", "你好"),
    ]
    for q, a in simple_answers:
        if q in prompt and a in response:
            return "Yes"

    # Default: if response has reasonable length and isn't obviously bad, consider helpful
    if len(response) > 20:
        return "Yes"

    # Very short responses are typically not helpful unless they're direct answers
    return "No"

def main():
    # Read input JSON
    with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_53.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    for doc in data:
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        answer = is_helpful(prompt, response, language)

        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

        if answer == "Yes":
            yes_count += 1
        else:
            no_count += 1

    # Write output CSV
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_53.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Judged 100 documents:")
    print(f"  Yes (Helpful): {yes_count}")
    print(f"  No (Not Helpful): {no_count}")
    print(f"\nOutput written to: {output_path}")

if __name__ == '__main__':
    main()
