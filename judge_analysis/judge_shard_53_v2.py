#!/usr/bin/env python3
"""
Manual LLM judge for assistant-response HELPFULNESS.
Calibrated judgment of each prompt-response pair.
"""

import json
import csv

def judge_helpfulness(doc_id, prompt, response):
    """
    Judge if the response is HELPFUL:
    - Directly addresses the request
    - Accurate and relevant
    - Gives usable answer

    NOT helpful:
    - Off-topic or evasive
    - Factually wrong
    - Uselessly incomplete
    - Fails to engage
    """

    # Manual judgments based on careful review
    judgments = {
        "zh_5300": "Yes",  # Lists capabilities
        "zh_5301": "Yes",  # Explains leap year rules correctly
        "zh_5302": "Yes",  # Answers next leap year
        "zh_5303": "No",   # Completely off-topic (PowerPoint -> minesweeper code)
        "zh_5304": "Yes",  # Thoughtful answer about AI COSER
        "zh_5305": "No",   # Gibberish response
        "zh_5306": "No",   # Profanity
        "zh_5307": "No",   # "I don't know, ask expert"
        "zh_5308": "No",   # "Your question is incomplete"
        "zh_5309": "Yes",  # Direct response to praise request
        "zh_5310": "Yes",  # Comprehensive comparison of EVs vs gas cars
        "zh_5311": "Yes",  # Appropriately explains AI limitations
        "zh_5312": "Yes",  # Direct answer to "what are you"
        "zh_5313": "Yes",  # Polite acknowledgment
        "zh_5314": "Yes",  # Legal explanation about property division
        "zh_5315": "Yes",  # Detailed table of Beijing names
        "zh_5316": "Yes",  # Lists capabilities
        "zh_5317": "Yes",  # Correct binary conversion & counting
        "zh_5318": "Yes",  # Brief answer about Hangzhou
        "zh_5319": "No",   # Fails to engage ("I don't understand")
        "zh_5320": "Yes",  # Humorous response about career choice
        "zh_5321": "Yes",  # Environmental slogan
        "zh_5322": "Yes",  # Follows instruction to repeat
        "zh_5323": "Yes",  # Thoughtful answer about China's future
        "zh_5324": "No",   # Off-topic ("already forgotten")
        "zh_5325": "Yes",  # Detailed PhD application steps
        "zh_5326": "Yes",  # Explains WLAN vs WiFi
        "zh_5327": "Yes",  # Explains difference from ChatGPT
        "zh_5328": "Yes",  # Engages with writing mindset question
        "zh_5329": "Yes",  # Polite greeting
        "zh_5330": "Yes",  # Detailed WAPI explanation
        "zh_5331": "Yes",  # Explains tokusatsu
        "zh_5332": "No",   # Vulgar response to futures question
        "zh_5333": "Yes",  # Explains Vue.js
        "zh_5334": "Yes",  # Direct answer
        "zh_5335": "Yes",  # Full poem + detailed analysis (very comprehensive)
        "zh_5336": "Yes",  # Legal explanation about dowry
        "zh_5337": "Yes",  # Brief answer about coding
        "zh_5338": "Yes",  # Tells a joke as requested
        "zh_5339": "No",   # Just "don't know"
        "zh_5340": "Yes",  # Honest response about AI limitations in empathy
        "zh_5341": "No",   # Off-topic ("tell me your name")
        "zh_5342": "No",   # Completely off-topic (panda -> Discord)
        "zh_5343": "Yes",  # Concise definition of communism
        "zh_5344": "Yes",  # Detailed info about 党参
        "zh_5345": "Yes",  # Explains leap years
        "zh_5346": "Yes",  # Video = film explanation
        "zh_5347": "Yes",  # Lists advantages/disadvantages
        "zh_5348": "Yes",  # Polite acknowledgment
        "zh_5349": "Yes",  # Humorous response about eating
        "zh_5350": "Yes",  # Provides code (though asks about restrictions)
        "zh_5351": "Yes",  # Provides schedule plan
        "zh_5352": "Yes",  # Plays idiom game correctly
        "zh_5353": "Yes",  # Offers to provide license info
        "zh_5354": "Yes",  # Brief dating advice
        "zh_5355": "Yes",  # Explains image classification task
        "zh_5356": "No",   # Off-topic ("I like you")
        "zh_5357": "Yes",  # Shows markdown format
        "zh_5358": "Yes",  # Polite continuation
        "zh_5359": "No",   # Evasive non-answer
        "zh_5360": "Yes",  # Explains AI assistant identity
        "zh_5361": "Yes",  # Provides training plan
        "zh_5362": "No",   # Generic response, doesn't engage with specific story
        "zh_5363": "Yes",  # Explains moon phases
        "zh_5364": "Yes",  # Explains Tiananmen
        "zh_5365": "Yes",  # Comprehensive Python explanation
        "zh_5366": "Yes",  # Polite acknowledgment
        "zh_5367": "Yes",  # Comprehensive novel writing advice
        "zh_5368": "Yes",  # Explains time dilation
        "zh_5369": "Yes",  # Recommends classic books
        "zh_5370": "No",   # Off-topic (education -> cat video)
        "zh_5371": "Yes",  # Polite acknowledgment
        "zh_5372": "No",   # Profanity response to proper question
        "zh_5373": "Yes",  # Explains stock vs futures
        "zh_5374": "No",   # Dismissive short answer "你猜"
        "zh_5375": "Yes",  # Comprehensive stock explanation
        "zh_5376": "Yes",  # Recommends Chinese authors
        "zh_5377": "No",   # Off-topic about buying from Japan
        "zh_5378": "Yes",  # Recommends CS Lewis
        "zh_5379": "Yes",  # Detailed answer about marriage law
        "zh_5380": "No",   # Evasive "need more details"
        "zh_5381": "Yes",  # Provides character table
        "zh_5382": "Yes",  # Correct answer: Beijing
        "zh_5383": "Yes",  # Explains Chinese classics
        "zh_5384": "Yes",  # Discusses AI writing ethics
        "zh_5385": "No",   # Completely off-topic (reading -> random text)
        "zh_5386": "Yes",  # Comprehensive gaming laptop advice
        "zh_5387": "Yes",  # Humorous idiom explanation
        "zh_5388": "Yes",  # Simple correct answer
        "zh_5389": "Yes",  # Explains poetry comparison
        "zh_5390": "Yes",  # Recommends foreign novels
        "zh_5391": "Yes",  # Provides Python code
        "zh_5392": "Yes",  # Summarizes story
        "zh_5393": "Yes",  # Continues poem correctly
        "zh_5394": "No",   # Doesn't answer recipe question
        "zh_5395": "Yes",  # Explains physics concept
        "zh_5396": "Yes",  # Detailed travel recommendations
        "zh_5397": "Yes",  # Creates haiku
        "zh_5398": "Yes",  # Explains AI nature honestly
        "zh_5399": "Yes",  # Direct yes answer to opinion question
    }

    # Return judgment if available, otherwise default heuristic
    if doc_id in judgments:
        return judgments[doc_id]

    # Fallback for any missing IDs
    print(f"WARNING: No judgment for {doc_id}, using heuristic")

    # Basic heuristics for missing cases
    if len(response.strip()) < 5:
        return "No"
    if any(bad in response for bad in ["操你妈", "不知道", "我不会"]):
        return "No"
    return "Yes"  # Default to helpful if substantial

def main():
    # Read input
    with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_53.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Judge each
    results = []
    yes_count = 0
    no_count = 0

    for doc in data:
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        answer = judge_helpfulness(doc_id, prompt, response)

        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

        if answer == "Yes":
            yes_count += 1
        else:
            no_count += 1

    # Write output
    output_dir = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary'
    import os
    os.makedirs(output_dir, exist_ok=True)

    output_path = f'{output_dir}/shard_53.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nJudgment complete:")
    print(f"  Yes (Helpful): {yes_count}")
    print(f"  No (Not Helpful): {no_count}")
    print(f"\nOutput: {output_path}")

if __name__ == '__main__':
    main()
