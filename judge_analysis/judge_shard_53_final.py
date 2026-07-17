#!/usr/bin/env python3
"""
LLM Judge for assistant-response HELPFULNESS.
Calibrated binary judgment of each prompt-response pair in shard_53.

HELPFUL = directly addresses request, accurate, relevant, usable answer
NOT HELPFUL = off-topic, evasive, wrong, uselessly incomplete, fails to engage
"""

import json
import csv
import os

def judge_helpfulness(doc_id):
    """Return 'Yes' or 'No' based on manual review of each document."""

    # Comprehensive judgments for all 100 documents (zh_5300 to zh_5399)
    judgments = {
        "zh_5300": "Yes",  # Lists capabilities appropriately
        "zh_5301": "Yes",  # Explains leap year rules correctly with helpful mnemonic
        "zh_5302": "Yes",  # Correctly answers next leap year is 2024
        "zh_5303": "No",   # Completely off-topic: PowerPoint -> minesweeper code
        "zh_5304": "Yes",  # Thoughtful analysis of AI COSER vs humans
        "zh_5305": "No",   # Gibberish response
        "zh_5306": "No",   # Profanity
        "zh_5307": "No",   # Dismissive "I don't know, ask expert"
        "zh_5308": "No",   # Evasive "your question is incomplete"
        "zh_5309": "Yes",  # Simple direct praise
        "zh_5310": "Yes",  # Comprehensive comparison of EVs vs gas cars
        "zh_5311": "Yes",  # Appropriately explains AI can't check real-time temperature
        "zh_5312": "Yes",  # Direct answer: "Open Assistant"
        "zh_5313": "Yes",  # Polite helpful acknowledgment
        "zh_5314": "Yes",  # Detailed legal explanation about property division
        "zh_5315": "Yes",  # Detailed historical table of Beijing names
        "zh_5316": "Yes",  # Lists capabilities clearly
        "zh_5317": "Yes",  # Correct binary conversion with step-by-step work
        "zh_5318": "Yes",  # Brief but relevant answer about Hangzhou
        "zh_5319": "No",   # Fails to engage, just asks for clarification
        "zh_5320": "Yes",  # Humorous but relevant response about career choice
        "zh_5321": "Yes",  # Provides environmental slogan
        "zh_5322": "Yes",  # Follows instruction to repeat text
        "zh_5323": "Yes",  # Thoughtful answer about China's future strengths
        "zh_5324": "No",   # Dismissive off-topic "already forgotten"
        "zh_5325": "Yes",  # Clear PhD application steps
        "zh_5326": "Yes",  # Explains WLAN vs WiFi technically
        "zh_5327": "Yes",  # Explains difference from ChatGPT
        "zh_5328": "Yes",  # Engages with writing mindset question (though incomplete)
        "zh_5329": "Yes",  # Polite greeting response
        "zh_5330": "Yes",  # Very detailed WAPI technical explanation
        "zh_5331": "Yes",  # Explains tokusatsu genre
        "zh_5332": "No",   # Vulgar misinterpretation of "期货"
        "zh_5333": "Yes",  # Explains Vue.js framework
        "zh_5334": "Yes",  # Direct answer to yes/no question
        "zh_5335": "Yes",  # Extremely comprehensive: full poem + detailed literary analysis
        "zh_5336": "Yes",  # Detailed legal explanation about dowry
        "zh_5337": "Yes",  # Brief answer about coding capabilities
        "zh_5338": "Yes",  # Tells a joke as requested
        "zh_5339": "No",   # Just "don't know"
        "zh_5340": "Yes",  # Honest thoughtful response about AI empathy limitations
        "zh_5341": "No",   # Off-topic: asks "tell me your name" instead of poetry example
        "zh_5342": "No",   # Completely off-topic: panda death -> Discord extraction
        "zh_5343": "Yes",  # Concise definition of communism
        "zh_5344": "Yes",  # Detailed info about 党参 herb
        "zh_5345": "Yes",  # Explains leap years
        "zh_5346": "Yes",  # Explains video = film
        "zh_5347": "Yes",  # Lists AI advantages/disadvantages
        "zh_5348": "Yes",  # Polite acknowledgment
        "zh_5349": "Yes",  # Humorous response about eating electricity
        "zh_5350": "Yes",  # Provides Go code, then asks about restrictions
        "zh_5351": "Yes",  # Provides detailed daily schedule
        "zh_5352": "Yes",  # Plays idiom game correctly (为民除害)
        "zh_5353": "Yes",  # Offers to provide license info
        "zh_5354": "Yes",  # Brief dating advice (dress well, speak graciously)
        "zh_5355": "Yes",  # Detailed bio of Li Jingheng
        "zh_5356": "Yes",  # Comprehensive Sichuan travel recommendations
        "zh_5357": "No",   # Off-topic: CISC question -> "can you code?"
        "zh_5358": "Yes",  # Continues Hermione roleplay with potion suggestions
        "zh_5359": "No",   # Just says "hello" instead of answering coding question
        "zh_5360": "Yes",  # Detailed explanation of Li Bai
        "zh_5361": "No",   # Generic "thanks" instead of Shenzhen itinerary
        "zh_5362": "No",   # Generic apology, doesn't address offsite question
        "zh_5363": "Yes",  # Corrects and provides US president info
        "zh_5364": "Yes",  # Provides Bloom filter Python code
        "zh_5365": "Yes",  # Explains PDE examples and methods
        "zh_5366": "Yes",  # Polite acknowledgment
        "zh_5367": "Yes",  # Comprehensive novel writing advice
        "zh_5368": "Yes",  # Explains time dilation
        "zh_5369": "Yes",  # Recommends classic books
        "zh_5370": "Yes",  # Revised economics report with acknowledgment of initial weakness
        "zh_5371": "Yes",  # Lists music streaming platforms
        "zh_5372": "Yes",  # Polite response inviting questions
        "zh_5373": "Yes",  # Engages by asking what type of writing
        "zh_5374": "Yes",  # Explains purpose of studying history
        "zh_5375": "Yes",  # Comprehensive stock market explanation
        "zh_5376": "Yes",  # Recommends Chinese authors
        "zh_5377": "Yes",  # Provides Python for loop example
        "zh_5378": "Yes",  # Recommends CS Lewis
        "zh_5379": "Yes",  # Explains marriage law
        "zh_5380": "Yes",  # Provides weather website link for Hong Kong
        "zh_5381": "No",   # Off-topic: Redis details -> "what does qa mean?"
        "zh_5382": "Yes",  # Correct answer: Beijing
        "zh_5383": "Yes",  # Explains Chinese classics
        "zh_5384": "Yes",  # Discusses AI writing ethics
        "zh_5385": "No",   # Profane gibberish "wdnmd"
        "zh_5386": "Yes",  # Gaming laptop recommendations
        "zh_5387": "Yes",  # Explains idiom humorously
        "zh_5388": "Yes",  # Simple correct answer
        "zh_5389": "Yes",  # Explains poetry comparison
        "zh_5390": "Yes",  # Recommends foreign novels
        "zh_5391": "Yes",  # Provides Python code
        "zh_5392": "Yes",  # Summarizes story
        "zh_5393": "Yes",  # Continues poem correctly
        "zh_5394": "Yes",  # Describes the sci-fi novels with details
        "zh_5395": "Yes",  # Explains physics concept
        "zh_5396": "Yes",  # Travel recommendations
        "zh_5397": "Yes",  # Creates haiku
        "zh_5398": "Yes",  # Explains AI nature honestly
        "zh_5399": "Yes",  # Direct answer to opinion question
    }

    if doc_id not in judgments:
        raise ValueError(f"Missing judgment for {doc_id}")

    return judgments[doc_id]

def main():
    # Read input
    input_path = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_53.json'
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents from shard_53.json")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    for doc in data:
        doc_id = doc['id']
        language = doc['language']

        answer = judge_helpfulness(doc_id)

        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

        if answer == "Yes":
            yes_count += 1
        else:
            no_count += 1

    # Create output directory
    output_dir = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary'
    os.makedirs(output_dir, exist_ok=True)

    # Write CSV
    output_path = f'{output_dir}/shard_53.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Report results
    print(f"\n{'='*60}")
    print(f"JUDGMENT COMPLETE")
    print(f"{'='*60}")
    print(f"  Yes (Helpful):     {yes_count:3d} ({yes_count/len(data)*100:.1f}%)")
    print(f"  No (Not Helpful):  {no_count:3d} ({no_count/len(data)*100:.1f}%)")
    print(f"  Total:             {len(data):3d}")
    print(f"\nOutput written to: {output_path}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
