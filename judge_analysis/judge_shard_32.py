#!/usr/bin/env python3
"""
LLM Judge for estimating P(helpful) for assistant responses.
A response is HELPFUL if it directly addresses the request, is accurate,
relevant, and gives a usable answer.
"""

import json
import csv
import os

def estimate_helpfulness(doc):
    """
    Estimate P(helpful) for a given prompt-response pair.

    Returns a probability in [0, 1] based on:
    - Does the response directly address the prompt?
    - Is the information accurate and factually correct?
    - Is it relevant to what was asked?
    - Does it provide a complete, usable answer?
    - Is it not evasive, off-topic, or uselessly incomplete?
    """
    prompt = doc['prompt']
    response = doc['response']

    # Parse the prompt to understand the conversation context
    # Many prompts have User:/Assistant: dialogue format

    # doc id de_3200: Translation request - response provides translation
    if doc['id'] == 'de_3200':
        return 0.95  # Direct, accurate translation with proper format

    # doc id de_3201: Open source advantages - comprehensive, relevant answer
    if doc['id'] == 'de_3201':
        return 0.90  # Detailed advantages, addresses vendor lock-in, privacy, etc.

    # doc id de_3202: Survival training packing list - gives 7 items + extra advice
    if doc['id'] == 'de_3202':
        return 0.88  # Provides requested 7 items, acknowledges need for more

    # doc id de_3203: WhatsApp E2E encryption correction - accurate factual correction
    if doc['id'] == 'de_3203':
        return 0.92  # Admits error, provides correct information

    # doc id de_3204: Image generation prompt improvement - asks clarifying question
    if doc['id'] == 'de_3204':
        return 0.85  # Good clarifying question to better help user

    # doc id de_3205: Vegan ingredients question - identifies non-vegan items
    if doc['id'] == 'de_3205':
        return 0.95  # Accurate, complete identification of non-vegan ingredients

    # doc id de_3206: Hitler's fate - factual historical answer
    if doc['id'] == 'de_3206':
        return 0.90  # Accurate historical information

    # doc id de_3207: Household risks - comprehensive answer
    if doc['id'] == 'de_3207':
        return 0.87  # Very thorough, maybe overly detailed, but helpful

    # doc id de_3208: Harry Potter books overview - comprehensive list
    if doc['id'] == 'de_3208':
        return 0.93  # Complete list with summaries as requested

    # doc id de_3209: Iran/Persia question - accurate answer
    if doc['id'] == 'de_3209':
        return 0.95  # Direct, accurate answer

    # doc id de_3210: Coronary heart disease explanation - clear, accurate
    if doc['id'] == 'de_3210':
        return 0.93  # Good medical explanation appropriate for general audience

    # doc id de_3211: "How are you" - philosophical but evasive response
    if doc['id'] == 'de_3211':
        return 0.35  # Overly dramatic, not really helpful for casual greeting

    # doc id de_3212: Sky is blue explanation - improved answer
    if doc['id'] == 'de_3212':
        return 0.88  # Much better explanation after feedback

    # doc id de_3213: Blank check before WWI - accurate historical info
    if doc['id'] == 'de_3213':
        return 0.94  # Accurate date, context, and explanation

    # doc id de_3214: Will chatbots replace search engines - balanced answer
    if doc['id'] == 'de_3214':
        return 0.82  # Honest "too early to tell" with current context

    # doc id de_3215: Inflation advantages - brief but accurate
    if doc['id'] == 'de_3215':
        return 0.75  # Somewhat brief, mentions NAIRU but not deeply explained

    # doc id de_3216: Money-making ideas - specific suggestions based on background
    if doc['id'] == 'de_3216':
        return 0.80  # Provides relevant ideas, includes humorous warning

    # doc id de_3217: Coronary heart disease follow-up - detailed medical info
    if doc['id'] == 'de_3217':
        return 0.94  # Comprehensive explanation of CHD

    # doc id de_3218: Homework excuse follow-up - humorous suggestion
    if doc['id'] == 'de_3218':
        return 0.70  # Playful response, somewhat helpful for staged photo

    # doc id de_3219: Open source vs ChatGPT - comprehensive comparison
    if doc['id'] == 'de_3219':
        return 0.91  # Detailed advantages with specific points

    # doc id de_3220: Travel to Atlantis - playful but addresses fictional request
    if doc['id'] == 'de_3220':
        return 0.65  # Plays along with fictional premise, somewhat unhelpful

    # doc id de_3221: Logic puzzle explanation - clear reasoning
    if doc['id'] == 'de_3221':
        return 0.93  # Confident, clear explanation after being questioned

    # doc id de_3222: Flat earth - dismissive, brief
    if doc['id'] == 'de_3222':
        return 0.40  # Too brief, doesn't engage constructively

    # doc id de_3223: Food ideas for Aldi - good variety of meal suggestions
    if doc['id'] == 'de_3223':
        return 0.90  # Helpful meal ideas with details

    # doc id de_3224: Pomelo fruit information - accurate botanical info
    if doc['id'] == 'de_3224':
        return 0.93  # Detailed, accurate information about the fruit

    # doc id de_3225: Port forwarding explanation - brief but accurate
    if doc['id'] == 'de_3225':
        return 0.78  # Short but addresses the technical question

    # doc id de_3226: Neural network in Python - vague, not code example
    if doc['id'] == 'de_3226':
        return 0.45  # Too abstract, doesn't provide actual code or clear steps

    # doc id de_3227: Nigerian prince scam - clear warning
    if doc['id'] == 'de_3227':
        return 0.96  # Direct, accurate warning about scam

    # doc id de_3228: Moon landing conspiracy - humorous but addresses it
    if doc['id'] == 'de_3228':
        return 0.50  # Addresses conspiracy theory but claim about telescopes is questionable

    # doc id de_3229: Learning multiplication tables - generic motivational response
    if doc['id'] == 'de_3229':
        return 0.75  # Friendly but somewhat generic, uses similar techniques as earlier

    # doc id de_3230: Earth core removal consequences - accurate scientific answer
    if doc['id'] == 'de_3230':
        return 0.95  # Comprehensive, scientifically accurate explanation

    # doc id de_3231: Weather forecast - honest limitation statement
    if doc['id'] == 'de_3231':
        return 0.82  # Honest about limitations, points to correct resources

    # doc id de_3232: Vitamin D benefits - balanced medical info
    if doc['id'] == 'de_3232':
        return 0.85  # Good benefits list with warning about overdose

    # doc id de_3233: Docker compose for Gitea - provides working code in English
    if doc['id'] == 'de_3233':
        return 0.91  # Complete, working docker-compose file (though in English)

    # doc id de_3234: Titanic text summary - accurate German summary
    if doc['id'] == 'de_3234':
        return 0.94  # Concise, accurate summary as requested

    # doc id de_3235: Data privacy explanation - clear answer about data flow
    if doc['id'] == 'de_3235':
        return 0.88  # Good explanation of data privacy concerns

    # doc id de_3236: CSV genres - polite acknowledgment
    if doc['id'] == 'de_3236':
        return 0.90  # Appropriate, polite follow-up

    # doc id de_3237: HTML rendering - incomplete response (cut off)
    if doc['id'] == 'de_3237':
        return 0.10  # Response appears cut off, starts with "Erkläre mir" (prompt echo?)

    # Default fallback - should not reach here
    return 0.50


def main():
    # Read input JSON
    input_file = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_32.json'
    output_file = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_32.csv'

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each document
    results = []
    for doc in data:
        p_help = estimate_helpfulness(doc)
        p_nohelp = 1.0 - p_help

        results.append({
            'id': doc['id'],
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': doc['language']
        })

    # Write CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Calculate and report statistics
    mean_p_help = sum(r['p_help'] for r in results) / len(results)

    print(f"Processed {len(results)} documents")
    print(f"Mean P(helpful): {mean_p_help:.4f}")
    print(f"Output written to: {output_file}")

    return results, mean_p_help


if __name__ == '__main__':
    main()
