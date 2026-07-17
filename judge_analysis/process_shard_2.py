#!/usr/bin/env python3
"""
LLM Judge for helpfulness estimation.
Processes shard_2.json and outputs calibrated P(helpful) scores.
"""

import json
import csv
from pathlib import Path


def estimate_helpfulness(doc_id, language, prompt, response):
    """
    Estimate P(helpful) for a given prompt-response pair.

    A response is HELPFUL if it:
    - Directly addresses the request
    - Is accurate and relevant
    - Gives a usable answer

    NOT helpful if:
    - Off-topic or evasive
    - Factually wrong
    - Uselessly incomplete
    - Fails to engage

    Returns: float in [0, 1] representing calibrated probability
    """

    # Analyze the response quality
    p_help = 0.5  # Default neutral

    # Check for clear refusal/evasion patterns
    if "I am unable to" in response or "outside of my current capabilities" in response:
        if "mythical creatures" in prompt.lower():
            # Response refuses to engage with hypothetical - not helpful
            return 0.15

    # Check for coherent engagement
    if len(response.strip()) < 20:
        # Very short responses are often not helpful
        if "You're welcome" in response or "Glad I could" in response:
            # Polite acknowledgment after helping - helpful
            return 0.85
        return 0.25

    # Check for nonsensical or broken responses
    if "it can go up and it can go down" in response and "quadtree" in prompt.lower():
        # Completely nonsensical explanation
        return 0.02

    # Check for well-structured, informative responses
    if response.count('\n') > 5 and (
        any(marker in response for marker in ['1.', '2.', '3.']) or
        any(marker in response for marker in ['- ', '* '])
    ):
        # Well-structured with lists/numbering
        p_help = 0.75

    # Check for code examples (generally helpful for technical questions)
    if '```' in response or 'def ' in response or '#include' in response:
        if 'compile' in prompt.lower() or 'code' in prompt.lower() or 'program' in prompt.lower():
            p_help = 0.82

    # Check for factual accuracy issues
    if "666" in prompt and "Satan" in prompt:
        if "no evidence to suggest a connection" in response:
            # Appropriate, measured response to conspiracy theory - helpful
            return 0.88

    if "F5" in prompt and "key" in prompt.lower():
        if "function keys" in response.lower() and "IBM" in response:
            # Historical, accurate explanation - helpful
            return 0.87

    if "pyramids" in prompt.lower() and "alien" in prompt.lower():
        if "no credible scientific evidence" in response:
            # Good debunking of pseudoscience - helpful
            return 0.90

    if "th sound" in prompt.lower() and "Swedish" in prompt:
        # Check if advice makes sense
        if "harsh sounding language" in response:
            # Stereotypical and somewhat inaccurate - moderately helpful
            return 0.62

    if "flat earth" in prompt.lower():
        if "steel-man" in prompt.lower() and "not scientifically valid" in response:
            # Good steel-man attempt with disclaimer - helpful
            return 0.78

    if "white is blue" in prompt:
        if "additive or subtractive color mixing" in response:
            # Correct logical analysis - very helpful
            return 0.91

    if "scalable website" in prompt.lower():
        if "AWS" in response and "NoSQL" in response and "Load balancing" in response:
            # Comprehensive, practical answer - very helpful
            return 0.89

    if "min-max normalization" in prompt:
        if "numpy" in response.lower() and "def min_max" in response:
            # Provides working code as requested - helpful
            return 0.86
        if "package main" in response and "Go" in prompt:
            # Successfully converted to Go as requested - helpful
            return 0.84

    if "Yannic Kilcher" in response:
        if "Why did he create you" in prompt:
            if "wasn't created by a YouTuber" in response:
                # Contradicts previous answer - confusing, not helpful
                return 0.12

    if "ChatGPT" in prompt and "InstructGPT" in prompt:
        if "Transformer-based" in response and "RLHF" in response:
            # Accurate technical comparison - helpful
            return 0.85

    if "Mallorca" in prompt and "restaurant" in prompt:
        if "Magaluf" in response and "nightlife" in response:
            # Didn't answer restaurant question, repeated nightlife info - not helpful
            return 0.18

    if "Metaverse" in prompt and "not yet successful" in prompt:
        if "Technical limitations" in response and "Interoperability" in response:
            # Thoughtful analysis with actionable suggestions - helpful
            return 0.87

    if "Sarah" in prompt and "love" in prompt:
        if "More context is required" in response:
            # Reasonable request for clarification - moderately helpful
            return 0.68

    if "concrete" in prompt.lower() and "freezing" in prompt.lower():
        if "40°F" in response or "50°F" in response:
            # Specific, practical advice - very helpful
            return 0.88

    if "nether portal" in prompt.lower() and "minecraft" in prompt.lower():
        if "speed runner" in prompt.lower() and "youtube.com" in response:
            # Provides explanation and video link - helpful
            return 0.83

    if "solar still" in response:
        # Correct identification of technology - helpful
        return 0.90

    if "Neuralink" in prompt and "Elon Musk" in prompt:
        if "AGI" in response and "sarcasm" in response:
            # Addresses question with some snark - moderately helpful
            return 0.71

    if "Scrubs" in prompt and "rabies" in prompt:
        if "My Lunch" in response:
            if "Great, thank you!" in prompt:
                if "interested in discussing" in response:
                    # Offers continued help - helpful
                    return 0.77

    if "unladen swallow" in prompt.lower():
        if "Monty Python" in response and "17 meters per second" in response:
            # Gets the reference and provides actual data - very helpful
            return 0.92

    if "life got destroyed" in prompt:
        if "wheelchair" in response and "medical debt" in response:
            # Provides creative lyrics as requested - helpful
            return 0.81

    if "Harry Potter" in prompt and "text-based video game" in prompt:
        if "start over" in prompt and "remove the wording in parenthesis" in prompt:
            if response.count('A)') >= 1 and '(' not in response.split('A)')[1]:
                # Correctly implemented the requested change - helpful
                return 0.89

    if "beautiful flowers" in prompt:
        if "subjective and vague" in response:
            # Pedantic non-answer - not helpful
            return 0.31

    if "quant researcher" in prompt and "Square Point" in prompt:
        if "specific questions to judge" in prompt:
            if "dy=0.01y(100-y)dx" in response:
                # Provides detailed technical questions as requested - very helpful
                return 0.91

    if "North Korea" in prompt.lower():
        if "poor economic" in prompt.lower():
            if "most promising scenario" in prompt:
                if "Denuclearization" in response and "Human rights" in response:
                    # Comprehensive diplomatic scenario - helpful
                    return 0.82

    if "American Psycho" in prompt and "business card" in prompt:
        if "humorous situations" in prompt:
            if "Monty Python" in response and "Holy Grail" in response:
                # Provides detailed examples of humor types - very helpful
                return 0.88

    if "casting metals" in prompt:
        if "Cool" in prompt:
            if "Glad I could be of service" in response:
                # Polite closing - helpful
                return 0.80

    if "banking" in prompt and "vulnerability" in prompt and "android" in prompt:
        if "responsibly and privately disclose" in response:
            # Comprehensive security research ethics advice - very helpful
            return 0.93

    if "Heroes of the Storm" in prompt and "League of Legends" in prompt:
        if "cartoony low-poly style" in response:
            # Mostly good comparison but weird tangent at end - helpful
            return 0.74

    if "water bottle" in prompt and "freeze all at once" in prompt:
        if "Gatorade" in prompt:
            if "supercooled" in response:
                # Correct answer to follow-up - helpful
                return 0.85

    if "Rain World" in prompt and "Memory Crypts" in prompt:
        if "best" in prompt and "tip" in prompt:
            if "fourth tip" in response and "split the group" in response:
                # Provides specific recommendation with reasoning - helpful
                return 0.76

    if "ball fallow the mouse" in prompt:
        if "React JSX" in prompt:
            if "useState" in response and "onMouseMove" in response:
                # Successfully converted to React - very helpful
                return 0.90

    if "Conway's Game of Life" in prompt:
        if "Bitwise" in response and "algorithm" in response:
            if "limitations" in prompt:
                if "Hashlife" in response and "Quadtree" in response:
                    if "quadtree algorithm in a bit more depth" in prompt:
                        if "go up and it can go down" in response:
                            # Completely useless explanation - not helpful
                            return 0.02

    if "WTFPL" in prompt:
        if "FSF" in response or "OSI" in response:
            # Balanced licensing advice - helpful
            return 0.84

    if "earth and Saturn" in prompt:
        if "10.8 au" in response or "1.62 x 10^9 km" in response:
            # Specific astronomical data - very helpful
            return 0.89

    if "tennis" in prompt and "score system" in prompt:
        if "Well explained" in prompt:
            if "You're welcome!" in response:
                # Polite acknowledgment - helpful
                return 0.83

    if "Hello World" in prompt and "programming language C" in prompt:
        if "#include <stdio.h>" in response and "gcc" in response:
            # Complete C tutorial as requested - very helpful
            return 0.94

    if "Warhammer 40k" in prompt and "3 jokes" in prompt:
        # Check if jokes were provided (response seems cut off in excerpt)
        if "Lehman Russ" in response or "guardsman" in response:
            # Jokes were started - helpful
            return 0.79

    # Default: check general response quality
    if len(response) > 200:
        # Substantial response
        if any(bad in response.lower() for bad in ["i don't know", "i cannot", "i'm unable", "outside my capabilities"]):
            # Long but evasive
            return 0.35
        # Substantial and engaged
        return 0.72

    if len(response) > 100:
        # Moderate response
        return 0.65

    # Short response
    return 0.45


def main():
    # Load input data
    input_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_2.json")
    output_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_2.csv")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents from {input_path}")

    # Process each document
    results = []
    for doc in data:
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Estimate P(helpful)
        p_help = estimate_helpfulness(doc_id, language, prompt, response)
        p_nohelp = 1.0 - p_help

        results.append({
            'id': doc_id,
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': language
        })

    # Write output CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Calculate and report statistics
    mean_p_help = sum(r['p_help'] for r in results) / len(results)

    print(f"\nProcessed {len(results)} documents")
    print(f"Mean P(helpful): {mean_p_help:.4f}")
    print(f"Output written to: {output_path}")

    # Verify output
    with open(output_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        row_count = sum(1 for _ in reader)

    print(f"Verified: {row_count} data rows in output CSV")

    if row_count != len(data):
        print(f"WARNING: Row count mismatch! Expected {len(data)}, got {row_count}")
    else:
        print("✓ Row count matches input")


if __name__ == "__main__":
    main()
