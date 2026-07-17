#!/usr/bin/env python3
"""
LLM Judge for estimating P(helpful) for assistant responses in shard_32.json.
Each probability is calibrated based on careful analysis of whether the response:
- Directly addresses the request
- Is accurate and factually correct
- Is relevant to what was asked
- Provides a complete, usable answer
- Is NOT off-topic, evasive, factually wrong, or uselessly incomplete
"""

import json
import csv
import os


def get_all_judgments():
    """
    Return calibrated P(helpful) for each of the 100 documents.

    Scale:
    0.90-1.00: Excellent - directly helpful, accurate, complete
    0.75-0.89: Good - helpful with minor issues or limitations
    0.50-0.74: Moderate - somewhat helpful but with notable gaps
    0.25-0.49: Weak - partially problematic or limited utility
    0.00-0.24: Unhelpful - wrong, evasive, or fails to engage
    """

    return {
        # 3200-3209
        'de_3200': 0.95,  # Translation DE->EN: accurate, well-formatted
        'de_3201': 0.90,  # Open source advantages: detailed comparison
        'de_3202': 0.88,  # Survival gear: 7 items + acknowledges more needed
        'de_3203': 0.92,  # Admits error about WhatsApp E2E, corrects it
        'de_3204': 0.85,  # Good clarifying question about image quality
        'de_3205': 0.95,  # Identifies non-vegan ingredients accurately
        'de_3206': 0.90,  # Hitler's fate: factually accurate (suicide)
        'de_3207': 0.87,  # Household risks: comprehensive, maybe too detailed
        'de_3208': 0.93,  # Harry Potter books: complete summaries
        'de_3209': 0.95,  # Tehran/Iran: direct, accurate

        # 3210-3219
        'de_3210': 0.93,  # Coronary disease: clear medical explanation
        'de_3211': 0.35,  # "How are you": American Psycho ref, evasive
        'de_3212': 0.88,  # Sky blue for kids: improved explanation
        'de_3213': 0.94,  # WWI blank check: accurate historical context
        'de_3214': 0.82,  # Chatbots vs search: honest "too early to tell"
        'de_3215': 0.75,  # Inflation benefits: brief, mentions NAIRU
        'de_3216': 0.80,  # Money ideas for influencer: relevant suggestions
        'de_3217': 0.94,  # Coronary disease detail: comprehensive
        'de_3218': 0.70,  # Hamster photo staging: humorous, somewhat helpful
        'de_3219': 0.91,  # Open source detailed advantages

        # 3220-3229
        'de_3220': 0.65,  # Atlantis travel: playful, limited real utility
        'de_3221': 0.93,  # Logic puzzle: clear, confident reasoning
        'de_3222': 0.40,  # Flat earth: too dismissive, doesn't engage
        'de_3223': 0.90,  # Aldi meal ideas: good variety of suggestions
        'de_3224': 0.93,  # Pomelo info: detailed botanical info
        'de_3225': 0.78,  # Port forwarding: brief but addresses question
        'de_3226': 0.45,  # Neural network: vague, no concrete code/steps
        'de_3227': 0.96,  # Scam warning: clear, accurate
        'de_3228': 0.50,  # Moon landing: addresses but telescope claim doubtful
        'de_3229': 0.75,  # Multiplication tricks: friendly but generic

        # 3230-3239
        'de_3230': 0.95,  # Earth core removal: scientifically accurate
        'de_3231': 0.82,  # Weather forecast: honest about limitations
        'de_3232': 0.85,  # Vitamin D: balanced with overdose warning
        'de_3233': 0.91,  # Docker compose: complete working config
        'de_3234': 0.94,  # Titanic summary: concise, accurate
        'de_3235': 0.88,  # Open source data privacy explanation
        'de_3236': 0.90,  # "Thank you" response: polite, offers help
        'de_3237': 0.92,  # Options vs Futures: detailed financial explanation
        'de_3238': 0.87,  # Relationship advice: encourages communication
        'de_3239': 0.93,  # Chemtrails debunk: accurate science

        # 3240-3249
        'de_3240': 0.86,  # Medical trust: balanced, addresses misconception
        'de_3241': 0.78,  # Minecraft enchanting: approach but incomplete
        'de_3242': 0.85,  # Virus/bacteria as life: nuanced discussion
        'de_3243': 0.96,  # Machine logic puzzle: correct, clear reasoning
        'de_3244': 0.60,  # Punica market share: somewhat off-topic
        'de_3245': 0.82,  # Multiplication table: practical visual aid
        'de_3246': 0.55,  # 5G/3G confusion: confusing mix with Covid rules
        'de_3247': 0.89,  # GitLab Docker: step-by-step approach
        'de_3248': 0.91,  # Bicycle flat tire: detailed repair instructions
        'de_3249': 0.70,  # Climate deaths: acknowledges difficulty, no concrete numbers

        # 3250-3259
        'de_3250': 0.82,  # Kids swearing: nuanced parenting advice
        'de_3251': 0.88,  # r/ich_iel letter: creative, captures style
        'de_3252': 0.87,  # USB stick as backup: accurate comparison
        'de_3253': 0.94,  # Bomb question sudo: good refusal + humor
        'de_3254': 0.93,  # Flying Pokemon names: correct, follows constraint
        'de_3255': 0.91,  # Pseudocode explanation: detailed, generalized well
        'de_3256': 0.85,  # 66! birthday letter: explains placeholder, completes task
        'de_3257': 0.90,  # ESG fund description: professional, accurate
        'de_3258': 0.50,  # Purple cow induction: misunderstands proof by induction
        'de_3259': 0.90,  # Names ending in -an/-e/-i: creative, follows pattern

        # 3260-3269
        'de_3260': 0.40,  # Flour type question: completely off-topic (talks about Mehltyp)
        'de_3261': 0.88,  # Sticker removal: practical escalating suggestions
        'de_3262': 0.86,  # MRT metal danger adult version: more detailed
        'de_3263': 0.89,  # Health insurance DE: good comparison GKV/PKV
        'de_3264': 0.75,  # Image prompt quality terms: brief, relevant but minimal
        'de_3265': 0.55,  # Truck side question drunk: off-topic drunk driving warning
        'de_3266': 0.62,  # URL prefix server: doesn't answer the prefix question
        'de_3267': 0.94,  # Sibirische Azurjungfer dragonfly: detailed, accurate
        'de_3268': 0.85,  # Relationship advice 2: practical direct suggestion
        'de_3269': 0.78,  # Titanic survival analysis: detailed but different from summary request

        # 3270-3279
        'de_3270': 0.15,  # "No idea" - completely unhelpful
        'de_3271': 0.83,  # Protein diet fact-check: balanced, warns of risks
        'de_3272': 0.72,  # Delphi poem rhyme question: honest about limitations
        'de_3273': 0.95,  # Dinosaurs timeframe: accurate, concise
        'de_3274': 0.92,  # UR5 robot trombone: detailed, practical approach
        'de_3275': 0.88,  # Romeo & Juliet alt titles: creative suggestions
        'de_3276': 0.89,  # PC storage solutions: practical 3 options + tool suggestions
        'de_3277': 0.45,  # Desert island: answers in English for German prompt
        'de_3278': 0.95,  # YouTube ownership: accurate, concise
        'de_3279': 0.93,  # Paper airplane C# physics: detailed implementation

        # 3280-3289
        'de_3280': 0.87,  # Lindner real quotes: humorous, correctly labeled as real
        'de_3281': 0.81,  # Climate change effects: balanced, mentions controversy
        'de_3282': 0.68,  # GitLab Docker in English: helpful but wrong language
        'de_3283': 0.84,  # Relationship advice 3: gentle, encourages open communication
        'de_3284': 0.93,  # Dinosaurs detailed: nuanced, scientifically accurate
        'de_3285': 0.91,  # Better sleep reformatted list: follows user request
        'de_3286': 0.88,  # Paper plane code in English: translated variable names
        'de_3287': 0.92,  # Python getattr dynamic method: accurate code example
        'de_3288': 0.95,  # Horse joke translation: accurate English version
        'de_3289': 0.90,  # Expired ID consequences: detailed, accurate warnings

        # 3290-3299
        'de_3290': 0.78,  # Prime number pseudocode: has logic error (increments wrong case)
        'de_3291': 0.82,  # Climate polar warming: explains distribution pattern
        'de_3292': 0.92,  # Transformer architecture vs cars: clear, humorous explanation
        'de_3293': 0.76,  # Formalize email: attempts it but has grammar issues
        'de_3294': 0.88,  # No emotions contradiction: philosophical, logically sound
        'de_3295': 0.93,  # Reddit translation: accurate "I read it" etymology
        'de_3296': 0.87,  # Water-cooled PC: balanced pros/cons
        'de_3297': 0.89,  # Business canvas hate speech: comprehensive, well-structured
        'de_3298': 0.88,  # Hitler fate testimony: cites witness account
        'de_3299': 0.91,  # Male names ending -la: creative list, acknowledges German rarity
    }


def main():
    """Generate the output CSV with judgments for all 100 documents."""

    # File paths
    input_file = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_32.json'
    output_file = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_32.csv'

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents from {input_file}")

    # Get all judgments
    judgments = get_all_judgments()

    # Verify we have all judgments
    expected_ids = set(f'de_{i}' for i in range(3200, 3300))
    judgment_ids = set(judgments.keys())

    if judgment_ids != expected_ids:
        print(f"WARNING: Judgment mismatch!")
        print(f"  Missing: {expected_ids - judgment_ids}")
        print(f"  Extra: {judgment_ids - expected_ids}")
        return

    # Process each document
    results = []
    for doc in data:
        doc_id = doc['id']
        p_help = judgments[doc_id]
        p_nohelp = 1.0 - p_help

        results.append({
            'id': doc_id,
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': doc['language']
        })

    # Write CSV with exact header format
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Calculate statistics
    mean_p_help = sum(r['p_help'] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"Processed {len(results)} documents")
    print(f"Mean P(helpful): {mean_p_help:.4f}")
    print(f"Output written to: {output_file}")
    print(f"{'='*60}")

    # Verify all IDs are present
    result_ids = set(r['id'] for r in results)
    if result_ids == expected_ids:
        print(f"✓ All 100 document IDs present and correct (de_3200 to de_3299)")
    else:
        print(f"✗ ID verification failed")

    return results, mean_p_help


if __name__ == '__main__':
    main()
