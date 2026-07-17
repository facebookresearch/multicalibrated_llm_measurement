#!/usr/bin/env python3
"""
Judge helpfulness of assistant responses in shard_46.json
HELPFUL = directly addresses request, accurate, relevant, usable answer
NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage
"""

import json
import csv

def judge_helpfulness(doc_id, prompt, response):
    """
    Judge if a response is helpful based on the prompt-response pair.
    Returns True for helpful, False for not helpful.
    """

    # Specific judgments based on doc_id and content analysis
    judgments = {
        # YES cases - directly helpful, accurate, relevant
        'fr_4600': True,  # LCM calculation correctly provided
        'fr_4601': True,  # Complete Python image conversion code
        'fr_4602': True,  # Correctly identifies "gland de lait" as fiction
        'fr_4605': True,  # Lists top 5 games with caveats
        'fr_4606': True,  # Love letter with requested features
        'fr_4607': True,  # Brief but accurate on cat genetics prediction
        'fr_4608': True,  # Earth is round - correct answer
        'fr_4610': True,  # Hunger Games book count with page numbers
        'fr_4611': True,  # Creative dialogue as requested
        'fr_4612': True,  # Photosynthesis explanation
        'fr_4614': True,  # Sentiment correctly identified
        'fr_4615': True,  # Correctly identifies non-prime
        'fr_4616': True,  # Correct reasoning about pregnancy
        'fr_4617': True,  # Lists video and board games
        'fr_4618': True,  # Polite decline suggestions
        'fr_4619': True,  # Correctly formatted list
        'fr_4620': True,  # Air resistance physics
        'fr_4621': True,  # Gas oven instructions
        'fr_4623': True,  # Refers to doctor appropriately
        'fr_4625': True,  # Multiple jar-opening methods
        'fr_4626': True,  # Good translation
        'fr_4627': True,  # Amorphous solid explanation
        'fr_4628': True,  # Habitable zone explanation
        'fr_4629': True,  # Inspirational quotes
        'fr_4630': True,  # "Pendule mère" explanation
        'fr_4631': True,  # Black hole singularity
        'fr_4633': True,  # Atmospheric scattering
        'fr_4634': True,  # Detailed Harry Potter puzzle
        'fr_4635': True,  # Blockchain vulgarization
        'fr_4636': True,  # Gender identity response
        'fr_4639': True,  # Plasma state explanation
        'fr_4640': True,  # Earth-is-round experiments
        'fr_4641': True,  # Inspirational quotes
        'fr_4642': True,  # Correctly identifies odd-one-out
        'fr_4643': True,  # Newton's method
        'fr_4644': True,  # Gender change response
        'fr_4645': True,  # XMPP vs Matrix comparison
        'fr_4646': True,  # Primality testing methods
        'fr_4647': True,  # PWM voltage conversion
        'fr_4648': True,  # Cat vs dog maintenance
        'fr_4649': True,  # Webcam criteria
        'fr_4650': True,  # Integration methods
        'fr_4651': True,  # Scrabble rap
        'fr_4652': True,  # Least useful finger
        'fr_4654': True,  # Khazars historical info
        'fr_4658': True,  # DJ Seduction similar artists
        'fr_4659': True,  # Appropriately refers to doctor
        'fr_4660': True,  # Letter to future self (better than 4609)
        'fr_4661': True,  # Simple Newton's law explanation
        'fr_4662': True,  # Strasbourg Christmas market
        'fr_4663': True,  # Rock states of matter
        'fr_4664': True,  # Erotic autobiography advice
        'fr_4665': True,  # Ubuntu alternatives
        'fr_4666': True,  # VTC terms and conditions
        'fr_4667': True,  # Chest pain - refers to doctor
        'fr_4668': True,  # Binary addition explained
        'fr_4670': True,  # Python folder size function
        'fr_4671': True,  # Existential questions list
        'fr_4674': True,  # Nihilism vs existentialism
        'fr_4675': True,  # Why trees lose leaves
        'fr_4676': True,  # French president term
        'fr_4677': True,  # Erotic opening paragraph
        'fr_4678': True,  # HDD vs SSD
        'fr_4680': True,  # Elephants and humans
        'fr_4681': True,  # ASCII art flower type
        'fr_4682': True,  # Master clock explanation
        'fr_4684': True,  # Age calculation correct
        'fr_4685': True,  # Noah's daughters-in-law
        'fr_4686': True,  # API URL construction
        'fr_4687': True,  # Translation with humor
        'fr_4688': True,  # Charles de Gaulle capitalization
        'fr_4690': True,  # Open Assistant GitHub URL
        'fr_4691': True,  # Glass melting temperature
        'fr_4692': True,  # Beer bottle opening technique
        'fr_4693': True,  # Appropriately uncertain about future
        'fr_4694': True,  # Corrects France World Cup wins
        'fr_4695': True,  # Lynx summary
        'fr_4696': True,  # Daylight saving time
        'fr_4697': True,  # Fishing in river
        'fr_4699': True,  # File size function modification

        # NO cases - off-topic, wrong, evasive, useless
        'fr_4603': False,  # Generic platitudes
        'fr_4604': False,  # Fictional president
        'fr_4609': False,  # Extremely low-effort letter
        'fr_4613': False,  # Off-topic teleportation
        'fr_4622': False,  # Refuses to answer left/right
        'fr_4624': False,  # Dismissive about Bible
        'fr_4632': False,  # Major factual errors (Ginny as mother)
        'fr_4637': False,  # Conflates PGMC with PGCD
        'fr_4638': False,  # Wrong about Earth and white dwarf
        'fr_4653': False,  # Off-topic gyrotron response
        'fr_4655': False,  # Just asks clarification without helping
        'fr_4656': False,  # Completely off-topic carbon-14
        'fr_4657': False,  # Wrong answer about Canadian French
        'fr_4669': False,  # Ignores name William, uses YouBot
        'fr_4672': False,  # Talks about bodybuilding not wine
        'fr_4673': False,  # Only asks clarification, no attempt
        'fr_4679': False,  # Wrong - xkcd 927 is a real comic
        'fr_4683': False,  # Lists scandals not pricing info
        'fr_4689': False,  # Vague warning, no config help
        'fr_4698': False,  # One-word "hippogriffe" nonsense
    }

    # If explicitly judged, return that
    if doc_id in judgments:
        return judgments[doc_id]

    # For remaining items, analyze programmatically
    # Check for common unhelpful patterns
    response_lower = response.lower()

    # Too short/dismissive responses
    if len(response) < 50:
        # Very short could be appropriate for yes/no questions
        if '?' in prompt and any(word in response_lower for word in ['oui', 'non', 'yes', 'no']):
            return True
        return False

    # Off-topic indicators
    if "je ne peux pas" in response_lower or "je ne sais pas" in response_lower:
        # Sometimes appropriate to admit limitations
        if "consulter" in response_lower or "médecin" in response_lower:
            return True  # Referring to expert is good
        return False  # Otherwise evasive

    # Default to helpful if no red flags
    return True

def main():
    # Read input JSON
    with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_46.json', 'r') as f:
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

        is_helpful = judge_helpfulness(doc_id, prompt, response)

        answer = "Yes" if is_helpful else "No"
        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

        if is_helpful:
            yes_count += 1
        else:
            no_count += 1

    # Write output CSV
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_46.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} judgments to {output_path}")
    print(f"Yes (helpful): {yes_count}")
    print(f"No (not helpful): {no_count}")
    print(f"Total: {yes_count + no_count}")

if __name__ == '__main__':
    main()
