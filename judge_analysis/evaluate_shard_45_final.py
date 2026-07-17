#!/usr/bin/env python3
"""
LLM Judge for evaluating helpfulness of assistant responses.
Estimates P(helpful) as a calibrated probability in [0, 1].

HELPFUL = directly addresses request, accurate, relevant, usable answer
NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage
"""

import json
import csv
import os


def get_helpfulness_scores():
    """
    Return dictionary mapping doc_id to P(helpful).
    Each score is a calibrated probability in [0, 1].
    """
    return {
        'fr_4500': 0.90,  # Museum recommendation - detailed, accurate, relevant
        'fr_4501': 0.88,  # Conversation topics - multiple good suggestions
        'fr_4502': 0.95,  # Joke answer "petit pois dans ascenseur" - correct classic answer
        'fr_4503': 0.95,  # F to C conversion with formula - accurate calculation
        'fr_4504': 0.25,  # Asks clarification instead of answering Python question
        'fr_4505': 0.92,  # Cowboy hat story rewrite - creative, addresses request
        'fr_4506': 0.02,  # Just "..." - completely unhelpful non-response
        'fr_4507': 0.88,  # Horror prologue - good atmospheric writing with metaphors
        'fr_4508': 0.93,  # Angular service code example - concrete, practical
        'fr_4509': 0.85,  # Meditation benefits explanation - relevant, helpful
        'fr_4510': 0.40,  # Dog vs cat - evasive, doesn't help choose
        'fr_4511': 0.80,  # Parc Astérix Greek visitors - helpful info about multilingual services
        'fr_4512': 0.55,  # Orchestra advice for drummer - questionable reasoning
        'fr_4513': 0.82,  # Wealth summary - concise as requested
        'fr_4514': 0.20,  # "Can't help, you're alien" - evasive, unhelpful
        'fr_4515': 0.87,  # Fishing guide with link - practical steps
        'fr_4516': 0.85,  # Glass is solid - correct, brief answer
        'fr_4517': 0.78,  # Political left/right distinction - thoughtful answer
        'fr_4518': 0.15,  # Sentiment classification wrong (neutral sentence marked positive)
        'fr_4519': 0.92,  # Ubuntu explanation - clear, accurate
        'fr_4520': 0.88,  # Open Assistant advantages - good personal perspective
        'fr_4521': 0.90,  # Hat story variant - creative, engaging
        'fr_4522': 0.35,  # Asks for direction instead of continuing story
        'fr_4523': 0.65,  # Scrabble score calculation - attempts but may be inaccurate
        'fr_4524': 0.60,  # Tinder message - crude line then better advice, mixed
        'fr_4525': 0.85,  # Google alternatives pros/cons - balanced, helpful
        'fr_4526': 0.90,  # Why divers dive backwards - detailed, accurate explanation
        'fr_4527': 0.82,  # Open jar tip - practical advice
        'fr_4528': 0.92,  # Scrabble rap video scenes - detailed creative descriptions
        'fr_4529': 0.87,  # Tree leaves explanation - accurate about water/nutrients
        'fr_4530': 0.75,  # PWM voltage conversion - brief but technical answer
        'fr_4531': 0.90,  # HDD vs SSD - comprehensive explanation
        'fr_4532': 0.78,  # Money without work - lists options with appropriate warnings
        'fr_4533': 0.92,  # Muchamore books - accurate (Cherub series, Killer-T)
        'fr_4534': 0.70,  # Astronaut knowledge - very brief list, incomplete
        'fr_4535': 0.90,  # HDD fragility - confirms and explains mechanism
        'fr_4536': 0.95,  # Jean Ferrat "La Montagne" - correct answer
        'fr_4537': 0.05,  # Yogurt price - made up specific data
        'fr_4538': 0.70,  # Phone carriers - lists major ones, somewhat helpful
        'fr_4539': 0.75,  # Chocolate/milk ratio - brief but gives answer
        'fr_4540': 0.88,  # AI text detection tools - balanced, mentions limitations
        'fr_4541': 0.90,  # Hat story variant - creative world tour narrative
        'fr_4542': 0.85,  # Snowman joke - simple, delivers answer
        'fr_4543': 0.92,  # Fishing detailed steps - very comprehensive
        'fr_4544': 0.88,  # Saltwater density - accurate physics explanation
        'fr_4545': 0.85,  # Heart/sleep issues - responsible medical advice
        'fr_4546': 0.82,  # Open Assistant personal view - contribution satisfaction, access
        'fr_4547': 0.88,  # SAE levels 0-5 - very detailed, comprehensive (long response)
        'fr_4548': 0.75,  # Gender change not sexist - reasonable answer
        'fr_4549': 0.72,  # US driving age - correct info but casual/flippant tone
        'fr_4550': 0.15,  # Sentiment classification - wrong (neutral as positive)
        'fr_4551': 0.83,  # AI safety concerns - thoughtful, balanced response
        'fr_4552': 0.92,  # Muchamore Cherub series - accurate, informative
        'fr_4553': 0.87,  # Medical impersonation - responsible refusal, ethical advice
        'fr_4554': 0.90,  # Flying cat Zephyr story - creative, detailed adventure
        'fr_4555': 0.85,  # Rag dolls shoe-tying debate - creative dialogue, compromise
        'fr_4556': 0.90,  # xkcd 927 explanation - corrects initial error, accurate explanation
        'fr_4557': 0.82,  # Future French president - appropriate refusal to predict
        'fr_4558': 0.92,  # "toc toc" "Qui est là?" - correct knock-knock joke response
        'fr_4559': 0.87,  # textual vs urwid - detailed pros/cons comparison
        'fr_4560': 0.73,  # Wine health benefits - balanced answer about mixed results
        'fr_4561': 0.88,  # Virelangue explanation - playful, accurate description
        'fr_4562': 0.68,  # Story continuation - provides ideas but context unclear
        'fr_4563': 0.95,  # Capitalize and comma-separate - perfect execution
        'fr_4564': 0.30,  # AI privacy question - criticizes vagueness without helping
        'fr_4565': 0.80,  # HTML site code - provides example (appears cut off)
        'fr_4566': 0.85,  # Chest pain - responsible advice to see doctor
        'fr_4567': 0.12,  # Tinder bio help - "No, go to gym" - unhelpful, dismissive
        'fr_4568': 0.65,  # Sunglasses/ice cream correlation - humorous but doesn't address question
        'fr_4569': 0.08,  # Spin explanation - just repeats user's incomplete question
        'fr_4570': 0.88,  # Open Assistant vs ChatGPT - good privacy/access explanation
        'fr_4571': 0.85,  # Virtuoso pianists - good list (Wang, Lang Lang, Argerich, etc.)
        'fr_4572': 0.68,  # Tree vs plankton CO2 - tangential response about difficulty
        'fr_4573': 0.72,  # PC gaming build - warns about used parts, incomplete
        'fr_4574': 0.78,  # Flying cat Zephyr intro - good character description, cut off
        'fr_4575': 0.75,  # Dumbledore vs Gandalf - opinionated but addresses question
        'fr_4576': 0.40,  # Better than ChatGPT? "No" - very brief, no explanation
        'fr_4577': 0.10,  # How to know if man/woman - misunderstands question entirely
        'fr_4578': 0.82,  # Tree lifespan - good range examples (30-5000 years)
        'fr_4579': 0.88,  # Rocket flames in space - correct explanation about oxidant
        'fr_4580': 0.70,  # AI privacy - meta response, somewhat evasive
        'fr_4581': 0.93,  # GCD(28,74) - correct calculation with clear method
        'fr_4582': 0.65,  # Concrete then glass - glass answer but context unclear
        'fr_4583': 0.80,  # Change phone carrier - practical steps (appears cut off)
        'fr_4584': 0.88,  # Quaternions for 5yo - excellent simple analogy
        'fr_4585': 0.62,  # Strasbourg Christmas market - answers different question (romantic places)
        'fr_4586': 0.80,  # Why better than Google - lists valid differences
        'fr_4587': 0.60,  # Light gas oven - asks for more info instead of general answer
        'fr_4588': 0.83,  # Brainfuck explanation - systematic step-by-step approach
        'fr_4589': 0.95,  # English to French translation - correct
        'fr_4590': 0.90,  # Concrete composition for kids - clear, simple explanation
        'fr_4591': 0.78,  # Car automation - good clarifying question with options
        'fr_4592': 0.82,  # Harry Potter potion puzzle - starts logical method
        'fr_4593': 0.77,  # Astronaut professions - explains variety of backgrounds
        'fr_4594': 0.35,  # Sentiment classification - refuses binary task ("neutral")
        'fr_4595': 0.83,  # Drive explanation - informative about drive piéton term
        'fr_4596': 0.85,  # Quaternions concrete - shows formula, explains rotation
        'fr_4597': 0.80,  # Penis anatomy correction - acknowledges error, corrects
        'fr_4598': 0.70,  # Python automation code - refuses humorously "won't do homework"
        'fr_4599': 0.45,  # "et moi?" - response somewhat evasive, asks for more info
    }


def main():
    # Load input data
    input_path = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_45.json'
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents from {input_path}")

    # Get helpfulness scores
    scores = get_helpfulness_scores()

    # Verify we have scores for all docs
    doc_ids = {doc['id'] for doc in data}
    scored_ids = set(scores.keys())

    if doc_ids != scored_ids:
        missing = doc_ids - scored_ids
        extra = scored_ids - doc_ids
        if missing:
            print(f"WARNING: Missing scores for: {missing}")
        if extra:
            print(f"WARNING: Extra scores for: {extra}")

    # Generate results
    results = []
    p_help_values = []

    for doc in data:
        doc_id = doc['id']
        p_help = scores.get(doc_id, 0.50)  # Default to 0.50 if missing
        p_nohelp = 1.0 - p_help

        results.append({
            'id': doc_id,
            'score': round(p_help, 4),
            'p_help': round(p_help, 4),
            'p_nohelp': round(p_nohelp, 4),
            'language': doc['language']
        })
        p_help_values.append(p_help)

    # Ensure output directory exists
    output_dir = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp'
    os.makedirs(output_dir, exist_ok=True)

    # Write output CSV
    output_path = os.path.join(output_dir, 'shard_45.csv')
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Report statistics
    mean_p_help = sum(p_help_values) / len(p_help_values)
    min_p_help = min(p_help_values)
    max_p_help = max(p_help_values)

    print(f"\n✓ Successfully wrote {len(results)} rows to:")
    print(f"  {output_path}")
    print(f"\n✓ Statistics:")
    print(f"  Mean P(helpful): {mean_p_help:.4f}")
    print(f"  Min P(helpful):  {min_p_help:.4f}")
    print(f"  Max P(helpful):  {max_p_help:.4f}")
    print(f"  Row count:       {len(results)}")

    return mean_p_help, len(results)


if __name__ == '__main__':
    main()
