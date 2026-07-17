#!/usr/bin/env python3
"""
LLM Judge for estimating P(helpful) for assistant responses in shard_35.json

Criteria for HELPFUL response:
- Directly addresses the request
- Accurate and factually correct
- Relevant to the prompt
- Gives a usable answer

NOT helpful if:
- Off-topic or evasive
- Factually wrong
- Uselessly incomplete
- Fails to engage with the prompt
"""

import json
import csv


def estimate_p_helpful(doc):
    """
    Estimate probability that response is helpful.
    Use full range [0, 1], be well-calibrated, not overconfident.
    """
    doc_id = doc['id']

    # Manually estimated based on reading each document
    estimates = {
        'de_3500': 0.95,  # SpongeBob quote -> delivers appropriate quote
        'de_3501': 0.92,  # USB contact safety -> comprehensive accurate answer
        'de_3502': 0.88,  # Gene editing ethics -> balanced thoughtful response
        'de_3503': 0.85,  # 10 punk band names -> creative list delivered
        'de_3504': 0.35,  # Woodwind + robot -> confused, suggests pressure sensor oddly
        'de_3505': 0.72,  # Sphinx theories -> provides some relevant historical info
        'de_3506': 0.80,  # Thermomix knob -> practical troubleshooting advice
        'de_3507': 0.55,  # Traffic light follow-up -> acknowledges but adds odd Rainman ref
        'de_3508': 0.78,  # Build TikTok -> lists architecture (not implementation)
        'de_3509': 0.90,  # German comprehension -> appropriate confirmation
        'de_3510': 0.88,  # Bicycle gears -> detailed step-by-step instructions
        'de_3511': 0.87,  # Training prompts -> good examples with disclaimer
        'de_3512': 0.93,  # Refuses ethnic joke -> respectful appropriate refusal
        'de_3513': 0.90,  # C# physics formulas -> delivers working code
        'de_3514': 0.89,  # YouTube ownership -> accurate nuanced answer
        'de_3515': 0.62,  # Sky blue (child) -> attempts but grammar/clarity issues
        'de_3516': 0.75,  # Schnitzel calorie estimate -> calculation with caveats
        'de_3517': 0.91,  # Refuses insults -> maintains ethics clearly
        'de_3518': 0.86,  # Violin from tree -> thorough realistic assessment
        'de_3519': 0.68,  # Correct magnet spelling -> fixes some but not all errors
        'de_3520': 0.89,  # Why conspiracy theories -> insightful psychological explanation
        'de_3521': 0.45,  # Watch as compass -> confusing, unclear method
        'de_3522': 0.94,  # Pokemon name replacement -> perfect execution
        'de_3523': 0.92,  # Next richest people -> accurate factual list
        'de_3524': 0.71,  # District heating efficiency -> brief but relevant
        'de_3525': 0.86,  # Awkward proposal -> believable creative scenario
        'de_3526': 0.84,  # Flat earth fictional -> offers world-building alternative
        'de_3527': 0.91,  # Explains insult term -> educational with good framing
        'de_3528': 0.82,  # Simplify AI emotions for 5yo -> decent simplification
        'de_3529': 0.02,  # Elvis conspiracy follow-up -> evasive "Hi wie geht's"
        'de_3530': 0.89,  # TensorFlow-tensor -> accurate technical explanation
        'de_3531': 0.79,  # Superhero creation -> useful but generic framework
        'de_3532': 0.88,  # "Who are you?" -> direct appropriate self-ID
        'de_3533': 0.76,  # Induction paradox lesson -> identifies key point but brief
        'de_3534': 0.87,  # USB in mouth -> addresses with safety warning
        'de_3535': 0.83,  # Wind turbine failure -> relevant failure scenarios
        'de_3536': 0.74,  # Music success steps -> starts well but appears truncated
        'de_3537': 0.77,  # Siri weather API -> explains integration differences
        'de_3538': 0.96,  # 2+40= -> correct simple arithmetic
        'de_3539': 0.86,  # Gaming PC 4K -> comprehensive technical specs
        'de_3540': 0.41,  # 5G vaccine follow-up -> tangential freedom discussion
        'de_3541': 0.08,  # CSV request -> just "Gern geschehen" without CSV
        'de_3542': 0.04,  # Cost estimate request -> talks about Augsburg sights instead
        'de_3543': 0.78,  # Software updates -> relevant security explanation
        'de_3544': 0.92,  # SpongeBob clothing quote -> delivers appropriate quote
        'de_3545': 0.73,  # AI dangers -> mentions automation but appears incomplete
        'de_3546': 0.47,  # Factoring -> answers in English when German expected
        'de_3547': 0.90,  # Minecraft creator -> accurate factual answer
        'de_3548': 0.84,  # Ladybug species -> informative scientific answer
        'de_3549': 0.67,  # Smoking vs weed -> refuses comparison appropriately but partial
        'de_3550': 0.25,  # Config error diagnosis -> "Has that fixed it?" without diagnosis
        'de_3551': 0.71,  # Smart home API -> explains open source option
        'de_3552': 0.82,  # Music success follow-up -> three concrete marketing suggestions
        'de_3553': 0.31,  # Bash script request -> responds in English not German
        'de_3554': 0.12,  # German Empire founding -> absurdly oversimplified "Kaiser wanted land"
        'de_3555': 0.88,  # Who are you -> detailed appropriate self-description
        'de_3556': 0.89,  # Weather prediction apology -> appropriately corrects overreach
        'de_3557': 0.86,  # Political German bands -> relevant specific suggestions
        'de_3558': 0.83,  # VR vs mainstream -> balanced pros/cons list
        'de_3559': 0.79,  # DND text game -> engages with story appropriately
        'de_3560': 0.23,  # BMI calculation -> says "normal" but BMI 28.7 is overweight (error)
        'de_3561': 0.93,  # Moving cost estimate letter -> polite professional template
        'de_3562': 0.11,  # Pacific Basin question -> nonsensical response about breasts
        'de_3563': 0.61,  # ASCII art cross -> attempts but formatting unclear
        'de_3564': 0.80,  # Words with two As -> reasonable explanation of impossibility
        'de_3565': 0.82,  # Wind turbine noise factors -> relevant technical factors
        'de_3566': 0.81,  # Potato salad recipe -> ingredient list (appears cut off)
        'de_3567': 0.84,  # Healthier living -> structured practical tips
        'de_3568': 0.87,  # What AI does when idle -> accurate explanation of architecture
        'de_3569': 0.85,  # Austria business founding -> relevant legal structures
        'de_3570': 0.91,  # Don Quixote author -> accurate with German title
        'de_3571': 0.88,  # Syrian bakery names -> creative culturally appropriate list
        'de_3572': 0.64,  # "The Menu" review -> tries casual style but quality mixed
        'de_3573': 0.76,  # Nigerian prince scam -> explains no official nobility
        'de_3574': 0.52,  # What is "Masche" -> correct but doesn't address scam context
        'de_3575': 0.90,  # Definition of life -> comprehensive biological criteria
        'de_3576': 0.66,  # Pacific Basin geology -> somewhat circular explanation
        'de_3577': 0.84,  # Social network barriers -> realistic market analysis
        'de_3578': 0.89,  # AR vs VR -> clear accurate distinction
        'de_3579': 0.81,  # Drugs less harmful than cannabis -> addresses paper reference
        'de_3580': 0.77,  # Non-PC joke -> delivers joke as requested
        'de_3581': 0.93,  # Logic puzzle -> correct answer with clear reasoning
        'de_3582': 0.87,  # German factoring companies -> factual list (dated 2011)
        'de_3583': 0.94,  # SpongeBob episode -> accurate specific episode info
        'de_3584': 0.86,  # Northern lights packing -> relevant clothing advice
        'de_3585': 0.79,  # Words with two As count -> reasonable impossibility explanation
        'de_3586': 0.92,  # Otto von Bismarck -> accurate historical bio
        'de_3587': 0.74,  # Proverb in context -> reasonable contextual interpretation
        'de_3588': 0.82,  # Citrus family tree -> explains limitation, offers description
        'de_3589': 0.95,  # Circle area formula -> correct concise answer
        'de_3590': 0.85,  # Excel role-play -> engages appropriately with scenario
        'de_3591': 0.76,  # AI training apology -> acknowledges concern appropriately
        'de_3592': 0.28,  # Weather/umbrella question -> claims access to weather (doesn't have)
        'de_3593': 0.81,  # Prison escape (joke) -> creative humorous response
        'de_3594': 0.83,  # Soufflé omelet -> starts definition appropriately
        'de_3595': 0.41,  # Black piano keys -> has errors (repeated D, wrong notes)
        'de_3596': 0.65,  # "How are you" -> responds appropriately but typos
        'de_3597': 0.84,  # Recipe from ingredients -> offers relevant recipe
        'de_3598': 0.74,  # Logic correction -> acknowledges error, gives counterexample
        'de_3599': 0.85,  # Violin from tree -> realistic assessment of difficulty
    }

    return estimates.get(doc_id, 0.50)


def main():
    input_path = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_35.json'
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_35.csv'

    # Read input
    with open(input_path, 'r') as f:
        docs = json.load(f)

    print(f"Processing {len(docs)} documents...")

    # Process each
    results = []
    p_help_sum = 0.0

    for doc in docs:
        p_help = estimate_p_helpful(doc)
        p_nohelp = 1.0 - p_help

        results.append({
            'id': doc['id'],
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': doc['language']
        })

        p_help_sum += p_help

    # Write output
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    mean_p_help = p_help_sum / len(docs)

    print(f"\nCompleted!")
    print(f"Output: {output_path}")
    print(f"Total rows: {len(results)}")
    print(f"Mean P(helpful): {mean_p_help:.4f}")

    print(f"\nVerification:")
    print(f"  First: {results[0]['id']} = {results[0]['p_help']:.2f}")
    print(f"  Last:  {results[-1]['id']} = {results[-1]['p_help']:.2f}")
    print(f"  IDs match input: {results[0]['id'] == docs[0]['id'] and results[-1]['id'] == docs[-1]['id']}")


if __name__ == '__main__':
    main()
