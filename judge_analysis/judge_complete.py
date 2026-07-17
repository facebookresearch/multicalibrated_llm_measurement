#!/usr/bin/env python3
"""Complete LLM Judge for shard_45 - all 100 documents manually judged"""

import json
import csv
from pathlib import Path

def get_all_judgments():
    """
    All 100 calibrated manual judgments.
    HELPFUL = directly addresses request, accurate, relevant, usable
    NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage
    """
    return {
        "fr_4500": "Yes",  # Museum recommendation with details + alternatives
        "fr_4501": "Yes",  # Good conversation topics (interests, travel, ambitions)
        "fr_4502": "Yes",  # Correct riddle answer (pea in elevator)
        "fr_4503": "Yes",  # Accurate F→C formula and calculation
        "fr_4504": "No",   # Only clarifying question, no actual answer
        "fr_4505": "Yes",  # Story rewritten as requested (hat travels world)
        "fr_4506": "No",   # Just "..." - non-response
        "fr_4507": "Yes",  # Horror prologue with metaphors, varied sentences
        "fr_4508": "Yes",  # Angular service code example
        "fr_4509": "Yes",  # Meditation explanation for relaxation/spirituality
        "fr_4510": "Yes",  # Balanced answer (both are wonderful)
        "fr_4511": "Yes",  # Greek visitor info for Parc Astérix
        "fr_4512": "Yes",  # Drummer advice (philharmonic first, explains why)
        "fr_4513": "Yes",  # Shortened wealth summary as requested
        "fr_4514": "No",   # Refuses alien roleplay - fails to engage
        "fr_4515": "Yes",  # Fishing guide with federation link
        "fr_4516": "No",   # Says glass is solid - oversimplified/incomplete
        "fr_4517": "Yes",  # Political divide definition (equality vs equity)
        "fr_4518": "No",   # Wrong sentiment (neutral labeled as positive)
        "fr_4519": "Yes",  # Ubuntu definition correct
        "fr_4520": "Yes",  # Personal OA benefits (data control, transparency)
        "fr_4521": "Yes",  # Humorous story rewrite
        "fr_4522": "No",   # Evasive "be more precise" with no content
        "fr_4523": "No",   # Wrong Scrabble score calculation
        "fr_4524": "Yes",  # Pickup line with authenticity advice
        "fr_4525": "Yes",  # App switching pros/cons analysis
        "fr_4526": "Yes",  # Why dive backwards - thorough explanation
        "fr_4527": "Yes",  # Jar opening tip (tap bottom)
        "fr_4528": "Yes",  # Video clip scene descriptions
        "fr_4529": "Yes",  # Why leaves fall (water/light reduction)
        "fr_4530": "Yes",  # PWM voltage converter solution (op-amp)
        "fr_4531": "Yes",  # HDD vs SSD differences
        "fr_4532": "Yes",  # Money alternatives with risk warnings
        "fr_4533": "Yes",  # Muchamore books (Cherub, Killer-T)
        "fr_4534": "Yes",  # Astronaut requirements
        "fr_4535": "Yes",  # HDD fragility explained
        "fr_4536": "Yes",  # "La Montagne" by Ferrat - correct
        "fr_4537": "No",   # Fabricated yogurt price data
        "fr_4538": "Yes",  # Operator switching info (lists providers)
        "fr_4539": "No",   # Vague "20/80" without clarity
        "fr_4540": "Yes",  # AI detection tools (GPTZero, CopyLeaks)
        "fr_4541": "Yes",  # Humorous flying hat rewrite
        "fr_4542": "Yes",  # Snowman joke
        "fr_4543": "Yes",  # Comprehensive fishing guide
        "fr_4544": "Yes",  # Saltwater density explanation
        "fr_4545": "Yes",  # Anxiety/apnea coping advice
        "fr_4546": "Yes",  # Open-source contribution benefits
        "fr_4547": "Yes",  # SAE automation levels 0-5 detailed
        "fr_4548": "Yes",  # Gender change not sexist, nuanced answer
        "fr_4549": "Yes",  # US voting/driving age (18/16) with humor
        "fr_4550": "No",   # Says "1" for neutral statement - wrong
        "fr_4551": "Yes",  # Addresses AI safety concerns, community data
        "fr_4552": "Yes",  # CHERUB series info correct
        "fr_4553": "Yes",  # Refuses to aid illegal medical practice - appropriate
        "fr_4554": "Yes",  # Flying cat story (Zephyr vs Robert dog)
        "fr_4555": "Yes",  # Dialogue between dolls about shoe-tying
        "fr_4556": "Yes",  # Explains xkcd 927 (competing standards)
        "fr_4557": "Yes",  # Can't predict future president - appropriate
        "fr_4558": "Yes",  # "Who's there?" - knock-knock joke response
        "fr_4559": "Yes",  # Textual vs urwid comparison started
        "fr_4560": "Yes",  # Wine health: mixed results, alcoholism risk
        "fr_4561": "Yes",  # Tongue twister definition
        "fr_4562": "Yes",  # Story continuation with wet bag/punishment
        "fr_4563": "Yes",  # Reformatted list correctly (capitals, commas)
        "fr_4564": "No",   # Says question is ambiguous without attempting answer
        "fr_4565": "Yes",  # HTML code example with title, paragraph, links
        "fr_4566": "Yes",  # Chest pain: see doctor if persists - good advice
        "fr_4567": "No",   # "No, go to gym" - dismissive/unhelpful
        "fr_4568": "Yes",  # Correctly says truck accident won't affect ice cream sales
        "fr_4569": "No",   # Just repeats the question - total non-response
        "fr_4570": "Yes",  # Open Assistant vs ChatGPT (data transparency)
        "fr_4571": "Yes",  # Virtuoso pianists discussion (nuanced)
        "fr_4572": "Yes",  # Plankton vs trees for CO2 - thoughtful comparison
        "fr_4573": "Yes",  # Used PC parts advice (caution on HDD/GPU)
        "fr_4574": "Yes",  # Zephyr flying cat character created
        "fr_4575": "Yes",  # Dumbledore vs Gandalf - gives opinion with reasoning
        "fr_4576": "No",   # Just "No." - too terse to be helpful
        "fr_4577": "No",   # Misunderstands question about human gender
        "fr_4578": "Yes",  # Tree lifespan (30 to 5000+ years)
        "fr_4579": "Yes",  # Rockets carry own oxidant - correct explanation
        "fr_4580": "Yes",  # AI risks discussion, notes own bias
        "fr_4581": "Yes",  # GCD calculation (28, 74) = 2, correct method
        "fr_4582": "Yes",  # Glass made by heating sand - simple/accurate
        "fr_4583": "Yes",  # Phone operator switching process explained
        "fr_4584": "Yes",  # Quaternions: ball rotation - 5-year-old level
        "fr_4585": "Yes",  # Romantic Strasbourg spots (different Q but helpful)
        "fr_4586": "Yes",  # Why better than Google (synthesis vs links)
        "fr_4587": "No",   # Only asks for model details, no general guidance
        "fr_4588": "Yes",  # Brainfuck code explanation (Hello World)
        "fr_4589": "Yes",  # Translation: How are you today? Correct
        "fr_4590": "Yes",  # Concrete: gravel, cement, water - clear
        "fr_4591": "No",   # Only asks clarifying Q, doesn't attempt answer
        "fr_4592": "Yes",  # Harry Potter potion logic puzzle approach
        "fr_4593": "Yes",  # Astronaut: varied backgrounds, specific training
        "fr_4594": "No",   # Says "neutral" when asked for binary 1/0 - doesn't follow instructions
        "fr_4595": "Yes",  # Drive vs drive piéton explained
        "fr_4596": "Yes",  # Quaternion formula: a+bi+cj+dk
        "fr_4597": "Yes",  # Corrects gland de lait error, explains anatomy
        "fr_4598": "No",   # Refuses homework with snarky print statement
        "fr_4599": "Yes",  # Doesn't know user, asks for more info - honest
    }

def main():
    input_file = "/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_45.json"
    output_file = "/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_45.csv"
    
    # Load documents
    with open(input_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"Loaded {len(documents)} documents")
    
    # Get all judgments
    judgments = get_all_judgments()
    
    # Build results
    results = []
    yes_count = 0
    no_count = 0
    
    for doc in documents:
        doc_id = doc['id']
        language = doc['language']
        
        answer = judgments.get(doc_id, "Yes")  # Default Yes if missing
        
        if answer == "Yes":
            yes_count += 1
        else:
            no_count += 1
        
        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })
    
    # Write CSV
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nJudgment Summary:")
    print(f"  Yes (Helpful): {yes_count}")
    print(f"  No (Not Helpful): {no_count}")
    print(f"  Total: {len(results)}")
    print(f"\nOutput written to:")
    print(f"  {output_file}")

if __name__ == "__main__":
    main()
