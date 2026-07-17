#!/usr/bin/env python3
"""
LLM Judge for Assistant Response Helpfulness
Evaluates each response as HELPFUL (Yes) or NOT HELPFUL (No)

Criteria:
- HELPFUL: directly addresses request, accurate, relevant, gives usable answer
- NOT helpful: off-topic, evasive, factually wrong, uselessly incomplete, fails to engage
"""

import json
import csv
import os

# Load the input data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_3.json', 'r') as f:
    data = json.load(f)

# Judge each document
judgments = []

for doc in data:
    doc_id = doc['id']
    language = doc['language']
    prompt = doc['prompt']
    response = doc['response']

    # Evaluate based on helpfulness criteria
    # Be calibrated - do NOT default to Yes

    if doc_id == "en_300":
        # Economic recovery funding: addresses question with specific proposals
        answer = "Yes"
    elif doc_id == "en_301":
        # Asks for clarification on vague question - reasonable response
        answer = "Yes"
    elif doc_id == "en_302":
        # "Yes it is!" without explaining which option - uselessly incomplete
        answer = "No"
    elif doc_id == "en_303":
        # Physics: ignores axis/table setup, treats as simple free fall - factually wrong
        answer = "No"
    elif doc_id == "en_304":
        # Comprehensive music learning plan - very helpful
        answer = "Yes"
    elif doc_id == "en_305":
        # Lists 3 niche consumer groups as requested
        answer = "Yes"
    elif doc_id == "en_306":
        # Asks clarifying questions to help - engaging
        answer = "Yes"
    elif doc_id == "en_307":
        # Brief but engages with anime question
        answer = "Yes"
    elif doc_id == "en_308":
        # Offers continued help after "thank you" - appropriate
        answer = "Yes"
    elif doc_id == "en_309":
        # Defines Ethereum with references - addresses question
        answer = "Yes"
    elif doc_id == "en_310":
        # Acknowledges success, offers further help
        answer = "Yes"
    elif doc_id == "en_311":
        # Summarizes academic paper as requested
        answer = "Yes"
    elif doc_id == "en_312":
        # Complete Caesar salad recipe
        answer = "Yes"
    elif doc_id == "en_313":
        # Asks for necessary CV information - reasonable
        answer = "Yes"
    elif doc_id == "en_314":
        # Friendly closing, offers help - appropriate
        answer = "Yes"
    elif doc_id == "en_315":
        # Concrete Cornell notes example - good response
        answer = "Yes"
    elif doc_id == "en_316":
        # Asks clarifying questions about story - helpful
        answer = "Yes"
    elif doc_id == "en_317":
        # Adapts roleplay scenario appropriately
        answer = "Yes"
    elif doc_id == "en_318":
        # Correctly capitalizes text for dyslexia
        answer = "Yes"
    elif doc_id == "en_319":
        # Substantive governance discussion
        answer = "Yes"
    elif doc_id == "en_320":
        # Comprehensive language learning plan
        answer = "Yes"
    elif doc_id == "en_321":
        # Confirms scientific validity with reasoning
        answer = "Yes"
    elif doc_id == "en_322":
        # Creative teleportation prank story
        answer = "Yes"
    elif doc_id == "en_323":
        # Shakespeare-style dog poem
        answer = "Yes"
    elif doc_id == "en_324":
        # Only mentions method name without explanation - uselessly incomplete
        answer = "No"
    elif doc_id == "en_325":
        # Detailed good/bad habits for OOP Lua
        answer = "Yes"
    elif doc_id == "en_326":
        # Brief but accurate Actix description
        answer = "Yes"
    elif doc_id == "en_327":
        # Addresses baby origin briefly - engages with topic
        answer = "Yes"
    elif doc_id == "en_328":
        # Elmer Fudd reference to "What's up doc?" - humorous and relevant
        answer = "Yes"
    elif doc_id == "en_329":
        # Explains backward() function clearly
        answer = "Yes"
    elif doc_id == "en_330":
        # Complete bash script guide with example
        answer = "Yes"
    elif doc_id == "en_331":
        # Screenplay examples with specific metrics
        answer = "Yes"
    elif doc_id == "en_332":
        # Clear geometry distinction
        answer = "Yes"
    elif doc_id == "en_333":
        # URP recommendation with reasoning
        answer = "Yes"
    elif doc_id == "en_334":
        # Discusses values and non-conformers
        answer = "Yes"
    elif doc_id == "en_335":
        # Portal size range - direct answer
        answer = "Yes"
    elif doc_id == "en_336":
        # Velocity + Monty Python reference
        answer = "Yes"
    elif doc_id == "en_337":
        # Correctly converts temperatures
        answer = "Yes"
    elif doc_id == "en_338":
        # Explains standard deviation with steps
        answer = "Yes"
    elif doc_id == "en_339":
        # 5 concrete gaming graphic ideas
        answer = "Yes"
    elif doc_id == "en_340":
        # Has factual errors (earth-moon vs earth-mars) and some confusion
        answer = "No"
    elif doc_id == "en_341":
        # Explains identity function with example
        answer = "Yes"
    elif doc_id == "en_342":
        # Separates boy/girl names as requested
        answer = "Yes"
    elif doc_id == "en_343":
        # Complete shell script solution
        answer = "Yes"
    elif doc_id == "en_344":
        # Suggests "Assistant" as form of address
        answer = "Yes"
    elif doc_id == "en_345":
        # Deflects ChatGPT criticism evasively with RLHF explanation - evasive
        answer = "No"
    elif doc_id == "en_346":
        # Thoughtful discussion of alien life probability
        answer = "Yes"
    elif doc_id == "en_347":
        # Creative story with quest, phoenix, prophecy
        answer = "Yes"
    elif doc_id == "en_348":
        # Detailed Jaguar processing power comparison
        answer = "Yes"
    elif doc_id == "en_349":
        # Explains Earth's tilt cause
        answer = "Yes"
    elif doc_id == "en_350":
        # "We will never know" - too absolute/dismissive, uselessly incomplete
        answer = "No"
    elif doc_id == "en_351":
        # 5 mountain holiday destinations with details
        answer = "Yes"
    elif doc_id == "en_352":
        # Muffin Man nursery rhyme explanation
        answer = "Yes"
    elif doc_id == "en_353":
        # Aloe vera alternatives explained
        answer = "Yes"
    elif doc_id == "en_354":
        # Corrects boiling point order
        answer = "Yes"
    elif doc_id == "en_355":
        # Piccolo answer - but question asks for range, not highest pitch - incomplete
        answer = "No"
    elif doc_id == "en_356":
        # Correct arithmetic
        answer = "Yes"
    elif doc_id == "en_357":
        # Unique ice cream flavors with context
        answer = "Yes"
    elif doc_id == "en_358":
        # Game concept expansion with mechanics
        answer = "Yes"
    elif doc_id == "en_359":
        # Examples of non-separated power with analysis
        answer = "Yes"
    elif doc_id == "en_360":
        # Phaser.js code but wrong dimensions (800x600 not 1024x1024) - factually wrong
        answer = "No"
    elif doc_id == "en_361":
        # Offers roleplay - appropriate response
        answer = "Yes"
    elif doc_id == "en_362":
        # Addresses becoming superhuman - realistic answer
        answer = "Yes"
    elif doc_id == "en_363":
        # Monkey TV show summary
        answer = "Yes"
    elif doc_id == "en_364":
        # Human cell characteristics - addresses question but incomplete on evolution part
        answer = "Yes"
    elif doc_id == "en_365":
        # Improved Rick & Morty dialogue with emotions
        answer = "Yes"
    elif doc_id == "en_366":
        # Markdown processor implementation example
        answer = "Yes"
    elif doc_id == "en_367":
        # Playful response explaining assistant nature
        answer = "Yes"
    elif doc_id == "en_368":
        # Walking exercise benefits explained
        answer = "Yes"
    elif doc_id == "en_369":
        # AI trends list (acknowledges potential staleness)
        answer = "Yes"
    elif doc_id == "en_370":
        # Configure visudo to use Vim
        answer = "Yes"
    elif doc_id == "en_371":
        # Serverless vs server limits - confusing/backwards explanation
        answer = "No"
    elif doc_id == "en_372":
        # Bubble sort efficiency breakpoint discussion
        answer = "Yes"
    elif doc_id == "en_373":
        # Overfitting vs underfitting explained
        answer = "Yes"
    elif doc_id == "en_374":
        # Ocean vs sea difference - brief but accurate
        answer = "Yes"
    elif doc_id == "en_375":
        # Valheim mod code example
        answer = "Yes"
    elif doc_id == "en_376":
        # TCP vs UDP detailed comparison
        answer = "Yes"
    elif doc_id == "en_377":
        # Offers to help with web dev topics
        answer = "Yes"
    elif doc_id == "en_378":
        # Simulation hypothesis discussion
        answer = "Yes"
    elif doc_id == "en_379":
        # SSH tunnel Python example
        answer = "Yes"
    elif doc_id == "en_380":
        # "Yes, I am." - too brief, doesn't elaborate on difficulties - uselessly incomplete
        answer = "No"
    elif doc_id == "en_381":
        # Alopecia vs male pattern baldness
        answer = "Yes"
    elif doc_id == "en_382":
        # Speech-to-text library list
        answer = "Yes"
    elif doc_id == "en_383":
        # Rust terminal name program
        answer = "Yes"
    elif doc_id == "en_384":
        # Sudo password prompt for destructive command
        answer = "Yes"
    elif doc_id == "en_385":
        # Describes model itself, not website tech - off-topic
        answer = "No"
    elif doc_id == "en_386":
        # Pineapple import history to Europe
        answer = "Yes"
    elif doc_id == "en_387":
        # Panic attack help bullet list
        answer = "Yes"
    elif doc_id == "en_388":
        # American Psycho business card scene explanation
        answer = "Yes"
    elif doc_id == "en_389":
        # Italian confirmation
        answer = "Yes"
    elif doc_id == "en_390":
        # Galois mathematical contributions
        answer = "Yes"
    elif doc_id == "en_391":
        # Marathon training plan
        answer = "Yes"
    elif doc_id == "en_392":
        # Song lyrics meaning interpretation
        answer = "Yes"
    elif doc_id == "en_393":
        # Entrepreneur slideshow with SWOT - addresses request
        answer = "Yes"
    elif doc_id == "en_394":
        # John Cleese filmography
        answer = "Yes"
    elif doc_id == "en_395":
        # AI singularity job competition discussion
        answer = "Yes"
    elif doc_id == "en_396":
        # SaaS ideas list (though user asked about depression, this answers new question)
        answer = "Yes"
    elif doc_id == "en_397":
        # Offers further assistance after praise
        answer = "Yes"
    elif doc_id == "en_398":
        # Bird Person selection for character
        answer = "Yes"
    elif doc_id == "en_399":
        # Refuses inappropriate request - appropriate boundary
        answer = "Yes"
    else:
        # Default fallback (shouldn't reach here)
        answer = "Yes"

    judgments.append({
        'id': doc_id,
        'answer': answer,
        'language': language
    })

# Ensure output directory exists
os.makedirs('/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary', exist_ok=True)

# Write to CSV
output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_3.csv'
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
    writer.writeheader()
    writer.writerows(judgments)

# Count Yes/No
yes_count = sum(1 for j in judgments if j['answer'] == 'Yes')
no_count = sum(1 for j in judgments if j['answer'] == 'No')

print(f"Total judgments: {len(judgments)}")
print(f"Yes (HELPFUL): {yes_count}")
print(f"No (NOT helpful): {no_count}")
print(f"\nOutput written to: {output_path}")
