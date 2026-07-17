#!/usr/bin/env python3
"""
Judge helpfulness of AI assistant responses in shard_10.json
HELPFUL (Yes) = directly addresses request, accurate, relevant, gives useful information or usable answer
NOT HELPFUL (No) = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage with what was asked
"""

import json
import csv

# Load data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_10.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Manual judgments based on careful review
judgments = {
    "es_1000": "Yes",  # Vector multiplication - comprehensive, accurate explanation with examples
    "es_1001": "Yes",  # BJT vs MOSFET - brief but correct about charge carriers
    "es_1002": "Yes",  # Ser vs estar - brief but captures the basic distinction
    "es_1003": "No",   # OSI layers - WRONG! Mixes layers, only describes 5, confuses terminology
    "es_1004": "Yes",  # Vaccine safety - balanced, informative, suggests consulting professionals
    "es_1005": "Yes",  # Asturias cider making - accurate 3-step process
    "es_1006": "Yes",  # EIN for foreign LLC - specific phone number and hours
    "es_1007": "Yes",  # Skinner's evolution - addresses question, explains how thinking evolved
    "es_1008": "Yes",  # Medical case translation to Spanish - correct translation
    "es_1009": "Yes",  # AI for developers - 5-step comprehensive guide
    "es_1010": "Yes",  # Explaining pet death to child - extensive age-appropriate guidelines
    "es_1011": "Yes",  # Song recommendations - gives 5 similar difficulty songs with explanation
    "es_1012": "Yes",  # Chicken recipe - complete with ingredients and steps
    "es_1013": "No",   # IPv4 vs IPv6 - user gives long explanation, assistant just says "correct!" - not helpful
    "es_1014": "No",   # Richest person - vague unhelpful advice about "virality"
    "es_1015": "Yes",  # Salary increase email - good template with all requested elements
    "es_1016": "No",   # Spanish writers - lists Pablo Neruda who is CHILEAN not Spanish! Factual error
    "es_1017": "Yes",  # Learning HTML encouragement - supportive practical tip
    "es_1018": "Yes",  # Human evolution continuing - thoughtful balanced answer
    "es_1019": "Yes",  # Cat poem 4 lines - rhymes properly, poetic
    "es_1020": "Yes",  # Music and buying - cites studies, detailed
    "es_1021": "Yes",  # Absurd train problem - correctly identifies absurdity, adds humor
    "es_1022": "Yes",  # Short vs long term memory - brief but accurate
    "es_1023": "No",   # Fortnite battle royale - user asked for DIALOGUE and INTERACTIONS, only got descriptions
    "es_1024": "Yes",  # Amino acids - complete list of 20
    "es_1025": "Yes",  # PDF merging Python - detailed code with multiple examples
    "es_1026": "Yes",  # 4G frequencies Spain - specific bands listed
    "es_1027": "Yes",  # Exposure triangle - comprehensive photography tutorial
    "es_1028": "Yes",  # Argentina president 2018 - correct (Macri)
    "es_1029": "No",   # Dating fossils - says they don't exist! Completely wrong
    "es_1030": "Yes",  # dSEO.pro - admits not knowing, asks for info (appropriate)
    "es_1031": "No",   # Web design - vague, poorly structured, doesn't address finding agency
    "es_1032": "Yes",  # Barnacle reproduction - accurate detailed biological explanation
    "es_1033": "Yes",  # Jaywalking laws - accurate country list
    "es_1034": "Yes",  # WWI summary - comprehensive, age-appropriate
    "es_1035": "Yes",  # Web scraping with requests - gives needed libraries and approach
    "es_1036": "Yes",  # Death penalty arguments - balanced 3 for and 3 against
    "es_1037": "Yes",  # Philosophy in 21st century - thoughtful fundamental answer
    "es_1038": "Yes",  # OpenSCAD - accurate description of parametric 3D modeling
    "es_1039": "Yes",  # Shorter summary request - provides condensed theoretical physics summary
    "es_1040": "Yes",  # What day is today - gives date with humor disclaimer
    "es_1041": "Yes",  # Thanks - appropriate "you're welcome" response
    "es_1042": "Yes",  # Existence poem - octosyllabic with rhyme as requested
    "es_1043": "No",   # Hogwarts Legacy - WRONG! It's a video game, not a book
    "es_1044": "Yes",  # Textures.com clarification - explains free vs paid
    "es_1045": "Yes",  # Prions - accurate explanation of misfolded proteins
    "es_1046": "Yes",  # "Vale" - appropriate casual acknowledgment
    "es_1047": "Yes",  # Centrum multivitamin - explains varying formulations
    "es_1048": "Yes",  # Hierro vs fierro - correct (synonyms)
    "es_1049": "Yes",  # Futuristic city - engages with prompt about technology and resources
    "es_1050": "Yes",  # Indigenous philosophy - holistic worldview explanation
    "es_1051": "Yes",  # Modern Age Spain - comprehensive summary with people, places, events
    "es_1052": "Yes",  # Murcia earthquakes - correct (plate tectonics)
    "es_1053": "Yes",  # Center button CSS - provides correct CSS code
    "es_1054": "Yes",  # Best rum accompaniment - thoughtful about sharing vs specific mixer
    "es_1055": "Yes",  # Traffic and weather - correctly states no real-time access
    "es_1056": "Yes",  # Bakery signage Bogotá - specific required signs
    "es_1057": "Yes",  # Existential questions - honest about limitations, addresses being human
    "es_1058": "Yes",  # AI for email - suggests automated classification and response
    "es_1059": "Yes",  # What is Spain - basic geography and culture answer
    "es_1060": "Yes",  # SOAP call Java - provides working code example
    "es_1061": "Yes",  # Three boots pairs - clever answer with Argentine idiom
    "es_1062": "Yes",  # Poem structure - analyzes rhyme scheme ABAB ABCD etc
    "es_1063": "Yes",  # Marvel board games - confirms list provided, asks if more needed
    "es_1064": "Yes",  # Where to shop - balanced answer about different stores for different needs
    "es_1065": "Yes",  # Meaning of life - thoughtful personal recommendations
    "es_1066": "Yes",  # Role reversal - correctly clarifies it's the AI helping user
    "es_1067": "Yes",  # Reformulate Spanish history - good paraphrase
    "es_1068": "Yes",  # Firebase auth Kotlin - starts explaining connection process
    "es_1069": "Yes",  # AGI methods future - balanced discussion of uncertainty
    "es_1070": "Yes",  # Father of psychoanalysis - correct (Freud)
    "es_1071": "Yes",  # Décima espinela about homeland - proper format and theme
    "es_1072": "Yes",  # Working together - appropriate enthusiastic response
    "es_1073": "Yes",  # Olive oil price - correctly notes variability by factors
    "es_1074": "Yes",  # States of matter - lists solid, liquid, gas, plasma etc
    "es_1075": "No",   # AI training data difference - says only "spontaneity" - uselessly incomplete
    "es_1076": "Yes",  # Stock quote value - correctly notes data may be outdated
    "es_1077": "Yes",  # Largest AI models - lists GPT and others
    "es_1078": "Yes",  # Freezer vs fridge - explains same system, different temps
    "es_1079": "No",   # "Coste mensual" - bizarre claim about letter repetition - wrong reasoning
    "es_1080": "Yes",  # Cappuccino vs normal coffee - correct (milk proportion)
    "es_1081": "Yes",  # Game Awards 2020 - correct (Last of Us Part 2)
    "es_1082": "Yes",  # C++ standard libraries - acknowledges user info, offers more help
    "es_1083": "Yes",  # Filter AC air - provides multiple suggestions
    "es_1084": "Yes",  # Turing machine as AI - correct distinction (mathematical model vs AI)
    "es_1085": "Yes",  # Toilet leak repair - gives specific repair steps
    "es_1086": "Yes",  # Cat nine lives origin - notes cultural variation
    "es_1087": "Yes",  # Python sum function - correct simple function
    "es_1088": "Yes",  # Python programming - installation and getting started steps
    "es_1089": "Yes",  # Indie games like Hollow Knight - recommends Dead Cells with explanation
    "es_1090": "Yes",  # Sumo wrestlers throw salt - correct purification ritual
    "es_1091": "Yes",  # 10 dog names - provides 10 names
    "es_1092": "Yes",  # Home automation - starts with assessment advice
    "es_1093": "Yes",  # Babies cry when sleepy - correct explanation of expressing need
    "es_1094": "No",   # Medio ambiente vs medioambiente - WRONG! Says "medio ambiente" but both are correct, RAE accepts both
    "es_1095": "Yes",  # Vegan lunch meal plan - provides full week plan
    "es_1096": "Yes",  # Prevention advice - cautious about medical advice, gives general tips
    "es_1097": "Yes",  # Baby congratulations message - heartfelt appropriate message
    "es_1098": "No",   # Open Assistant offline - WRONG! It requires internet, this is factually incorrect
    "es_1099": "Yes",  # Estimate age via historical events - clever use of Berlin Wall 1989
}

# Verify we have all 100
assert len(judgments) == 100, f"Expected 100 judgments, got {len(judgments)}"

# Create results
results = []
yes_count = 0
no_count = 0

for doc in data:
    doc_id = doc['id']
    language = doc['language']
    answer = judgments[doc_id]

    results.append({
        'id': doc_id,
        'answer': answer,
        'language': language
    })

    if answer == "Yes":
        yes_count += 1
    else:
        no_count += 1

# Create output directory if needed
import os
os.makedirs('/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary', exist_ok=True)

# Write CSV
output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_10.csv'
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
    writer.writeheader()
    writer.writerows(results)

print(f"Processed {len(results)} documents")
print(f"Yes (helpful): {yes_count}")
print(f"No (not helpful): {no_count}")
print(f"Data rows written: {len(results)}")
print(f"Results written to: {output_path}")
