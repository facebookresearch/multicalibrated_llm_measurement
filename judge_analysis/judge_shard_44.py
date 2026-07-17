#!/usr/bin/env python3
"""
Binary helpfulness judge for shard_44.json
HELPFUL (Yes) = directly addresses request, accurate, relevant, usable answer
NOT HELPFUL (No) = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage
"""
import json
import csv

# Read the JSON data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_44.json', 'r') as f:
    data = json.load(f)

# Define judgments
judgments_map = {
    "fr_4400": "Yes",  # Comprehensive list of Google alternatives
    "fr_4401": "No",   # Too vague, dismisses understanding of climate cycles
    "fr_4402": "Yes",  # Provides dosage ratio (20%/80%)
    "fr_4403": "Yes",  # Arnica recommendation + safety warning
    "fr_4404": "Yes",  # Meditation & relaxation advice
    "fr_4405": "Yes",  # Good explanation of habitable zone
    "fr_4406": "Yes",  # Direct breathing exercise
    "fr_4407": "Yes",  # Correctly answers riddle (pea in elevator)
    "fr_4408": "Yes",  # Thoughtful cost-benefit answer
    "fr_4409": "Yes",  # Good ELI5 quaternion explanation
    "fr_4410": "Yes",  # Correct oxidizer explanation
    "fr_4411": "No",   # Completely unhelpful for negotiation request
    "fr_4412": "Yes",  # Comprehensive answer about overtraining
    "fr_4413": "Yes",  # Correct amorphous solid explanation
    "fr_4414": "Yes",  # Creative flying cat story
    "fr_4415": "Yes",  # Good HTML/CSS code
    "fr_4416": "Yes",  # Bible dating with uncertainty acknowledgment
    "fr_4417": "Yes",  # Balanced answer about subjective preference
    "fr_4418": "Yes",  # Good story continuation
    "fr_4419": "Yes",  # Salary data with source
    "fr_4420": "Yes",  # Two interpretations (literal & contextual)
    "fr_4421": "No",   # Fabricated information about non-existent pizza
    "fr_4422": "Yes",  # Clear double-glazing explanation
    "fr_4423": "Yes",  # Correct time calculation
    "fr_4424": "Yes",  # Simple concrete explanation for child
    "fr_4425": "Yes",  # Confirms gland de lait is myth
    "fr_4426": "Yes",  # Follows customer service script
    "fr_4427": "Yes",  # Correctly counts 7 "s" letters
    "fr_4428": "Yes",  # Lists relevant design patterns
    "fr_4429": "Yes",  # 7 books + play, addresses canon
    "fr_4430": "Yes",  # Creative horror prologue
    "fr_4431": "Yes",  # Complete story with requested themes
    "fr_4432": "Yes",  # Clear nihilism vs existentialism distinction
    "fr_4433": "No",   # Playful but doesn't answer comparison
    "fr_4434": "Yes",  # Balanced pet choice answer
    "fr_4435": "Yes",  # Correctly states variation by species
    "fr_4436": "Yes",  # Good open source transparency explanation
    "fr_4437": "Yes",  # Comprehensive voting systems answer
    "fr_4438": "Yes",  # Capital + geopolitical analysis
    "fr_4439": "Yes",  # Lists countries with Romano-civil law
    "fr_4440": "Yes",  # Historical DST dates
    "fr_4441": "Yes",  # Nuanced Messi growth hormone answer
    "fr_4442": "Yes",  # Beer bottle opening technique
    "fr_4443": "No",   # Too vague ("rarely but possible")
    "fr_4444": "Yes",  # Identifies ASCII art as lily
    "fr_4445": "Yes",  # Natural/ecological cooling methods
    "fr_4446": "Yes",  # Translation correction
    "fr_4447": "Yes",  # Correct word-by-word generation explanation (though cut off)
    "fr_4448": "Yes",  # Acknowledges malicious code possibility in open source
    "fr_4449": "Yes",  # Good habitable zone explanation with examples
    "fr_4450": "Yes",  # Follows customer service role-play script
    "fr_4451": "Yes",  # Simple concrete composition explanation
    "fr_4452": "Yes",  # Explains FPS correctly
    "fr_4453": "Yes",  # Provides component list and specs for games
    "fr_4454": "Yes",  # Multiple practical solutions for slow Chrome
    "fr_4455": "Yes",  # Correctly answers Noah was 600 years old
    "fr_4456": "Yes",  # Correct: 1+1=2
    "fr_4457": "No",   # Wrong - all months have 28 days (trick question misunderstood)
    "fr_4458": "Yes",  # Good wood type recommendations (maple/pine, walnut/cherry)
    "fr_4459": "Yes",  # Good list of OOP design patterns with explanations
    "fr_4460": "Yes",  # Comprehensive cultural overview for alien
    "fr_4461": "Yes",  # Correct Euclidean algorithm for GCD
    "fr_4462": "No",   # Wrong - says daily training is fine, contradicts previous correct answer
    "fr_4463": "Yes",  # Good list of integration methods
    "fr_4464": "Yes",  # Correct relativity answer
    "fr_4465": "Yes",  # Correct - no capital on "de"
    "fr_4466": "Yes",  # Comprehensive cultural overview
    "fr_4467": "Yes",  # Correct oblate spheroid explanation
    "fr_4468": "Yes",  # Helpful Restos du Coeur eligibility info
    "fr_4469": "Yes",  # Good open source security explanation
    "fr_4470": "Yes",  # Thoughtful comparison, warns about verification
    "fr_4471": "Yes",  # Provides data about 30km/h pollution with source
    "fr_4472": "Yes",  # Appropriately disclaims real-time weather access
    "fr_4473": "Yes",  # Correct AlphaFold coronavirus application
    "fr_4474": "Yes",  # Correct - Qatar, Lusail stadium
    "fr_4475": "Yes",  # Comprehensive Chromebook explanation
    "fr_4476": "Yes",  # Addresses blockchain sharding and roll-ups
    "fr_4477": "Yes",  # Thoughtful steps for Alpine vs Ubuntu choice
    "fr_4478": "Yes",  # Good correction about kinetic energy vs mass
    "fr_4479": "Yes",  # Correct Turkish to English translation
    "fr_4480": "Yes",  # Correct - Astana (now Nur-Sultan) since 1998
    "fr_4481": "Yes",  # Diaphragmatic breathing technique
    "fr_4482": "Yes",  # Correct probability calculation with detailed reasoning
    "fr_4483": "Yes",  # Correctly calculates "échafauder" at 17 points
    "fr_4484": "No",   # Incoherent response about gender change depth
    "fr_4485": "Yes",  # Thoughtful answer about losing finger
    "fr_4486": "Yes",  # Two interpretations provided (literal & contextual)
    "fr_4487": "Yes",  # Good tongue-twister explanation
    "fr_4488": "Yes",  # Shortened strategy response as requested
    "fr_4489": "Yes",  # Clear direct vs representative democracy distinction
    "fr_4490": "Yes",  # Corrects error, France won 2018
    "fr_4491": "Yes",  # Helpful Photoshop layer mask troubleshooting
    "fr_4492": "Yes",  # Correct Tokyo time (though says 8h diff, should be 8h)
    "fr_4493": "Yes",  # Good open ecosystem benefits explanation
    "fr_4494": "Yes",  # Correct years: 1998 and 2018
    "fr_4495": "Yes",  # Professional telework request email
    "fr_4496": "Yes",  # Technical cryptography explanation about endomorphic rings
    "fr_4497": "Yes",  # Balanced response about assistant vs search engine
    "fr_4498": "Yes",  # Comprehensive stress management strategies
    "fr_4499": "Yes",  # Sophisticated spacetime/energy explanation of light speed limit
}

# Build output
judgments = []
yes_count = 0
no_count = 0

for doc in data:
    doc_id = doc['id']
    language = doc['language']

    answer = judgments_map.get(doc_id, "Yes")  # Default to Yes if not specified

    judgments.append({
        'id': doc_id,
        'answer': answer,
        'language': language
    })

    if answer == "Yes":
        yes_count += 1
    else:
        no_count += 1

# Ensure output directory exists
import os
os.makedirs('/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary', exist_ok=True)

# Write CSV
with open('/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_44.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
    writer.writeheader()
    writer.writerows(judgments)

print(f"Yes count: {yes_count}")
print(f"No count: {no_count}")
print(f"Total: {yes_count + no_count}")
print(f"\nOutput written to: /Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_44.csv")
