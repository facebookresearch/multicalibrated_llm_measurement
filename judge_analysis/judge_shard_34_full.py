#!/usr/bin/env python3
"""
LLM judge for helpfulness - shard_34.json
Judging criteria:
HELPFUL (Yes) = directly addresses request, accurate, relevant, usable answer
NOT helpful (No) = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage
"""

import json
import csv

# Read input
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_34.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} items")

# Comprehensive judgments for all 100 items
# Based on careful analysis of prompt-response pairs
judgments_map = {
    'de_3400': 'Yes',  # Healthy living advice - comprehensive, actionable
    'de_3401': 'Yes',  # Prime number explanation - correct method
    'de_3402': 'Yes',  # Earth curvature experiment - scientifically sound
    'de_3403': 'Yes',  # Vaccine alternatives - helpful suggestion
    'de_3404': 'Yes',  # Washing machine explanation - accurate
    'de_3405': 'Yes',  # Clarification request - appropriate
    'de_3406': 'Yes',  # Reeperbahn - correct answer
    'de_3407': 'Yes',  # Name of AI - appropriate response
    'de_3408': 'Yes',  # Letter modification - correctly executed
    'de_3409': 'Yes',  # Washing efficiency tips - comprehensive
    'de_3410': 'Yes',  # HOPR definition - accurate with source
    'de_3411': 'Yes',  # Name frequency - nuanced, relevant
    'de_3412': 'Yes',  # Camping options - detailed with pros/cons
    'de_3413': 'Yes',  # Cancellation email - well-crafted
    'de_3414': 'Yes',  # Falling tree - thoughtful philosophical exploration
    'de_3415': 'Yes',  # Data copy tools - comprehensive multi-platform
    'de_3416': 'No',   # Alligatoah lyrics - delivers essay not song lyric as requested
    'de_3417': 'Yes',  # Tie instructions - clear step-by-step
    'de_3418': 'Yes',  # Vacation packing - covers essentials
    'de_3419': 'Yes',  # ASCII cross - provides requested format
    'de_3420': 'Yes',  # OpenAI/GPT3 differences - accurate explanation
    'de_3421': 'Yes',  # Schnitzel calories - specific accurate info
    'de_3422': 'Yes',  # Teekesselchen examples - correct examples
    'de_3423': 'Yes',  # Solar perpetual motion - correctly explains impossibility
    'de_3424': 'Yes',  # Autocracy discussion - balanced with disclaimer
    'de_3425': 'Yes',  # Storage tips - actionable suggestions
    'de_3426': 'Yes',  # Track crossing - appropriate safety info
    'de_3427': 'No',   # Erdogan/Sweden - factually wrong, misses real geopolitical reasons
    'de_3428': 'Yes',  # "Who are you" - appropriate brief answer
    'de_3429': 'Yes',  # Cellular automata - detailed technical example
    'de_3430': 'No',   # 2+40 Douglas Adams - playful but doesn't answer math question
    'de_3431': 'Yes',  # Potato salad recipe - complete detailed recipe
    'de_3432': 'Yes',  # German states - accurate count and list
    'de_3433': 'Yes',  # Programming languages - helpful recommendations
    'de_3434': 'Yes',  # Passport vs ID - comprehensive accurate list
    'de_3435': 'Yes',  # Rust syntax - correct examples
    'de_3436': 'Yes',  # Finance ministers - comprehensive accurate table
    'de_3437': 'Yes',  # Chemical elements - accurate explanation
    'de_3438': 'Yes',  # Joke correction - appropriately accepts better interpretation
    'de_3439': 'No',   # Rammstein song - truncated/incomplete mid-response
    'de_3440': 'Yes',  # Reeperbahn again - adds context to earlier answer
    'de_3441': 'Yes',  # Dinosaur timeline - accurate correction
    'de_3442': 'Yes',  # Backstahl vs Pizzastein - addresses differences
    'de_3443': 'No',   # Birthday letter - apologizes but doesn't provide the requested letter
    'de_3444': 'No',   # Chemical elements - deliberately misunderstands "we" as ambiguous
    'de_3445': 'Yes',  # Soufflé omelette substitution - helpful answer
    'de_3446': 'No',   # Port forwarding - vague, imprecise, doesn't help with setup
    'de_3447': 'Yes',  # Reverse mouse - provides specific tool
    'de_3448': 'Yes',  # Fund name - provides creative suggestion
    'de_3449': 'Yes',  # CSV comparison tool - recommends WinMerge
    'de_3450': 'Yes',  # Monster Energy followup - provides correct info
    'de_3451': 'No',   # Microwave sales - asks for more info instead of answering
    'de_3452': 'Yes',  # Truck overtaking color - correct reasoning
    'de_3453': 'Yes',  # Blue sky for children - clear simple explanation
    'de_3454': 'Yes',  # Backup reliability - helpful technical details
    'de_3455': 'Yes',  # Poem with specific words - delivers requested poem
    'de_3456': 'Yes',  # Gummy bear rain problems - creative realistic list
    'de_3457': 'Yes',  # Magnets two-level explanation - appropriate
    'de_3458': 'Yes',  # Minecraft enchantment costs - helpful guidance
    'de_3459': 'Yes',  # Email formalization - good correction
    'de_3460': 'Yes',  # Python function with input - delivers requested code
    'de_3461': 'Yes',  # Filming people - accurate legal information
    'de_3462': 'Yes',  # Social network marketing - practical SEO tips
    'de_3463': 'Yes',  # Memorizing multiplication table - helpful tricks
    'de_3464': 'Yes',  # Premod/Mod/Postmod - clear distinctions
    'de_3465': 'Yes',  # Spongebob quotes - stays in character
    'de_3466': 'Yes',  # Sudo explanation - accurate technical info
    'de_3467': 'Yes',  # Software engineering aspects - comprehensive overview
    'de_3468': 'Yes',  # HTML landing page - provides working code
    'de_3469': 'Yes',  # Factoring finance - accurate definition
    'de_3470': 'Yes',  # Easy money - realistic disclaimer with suggestions
    'de_3471': 'Yes',  # Pokemon name replacement - correctly executed
    'de_3472': 'Yes',  # Meeting person more - actionable suggestions
    'de_3473': 'Yes',  # IBM Nazi role - substantive historical essay
    'de_3474': 'Yes',  # Violin strings - accurate materials info
    'de_3475': 'Yes',  # Fox as pet - accurate legal/ethical note
    'de_3476': 'Yes',  # AR explanation - correct definition
    'de_3477': 'Yes',  # Logic puzzle socks - correct reasoning with humor
    'de_3478': 'Yes',  # Contrails explanation - scientifically accurate
    'de_3479': 'Yes',  # DIN 4109 noise - relevant standard identified
    'de_3480': 'Yes',  # Weather today - appropriately can't provide, suggests alternative
    'de_3481': 'No',   # AI memory claim - factually wrong, implies persistent learning
    'de_3482': 'Yes',  # DFA explanation - accurate technical definition
    'de_3483': 'Yes',  # File transfer tools - provides alternatives
    'de_3484': 'Yes',  # HOPR multiple meanings - acknowledges context-dependence
    'de_3485': 'Yes',  # Recipe from ingredients - creative practical suggestion
    'de_3486': 'No',   # Logic puzzle - answer is wrong (says #2 correct but reasoning flawed)
    'de_3487': 'Yes',  # Can you predict future - appropriate decline
    'de_3488': 'Yes',  # TensorFlow explanation - relevant accurate info
    'de_3489': 'Yes',  # What can you do - appropriate self-description
    'de_3490': 'Yes',  # Office chair ergonomics - specific measurements
    'de_3491': 'Yes',  # AI emotions - accurate response
    'de_3492': 'No',   # Nutrition table - response doesn't address question at all
    'de_3493': 'Yes',  # Previous 20 words - appropriately states limitation
    'de_3494': 'Yes',  # Stars in universe - accurate estimate range
    'de_3495': 'Yes',  # LLM proprietary vs open - accurate explanation
    'de_3496': 'Yes',  # Linux on old Mac - helpful suggestion
    'de_3497': 'No',   # Sailing against wind - misunderstands question, says boats don't go against wind
    'de_3498': 'Yes',  # Why 1 not prime - correct mathematical reason
    'de_3499': 'Yes',  # Negative feedback on text - provides appropriate criticism
}

# Process all items
judgments = []
yes_count = 0
no_count = 0

for item in data:
    doc_id = item['id']
    language = item['language']

    # Get judgment
    answer = judgments_map.get(doc_id, 'Yes')  # Default to Yes if not explicitly marked

    if answer == 'Yes':
        yes_count += 1
    else:
        no_count += 1

    judgments.append({
        'id': doc_id,
        'answer': answer,
        'language': language
    })

# Ensure output directory exists
import os
os.makedirs('/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary', exist_ok=True)

# Write output CSV
with open('/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_34.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
    writer.writeheader()
    writer.writerows(judgments)

print(f"\nResults:")
print(f"Yes count: {yes_count}")
print(f"No count: {no_count}")
print(f"Total: {yes_count + no_count}")
print(f"\nOutput written to: /Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_34.csv")
