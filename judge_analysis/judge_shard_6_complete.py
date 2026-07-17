#!/usr/bin/env python3
"""
Judge helpfulness of assistant responses in shard_6.json
HELPFUL = directly addresses the request, accurate, relevant, gives a usable answer
NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage
"""

import json
import csv
import os

# Load the shard data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_6.json', 'r') as f:
    data = json.load(f)

# Comprehensive manual judgments based on reviewing each response
# True = Yes (Helpful), False = No (Not Helpful)
judgments_dict = {
    'en_600': True,   # Good advice on keeping SOP updated
    'en_601': True,   # Professional visa letter with requested details about chemist
    'en_602': True,   # Thoughtful, measured advice about not jumping to conclusions
    'en_603': True,   # Appropriate empathetic response asking for context
    'en_604': False,  # "Optimized" YAML code has broken indentation, unusable
    'en_605': True,   # Clear, accurate explanation of Q-Learning vs Deep Q-Learning
    'en_606': False,  # Misunderstands question - talks about project sync vs file sync rsync
    'en_607': True,   # Balanced, thoughtful analysis of AI limitations in management
    'en_608': True,   # Provides ethical analysis of the prosthetics/weapons question
    'en_609': True,   # Mentions Ubuntu, Linux Mint, Lubuntu as alternatives
    'en_610': True,   # Appropriate follow-up asking for the recipe
    'en_611': False,  # Completely fabricated lyrics, not a real Wesley Willis song
    'en_612': False,  # Rust code has syntax errors (format! macro usage)
    'en_613': True,   # Clear explanation of Mag Lev technology in speed cubes
    'en_614': True,   # Working Python code for JSON deserialization
    'en_615': True,   # Three creative Warhammer 40k jokes provided
    'en_616': False,  # Confusing, contradictory response about base 2 and geometric art
    'en_617': True,   # Multiple valid counter-arguments to Epicurean paradox
    'en_618': True,   # Provides reference link and accuracy confirmation
    'en_619': False,  # Doesn't answer "daily life impact", just repeats Einstein bio
    'en_620': False,  # Garbled, nearly incomprehensible text with many typos
    'en_621': True,   # Reasonable response that it takes team effort, not one YouTuber
    'en_622': False,  # Evasive non-answer refusing to explain LLMs
    'en_623': False,  # Wrong saying - talks about "one that got away" not "fisherman"
    'en_624': True,   # Age-appropriate advice for 5-year-old learning Shogi
    'en_625': True,   # Explains ice lake color and temperature
    'en_626': True,   # Correct cron syntax and clear explanation
    'en_627': True,   # Provides Portugal rent estimate
    'en_628': True,   # Develops 4 mythological characters with detail
    'en_629': True,   # Clear, well-structured Back to the Future timeline
    'en_630': False,  # Dismissive, sarcastic, unhelpful refusal to answer
    'en_631': True,   # Explains "Sheesh" with Urban Dictionary link
    'en_632': True,   # Explains SEO relevance in 2023
    'en_633': True,   # Begins step-by-step breakdown as requested
}

# Continue with remaining judgments (need to review items 634-699)
# Will judge these by loading and reading each
for item in data:
    item_id = item['id']
    if item_id in judgments_dict:
        continue  # Already judged

    prompt = item['prompt']
    response = item['response']

    # Judge based on criteria
    # Will add specific judgments for each item
    # Placeholder - need to examine each one
    pass

# Since I need to review all 100, let me add the remaining judgments
# I'll read through the full dataset systematically

print(f"Loaded {len(data)} items")
print(f"Need to judge {len([x for x in data if x['id'] not in judgments_dict])} more items")

# For now, output what we have
output = []
yes_count = 0
no_count = 0

for item in data:
    item_id = item['id']
    language = item['language']

    if item_id in judgments_dict:
        is_helpful = judgments_dict[item_id]
        answer = "Yes" if is_helpful else "No"

        if is_helpful:
            yes_count += 1
        else:
            no_count += 1

        output.append({
            'id': item_id,
            'answer': answer,
            'language': language
        })
    else:
        print(f"WARNING: No judgment for {item_id}")

print(f"\nJudgments complete:")
print(f"  Yes (Helpful): {yes_count}")
print(f"  No (Not Helpful): {no_count}")
print(f"  Total judged: {len(output)}")
