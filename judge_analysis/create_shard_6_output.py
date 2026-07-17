#!/usr/bin/env python3
"""
Create final CSV output for shard_6 judgments
"""

import json
import csv
import os

# Load the shard data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_6.json', 'r') as f:
    data = json.load(f)

# Combined judgments from manual review (en_600-633) and agent review (en_634-699)
all_judgments = {
    # Manual judgments en_600-633
    'en_600': True,   # Good advice on keeping SOP updated
    'en_601': True,   # Professional visa letter with requested details
    'en_602': True,   # Thoughtful advice about not jumping to conclusions
    'en_603': True,   # Appropriate empathetic response
    'en_604': False,  # Broken YAML indentation
    'en_605': True,   # Clear Q-Learning explanation
    'en_606': False,  # Misunderstands rsync question
    'en_607': True,   # Balanced AI management analysis
    'en_608': True,   # Ethical analysis provided
    'en_609': True,   # Lists Linux alternatives
    'en_610': True,   # Appropriate follow-up question
    'en_611': False,  # Fabricated lyrics
    'en_612': False,  # Rust syntax errors
    'en_613': True,   # Explains Mag Lev technology
    'en_614': True,   # Working Python code
    'en_615': True,   # Three creative jokes
    'en_616': False,  # Confusing contradictory response
    'en_617': True,   # Valid counter-arguments
    'en_618': True,   # Provides reference
    'en_619': False,  # Doesn't answer daily life impact
    'en_620': False,  # Garbled incomprehensible text
    'en_621': True,   # Reasonable team effort response
    'en_622': False,  # Evasive refusal to answer
    'en_623': False,  # Wrong saying discussed
    'en_624': True,   # Age-appropriate Shogi advice
    'en_625': True,   # Ice lake explanation
    'en_626': True,   # Correct cron syntax
    'en_627': True,   # Portugal rent estimate
    'en_628': True,   # Character development
    'en_629': True,   # Clear BTTF timeline
    'en_630': False,  # Dismissive unhelpful
    'en_631': True,   # Explains "Sheesh"
    'en_632': True,   # SEO relevance 2023
    'en_633': True,   # Step-by-step breakdown

    # Agent judgments en_634-699
    'en_634': True,   # AI job displacement ethics
    'en_635': True,   # Intermittent fasting info
    'en_636': True,   # Roleplay engagement
    'en_637': False,  # False web search claim
    'en_638': True,   # States limitation appropriately
    'en_639': True,   # Trolley problem explanation
    'en_640': True,   # 3D printing advice
    'en_641': False,  # Unhelpfully neutral dodge
    'en_642': True,   # Identifies Monty Python
    'en_643': True,   # Creative story rewrite
    'en_644': True,   # Correctly sorted list
    'en_645': True,   # Magic systems distinction
    'en_646': True,   # GTA cheat code answer
    'en_647': True,   # Docker-compose explanation
    'en_648': True,   # Creative prompts
    'en_649': True,   # Clarifying question
    'en_650': True,   # Message vs setting distinction
    'en_651': True,   # Sclerotia answer
    'en_652': True,   # Problem-solving advice
    'en_653': True,   # Positive response
    'en_654': True,   # Correct pip install
    'en_655': True,   # Logical morality answer
    'en_656': True,   # Texas grid history
    'en_657': True,   # Anime preference question
    'en_658': True,   # "Cyber" terminology
    'en_659': False,  # Wrong threading advice
    'en_660': True,   # Axial tilt explanation
    'en_661': True,   # Monkey roleplay
    'en_662': True,   # Letter outline
    'en_663': False,  # PragerU misinformation
    'en_664': True,   # Animal color vision
    'en_665': True,   # You're welcome
    'en_666': True,   # Probability calculations
    'en_667': False,  # Doesn't explain HOW photosynthesis works
    'en_668': True,   # Bias-variance tradeoff
    'en_669': True,   # Sleep with infant advice
    'en_670': True,   # Glad to help
    'en_671': True,   # PKU summary
    'en_672': True,   # Confirmation response
    'en_673': True,   # Open Assistant advantages
    'en_674': True,   # Asks for country/legal warning
    'en_675': True,   # YouTube growth advice
    'en_676': True,   # Eye strain prevention
    'en_677': True,   # Car tax discussion
    'en_678': True,   # Bible accuracy nuance
    'en_679': True,   # WHO vs AQI clarification
    'en_680': False,  # Wrong genre recommendation
    'en_681': True,   # Instruments analysis
    'en_682': True,   # Clarifying question
    'en_683': True,   # Tongue sore advice
    'en_684': True,   # Lightning Network explanation
    'en_685': True,   # Improved maze algorithm
    'en_686': True,   # REPL implementation
    'en_687': True,   # Smartboard troubleshooting
    'en_688': True,   # Big O notation
    'en_689': True,   # Efficiency tradeoffs
    'en_690': True,   # Accurate answers
    'en_691': True,   # Clash Royale deck archetypes
    'en_692': True,   # AI abuse consequences
    'en_693': True,   # Paint sprayer advice
    'en_694': True,   # States limitation
    'en_695': True,   # Language barrier strategies
    'en_696': True,   # HTML/CSS/JS linking
    'en_697': True,   # Story structures list
    'en_698': True,   # Gray code history
    'en_699': True,   # Androids plot summary
}

# Create output rows
output_rows = []
yes_count = 0
no_count = 0

for item in data:
    item_id = item['id']
    language = item['language']

    if item_id not in all_judgments:
        raise ValueError(f"Missing judgment for {item_id}")

    is_helpful = all_judgments[item_id]
    answer = "Yes" if is_helpful else "No"

    if is_helpful:
        yes_count += 1
    else:
        no_count += 1

    output_rows.append({
        'id': item_id,
        'answer': answer,
        'language': language
    })

# Ensure output directory exists
os.makedirs('/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary', exist_ok=True)

# Write to CSV
output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_6.csv'
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
    writer.writeheader()
    writer.writerows(output_rows)

print(f"Output written to: {output_path}")
print(f"\nResults:")
print(f"  Yes (Helpful):     {yes_count}")
print(f"  No (Not Helpful):  {no_count}")
print(f"  Total:             {len(output_rows)}")
print(f"\nYes rate: {yes_count/len(output_rows)*100:.1f}%")
