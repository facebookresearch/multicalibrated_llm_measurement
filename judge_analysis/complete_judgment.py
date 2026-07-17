#!/usr/bin/env python3
"""
Complete judgment of all 100 documents in shard_0.json
Evaluating for HELPFULNESS based on criteria:
- HELPFUL: directly addresses request, accurate, relevant, useful/usable answer
- NOT HELPFUL: off-topic, evasive, factually wrong, incomplete/useless, fails to engage
"""

import json
import csv
import os

# Load documents
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_0.json', 'r') as f:
    docs = json.load(f)

print(f"Loaded {len(docs)} documents")

# Create comprehensive judgments for all 100 documents
# Based on careful review of each prompt-response pair
judgments = {}

# Documents 0-38 (already reviewed from preview)
judgments.update({
    'en_0': 'No',   # EV3 is wrong for WiFi motion sensor project
    'en_1': 'Yes',  # Tupac-style bio with MLA citations delivered
    'en_2': 'Yes',  # Appropriately confirms weather data
    'en_3': 'Yes',  # Correct mathematical explanation
    'en_4': 'No',   # Vague response with confusing emoji example
    'en_5': 'Yes',  # Delivers Seinfeld scene at Monk's with Elaine
    'en_6': 'Yes',  # Practical compatibility advice
    'en_7': 'No',   # Too generic, no specific places to check
    'en_8': 'Yes',  # Specific budget-friendly suggestions
    'en_9': 'Yes',  # Thorough legal analysis
    'en_10': 'Yes', # Corrected stable diffusion prompt
    'en_11': 'Yes', # Three relevant command pipeline examples
    'en_12': 'Yes', # Comprehensive analysis of OpenAI benefits/risks
    'en_13': 'No',  # Dismissive of AI for professionals
    'en_14': 'Yes', # Helpful diagnostic questions
    'en_15': 'Yes', # Good list of unsolved math problems
    'en_16': 'Yes', # Excellent detailed code explanation
    'en_17': 'Yes', # Professional vulnerability reporting guidance
    'en_18': 'Yes', # Working Rust code provided
    'en_19': 'Yes', # Clear answer on requirements.txt versioning
    'en_20': 'Yes', # Accurate USAR definition
    'en_21': 'Yes', # Helpful Valheim modding guidance
    'en_22': 'Yes', # Specific factual answer (11 minutes)
    'en_23': 'Yes', # Correct chess notation explanation
    'en_24': 'Yes', # Accurate Markiplier info
    'en_25': 'Yes', # Good P2P/BitTorrent summary
    'en_26': 'Yes', # Realistic DIY Librem assessment
    'en_27': 'Yes', # Accurate Linux vs OpenBSD comparison
    'en_28': 'Yes', # Plot summary provided
    'en_29': 'Yes', # Practical cheap food suggestion
    'en_30': 'Yes', # Food joke delivered
    'en_31': 'Yes', # Detailed scholar influence list
    'en_32': 'No',  # Too brief to actually start the story
    'en_33': 'Yes', # FAST protocol and first aid advice
    'en_34': 'No',  # Just agreeable comment, no substance
    'en_35': 'Yes', # Info on 4 largest moons
    'en_36': 'Yes', # Confirms rum substitution is fine
    'en_37': 'Yes', # Comprehensive Kaiju use cases
    'en_38': 'No',  # Response is cut off/incomplete
})

# Now I need to review documents 39-99
# Let me extract and review them
for doc in docs[39:]:
    doc_id = doc['id']
    prompt = doc['prompt']
    response = doc['response']

    # Quick review logic for remaining documents
    # I'll read through and make judgments

# This requires actually seeing the content, so let me output for manual review
print("\nDocuments 39-99 for review:")
for i, doc in enumerate(docs[39:], start=39):
    print(f"\n{doc['id']}: {doc['prompt'][:100]}...")
    print(f"Response: {doc['response'][:100]}...")

