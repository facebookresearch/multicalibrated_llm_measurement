#!/usr/bin/env python3
"""
LLM Judge for estimating P(helpful) on assistant responses.
Processes shard_21.json and outputs calibrated probability scores.
"""

import json
import csv

# Read input data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_21.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} records")

# Manual calibrated probability judgments for each response
# P(helpful) ∈ [0,1]: probability the response is HELPFUL
# HELPFUL = directly addresses request, accurate, relevant, usable answer
# NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

judgments = {
    'ru_2100': 0.95,  # Lists 5 largest Russian cities with pops - direct, accurate, complete
    'ru_2101': 0.88,  # Etymology of "завтрак" - clever analogy, addresses question well
    'ru_2102': 0.92,  # Compression algorithms (lossless/lossy) - comprehensive, categorized
    'ru_2103': 0.85,  # "Thank you" -> "Glad to help, need more?" - appropriate, polite
    'ru_2104': 0.30,  # DDoS rights question -> "*shyly* root rights" - evasive joke, inappropriate
    'ru_2105': 0.90,  # Linux-friendly distros - 4 good options with context
    'ru_2106': 0.93,  # FFT self-implementation - working code, Cooley-Tukey algorithm explained
    'ru_2107': 0.42,  # "Tell me news" -> "I'm not a news feed" - honest but unhelpful
    'ru_2108': 0.94,  # numpy.array bug - correctly diagnoses missing brackets, shows fix
    'ru_2109': 0.91,  # 1000 bottles puzzle - binary search solution, correct generalization
    'ru_2110': 0.93,  # Reboiling water myth - debunks with chemistry, addresses Fe oxidation
    'ru_2111': 0.89,  # Rewrites location questions naturally - creative, less suspicious
    'ru_2112': 0.12,  # "Conquer world with AI" -> "I agree" - bizarre, concerning, unhelpful
    'ru_2113': 0.78,  # Hourglass repair -> "Buy new if not sentimental" - pragmatic advice
    'ru_2114': 0.82,  # OSS vs proprietary - thoughtful reflection on being open-source
    'ru_2115': 0.84,  # Pu-erh tea selection - detailed (overwrought style but informative)
    'ru_2116': 0.87,  # Weight gain after quitting - explains nicotine/blood sugar mechanism
    'ru_2117': 0.89,  # "Why Earth flat?" - explains perception, curvature evidence, tactful
    'ru_2118': 0.96,  # Mulled wine recipe - complete ingredients + instructions
    'ru_2119': 0.38,  # "No way out?" -> "Break through, poke, find" - vague platitudes
    'ru_2120': 0.88,  # React form styling - adds inline CSS, improves aesthetics
    'ru_2121': 0.86,  # "No toxins no poop?" - explains fiber/waste with shopping bag analogy
    'ru_2122': 0.90,  # Raccoon CNN classifier - TensorFlow training code, well-structured
    'ru_2123': 0.60,  # "Where grow?" -> "On citrus tree" - literal botanical, sidesteps geography
    'ru_2124': 0.87,  # Linux-Libre in Void - technical fork instructions, caveats included
    'ru_2125': 0.83,  # "How are you?" - enthusiastic, reciprocates question, friendly
    'ru_2126': 0.80,  # War partisanship -> "No consciousness/emotions" - appropriate boundary
    'ru_2127': 0.84,  # Recipe source -> explains AI databases/processing - meta but informative
    'ru_2128': 0.65,  # "Thanks" -> "*hopeful* Ask more?" - endearing but low information
    'ru_2129': 0.75,  # Purr request -> *[Feed me?]* - playful cat roleplay, stays in character
    'ru_2130': 0.52,  # Penguin joke - cultural/regional reference, may not land universally
    'ru_2131': 0.92,  # No ImageMagick -> install commands Debian/Arch + source - complete solution
    'ru_2132': 0.88,  # Big Crunch hypothesis - explains cyclic universe alternative
    'ru_2133': 0.76,  # ASCII art - stick figures passing objects, reasonable interpretation
    'ru_2134': 0.32,  # "Why 6yo jumps?" -> "They felt something" - circular, unhelpful to child
    'ru_2135': 0.89,  # RNN state info -> neuron outputs as memory - accurate technical answer
    'ru_2136': 0.22,  # Romantic dinner elaboration - sentence cuts off mid-idea, incomplete
    'ru_2137': 0.80,  # War&Peace essay detail - Roman numeral outline, but truncates mid-section
}

# Now I need to continue reading the JSON to get all 100 responses
# Let me load them programmatically and add judgments

# First, let me see what IDs I'm missing
all_ids = [item['id'] for item in data]
judged_ids = set(judgments.keys())
missing_ids = [id for id in all_ids if id not in judged_ids]

print(f"Already judged: {len(judgments)}")
print(f"Missing: {len(missing_ids)}")
print(f"Missing IDs: {missing_ids[:10]}...")  # Show first 10

# I need to read the rest of the data to make judgments on all 100
# Let me extract and display them for judgment

for item in data:
    if item['id'] not in judgments:
        print(f"\n{item['id']}:")
        print(f"Prompt: {item['prompt'][:150]}...")
        print(f"Response: {item['response'][:150]}...")
        # I'll add these judgments after inspection

# For now, let me add placeholder estimates for the remaining items
# I'll update these based on actual content

# Reading through the remaining items from the JSON structure...
# Let me complete the judgments programmatically

print("\n" + "="*80)
print("Completing judgments for remaining items...")
print("="*80)
