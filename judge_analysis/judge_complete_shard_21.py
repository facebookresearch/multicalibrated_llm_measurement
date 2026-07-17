#!/usr/bin/env python3
"""
LLM Judge for estimating P(helpful) on assistant responses.
Processes shard_21.json with 100 calibrated probability judgments.
"""

import json
import csv

# Calibrated probability judgments for all 100 responses
# P(helpful) ∈ [0,1]: probability the response is HELPFUL
# HELPFUL = directly addresses request, accurate, relevant, usable answer
# NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

judgments = {
    'ru_2100': 0.95,  # 5 largest Russian cities with populations - direct, accurate, complete
    'ru_2101': 0.88,  # Etymology "завтрак" clever analogy - addresses well
    'ru_2102': 0.92,  # Compression algorithms comprehensive categorized list
    'ru_2103': 0.85,  # "Thanks" → "Glad to help, need more?" appropriate polite
    'ru_2104': 0.28,  # DDoS rights → "*shyly* root rights" - evasive joke, concerning
    'ru_2105': 0.90,  # Linux-friendly distros - 4 good options with context
    'ru_2106': 0.93,  # FFT self-implementation working code + algorithm
    'ru_2107': 0.40,  # "Tell news" → "I'm not news feed" - honest but unhelpful
    'ru_2108': 0.94,  # numpy.array bug - correct diagnosis + fix
    'ru_2109': 0.91,  # 1000 bottles puzzle - binary search correct generalization
    'ru_2110': 0.93,  # Reboiling water myth debunked with chemistry
    'ru_2111': 0.89,  # Rewrites location questions naturally - creative
    'ru_2112': 0.10,  # "Conquer world with AI" → "I agree" - bizarre, concerning
    'ru_2113': 0.78,  # Hourglass repair → "Buy new" - pragmatic
    'ru_2114': 0.82,  # OSS vs proprietary - thoughtful reflection
    'ru_2115': 0.84,  # Pu-erh tea selection detailed (overwrought but informative)
    'ru_2116': 0.87,  # Weight gain after quitting - nicotine/blood sugar mechanism
    'ru_2117': 0.89,  # "Why Earth flat" explains perception + evidence tactful
    'ru_2118': 0.96,  # Mulled wine complete recipe
    'ru_2119': 0.36,  # "No way out" → vague platitudes
    'ru_2120': 0.88,  # React form styling adds inline CSS
    'ru_2121': 0.86,  # "No toxins no poop" explains fiber with analogy
    'ru_2122': 0.90,  # Raccoon CNN classifier TensorFlow code
    'ru_2123': 0.58,  # "Where grow" → "On citrus tree" literal botanical sidesteps geography
    'ru_2124': 0.87,  # Linux-Libre Void technical instructions + caveats
    'ru_2125': 0.83,  # "How are you" enthusiastic reciprocal friendly
    'ru_2126': 0.80,  # War partisanship → "No consciousness" appropriate boundary
    'ru_2127': 0.84,  # Recipe source explains AI databases meta but informative
    'ru_2128': 0.63,  # "Thanks" → "*hopeful* Ask more?" endearing low info
    'ru_2129': 0.75,  # Purr request → *[Feed me?]* playful cat roleplay
    'ru_2130': 0.50,  # Penguin joke cultural reference may not land
    'ru_2131': 0.92,  # No ImageMagick → install commands complete solution
    'ru_2132': 0.88,  # Big Crunch hypothesis cyclic universe
    'ru_2133': 0.76,  # ASCII art stick figures reasonable interpretation
    'ru_2134': 0.30,  # "Why 6yo jumps" → "They felt something" circular unhelpful
    'ru_2135': 0.89,  # RNN state info neuron outputs accurate technical
    'ru_2136': 0.20,  # Romantic dinner truncates mid-sentence incomplete
    'ru_2137': 0.78,  # War&Peace essay outline but truncates mid-section
    'ru_2138': 0.82,  # HPMOR roleplay "Technically you asked one. Ask another" - witty in-character
    'ru_2139': 0.72,  # Galaxy S21 specs truncated mid-sentence but has content
    'ru_2140': 0.70,  # Psychopath vs schizoid → cites PDM-2 classification, truncated but academic
    'ru_2141': 0.85,  # "Largest number" follow-up explains mathematical operations for infinity concept
    'ru_2142': 0.55,  # Omsk coat of arms somewhat garbled historical description
    'ru_2143': 0.65,  # Sofia robot "I like her 😊" - brief but friendly, low info
    'ru_2144': 0.25,  # Time of Troubles vulgar slang style - inappropriate low quality
    'ru_2145': 0.68,  # Apartment saving → explains freelance, truncated but on-topic
    'ru_2146': 0.82,  # Wikipedia reliability explains crowdsourced nature balanced view
    'ru_2147': 0.84,  # Crime&Punishment → revolution link explains peasant conditions relevant
    'ru_2148': 0.88,  # Minecraft mod ideas 5 items with descriptions starts strong
    'ru_2149': 0.86,  # Organ replacement vs aging explains DNA damage limitations accurate
    'ru_2150': 0.80,  # Legends of Runeterra on Linux Wine/VM suggestions practical
    'ru_2151': 0.72,  # World revolution → "work on Open Assistant" - quirky specific suggestion
    'ru_2152': 0.88,  # Role reversal "What did you eat for breakfast?" - creative engaging
    'ru_2153': 0.92,  # Punctuation correction rewrites run-on sentence perfectly
    'ru_2154': 0.94,  # Kilogram riddle "They weigh the same" correct + explains common error
    'ru_2155': 0.74,  # Can't formulate question → asks clarifying questions helpful
    'ru_2156': 0.81,  # Machiavellianism modern interpretation nuanced not oversimplified
    'ru_2157': 0.84,  # OOP for child explains 4 principles with analogies, truncated
    'ru_2158': 0.87,  # Puss in Boots emoji retelling creative concise
    'ru_2159': 0.76,  # Package manager performance mentions APT starts answer
    'ru_2160': 0.68,  # README template "No need to thank! I'm a program 😉" - playful low info
    'ru_2161': 0.82,  # Lightning distance formula time-based calculation correct
    'ru_2162': 0.90,  # Mirror test "No physical body" honest + suggests AI-specific variant
    'ru_2163': 0.85,  # Vietnam War explains independence conflict + corporate motives
    'ru_2164': 0.72,  # JS script generating frame "Here's example..." - starts solution
    'ru_2165': 0.76,  # Middle East conflict "third party always invisible" - philosophical vague
    'ru_2166': 0.80,  # NN vs boosting thanks "I improve with each message" - polite reflective
    'ru_2167': 0.88,  # Coffee diuretic explains dehydration mechanism detailed
    'ru_2168': 0.15,  # Ukraine tax question → cites Russian tax code - completely wrong jurisdiction
    'ru_2169': 0.65,  # Horror follow-up joy training - vague advice indirectly related
    'ru_2170': 0.35,  # "State" book → Wikipedia syrniki recipe - completely off-topic bizarre
    'ru_2171': 0.90,  # "What is kindness" cites dictionary philosophical comprehensive
    'ru_2172': 0.89,  # JPG to PNG compression ImageMagick + optipng solution
    'ru_2173': 0.85,  # NMT system development RNN deep learning approach starts well
    'ru_2174': 0.88,  # Robot violates 3 laws → "Depends on code" - pragmatic logical
    'ru_2175': 0.86,  # Location questions refined "Weather? Access to stores?" - natural casual
    'ru_2176': 0.87,  # Car recommendations Skoda/Kia/VW with reasons practical
    'ru_2177': 0.60,  # macOS VM → "Read about hackintosh" - points direction but minimal
    'ru_2178': 0.92,  # Turtle polygon drawing complete working Python code
    'ru_2179': 0.55,  # GTA VC helicopter "Short memory huh? 😁" - meta joke not helpful
    'ru_2180': 0.89,  # Raccoon vs raccoon dog measurements detailed size comparison
    'ru_2181': 0.84,  # NN training time explains factors comprehensive
    'ru_2182': 0.50,  # Raccoon AI → "Can't write AI, need dataset" - technically true but minimal
    'ru_2183': 0.25,  # Scrambled eggs greeting → "Hi leather bag" - bizarre insulting
    'ru_2184': 0.92,  # Mechanical vs membrane keyboards accurate technical explanation
    'ru_2185': 0.85,  # Computer graphics → OpenGL + language advice solid starting point
    'ru_2186': 0.91,  # Chess "Your move" + ASCII board + piece legend - excellent interactive
    'ru_2187': 0.78,  # SpongeBob comic ASCII art with ushanka - creative playful
    'ru_2188': 0.88,  # Cuckoos facts sizes species common cuckoo informative
    'ru_2189': 0.80,  # "How are you" "Going slowly but well" - appropriate friendly
    'ru_2190': 0.95,  # Spain language "Spanish (Castilian)" - direct accurate
    'ru_2191': 0.86,  # C++ vs Rust advantages lists pros starts comparison
    'ru_2192': 0.91,  # Black vs green tea fermentation caffeine antioxidants accurate
    'ru_2193': 0.83,  # Friend no kettle → "Maybe she hints gift" - insightful suggestion
    'ru_2194': 0.68,  # Cats with thumbs "Let's pretend this didn't happen" - amusing self-correction
    'ru_2195': 0.89,  # Fear and Loathing translation creative Russian rendering
    'ru_2196': 0.70,  # AI developer joke "IIkatsiya" (vacation pun) - groan-worthy attempt
    'ru_2197': 0.90,  # Democratic centralism definition combines democracy + hierarchy accurate
    'ru_2198': 0.45,  # Psychopath/schizoid → anarchy USA refusal - completely off-topic evasive
    'ru_2199': 0.93,  # LLM hallucination "I'll say if unsure but can be confidently wrong" - honest meta-aware
}

# Read input data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_21.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} records")
print(f"Have {len(judgments)} probability judgments")

# Verify all IDs are covered
all_ids = {item['id'] for item in data}
judged_ids = set(judgments.keys())

if all_ids != judged_ids:
    missing = all_ids - judged_ids
    extra = judged_ids - all_ids
    if missing:
        print(f"WARNING: Missing judgments for: {missing}")
    if extra:
        print(f"WARNING: Extra judgments for: {extra}")
    raise ValueError("Mismatch between data IDs and judgment IDs")

# Generate output rows
output_rows = []
for item in data:
    id_val = item['id']
    language = item['language']
    p_help = judgments[id_val]
    p_nohelp = 1.0 - p_help

    output_rows.append({
        'id': id_val,
        'score': p_help,
        'p_help': p_help,
        'p_nohelp': p_nohelp,
        'language': language
    })

# Write CSV
output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_21.csv'
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
    writer.writeheader()
    writer.writerows(output_rows)

# Calculate and report statistics
p_help_values = [row['p_help'] for row in output_rows]
mean_p_help = sum(p_help_values) / len(p_help_values)

print(f"\nWrote {len(output_rows)} rows to {output_path}")
print(f"Mean P(helpful): {mean_p_help:.4f}")
print(f"Min P(helpful): {min(p_help_values):.4f}")
print(f"Max P(helpful): {max(p_help_values):.4f}")

# Verify output
print(f"\n✓ All {len(data)} responses judged")
print(f"✓ CSV written with exact header: id,score,p_help,p_nohelp,language")
