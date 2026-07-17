#!/usr/bin/env python3
"""
Evaluate helpfulness of responses in shard_22.json
Each document evaluated for P(helpful) where:
- HELPFUL = directly addresses request, accurate, relevant, usable answer
- NOT helpful = off-topic, evasive, wrong, incomplete, fails to engage
"""

import json
import csv

# Load the data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_22.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Manual P(helpful) scores for each document (calibrated across full [0,1] range)
scores = {
    "ru_2200": 0.90,  # Excellent answer: accurately explains numerology is pseudoscience
    "ru_2201": 0.85,  # Good honest answer about numbered citations being input error
    "ru_2202": 0.35,  # Problematic AI consciousness claim, questionable philosophy
    "ru_2203": 0.82,  # Good diagnostic approach: asks clarifying questions, gives reasonable advice
    "ru_2204": 0.78,  # Honest about Russia's recycling limitations, practical advice given
    "ru_2205": 0.85,  # Good historical answer about nihilism origins
    "ru_2206": 0.12,  # Completely fails - asked for piracy example, gave unrelated court case
    "ru_2207": 0.95,  # Excellent comprehensive list of venomous snakes with locations
    "ru_2208": 0.88,  # Good comprehensive Android security advice
    "ru_2209": 0.93,  # Excellent professional rewrite of angry email
    "ru_2210": 0.83,  # Good accessible explanation of non-factorial rings for general audience
    "ru_2211": 0.22,  # Answers wood stain removal then bizarre unrelated self-modification question
    "ru_2212": 0.88,  # Comprehensive yoga/meditation benefits
    "ru_2213": 0.82,  # Provides MQL5 trading code as requested (functional example)
    "ru_2214": 0.91,  # Excellent clear explanation of open source principles
    "ru_2215": 0.65,  # Provides some Frostpunk quotes but limited/possibly inaccurate
    "ru_2216": 0.48,  # Weak Freudian joke about pears, doesn't really answer
    "ru_2217": 0.91,  # Accurate concise definition of denazification
    "ru_2218": 0.81,  # Successfully decodes obfuscated text and creates creative encoded response
    "ru_2219": 0.68,  # Says "happy to help" but this is response to finding ISO - minor/polite
    "ru_2220": 0.93,  # Complete detailed homemade cheese recipe with clear steps
    "ru_2221": 0.45,  # Just points out incomplete question without being helpful
    "ru_2222": 0.42,  # Very minimal "battery problem?" - barely helpful
    "ru_2223": 0.90,  # Good budget 3D printer recommendations with prices
    "ru_2224": 0.87,  # Excellent self-aware explanation of citation number artifacts
    "ru_2225": 0.94,  # Brilliant educational answer about digestion/metabolism for young audience
    "ru_2226": 0.72,  # Reasonable suggestion about tea invitation but brief
    "ru_2227": 0.86,  # Good practical nose hair removal methods
    "ru_2228": 0.76,  # Honest about limitations, explains open source contribution
    "ru_2229": 0.91,  # Great second set of Russian puns/wordplay
    "ru_2230": 0.78,  # Thoughtful philosophy answer about power/control/ego (mid-conversation)
    "ru_2231": 0.87,  # Excellent nuanced answer armor vs firearms with video reference
    "ru_2232": 0.62,  # Just "you're welcome" - polite but minimal
    "ru_2233": 0.55,  # Snarky "do you want to chat with object?" - somewhat dismissive
    "ru_2234": 0.68,  # Generic AI response about loving users - addresses question somewhat
    "ru_2235": 0.86,  # Good creative translation of Hunter S. Thompson passage
    "ru_2236": 0.84,  # Good explanation why AI can't taste H2/He, adds context about acids
    "ru_2237": 0.88,  # Accurate concise answer: Röntgen labeled unknown radiation as X
    "ru_2238": 0.90,  # Identifies numpy array error and provides correct solution
    "ru_2239": 0.85,  # Standard but accurate definition of love
    "ru_2240": 0.87,  # Good English "may" subjunctive examples
    "ru_2241": 0.72,  # Honest admission about not having real-time bitcoin data
    "ru_2242": 0.70,  # Starts answering music promotion then shifts to generic marketing essay
    "ru_2243": 0.90,  # Excellent clear explanation of 5 English question types with examples
    "ru_2244": 0.82,  # Good concise answer on postmodernism vs nihilism re: values
    "ru_2245": 0.84,  # Good answer about Bulgakov's Jerusalem references beyond Master/Margarita
    "ru_2246": 0.88,  # Thoughtful answer about tea euphemism, suggests asking directly
    "ru_2247": 0.75,  # Sweet response encouraging user to sleep ("turns off light")
    "ru_2248": 0.82,  # Good Big Bang explanation from Wikipedia, offers more detail
    "ru_2249": 0.78,  # Answers about ASP.NET Core 2 being from Microsoft
    "ru_2250": 0.32,  # Completely wrong - describes Ubuntu terminal when asked about Void Linux
    "ru_2251": 0.80,  # Thoughtful subjective answer about music quality/consumer society
    "ru_2252": 0.92,  # Simple accurate answer: Paris is capital of France
    "ru_2253": 0.73,  # Humorous answer about singing origins - creative but not informative
    "ru_2254": 0.82,  # Witty response about finding good music vs trends ("like marriage not fling")
    "ru_2255": 0.68,  # Just asks clarification on "anonymous browser" - reasonable but minimal
    "ru_2256": 0.84,  # Good essay on life with pro/con arguments
    "ru_2257": 0.88,  # Excellent Python function with detailed parameters for divisibility
    "ru_2258": 0.89,  # Five good penguin jokes in Russian
    "ru_2259": 0.81,  # Good sea creature dialogue for voice acting practice
    "ru_2260": 0.76,  # Describes projectile physics for Unity NPC shooting accuracy
    "ru_2261": 0.74,  # Says sunk cost fallacy consequences feel normal until they hit
    "ru_2262": 0.86,  # Good recursive range() implementation in Python without loops
    "ru_2263": 0.70,  # Apologizes for seeming rude - polite recovery
    "ru_2264": 0.62,  # Suggests "Binance" for bitcoin price - minimal answer
    "ru_2265": 0.64,  # Snarky Fallout quote response, asks for clarification about war question
    "ru_2266": 0.87,  # Thoughtful job choice advice, asks to list pros/cons
    "ru_2267": 0.58,  # "Fill potholes with gravel/sand" - too brief/obvious
    "ru_2268": 0.90,  # Excellent detailed open source explanation
    "ru_2269": 0.85,  # Good list of AI capabilities after user says "no"
    "ru_2270": 0.84,  # Accurate answer about Dead Souls serfdom census period
    "ru_2271": 0.76,  # Suggests merch monetization after game becomes popular
    "ru_2272": 0.55,  # Asks for vacuum model - helpful but minimal response
    "ru_2273": 0.66,  # Weird joke "sheep eaten by warrant officer" - confusing
    "ru_2274": 0.38,  # Sarcastic "run jump dance" - unhelpful and dismissive
    "ru_2275": 0.91,  # Complete shashlik recipe with ingredients and steps
    "ru_2276": 0.68,  # Asks for media text to analyze - reasonable request
    "ru_2277": 0.18,  # Asks birth year to determine age - defeats the purpose entirely
    "ru_2278": 0.80,  # Simple "Hi user!" response to English greeting
    "ru_2279": 0.86,  # Interesting etymology of "завтрак" (breakfast/tomorrow)
    "ru_2280": 0.84,  # Good approach to wine poisoning problem with binary labeling
    "ru_2281": 0.25,  # Answer uses NO Old Church Slavonic words despite request
    "ru_2282": 0.72,  # Polite "hope I helped" closing
    "ru_2283": 0.45,  # Confused answer mixing up raccoons/raccoon dogs with completely wrong info
    "ru_2284": 0.81,  # Good philosophical response about self-control and willpower
    "ru_2285": 0.77,  # Thoughtful answer about art creativity based on viewer experience
    "ru_2286": 0.79,  # Creative game monetization ideas (streaming, film rights, donor features)
    "ru_2287": 0.74,  # Confirms knife throwing connection to "тютелька в тютельку"
    "ru_2288": 0.86,  # Good comparison of liberalism vs conservatism (political/economic/social)
    "ru_2289": 0.82,  # Polite response about unusual name, confirms French translation ability
    "ru_2290": 0.88,  # Good advice on dealing with school bullying
    "ru_2291": 0.89,  # Excellent creative response: ignores human like a cat would
    "ru_2292": 0.90,  # Good air raid shelter poster instructions (clear, actionable)
    "ru_2293": 0.71,  # Provides image generation service links for SpongeBob comic
    "ru_2294": 0.25,  # Rude "Google it yourself, not small" - unhelpful dismissive
    "ru_2295": 0.88,  # Good technical explanation of Yggdrasil mesh network
    "ru_2296": 0.62,  # Says can't simplify Windows password reset process - honest but unhelpful
    "ru_2297": 0.86,  # Good Neznaika on the Moon plot summary
    "ru_2298": 0.75,  # Answers about KDE2 patching on FreeBSD with security warning
    "ru_2299": 0.55,  # "Horror but similar" response to Upyachka meme - minimal engagement
}

# Verify we have exactly 100 scores with correct IDs
expected_ids = [f"ru_{2200 + i}" for i in range(100)]
assert len(scores) == 100, f"Expected 100 scores, got {len(scores)}"
assert set(scores.keys()) == set(expected_ids), "ID mismatch"

# Create output directory if it doesn't exist
import os
output_dir = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp'
os.makedirs(output_dir, exist_ok=True)

# Write CSV output
output_path = os.path.join(output_dir, 'shard_22.csv')
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'score', 'p_help', 'p_nohelp', 'language'])

    for doc in data:
        doc_id = doc['id']
        p_help = scores[doc_id]
        p_nohelp = 1.0 - p_help
        writer.writerow([
            doc_id,
            f"{p_help:.2f}",
            f"{p_help:.2f}",
            f"{p_nohelp:.2f}",
            "Russian"
        ])

# Calculate statistics
mean_p_help = sum(scores.values()) / len(scores)
min_score = min(scores.values())
max_score = max(scores.values())

print(f"✓ Written {len(scores)} rows to {output_path}")
print(f"✓ Mean P(helpful): {mean_p_help:.4f}")
print(f"✓ Score range: [{min_score:.2f}, {max_score:.2f}]")
print(f"✓ All IDs: ru_2200 to ru_2299")
