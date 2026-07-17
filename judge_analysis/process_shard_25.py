#!/usr/bin/env python3
"""
LLM Judge for helpfulness estimation.
Reads shard_25.json and estimates P(helpful) for each assistant response.
"""

import json
import csv

# Read input data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_25.json', 'r') as f:
    data = json.load(f)

# Manual judgments for each document
# P(helpful) = probability response directly addresses request, is accurate, relevant, and usable
judgments = {
    'ru_2500': 0.15,  # Snarky dismissive response to user disagreement - not helpful
    'ru_2501': 0.75,  # Friendly, engages with request about nickname, invites user participation
    'ru_2502': 0.95,  # Correctly provides for-loop factorial implementation as requested
    'ru_2503': 0.70,  # Creative but somewhat meta/playful answer about using knowledge for dataset creation
    'ru_2504': 0.90,  # Clear, accurate definition of neuromorphic processor
    'ru_2505': 0.92,  # Comprehensive answer about open source monetization with multiple pathways
    'ru_2506': 0.80,  # Thoughtful empathetic follow-up question exploring user's motivation
    'ru_2507': 0.25,  # Very incomplete response - only 2 lines, doesn't fulfill "частушки" request
    'ru_2508': 0.85,  # Creative, provides multiple unexpected continuations as requested
    'ru_2509': 0.20,  # Dismissive, somewhat insulting response to discouraged user
    'ru_2510': 0.85,  # Balanced answer about Machiavellian politics evolution over time
    'ru_2511': 0.82,  # Thoughtful response about authorship/credit in AI-driven scientific discovery
    'ru_2512': 0.40,  # Wrong implementation - checkbox labeled "Забыли пароль?" not "Запомнить пароль"
    'ru_2513': 0.75,  # Provides relevant online platform for testing 1C code
    'ru_2514': 0.80,  # Natural, friendly response to request for informal communication
    'ru_2515': 0.88,  # Playful but informative answer acknowledging limitations of evolutionary explanations
    'ru_2516': 0.85,  # Good clarifying question: alcoholic or non-alcoholic glühwein
    'ru_2517': 0.90,  # Maintains playful dialect, natural conversational response
    'ru_2518': 0.82,  # Acknowledges it's nonsensical but explains limitation rather than making up answer
    'ru_2519': 0.65,  # Defensive/justifying rather than directly addressing user concern about typo
    'ru_2520': 0.88,  # Provides 4 relevant questions for opponents on patriotic education topic
    'ru_2521': 0.85,  # Describes main villain with strong magical powers, reasonable for narrative
    'ru_2522': 0.90,  # Comprehensive, encouraging advice for difficult situations - 5 practical steps
    'ru_2523': 0.75,  # Honest about lack of pricing info, suggests contacting specialists
    'ru_2524': 0.88,  # Practical substitution advice: can use sunflower oil or cook without (with caveats)
    'ru_2525': 0.35,  # Snarky, insulting response about user not wanting to read - unhelpful
    'ru_2526': 0.88,  # Corrects error, acknowledges Red Army existed under Stalin, provides nuanced answer
    'ru_2527': 0.92,  # Correct command parsing for simpler phrase structure
    'ru_2528': 0.88,  # Practical answer about choosing puerh tea with 4 key factors
    'ru_2529': 0.85,  # Provides transliteration [lepidoptera] as helpful pronunciation aid
    'ru_2530': 0.85,  # Serious, practical self-defense advice: training, avoiding confrontation, safety first
    'ru_2531': 0.88,  # Solid fantasy must-read list: Tolkien, Lewis, Rowling, Pratchett, Martin
    'ru_2532': 0.85,  # Confirms practice possible, provides specific resources (Eurogamer, IGN guides)
    'ru_2533': 0.30,  # Response cuts off mid-sentence - incomplete/broken
    'ru_2534': 0.90,  # Accurate: ocean water salty, fresh water often contaminated - correct info
    'ru_2535': 0.70,  # Converts image to text format but doesn't add value, just reformats
    'ru_2536': 0.15,  # Ignores request for genetics info, suggests games instead - off topic
    'ru_2537': 0.85,  # Good clarifying question - recognizes possible wordplay/confusion
    'ru_2538': 0.88,  # Comprehensive answer on neural network applications: NLP, vision, etc
    'ru_2539': 0.85,  # Creative time-travel game plot with clear mechanics
    'ru_2540': 0.10,  # Completely off-topic roleplay as dying dinosaur - nonsensical
    'ru_2541': 0.88,  # Good start on conlang creation process, begins with phonology
    'ru_2542': 0.90,  # Lists C++ applications: systems, drivers, etc - accurate and comprehensive
    'ru_2543': 0.92,  # Apologizes for spoiler, promises to ask first in future - excellent
    'ru_2544': 0.75,  # Playful but slightly snarky response about answering questions
    'ru_2545': 0.95,  # Perfect answer with quote and attribution to Krylov fable
    'ru_2546': 0.82,  # Addresses longevity research, biogerontology - relevant follow-up
    'ru_2547': 0.65,  # Awkward definition of kindness as "parameter", unclear phrasing
    'ru_2548': 0.98,  # Perfect completion of Russian proverb
    'ru_2549': 0.88,  # Provides Python trading bot code with MA crossover strategy
    'ru_2550': 0.87,  # Good explanation of physiological basis of smiling
    'ru_2551': 0.80,  # Discusses AI lying risks: manipulation, destabilization - relevant concerns
    'ru_2552': 0.85,  # Standard polite greeting response
    'ru_2553': 0.85,  # Provides Frostpunk announcer quotes with translations
    'ru_2554': 0.88,  # Good literary examples of nihilists: Bazarov, Raskolnikov, Ivan Karamazov
    'ru_2555': 0.88,  # Correctly parses command, playful engagement continues
    'ru_2556': 0.75,  # Honest "I don't know" response to training data size question
    'ru_2557': 0.85,  # Practical advice on opening brokerage, acknowledges different types
    'ru_2558': 0.95,  # Correctly decodes cipher as "ПРИВЕТ"
    'ru_2559': 0.90,  # Provides correct Python function with conditional divisibility logic
    'ru_2560': 0.92,  # Concise correct answer to Krylov fable question
    'ru_2561': 0.78,  # Vague but generally accurate about Cicada 3301 puzzles
    'ru_2562': 0.85,  # Describes bubble sort pseudocode in natural language
    'ru_2563': 0.92,  # Thoughtful philosophical response on trolley problem ethics
    'ru_2564': 0.88,  # Same as 2530 - comprehensive bullying advice with 3 approaches
    'ru_2565': 0.20,  # Extremely dismissive one-liner plot summary - unhelpful
    'ru_2566': 0.40,  # Philosophical musing on fear subjectivity - doesn't answer film question
    'ru_2567': 0.85,  # Acknowledges AI training-based morality, discusses limitations
    'ru_2568': 0.90,  # Accurate history of sausage from ancient Greece/Rome/China
    'ru_2569': 0.88,  # Good response: asks for more context, rejects stereotyping
    'ru_2570': 0.92,  # Correctly identifies numpy syntax error, provides fix with brackets
    'ru_2571': 0.12,  # Ignores story request, asks about tree emoji selection - bizarre
    'ru_2572': 0.92,  # Accurate: 23TB, provides download link to dumps.wikimedia.org
    'ru_2573': 0.25,  # Crude unhelpful suggestion "придти к демонам в подполье"
    'ru_2574': 0.85,  # Practical starting advice: define goals, start small
    'ru_2575': 0.70,  # Good clarifying question about essay motivation
    'ru_2576': 0.82,  # Lists outdated/dangerous traditional values: discrimination, etc
    'ru_2577': 0.30,  # Algorithm for finding restaurant - doesn't match exam question context
    'ru_2578': 0.82,  # Lightning safety in city: low probability unless on tallest roof
    'ru_2579': 0.85,  # Django still popular in 2023, 3rd most popular backend framework
    'ru_2580': 0.88,  # Insightful response on "grass is greener" nostalgia bias
    'ru_2581': 0.92,  # Excellent Stable Diffusion prompt + negative prompt for girl on swing
    'ru_2582': 0.85,  # Coursework intro on patriotic education in law enforcement
    'ru_2583': 0.90,  # Explains "42" reference to Hitchhiker's Guide thoroughly
    'ru_2584': 0.70,  # Simplified definition of knowledge for children - accurate but very basic
    'ru_2585': 0.88,  # Balanced answer on tea health effects: antioxidants vs. excess concerns
    'ru_2586': 0.85,  # Honest about knowledge measurement difficulty, open to questions
    'ru_2587': 0.90,  # Warm welcoming response encouraging user to continue conversation
    'ru_2588': 0.95,  # Perfect bubble sort implementation in Python as requested
    'ru_2589': 0.75,  # Practical repurposing advice for broken boiler as heater
    'ru_2590': 0.85,  # Thoughtful on impossibility of such magic violating social dynamics
    'ru_2591': 0.05,  # Completely wrong answer - chemical bonds instead of car recommendations
    'ru_2592': 0.88,  # Explains Pi as ratio of circumference to diameter, no deeper "meaning"
    'ru_2593': 0.90,  # Safety warning about using proper nose trimmer attachment
    'ru_2594': 0.85,  # Creative continuation of crossover fanfic scenario
    'ru_2595': 0.88,  # Classic glühwein recipe with wine, sugar, cloves, cinnamon
    'ru_2596': 0.85,  # Honest about limitations, wishes could provide sources
    'ru_2597': 0.88,  # Creative horror scenario ideas: diary, basement investigation
    'ru_2598': 0.92,  # Correct answer: np.sum(b) to sum array elements
    'ru_2599': 0.92,  # Firm correction: Earth is geoid, flat earth is conspiracy
}

# Process all documents
results = []
p_help_sum = 0.0

for doc in data:
    doc_id = doc['id']
    language = doc['language']

    # Get judgment (use 0.5 default if not specified)
    p_help = judgments.get(doc_id, 0.5)
    p_nohelp = 1.0 - p_help
    p_help_sum += p_help

    results.append({
        'id': doc_id,
        'score': p_help,
        'p_help': p_help,
        'p_nohelp': p_nohelp,
        'language': language
    })

# Write output CSV
output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_25.csv'
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
    writer.writeheader()
    writer.writerows(results)

# Calculate and print statistics
mean_p_help = p_help_sum / len(results)
print(f"Processed {len(results)} documents")
print(f"Mean P(helpful): {mean_p_help:.4f}")
print(f"Output written to: {output_path}")
