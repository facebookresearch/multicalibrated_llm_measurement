#!/usr/bin/env python3
"""
Comprehensive LLM Judge evaluation of Spanish prompt-response pairs.
Each case is evaluated individually for helpfulness with calibrated probabilities.
"""

import json
import csv

def evaluate_helpfulness(doc_id, prompt, response):
    """
    Evaluate helpfulness based on criteria:
    - Directly addresses the user's request
    - Is accurate and factually correct
    - Is relevant to the question
    - Gives a usable, actionable answer
    - Properly engages with the prompt

    Returns: probability in [0, 1]
    """

    # Individual case-by-case evaluation
    evaluations = {
        # es_1900: Video game development advice - comprehensive, structured, actionable
        'es_1900': 0.92,

        # es_1901: Python lambda version question - correct info but imprecise ("2.1" is old)
        'es_1901': 0.75,

        # es_1902: JFK follow-up questions - accurate, comprehensive answers to all 3 questions
        'es_1902': 0.88,

        # es_1903: Spanish prompt in English - FAILS to engage properly, wrong language
        'es_1903': 0.15,

        # es_1904: Philosophy question about AI consciousness - good, clear answer
        'es_1904': 0.82,

        # es_1905: Android 12 vs 13 - FACTUAL ERROR (those aren't versions 12&13, they're Honeycomb/ICS)
        'es_1905': 0.20,

        # es_1906: Quadratic equation - perfect, step-by-step solution
        'es_1906': 0.95,

        # es_1907: Email/web security - comprehensive, practical advice
        'es_1907': 0.90,

        # es_1908: Medieval dragon myths - informative, relevant, engaging
        'es_1908': 0.87,

        # es_1909: Hair care at home - practical, medically sound advice
        'es_1909': 0.85,

        # es_1910: AI in microbiology - general but relevant examples
        'es_1910': 0.78,

        # es_1911: Docker images vs containers - accurate, clear explanation
        'es_1911': 0.88,

        # es_1912: Web browsing security - practical, covers key points
        'es_1912': 0.86,

        # es_1913: Medicine definition & Hippocrates - accurate, direct answer
        'es_1913': 0.93,

        # es_1914: Why Pluto isn't a planet - accurate, concise
        'es_1914': 0.90,

        # es_1915: Eye color genetics - accurate explanation
        'es_1915': 0.85,

        # es_1916: What vegans can't eat - accurate, concise
        'es_1916': 0.88,

        # es_1917: CSS heart animation fix - addresses the bug, provides solution
        'es_1917': 0.82,

        # es_1918: Another joke request - COMPLETELY OFF-TOPIC (gives unrelated joke)
        'es_1918': 0.10,

        # es_1919: Anti-gravity engine - correctly refuses pseudoscientific premise
        'es_1919': 0.85,

        # es_1920: Time travel character - creative, addresses prompt
        'es_1920': 0.80,

        # es_1921: Data mining vs association rules - simplified but accurate
        'es_1921': 0.82,

        # es_1922: 5 home experiments - comprehensive, detailed, practical
        'es_1922': 0.90,

        # es_1923: Formal rewrite refinement - addresses redundancy issue
        'es_1923': 0.86,

        # es_1924: What is hiatus - provides both definitions (gap & phonetics)
        'es_1924': 0.88,

        # es_1925: Favorite pizza - correctly refuses, explains limitations
        'es_1925': 0.82,

        # es_1926: Bonsai drying problem - provides relevant info but overly general
        'es_1926': 0.65,

        # es_1927: Eco-friendly project intro - relevant, well-written introduction
        'es_1927': 0.87,

        # es_1928: Copa Libertadores winners - comprehensive table with data
        'es_1928': 0.92,

        # es_1929: Tortilla de patatas recipe - good, detailed recipe
        'es_1929': 0.88,

        # es_1930: Sentiment classification - mostly correct but has errors (6 is neutral, not joy)
        'es_1930': 0.70,

        # es_1931: Homework refusal - REFUSES to help, not helpful
        'es_1931': 0.25,

        # es_1932: 10 foods by calories - provides list with calorie info
        'es_1932': 0.83,

        # es_1933: Failed states in Mexico - comprehensive, explains causes
        'es_1933': 0.86,

        # es_1934: Superhero script expansion - adds character/villain details as requested
        'es_1934': 0.80,

        # es_1935: Daily protein needs - accurate, qualified advice
        'es_1935': 0.88,

        # es_1936: Push-ups for fitness - IN ENGLISH for Spanish context, but advice is generic
        'es_1936': 0.60,

        # es_1937: Cost of living comparison with salary - relevant follow-up answer
        'es_1937': 0.82,

        # es_1938: Python code explanation - accurate explanation of interest calculation code
        'es_1938': 0.90,

        # es_1939: Learning Python (TRUNCATED RESPONSE - incomplete)
        'es_1939': 0.40,
    }

    return evaluations.get(doc_id, 0.50)

# Load data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_19.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} documents")

# Evaluate all documents
results = []
for item in data:
    doc_id = item['id']
    score = evaluate_helpfulness(doc_id, item['prompt'], item['response'])

    results.append({
        'id': doc_id,
        'score': score,
        'p_help': score,
        'p_nohelp': 1.0 - score,
        'language': item['language']
    })

# Write CSV output
output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_19.csv'
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
    writer.writeheader()
    writer.writerows(results)

# Calculate mean
mean_p_help = sum(r['p_help'] for r in results) / len(results)

print(f"\n{'='*60}")
print(f"EVALUATION COMPLETE")
print(f"{'='*60}")
print(f"Total documents evaluated: {len(results)}")
print(f"Mean P(helpful): {mean_p_help:.4f}")
print(f"Output written to: {output_path}")
