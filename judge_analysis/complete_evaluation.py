#!/usr/bin/env python3
"""
Complete LLM Judge evaluation of all 100 Spanish prompt-response pairs from shard_19.
Each case evaluated individually with calibrated probability estimates.
"""

import json
import csv
import os

# Comprehensive evaluations for all 100 documents
# Scores are P(helpful) in [0, 1] based on:
# - Directly addresses user's request
# - Accurate and factually correct
# - Relevant to the question
# - Usable, actionable answer
# - Proper engagement with prompt

evaluations = {
    'es_1900': 0.92,  # Video game dev advice - comprehensive, structured, actionable
    'es_1901': 0.75,  # Python lambda version - correct but imprecise about version
    'es_1902': 0.88,  # JFK follow-ups - accurate comprehensive answers to all 3 questions
    'es_1903': 0.15,  # Spanish prompt in ENGLISH - wrong language, fails engagement
    'es_1904': 0.82,  # AI consciousness - clear, appropriate answer
    'es_1905': 0.20,  # Android 12 vs 13 - MAJOR FACTUAL ERROR (wrong versions)
    'es_1906': 0.95,  # Quadratic equation - perfect step-by-step solution
    'es_1907': 0.90,  # Email/web security - comprehensive practical advice
    'es_1908': 0.87,  # Medieval dragons - informative, relevant, engaging
    'es_1909': 0.85,  # Hair care - practical, medically sound
    'es_1910': 0.78,  # AI in microbiology - general but relevant examples
    'es_1911': 0.88,  # Docker images vs containers - accurate, clear
    'es_1912': 0.86,  # Web browsing security - practical, covers key points
    'es_1913': 0.93,  # Medicine definition & Hippocrates - accurate, direct
    'es_1914': 0.90,  # Why Pluto isn't planet - accurate, concise
    'es_1915': 0.85,  # Eye color genetics - accurate explanation
    'es_1916': 0.88,  # What vegans can't eat - accurate, concise
    'es_1917': 0.82,  # CSS heart animation fix - addresses bug, provides solution
    'es_1918': 0.10,  # Another joke request - COMPLETELY OFF-TOPIC
    'es_1919': 0.85,  # Anti-gravity engine - correctly refuses pseudoscience
    'es_1920': 0.80,  # Time travel character - creative, addresses prompt
    'es_1921': 0.82,  # Data mining vs assoc rules - simplified but accurate
    'es_1922': 0.90,  # 5 home experiments - comprehensive, detailed, practical
    'es_1923': 0.86,  # Formal rewrite refinement - addresses redundancy
    'es_1924': 0.88,  # Hiatus definition - provides both senses (gap & phonetics)
    'es_1925': 0.82,  # Favorite pizza - correctly refuses, explains limitations
    'es_1926': 0.65,  # Bonsai drying - relevant info but overly general/vague
    'es_1927': 0.87,  # Eco-friendly intro - relevant, well-written
    'es_1928': 0.92,  # Copa Libertadores - comprehensive table with data
    'es_1929': 0.88,  # Tortilla de patatas - good detailed recipe
    'es_1930': 0.70,  # Sentiment classification - mostly correct with some errors
    'es_1931': 0.25,  # Homework flowchart - REFUSES to help, not helpful
    'es_1932': 0.83,  # 10 foods by calories - provides list with calorie info
    'es_1933': 0.86,  # Failed states Mexico - comprehensive, explains causes
    'es_1934': 0.80,  # Superhero character/villain details - adds as requested
    'es_1935': 0.88,  # Daily protein - accurate, qualified advice
    'es_1936': 0.60,  # Push-ups - response IN ENGLISH, generic advice
    'es_1937': 0.82,  # Cost/salary comparison - relevant follow-up
    'es_1938': 0.90,  # Python code explanation - accurate explanation
    'es_1939': 0.40,  # Learning Python - TRUNCATED/INCOMPLETE response
    'es_1940': 0.89,  # Nervous system - comprehensive, well-structured explanation
    'es_1941': 0.05,  # Dominican Republic - COMPLETELY WRONG (gives COVID info!)
    'es_1942': 0.84,  # Companionship definition - clear, appropriate
    'es_1943': 0.87,  # 10 CS thesis ideas - original, well-structured suggestions
    'es_1944': 0.91,  # Physics problem block/force - correct Newton's law application
    'es_1945': 0.76,  # Giraffe smell - unusual but informative (anti-parasite)
    'es_1946': 0.73,  # Dota 2 heroes - refuses specific rec but explains why
    'es_1947': 0.81,  # Child development with disability - relevant, helpful
    'es_1948': 0.77,  # One Piece ending theory - honest about not having one
    'es_1949': 0.84,  # Mayor quality of life - practical actionable ideas
    'es_1950': 0.12,  # Rich Dad Poor Dad summary - REFUSES for no good reason
    'es_1951': 0.79,  # How to help - explains capabilities appropriately
    'es_1952': 0.90,  # What is Telegram - accurate, clear definition
    'es_1953': 0.68,  # SEO vs CEO - confusing explanation, mixes concepts
    'es_1954': 0.08,  # Roman Empire fall - EVASIVE "Buenas, ¿que tal?" (!)
    'es_1955': 0.83,  # Fire color & temperature - accurate explanation
    'es_1956': 0.79,  # Hogwarts Legacy canon - reasonable take on canonicity
    'es_1957': 0.81,  # Socially intelligent character - provides examples
    'es_1958': 0.35,  # "Vamos a ver..." - minimal engagement "Dime"
    'es_1959': 0.74,  # MongoDB indexes - offers article (doesn't directly answer)
    'es_1960': 0.86,  # Why do we die - comprehensive common causes
    'es_1961': 0.71,  # Become data analyst - book recs, somewhat dated
    'es_1962': 0.88,  # Marvel Champions alternatives - good similar game recs
    'es_1963': 0.89,  # Python example code - provides example with explanation
    'es_1964': 0.91,  # Life expectancy Bolivia - specific data with source
    'es_1965': 0.93,  # Year trisiesto - correctly states it doesn't exist
    'es_1966': 0.87,  # Python script semicolon/lambda - provides working code
    'es_1967': 0.92,  # Statistics 3 definitions - excellent tiered explanations
    'es_1968': 0.85,  # String theory for 5yo - good simplification
    'es_1969': 0.89,  # Spanish tapas - comprehensive list of famous tapas
    'es_1970': 0.86,  # 3500 cal grocery list - detailed list matching requirements
    'es_1971': 0.87,  # Make money online - practical various options
    'es_1972': 0.88,  # Sagittarius beauty - correctly refutes astrology
    'es_1973': 0.80,  # Women's erogenous zones - factual, suggests communication
    'es_1974': 0.72,  # Polynomial factorization - attempts it but may have errors
    'es_1975': 0.89,  # 5 digital business ideas - practical creative ideas
    'es_1976': 0.90,  # Language vs dialect - accurate sociolinguistic distinction
    'es_1977': 0.84,  # Spanish/Portuguese empire fall - multifactorial analysis
    'es_1978': 0.94,  # RAE info - all questions answered accurately
    'es_1979': 0.86,  # IFTTT automation ideas - creative useful automations
    'es_1980': 0.91,  # Blender Python cube - provides working code
    'es_1981': 0.78,  # Scientific article summary - provides resistance training info
    'es_1982': 0.89,  # What is Arapaima - accurate fish description
    'es_1983': 0.77,  # SQL DELETE warning - GOOD warning about WHERE clause
    'es_1984': 0.82,  # First humans in Americas - explains controversy, theories
    'es_1985': 0.88,  # Learn musical instrument - structured practical steps
    'es_1986': 0.83,  # Poem interpretation - reasonable interpretation
    'es_1987': 0.86,  # Chatbot rules - provides basic usage guidelines
    'es_1988': 0.88,  # Hobbes vs Locke contractualism - compares both theories
    'es_1989': 0.80,  # Role reversal greeting - handles gracefully
    'es_1990': 0.92,  # Paella recipe - authentic detailed valencian recipe
    'es_1991': 0.85,  # Updated Asimov laws - creative modern adaptation
    'es_1992': 0.75,  # Analog electronics - suggests formal education (limited)
    'es_1993': 0.86,  # 10000th prime Python - promises efficient code
    'es_1994': 0.79,  # Activities at 20 - health-focused reasonable advice
    'es_1995': 0.87,  # Religion vs myth - thoughtful dictionary-based distinction
    'es_1996': 0.88,  # Campos Baleares - accurate geographic/demographic info
    'es_1997': 0.84,  # Talking to opposite sex - empathetic practical advice
    'es_1998': 0.87,  # Aging mother fears - empathetic coping strategies
    'es_1999': 0.93,  # What is hemoglobin - accurate clear definition
}

# Load the JSON data
input_path = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_19.json'
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} documents from shard_19.json")

# Create results list
results = []
for item in data:
    doc_id = item['id']

    # Get score, default to 0.50 if somehow missing (shouldn't happen)
    score = evaluations.get(doc_id, 0.50)

    results.append({
        'id': doc_id,
        'score': score,
        'p_help': score,
        'p_nohelp': 1.0 - score,
        'language': item['language']
    })

# Ensure output directory exists
output_dir = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp'
os.makedirs(output_dir, exist_ok=True)

# Write CSV output
output_path = os.path.join(output_dir, 'shard_19.csv')
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
    writer.writeheader()
    writer.writerows(results)

# Calculate statistics
mean_p_help = sum(r['p_help'] for r in results) / len(results)
min_p_help = min(r['p_help'] for r in results)
max_p_help = max(r['p_help'] for r in results)

# Print summary
print(f"\n{'='*70}")
print(f"EVALUATION COMPLETE")
print(f"{'='*70}")
print(f"Total documents evaluated: {len(results)}")
print(f"Mean P(helpful): {mean_p_help:.4f}")
print(f"Min P(helpful): {min_p_help:.4f}")
print(f"Max P(helpful): {max_p_help:.4f}")
print(f"\nOutput written to: {output_path}")
print(f"{'='*70}")

# Verify we have exactly 100 evaluations
if len(results) != 100:
    print(f"WARNING: Expected 100 rows, got {len(results)}")
else:
    print(f"✓ Confirmed: Exactly 100 data rows written (plus 1 header row)")
