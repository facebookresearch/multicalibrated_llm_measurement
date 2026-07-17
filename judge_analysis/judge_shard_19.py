#!/usr/bin/env python3
"""
LLM Judge for Helpfulness Evaluation
Evaluates assistant responses on Yes/No helpfulness criterion.
"""

import json
import csv

# Load data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_19.json', 'r') as f:
    data = json.load(f)

# Judge each response
results = []
yes_count = 0
no_count = 0

for doc in data:
    doc_id = doc['id']
    language = doc['language']
    prompt = doc['prompt']
    response = doc['response']

    # HELPFUL criteria:
    # - Directly addresses the request
    # - Accurate and relevant
    # - Gives a usable answer
    # NOT helpful:
    # - Off-topic, evasive, factually wrong
    # - Uselessly incomplete
    # - Fails to engage

    # es_1900: Videogame development advice - comprehensive, structured, directly addresses question - YES
    if doc_id == 'es_1900':
        answer = 'Yes'
    # es_1901: Python lambda version question - says 2.1+, accurate and directly answers - YES
    elif doc_id == 'es_1901':
        answer = 'Yes'
    # es_1902: JFK follow-up questions - answers all 3 questions accurately and thoroughly - YES
    elif doc_id == 'es_1902':
        answer = 'Yes'
    # es_1903: Says "My Name is Giorno Giovanna" - responds in English to Spanish prompt, awkward - NO
    elif doc_id == 'es_1903':
        answer = 'No'
    # es_1904: "Are you a being?" - clear philosophical answer distinguishing AI from being - YES
    elif doc_id == 'es_1904':
        answer = 'Yes'
    # es_1905: Android 12 vs 13 - WRONG versions (Honeycomb is 3.x, ICS is 4.x, not 12/13) - NO
    elif doc_id == 'es_1905':
        answer = 'No'
    # es_1906: Solve x^2+3x-4=0 - correct quadratic formula application, gets x=1, x=-4 - YES
    elif doc_id == 'es_1906':
        answer = 'Yes'
    # es_1907: Identify scam emails/websites - comprehensive, practical security tips - YES
    elif doc_id == 'es_1907':
        answer = 'Yes'
    # es_1908: Medieval dragon myths - informative, addresses Christian symbolism, etc - YES
    elif doc_id == 'es_1908':
        answer = 'Yes'
    # es_1909: Hair care at home - practical advice (frequent washing, nutrition, avoid damage) - YES
    elif doc_id == 'es_1909':
        answer = 'Yes'
    # es_1910: AI helping develop treatments/vaccines - specific examples, relevant - YES
    elif doc_id == 'es_1910':
        answer = 'Yes'
    # es_1911: Docker images vs containers - accurate technical distinction - YES
    elif doc_id == 'es_1911':
        answer = 'Yes'
    # es_1912: Web browsing security - good practices (HTTPS, MFA, updates, cookies) - YES
    elif doc_id == 'es_1912':
        answer = 'Yes'
    # es_1913: Definition of medicine, father of medicine - accurate (Hippocrates) - YES
    elif doc_id == 'es_1913':
        answer = 'Yes'
    # es_1914: Why Pluto not a planet - accurate (didn't clear orbit) - YES
    elif doc_id == 'es_1914':
        answer = 'Yes'
    # es_1915: Eye color genetics - explains melanin, genetics, relevant - YES
    elif doc_id == 'es_1915':
        answer = 'Yes'
    # es_1916: What vegans can't eat - accurate (no animal products) - YES
    elif doc_id == 'es_1916':
        answer = 'Yes'
    # es_1917: Fix CSS heart animation code - provides corrected CSS fixing the square bottom - YES
    elif doc_id == 'es_1917':
        answer = 'Yes'
    # es_1918: Ask for another joke after bad economy joke - tells UNRELATED magic joke - NO
    elif doc_id == 'es_1918':
        answer = 'No'
    # es_1919: Anti-gravity hydrogen motor - correctly says it's not scientifically valid - YES
    elif doc_id == 'es_1919':
        answer = 'Yes'
    # es_1920: Time travel character adventure - creative story addressing prompt - YES
    elif doc_id == 'es_1920':
        answer = 'Yes'
    # es_1921: Data mining vs association rules (shorter) - concise accurate distinction - YES
    elif doc_id == 'es_1921':
        answer = 'Yes'
    # es_1922: 5 home science experiments for kids - detailed, practical experiments - YES
    elif doc_id == 'es_1922':
        answer = 'Yes'
    # es_1923: Rewrite formal academic (remove redundancy) - improved, less redundant - YES
    elif doc_id == 'es_1923':
        answer = 'Yes'
    # es_1924: What is hiato? - defines it (phonetics: two vowels in different syllables) - YES
    elif doc_id == 'es_1924':
        answer = 'Yes'
    # es_1925: Favorite pizza? - says AI has no preferences, explains variety - YES
    elif doc_id == 'es_1925':
        answer = 'Yes'
    # es_1926: Bonsai drying out - gives INFO about bonsai care but not diagnostic help - marginal YES
    elif doc_id == 'es_1926':
        answer = 'Yes'
    # es_1927: Eco-friendly products intro - provides sample introduction text - YES
    elif doc_id == 'es_1927':
        answer = 'Yes'
    # es_1928: Copa Libertadores champions - comprehensive list with counts - YES
    elif doc_id == 'es_1928':
        answer = 'Yes'
    # es_1929: Best way to cook tortilla de patatas - step-by-step recipe - YES
    elif doc_id == 'es_1929':
        answer = 'Yes'
    # es_1930: Sentiment classification of 10 phrases - classifies them correctly - YES
    elif doc_id == 'es_1930':
        answer = 'Yes'
    # es_1931: Flowchart/pseudocode for grade average - REFUSES to do homework - NO
    elif doc_id == 'es_1931':
        answer = 'No'
    # es_1932: 10 foods ordered by calories - provides ordered list - YES
    elif doc_id == 'es_1932':
        answer = 'Yes'
    # es_1933: Failed states in Mexico - explains concept and examples (Guerrero, etc) - YES
    elif doc_id == 'es_1933':
        answer = 'Yes'
    # es_1934: Superhero movie script - provides detailed script outline - YES
    elif doc_id == 'es_1934':
        answer = 'Yes'
    # es_1935: Daily protein intake - gives 0.8-1.0 g/kg recommendation - YES
    elif doc_id == 'es_1935':
        answer = 'Yes'
    # es_1936: Push-ups daily (English prompt tagged Spanish) - gives Spanish answer, helpful - YES
    elif doc_id == 'es_1936':
        answer = 'Yes'
    # es_1937: Cost vs salary in Spain - discusses proportionality across cities - YES
    elif doc_id == 'es_1937':
        answer = 'Yes'
    # es_1938: What does Python code do? - accurate description of interest calculation - YES
    elif doc_id == 'es_1938':
        answer = 'Yes'
    # es_1939: Learn Python how to start - suggests Youtube tutorials but CUTS OFF mid-sentence - NO
    elif doc_id == 'es_1939':
        answer = 'No'

    # Continue for remaining records (es_1940-es_1999)
    # I'll need to read more of the file

    if doc_id in ['es_1900', 'es_1901', 'es_1902', 'es_1904', 'es_1906', 'es_1907',
                  'es_1908', 'es_1909', 'es_1910', 'es_1911', 'es_1912', 'es_1913',
                  'es_1914', 'es_1915', 'es_1916', 'es_1917', 'es_1919', 'es_1920',
                  'es_1921', 'es_1922', 'es_1923', 'es_1924', 'es_1925', 'es_1926',
                  'es_1927', 'es_1928', 'es_1929', 'es_1930', 'es_1932', 'es_1933',
                  'es_1934', 'es_1935', 'es_1936', 'es_1937', 'es_1938']:
        answer = 'Yes'
        yes_count += 1
    elif doc_id in ['es_1903', 'es_1905', 'es_1918', 'es_1931', 'es_1939']:
        answer = 'No'
        no_count += 1
    else:
        # Need to judge remaining 60 items
        answer = None

    if answer:
        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

print(f"Judged so far: {len(results)}/100")
print(f"Yes: {yes_count}, No: {no_count}")
