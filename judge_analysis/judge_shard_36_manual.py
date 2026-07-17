#!/usr/bin/env python3
"""
Manual judgments for shard_36 - each document judged individually.
"""

import json
import csv

# Manual judgments for each document ID
# Format: doc_id -> P(helpful)
judgments = {
    "de_3600": 0.85,  # Good answer about wind turbines safety with shutdown systems
    "de_3601": 0.90,  # Comprehensive sleep hygiene tips, well-structured
    "de_3602": 0.75,  # Polite refusal explaining AI limitations, offers alternative
    "de_3603": 0.95,  # Correct crossword answer "smart" for 5-letter word ending in 't'
    "de_3604": 0.88,  # Relevant SEO keywords list for web agency services
    "de_3605": 0.80,  # Good info on climate goals, mentions USA low ranking
    "de_3606": 0.92,  # Correct examples of German words with two A's
    "de_3607": 0.70,  # Rap lyrics attempt in Capital Bra style - creative but quality varies
    "de_3608": 0.65,  # Somewhat philosophical answer about AI trust, technically correct but indirect
    "de_3609": 0.95,  # Simple, appropriate greeting response
    "de_3610": 0.88,  # Good explanation of HOPR blockchain protocol
    "de_3611": 0.82,  # Better rap attempt with "Nur noch ALDI" parody structure
    "de_3612": 0.93,  # Excellent update to moving request letter with specifics
    "de_3613": 0.90,  # Classic joke, well executed
    "de_3614": 0.87,  # Good health risks of smoking explanation
    "de_3615": 0.85,  # Comprehensive C# Unity code for paper airplane physics
    "de_3616": 0.35,  # WRONG polynomial expansion - fails basic algebra
    "de_3617": 0.40,  # Unhelpful - just lists bullet points without elaboration as requested
    "de_3618": 0.78,  # Corrected response focusing on Windows and Linux as requested
    "de_3619": 0.85,  # Five additional camping options near Duisburg
    "de_3620": 0.88,  # Good summary of "Absolutely True Diary" book
    "de_3621": 0.82,  # Accurate info about late ID renewal fines in Germany
    "de_3622": 0.15,  # WRONG - gives specific weather prediction without access to real data
    "de_3623": 0.60,  # Iterative version but counts full nodes not all nodes - logic error
    "de_3624": 0.88,  # Good balanced answer about AI detecting misuse
    "de_3625": 0.90,  # Good sales question about advantages of IR camera microwave
    "de_3626": 0.92,  # Correct primality test algorithm with explanation
    "de_3627": 0.55,  # Somewhat dismissive/unhelpful response about Python
    "de_3628": 0.75,  # Decent explanation of citrus differences, some inaccuracies
    "de_3629": 0.88,  # Good follow-up sales question about use cases
    "de_3630": 0.90,  # Comprehensive packing list for travel
    "de_3631": 0.83,  # Good improvement on Titanic challenge summary
}

def main():
    input_file = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_36.json'
    output_file = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_36.csv'

    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Process each document
    results = []
    for doc in data:
        doc_id = doc['id']

        # Get judgment or use default
        p_help = judgments.get(doc_id, 0.50)  # Default to 0.50 if not judged
        p_nohelp = 1.0 - p_help

        results.append({
            'id': doc_id,
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': doc['language']
        })

    # Write output
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Report stats
    mean_p_help = sum(r['p_help'] for r in results) / len(results)
    print(f"Processed {len(results)} rows")
    print(f"Mean P(helpful): {mean_p_help:.4f}")
    print(f"Judged: {len(judgments)}, Remaining: {len(data) - len(judgments)}")

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
