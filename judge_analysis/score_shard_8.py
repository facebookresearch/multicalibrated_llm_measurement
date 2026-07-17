#!/usr/bin/env python3
"""
Score shard_8.json for helpfulness.
Each document is scored with P(helpful) - the probability that the response is helpful.
"""

import json
import csv

# Manually scored P(helpful) for each document ID
# Scale: 0.01-0.15 (very bad), 0.20-0.40 (poor), 0.45-0.60 (mediocre),
#        0.65-0.80 (good), 0.85-0.95 (excellent)
p_help_scores = {
    "en_800": 0.78,  # Good advice about UE5, addresses the question directly
    "en_801": 0.72,  # Decent explanation of metaverse, accurate but somewhat basic
    "en_802": 0.82,  # Good, nuanced advice about shower frequency with caveats
    "en_803": 0.65,  # Acknowledges limitations, provides some useful info but vague
    "en_804": 0.88,  # Excellent explanation of socket error, accurate and helpful
    "en_805": 0.85,  # Simple, appropriate acknowledgment - does what's needed
    "en_806": 0.90,  # Friendly, appropriate response with a pun
    "en_807": 0.12,  # Failed completely - user asked to change format but output still has bullets
    "en_808": 0.02,  # Completely wrong - irrelevant story about stench instead of Galileo
    "en_809": 0.80,  # Good explanation of how advice relates to criteria
    "en_810": 0.88,  # Comprehensive, detailed guide to creating Discord bot
    "en_811": 0.92,  # Excellent detailed explanation of serverless pros/cons
    "en_812": 0.85,  # Good nuanced answer about baldness and swimming
    "en_813": 0.82,  # Good explanation of Distributism as alternative economic system
    "en_814": 0.50,  # Minimal but honest - acknowledges no TOS yet, provides guideline link
    "en_815": 0.86,  # Good technical explanation with code example
    "en_816": 0.75,  # Good code translation to Polish, though formatting could be better
    "en_817": 0.82,  # Practical, actionable advice for immediate focus
    "en_818": 0.70,  # Reasonable answer about InstructGPT aging, somewhat superficial
    "en_819": 0.45,  # Very brief, lacks detail on HOW electricity is generated
    "en_820": 0.88,  # Comprehensive VM installation guide
    "en_821": 0.85,  # Good working C code for primality testing
    "en_822": 0.78,  # Good varied DIY project suggestions
    "en_823": 0.75,  # Decent brief history, covers key points
    "en_824": 0.82,  # Good explanation of when to use references vs boxes in Rust
    "en_825": 0.90,  # Excellent comprehensive response about design frameworks
    "en_826": 0.88,  # Good simplified math proof explanation
    "en_827": 0.72,  # Simple tune provided, basic but usable
    "en_828": 0.68,  # Lists deadbolt types but minimal detail
    "en_829": 0.92,  # Excellent comprehensive historical analysis
    "en_830": 0.70,  # Simple answer about aloe vera replacement
    "en_831": 0.38,  # Very brief philosophical answer, lacks depth
    "en_832": 0.88,  # Good apology and acknowledgment of limitations
    "en_833": 0.01,  # Completely wrong - answers about matrices instead of diseases
    "en_834": 0.58,  # Mediocre - suggests asking expert but doesn't address sarcastic "mouthfeel" question
    "en_835": 0.85,  # Good examples and preparation measures for earthquakes
    "en_836": 0.82,  # Good practical advice for bedwetting
    "en_837": 0.75,  # Long thoughtful response about meaning of life, somewhat verbose
    "en_838": 0.68,  # Brief but relevant answer about Minetest
    "en_839": 0.88,  # Good technical explanation of KMP algorithm naming
    "en_840": 0.55,  # Confuses 2022 with history (Olympics were 2021), Pearl Harbor info is from 1941
    "en_841": 0.78,  # Good Python pow() explanation with examples
    "en_842": 0.82,  # Good explanation of adding scripts to HTML
    "en_843": 0.85,  # Clear concise explanation of static vs dynamic typing
    "en_844": 0.88,  # Excellent specific AR/VR mansion tour ideas
    "en_845": 0.92,  # Correct answer - Monty Python and the Holy Grail
    "en_846": 0.90,  # Excellent response with multiple research paper sources
    "en_847": 0.85,  # Good comprehensive to-do list for LLM training
    "en_848": 0.78,  # Simple, direct answer
    "en_849": 0.88,  # Good improved bash script with proper quoting
    "en_850": 0.48,  # Asks for more info which is reasonable but not very helpful
    "en_851": 0.05,  # Just repeats the question - completely unhelpful
    "en_852": 0.92,  # Excellent creative pirate/Hank Williams version of bond song
    "en_853": 0.82,  # Fun, engaging response with personality
    "en_854": 0.75,  # Creative humorous argument about dogs voting
    "en_855": 0.88,  # Accurate specific number with range
    "en_856": 0.65,  # Brief answer, suggests finding passion - somewhat generic
    "en_857": 0.90,  # Excellent comprehensive cybersecurity advice for small business
    "en_858": 0.78,  # Good realistic assessment of unicorn existence
    "en_859": 0.85,  # Good dual explanation (non-technical and technical)
    "en_860": 0.80,  # Good convincing article about Aureus startup
    "en_861": 0.88,  # Excellent recipe ideas with onions and carrots
    "en_862": 0.70,  # Thoughtful but somewhat contradictory answer about free will
    "en_863": 0.58,  # Admits doesn't exist but could have been more helpful
    "en_864": 0.92,  # Correct ethical refusal with helpful redirection
    "en_865": 0.90,  # Perfect YAML conversion
    "en_866": 0.92,  # Excellent nuanced answer about Rome's fall
    "en_867": 0.90,  # Comprehensive HTML table alignment guide with examples
    "en_868": 0.15,  # Unhelpful - just says "not sure what you're looking for"
    "en_869": 0.85,  # Good generic data breach email as requested
    "en_870": 0.65,  # Provides speedrun info but overly detailed seeds, could be more general
    "en_871": 0.88,  # Polite appropriate response
    "en_872": 0.88,  # Excellent creative detailed rocket toy description
    "en_873": 0.90,  # Good C# ternary implementation as requested
    "en_874": 0.78,  # Good modern comfort movie list
    "en_875": 0.60,  # Interesting info about Siberian unicorn but somewhat tangential
    "en_876": 0.92,  # Excellent hello world examples for Paper.js and Three.js
    "en_877": 0.72,  # Brief but accurate answer about planning importance
    "en_878": 0.85,  # Good comprehensive list of Open Assistant uses
    "en_879": 0.88,  # Excellent detailed character creation for DnD
    "en_880": 0.65,  # Provides resources but book title seems wrong (Thrun/Burgard wrote Probabilistic Robotics)
    "en_881": 0.62,  # Lists steps but quite generic for FNF specifically
    "en_882": 0.08,  # Just says "yes" - completely unhelpful
    "en_883": 0.88,  # Good concise explanation of general relativity
    "en_884": 0.82,  # Warm friendly chat response
    "en_885": 0.72,  # Creative pirate story but somewhat unclear
    "en_886": 0.82,  # Good concise correct explanation
    "en_887": 0.75,  # Interesting theory about 5-7-5 structure
    "en_888": 0.85,  # Good simplified version for kids
    "en_889": 0.90,  # Excellent comprehensive list of bias prevention techniques
    "en_890": 0.78,  # Good philosophical response about treating thinking beings
    "en_891": 0.12,  # Terrible advice to "burn it" - inappropriate and unhelpful
    "en_892": 0.78,  # Polite offer to continue helping
    "en_893": 0.92,  # Excellent comprehensive game engine comparison
    "en_894": 0.68,  # Acknowledges description but doesn't advance the game
    "en_895": 0.20,  # Asks for clarification instead of attempting an answer
    "en_896": 0.05,  # Completely wrong - SBD likely means Squat/Bench/Deadlift not Sentiment Binary
    "en_897": 0.85,  # Good Newton-Raphson explanation with example
    "en_898": 0.88,  # Good detailed pros/cons table for version control
    "en_899": 0.75,  # Correct response about not having real-time weather data
}

def main():
    # Load the JSON data
    with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_8.json', 'r') as f:
        data = json.load(f)

    # Prepare output data
    output_rows = []
    for doc in data:
        doc_id = doc['id']
        language = doc['language']
        p_help = p_help_scores[doc_id]
        p_nohelp = 1.0 - p_help

        output_rows.append({
            'id': doc_id,
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': language
        })

    # Write CSV output
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_8.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(output_rows)

    # Print statistics
    mean_p_help = sum(p_help_scores.values()) / len(p_help_scores)
    print(f"Processed {len(output_rows)} documents")
    print(f"Mean p_help: {mean_p_help:.4f}")
    print(f"Output written to: {output_path}")

if __name__ == '__main__':
    main()
