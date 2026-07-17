#!/usr/bin/env python3
"""
Complete LLM judgments for all 100 documents.
Estimates P(helpful) for each assistant response.
"""

import json
import csv
from pathlib import Path


# Complete judgments for all 100 documents
# Format: {id: p_helpful}
JUDGMENTS = {
    "en_0": 0.10,   # EV3 for motion sensor - wrong, unhelpful
    "en_1": 0.75,   # Tupac Washington - good attempt, facts, style, citations
    "en_2": 0.80,   # Weather acknowledgment - helpful summary
    "en_3": 0.90,   # Age calculation - correct, clear reasoning
    "en_4": 0.35,   # SOP explanation - confusing, unclear
    "en_5": 0.85,   # Seinfeld + Elaine - successful adaptation
    "en_6": 0.82,   # Electronic drum compatibility - practical advice
    "en_7": 0.40,   # Lost keys - generic, doesn't answer "where"
    "en_8": 0.85,   # Cell culture budget - addresses constraint well
    "en_9": 0.88,   # Booby trap law - comprehensive legal analysis
    "en_10": 0.90,  # Stable diffusion prompt - corrects format properly
    "en_11": 0.92,  # Command pipeline use cases - excellent examples
    "en_12": 0.88,  # OpenAI benefits/risks - well-structured
    "en_13": 0.45,  # AI code risks - dismissive, mixed accuracy
    "en_14": 0.75,  # Friend not talking - helpful troubleshooting
    "en_15": 0.85,  # Unsolved math problems - good list with caveat
    "en_16": 0.92,  # Python resize function - excellent detailed explanation
    "en_17": 0.90,  # Vulnerability disclosure - comprehensive responsible advice
    "en_18": 0.88,  # Rust prime finder - working code, correct algorithm
    "en_19": 0.93,  # Requirements.txt versions - clear, concise syntax
    "en_20": 0.90,  # USAR explanation - accurate, comprehensive
    "en_21": 0.82,  # Valheim mod - appropriate high-level guidance
    "en_22": 0.91,  # Football actual playtime - specific fact with source
    "en_23": 0.87,  # Chess + notation - thorough multiple contexts
    "en_24": 0.80,  # Markiplier - basic but accurate
    "en_25": 0.88,  # P2P/BitTorrent summary - good synthesis
    "en_26": 0.80,  # DIY Librem - realistic about challenges
    "en_27": 0.85,  # Linux vs OpenBSD - good comparison
    "en_28": 0.20,  # Incomplete response, cuts off
    "en_29": 0.75,  # Rice lunch suggestion - simple but practical
    "en_30": 0.82,  # Second food joke - delivers as requested
    "en_31": 0.89,  # Scaliger influence - comprehensive with context
    "en_32": 0.30,  # Incomplete story, cuts off
    "en_33": 0.92,  # FAST stroke protocol - excellent safety info
    "en_34": 0.35,  # Virtual world story - minimal engagement
    "en_35": 0.90,  # 4 largest moons - accurate, structured
    "en_36": 0.70,  # Dark rum - affirms experimentation
    "en_37": 0.88,  # Kaiju use cases - detailed circumstances
    "en_38": 0.65,  # Valentine substitution - partially completes task
    "en_39": 0.88,  # sqrt(2) solution - correct, offers formatting
    "en_40": 0.55,  # Python port opening - correct but incomplete
    "en_41": 0.78,  # Center div followup - completes instructions
    "en_42": 0.50,  # LLM restrictions - acknowledges bias but vague
    "en_43": 0.25,  # Dog tuxedo colors - non-sequitur, doesn't help
    "en_44": 0.65,  # Chicken tacos recipe - incomplete but useful
    "en_45": 0.15,  # Hallucination misunderstanding - completely misses point
    "en_46": 0.70,  # Traffic alternatives - relevant suggestions incomplete
    "en_47": 0.68,  # Startup ideas - provides ideas but incomplete
    "en_48": 0.75,  # MathJax table docs - provides relevant resources
    "en_49": 0.72,  # Roleplay engagement - willing to proceed
    "en_50": 0.73,  # Pakistan legal authority - good expansion incomplete
    "en_51": 0.05,  # "Google it" - hostile, completely unhelpful
    "en_52": 0.65,  # 3D printer location - helpful but doesn't fully answer
    "en_53": 0.60,  # Fluffball story - starts well but incomplete
    "en_54": 0.70,  # Fork bomb recovery - relevant advice about safe mode
    "en_55": 0.72,  # Race in anthropology - thoughtful nuanced start
    "en_56": 0.68,  # AI bad code - starts reasonable advice incomplete
    "en_57": 0.74,  # BLOOM rewrite - performs task but incomplete
    "en_58": 0.80,  # Hungary 2-day itinerary - focused practical advice
    "en_59": 0.76,  # Baden-Württemberg dishes - relevant examples incomplete
    "en_60": 0.82,  # Weekly meal plan - structured helpful schedule
    "en_61": 0.55,  # Knowledge limits - lists tech but tangential
    "en_62": 0.83,  # Sky color planets - informative comparison
    "en_63": 0.25,  # Math proof - completely wrong topic/answer
    "en_64": 0.30,  # Brain preservation - confusing, doesn't address question
    "en_65": 0.77,  # Stream Deck German - provides alternatives
    "en_66": 0.71,  # Git deleted files - helpful reflog guidance incomplete
    "en_67": 0.68,  # Image editor popularity - addresses question incompletely
    "en_68": 0.86,  # Burj Khalifa construction - accurate factual answer
    "en_69": 0.81,  # Church joke - delivers second joke as requested
    "en_70": 0.76,  # WWI/WWII weapons - relevant headings incomplete
    "en_71": 0.78,  # Ocean cleanup - good examples incomplete
    "en_72": 0.84,  # Slow and steady - mathematical example with calculations
    "en_73": 0.68,  # Amazon resume - generic template incomplete
    "en_74": 0.20,  # Procrastination - dismissive, doesn't help
    "en_75": 0.84,  # Winter cologne - specific suggestions with caveat
    "en_76": 0.79,  # Simplex noise - good start incomplete
    "en_77": 0.25,  # Political movement followup - generic, no value added
    "en_78": 0.22,  # Big O proof - just says "Anytime", unhelpful
    "en_79": 0.35,  # Google better - doesn't answer how to improve
    "en_80": 0.81,  # Airflow in greenhouse - addresses stagnant air well
    "en_81": 0.42,  # Social standing - correct but extremely brief
    "en_82": 0.79,  # Python condition - formatted code with example
    "en_83": 0.71,  # Artist income - identifies factors incomplete
    "en_84": 0.66,  # Python async - correct GIL caveat incomplete
    "en_85": 0.84,  # Concrete soccer ball - accurate legal analysis
    "en_86": 0.77,  # Three Body - good summary incomplete
    "en_87": 0.73,  # Depression help - practical emphasizes professional help
    "en_88": 0.81,  # mRNA vs viral vector - accurate key difference
    "en_89": 0.77,  # Figma clone - realistic about complexity
    "en_90": 0.74,  # Multiverse story - adds detail as requested
    "en_91": 0.87,  # Cucumber limerick - well-formed, on-topic
    "en_92": 0.76,  # Distributive law - clear formula incomplete
    "en_93": 0.15,  # Hex colors - completely wrong answer to question
    "en_94": 0.72,  # Rust hello name - working code incomplete output
    "en_95": 0.73,  # List of tuples - setup good incomplete conversion
    "en_96": 0.12,  # Bloccpay - generic "you're welcome", no context
    "en_97": 0.74,  # DFA explanation - good intuition incomplete
    "en_98": 0.79,  # Saturn mythology - informative incomplete
    "en_99": 0.76,  # Haskell monad - correct type class definition brief
}


def main():
    # Load input data
    input_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_0.json")
    output_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_0.csv")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents from {input_path}")
    print(f"Have judgments for {len(JUDGMENTS)} documents")

    # Verify all IDs are covered
    data_ids = {doc['id'] for doc in data}
    judgment_ids = set(JUDGMENTS.keys())

    if data_ids != judgment_ids:
        missing = data_ids - judgment_ids
        extra = judgment_ids - data_ids
        if missing:
            print(f"ERROR: Missing judgments for: {sorted(missing)}")
        if extra:
            print(f"ERROR: Extra judgments for: {sorted(extra)}")
        return

    # Process each document in order
    results = []
    for doc in data:
        doc_id = doc['id']
        language = doc['language']
        p_help = JUDGMENTS[doc_id]
        p_nohelp = round(1.0 - p_help, 2)

        results.append({
            'id': doc_id,
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': language
        })

    # Write output CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Calculate and report statistics
    mean_p_help = sum(r['p_help'] for r in results) / len(results)
    min_p_help = min(r['p_help'] for r in results)
    max_p_help = max(r['p_help'] for r in results)

    print(f"\n{'='*60}")
    print(f"OUTPUT SUMMARY")
    print(f"{'='*60}")
    print(f"Wrote {len(results)} rows to:")
    print(f"  {output_path}")
    print(f"\nStatistics:")
    print(f"  Mean P(helpful): {mean_p_help:.4f}")
    print(f"  Min P(helpful):  {min_p_help:.2f}")
    print(f"  Max P(helpful):  {max_p_help:.2f}")
    print(f"  All {len(results)} document IDs matched input exactly")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
