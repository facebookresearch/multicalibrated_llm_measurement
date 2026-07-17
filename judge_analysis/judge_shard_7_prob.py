#!/usr/bin/env python3
"""
LLM Judge for estimating P(helpful) on assistant responses.
A response is HELPFUL if it directly addresses the request, is accurate, relevant,
and gives a usable answer; NOT helpful if off-topic, evasive, factually wrong,
uselessly incomplete, or fails to engage.

This version outputs calibrated probabilities in [0,1], not binary yes/no.
"""

import json
import csv
import os


def estimate_p_helpful(doc_id, language, prompt, response):
    """
    Estimate P(helpful) for a single document.
    Returns a calibrated probability in [0, 1].

    I will carefully consider each response and estimate a probability,
    using the full range [0,1], not just 0 or 1.
    """

    # Document-specific probability estimates based on careful analysis
    # Higher values (0.9-1.0) = clearly helpful
    # Mid-high (0.7-0.89) = helpful with minor issues
    # Middle (0.4-0.69) = mixed / partially helpful
    # Low (0.1-0.39) = mostly unhelpful with some value
    # Very low (0.0-0.09) = clearly unhelpful

    probabilities = {
        "en_700": 0.92,  # Comprehensive caffeine info for Coca-Cola products - accurate and detailed
        "en_701": 0.75,  # Mentions relevant field but "Climate and Environmental Psychology" isn't standard terminology
        "en_702": 0.91,  # Good ~100 word summary as requested, covers key points accurately
        "en_703": 0.93,  # Clear props explanation with code examples - directly helpful
        "en_704": 0.20,  # Trivial response that just restates what's in the question
        "en_705": 0.38,  # Has correct synonyms but terrible repetition ("reject" itself listed, many duplicates)
        "en_706": 0.05,  # Completely off-topic: asked about depression, got SaaS ideas
        "en_707": 0.89,  # Good Python web framework recommendations with appropriate caveats
        "en_708": 0.35,  # Incomplete tree, doesn't cover all Wikipedia disciplines as requested
        "en_709": 0.94,  # Excellent: 3 relevant seminal papers on scaling laws
        "en_710": 0.91,  # Accurate explanation of blue rarity (anthocyanins, structural color)
        "en_711": 0.03,  # Completely off-topic: asked about phase modulation, got Unreal Engine
        "en_712": 0.87,  # Identifies 3 reasonable causes for Gradle wrapper error with solutions
        "en_713": 0.86,  # Direct, actionable answer about contributing without coding
        "en_714": 0.15,  # Incorrect formula: E(g(X)) ≠ ∫g(x)dx; misunderstands derivative question
        "en_715": 0.83,  # Thoughtful response about quantum computing impact with realistic timeline
        "en_716": 0.84,  # Explains OCR correctly for scanned documents
        "en_717": 0.86,  # Provides relevant script structure with cooperation themes as requested
        "en_718": 0.88,  # Good recommendation (Stable Diffusion) with practical guidance
        "en_719": 0.95,  # Excellent: accurate altitude dangers, hypoxia, Armstrong limit details
        "en_720": 0.78,  # Provides price with caveats (price may be outdated but structure good)
        "en_721": 0.89,  # Sincere apology template with good advice on listening
        "en_722": 0.90,  # Clear answer: NASA website and ApJ paper
        "en_723": 0.91,  # Accurate explanation of Arab Rashidun conquest and context
        "en_724": 0.93,  # Comprehensive answers to all 5 rhythm game questions
        "en_725": 0.94,  # Accurate, comprehensive Simón Bolívar biography
        "en_726": 0.92,  # Correct chain: Jill Biden → Hammonton NJ → Trenton capital → pop
        "en_727": 0.94,  # Correct C++03 template metaprogramming solution with clear explanation
        "en_728": 0.82,  # Thoughtful solutions to AI apologizing problem
        "en_729": 0.67,  # Offers workaround (calculate if given date) with humor about arithmetic
        "en_730": 0.95,  # Complete, detailed feijoada recipe - highly usable
        "en_731": 0.90,  # Accurate Java and Maven description with good detail level
        "en_732": 0.52,  # Provides 3D rotation matrices when 2D was requested; partially relevant
        "en_733": 0.89,  # Successfully explains quantum tunneling in caveman speak
        "en_734": 0.93,  # Direct answer: 0 caffeine, with contamination caveat
        "en_735": 0.91,  # Good explanation that evil person can be superficially nice
        "en_736": 0.91,  # Correct Haskell monad definition with code example
        "en_737": 0.02,  # Completely wrong: talks about blockchain, not audio sidechaining
        "en_738": 0.48,  # Has good content but massive duplication (points 1-5 repeated twice)
        "en_739": 0.93,  # Successfully adapts speech to informal tone as requested
        "en_740": 0.88,  # Answers filament question with multiple options and quality notes
        "en_741": 0.10,  # Refuses user's feedback, insists towel is critical (obstinate)
        "en_742": 0.40,  # Response cut off mid-sentence, incomplete troubleshooting
        "en_743": 0.88,  # Provides step-by-step OS writing guide (assuming continuation from context)
        "en_744": 0.85,  # Reformats caffeine/alcohol answer as requested
        "en_745": 0.87,  # Lists relevant resume improvements: internships, volunteering, etc.
        "en_746": 0.18,  # Vague and poorly written ("it depends. some are good some are bad")
        "en_747": 0.82,  # Confirms Aseprite trial version exists
        "en_748": 0.84,  # Provides FTB launcher download link as alternative
        "en_749": 0.86,  # Accurate info about University of Minho
        "en_750": 0.81,  # Explains GPT-NeoX Discord bot setup (may have technical issues)
        "en_751": 0.85,  # Provides team ratio guidance with reasonable ranges
        "en_752": 0.90,  # Comprehensive explanation of Irenaeus' soul-making theodicy
        "en_753": 0.89,  # Good explanation of OpenAI vs Open Assistant openness
        "en_754": 0.79,  # Asks relevant follow-up questions (appropriate engagement)
        "en_755": 0.87,  # Provides 5 novel game ideas as requested
        "en_756": 0.90,  # Shows Flask basic setup with virtual environment correctly
        "en_757": 0.88,  # Answers ftplib questions about commands and file upload
        "en_758": 0.91,  # Successfully removes hand/arm/finger related values as requested
        "en_759": 0.84,  # Discusses access to justice barriers in Pakistan
        "en_760": 0.25,  # Asked for specific GAN paper abstract, got generic GAN description
        "en_761": 0.89,  # Provides translation of Icelandic proverb
        "en_762": 0.87,  # Provides simplified CO2-eq explanation as requested
        "en_763": 0.72,  # Asks clarifying question about skating (reasonable but adds no info)
        "en_764": 0.86,  # Addresses copyright risk in AI code generation thoughtfully
        "en_765": 0.92,  # Explains different types of infinity in mathematics well
        "en_766": 0.88,  # Correctly refuses to emulate espionage (ethical response)
        "en_767": 0.87,  # Explains Google's indexing and search speed
        "en_768": 0.89,  # Lists popular lucid dreaming techniques
        "en_769": 0.93,  # Correct: Joe Biden is president (accurate for Feb 2023)
        "en_770": 0.08,  # Goes off on bizarre tangent about monsters attacking cabin
        "en_771": 0.76,  # Asks clarifying question about population vs area (helpful but adds no info)
        "en_772": 0.87,  # Confirms alarm works in Do Not Disturb mode
        "en_773": 0.91,  # Provides Australian tax table for 2022-23
        "en_774": 0.86,  # Outlines steps to use Stable Diffusion model
        "en_775": 0.74,  # States Open Assistant based on OPT (may be outdated info)
        "en_776": 0.83,  # Concise good vs evil explanation in ~100 words
        "en_777": 0.81,  # Reasonably declines 300 pages, offers outline (pragmatic)
        "en_778": 0.65,  # Simple acknowledgment (minimal but appropriate follow-up)
        "en_779": 0.87,  # Re-explains labor theory of value as requested
        "en_780": 0.90,  # Lists common Tinder abbreviations accurately
        "en_781": 0.88,  # Explains labor theory of value with appropriate nuance
        "en_782": 0.89,  # Explains why goalkeeper goals are rare
        "en_783": 0.82,  # Explains AI doesn't retain history for privacy (reasonable but may not be complete)
        "en_784": 0.90,  # Describes CFOP/Fridrich method for Rubik's cube
        "en_785": 0.84,  # Provides approach for Kivy particle explosion (may need more detail)
        "en_786": 0.88,  # Explains GPU transistor count increase since GeForce 256
        "en_787": 0.86,  # Identifies likely cause of SQL parameter error
        "en_788": 0.91,  # Shows C# LINQ sorting and unique filtering correctly
        "en_789": 0.85,  # Explains historical causes of antisemitism (sensitive topic, handled reasonably)
        "en_790": 0.80,  # Confirms strategies viable in Competitive mode (brief but direct)
        "en_791": 0.87,  # Provides tips for building consistency and discipline
        "en_792": 0.12,  # Nonsensical response ("Bob is not to he trusted because scammer")
        "en_793": 0.82,  # Lists 20 Latin American countries (Puerto Rico is debatable)
        "en_794": 0.89,  # Explains Elasticsearch snapshot creation process
        "en_795": 0.87,  # Explains Earth formation start as dust aggregation
        "en_796": 0.22,  # Incorrect: Ninja wasn't most popular streamer in Feb 2023
        "en_797": 0.79,  # Mentions GitLab, encourages experimentation (brief but helpful)
        "en_798": 0.88,  # Lists YouTube competitors appropriately
        "en_799": 0.92,  # Provides exact VScode regex find/replace steps
    }

    if doc_id in probabilities:
        return probabilities[doc_id]
    else:
        # Default to middle probability for unknown documents
        print(f"Warning: No probability estimate for {doc_id}, using 0.50")
        return 0.50


def main():
    input_file = "/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_7.json"
    output_file = "/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_7.csv"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} documents from {input_file}")

    # Process each document
    results = []
    p_help_sum = 0.0

    for doc in documents:
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Estimate P(helpful)
        p_help = estimate_p_helpful(doc_id, language, prompt, response)
        p_nohelp = 1.0 - p_help

        results.append({
            'id': doc_id,
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': language
        })

        p_help_sum += p_help

    # Calculate mean
    mean_p_help = p_help_sum / len(results)

    # Write to CSV with exact header format requested
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {output_file}")
    print(f"\nMean P(helpful): {mean_p_help:.4f}")
    print(f"Total documents: {len(results)}")

    # Verify row count
    assert len(results) == 100, f"Expected 100 rows, got {len(results)}"
    print(f"✓ Confirmed 100 data rows")


if __name__ == "__main__":
    main()
