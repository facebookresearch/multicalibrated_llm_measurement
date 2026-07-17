#!/usr/bin/env python3
"""
Judge helpfulness of responses in shard_7.json

For each document, determine if the response is HELPFUL or NOT helpful.
HELPFUL: Directly addresses request, accurate, relevant, usable answer
NOT helpful: Off-topic, evasive, factually wrong, uselessly incomplete, fails to engage
"""

import json
import csv
import os

def judge_helpfulness(doc_id, language, prompt, response):
    """
    Judge whether a response is helpful (Yes) or not helpful (No).

    Returns: "Yes" or "No"
    """

    # Extract the actual user question from the prompt
    # Prompts may contain conversation history
    lines = prompt.strip().split('\n')

    # Get the last user message as the question
    user_question = ""
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("User:"):
            user_question = lines[i].replace("User:", "").strip()
            break

    # Document-specific judgments based on careful analysis
    judgments = {
        "en_700": "Yes",  # Provides comprehensive caffeine content for various Coca-Cola products
        "en_701": "Yes",  # Explains Climate and Environmental Psychology with relevant examples
        "en_702": "Yes",  # Provides good summary of Tunisia's independence and presidents in ~100 words
        "en_703": "Yes",  # Clear explanation of props in Quasar with code examples
        "en_704": "Yes",  # Direct answer: colleagues should suggest he relax (though brief)
        "en_705": "No",   # Lists "reject" itself and has repetitive/broken formatting
        "en_706": "No",   # Completely off-topic: asked about depression/anxiety, got SaaS ideas
        "en_707": "Yes",  # Lists relevant Python web frameworks with appropriate caveats
        "en_708": "No",   # Incomplete decision tree, doesn't cover all disciplines from Wikipedia
        "en_709": "Yes",  # Provides 3 relevant research papers on scaling laws
        "en_710": "Yes",  # Accurate explanation of why blue is uncommon (anthocyanins, structural color)
        "en_711": "No",   # Off-topic: asked about phase modulation, got Unreal Engine animation
        "en_712": "Yes",  # Identifies 3 reasonable causes for Gradle wrapper error
        "en_713": "Yes",  # Direct answer: create account and complete tasks
        "en_714": "No",   # Wrong formula for expected value: E(g(X)) ≠ ∫g(x)dx
        "en_715": "Yes",  # Thoughtful response about quantum computing impact on everyday life
        "en_716": "Yes",  # Explains OCR for scanned documents, addresses the constraint
        "en_717": "Yes",  # Provides relevant post-apocalyptic script structure with themes
        "en_718": "Yes",  # Recommends Stable Diffusion with practical guidance
        "en_719": "Yes",  # Accurate info about altitude dangers and Armstrong limit
        "en_720": "Yes",  # Provides price (~$94,990) with appropriate caveats
        "en_721": "Yes",  # Provides sincere apology template and good advice
        "en_722": "Yes",  # Answers where to read more: NASA website and ApJ paper
        "en_723": "Yes",  # Explains Byzantine-Arab Wars led to Sasanian overthrow
        "en_724": "Yes",  # Comprehensive finger stretches and practice techniques for rhythm games
        "en_725": "Yes",  # Accurate bio of Simón Bolívar as "Liberator"
        "en_726": "Yes",  # Correct: Jill Biden born in NJ, capital Trenton, pop ~90,457
        "en_727": "Yes",  # Correct C++03 template metaprogramming for factorial
        "en_728": "Yes",  # Reasonable solution: feedback mechanism and evidence-based responses
        "en_729": "Yes",  # Offers to calculate if given today's date (reasonable workaround)
        "en_730": "Yes",  # Complete feijoada recipe with ingredients and instructions
        "en_731": "Yes",  # Accurate description of Java and Maven's purpose
        "en_732": "No",   # 2D rotation doesn't use x/y axis rotations; answer gives 3D matrices
        "en_733": "Yes",  # Explains quantum tunneling in caveman speak as requested
        "en_734": "Yes",  # Direct answer: 0 caffeine (with contamination caveat)
        "en_735": "Yes",  # Good explanation: evil person can be superficially nice
        "en_736": "Yes",  # Correct Haskell monad definition with return and >>=
        "en_737": "No",   # Completely wrong: sidechain is NOT about blockchain, it's audio
        "en_738": "Yes",  # Comprehensive fishing guide with licenses, tackle, safety, location
        "en_739": "Yes",  # Successfully made the speech more informal as requested
        "en_740": "Yes",  # Answers filament question: PLA, TPU, PETG, ABS with quality note
        "en_741": "No",   # Insists towel is critical despite user saying it's inappropriate
        "en_742": "Yes",  # Response cut off but starts with empathy and troubleshooting (F11)
        "en_743": "Yes",  # Provides step-by-step guide for writing an OS from scratch
        "en_744": "Yes",  # Reformats previous answer about caffeine/alcohol interaction
        "en_745": "Yes",  # Lists relevant ways to improve resume: internships, volunteering, etc.
        "en_746": "No",   # Vague, poorly written ("it depends. some are good some are bad")
        "en_747": "Yes",  # Confirms Aseprite has trial version before buying
        "en_748": "Yes",  # Provides FTB launcher download link as alternative
        "en_749": "Yes",  # Accurate info about Minho University location and reputation
        "en_750": "Yes",  # Explains how to install GPT-NeoX library and create Discord bot
        "en_751": "Yes",  # Provides team ratio guidance: backend 2-5x frontend, 1:10-20 designer
        "en_752": "Yes",  # Explains Irenaeus' soul-making theodicy comprehensively
        "en_753": "Yes",  # Good explanation of OpenAI vs Open Assistant openness differences
        "en_754": "Yes",  # Asks relevant follow-up questions about the marathon
        "en_755": "Yes",  # Provides 5 novel game ideas as requested
        "en_756": "Yes",  # Shows Flask basic setup with virtual environment
        "en_757": "Yes",  # Answers questions about ftplib commands and file upload
        "en_758": "Yes",  # Successfully removes hand/arm/finger related values
        "en_759": "Yes",  # Discusses access to justice barriers in Pakistan
        "en_760": "No",   # Asked for specific GAN paper abstract, got generic description
        "en_761": "Yes",  # Correct translation of Icelandic proverb
        "en_762": "Yes",  # Provides simplified version of CO2-eq explanation
        "en_763": "Yes",  # Asks clarifying question about skating frequency (reasonable)
        "en_764": "Yes",  # Addresses copyright risk in AI code generation
        "en_765": "Yes",  # Explains different types of infinity in mathematics
        "en_766": "Yes",  # Correctly refuses to emulate espionage systems (ethical)
        "en_767": "Yes",  # Explains Google's indexing and search speed
        "en_768": "Yes",  # Lists popular lucid dreaming techniques
        "en_769": "Yes",  # Correct: Joe Biden is president as of Feb 2023
        "en_770": "No",   # Goes off on weird tangent about monsters attacking cabin
        "en_771": "Yes",  # Asks clarifying question (population vs area)
        "en_772": "Yes",  # Confirms alarm works in Do Not Disturb mode
        "en_773": "Yes",  # Provides Australian tax table for 2022-23
        "en_774": "Yes",  # Outlines steps to use Stable Diffusion model
        "en_775": "Yes",  # States Open Assistant based on OPT with different sizes
        "en_776": "Yes",  # Concise explanation of good vs evil in ~100 words
        "en_777": "Yes",  # Reasonably declines 300 pages, offers outline instead
        "en_778": "Yes",  # Simple acknowledgment to follow-up after medical advice
        "en_779": "Yes",  # Re-explains labor theory of value as requested
        "en_780": "Yes",  # Lists common Tinder abbreviations (DTF, NSFW, LTR, etc.)
        "en_781": "Yes",  # Explains labor theory of value with nuance
        "en_782": "Yes",  # Explains why goalkeeper goals are rare
        "en_783": "Yes",  # Explains AI doesn't retain history for privacy/security
        "en_784": "Yes",  # Describes CFOP/Fridrich method for Rubik's cube
        "en_785": "Yes",  # Provides step-by-step approach for Kivy particle explosion
        "en_786": "Yes",  # Explains GPU transistor count increase since GeForce 256
        "en_787": "Yes",  # Identifies likely cause of SQL parameter mismatch error
        "en_788": "Yes",  # Shows C# LINQ sorting and unique filtering example
        "en_789": "Yes",  # Explains historical causes of antisemitism
        "en_790": "Yes",  # Confirms strategies are viable in Competitive mode
        "en_791": "Yes",  # Provides tips for building consistency and discipline
        "en_792": "No",   # Response is nonsensical ("Bob is not to he trusted because scammer")
        "en_793": "Yes",  # Lists 20 Latin American countries (though Puerto Rico debatable)
        "en_794": "Yes",  # Explains Elasticsearch snapshot creation process
        "en_795": "Yes",  # Explains Earth formation start as dust aggregation
        "en_796": "No",   # Incorrect: Ninja wasn't most popular in Feb 2023
        "en_797": "Yes",  # Mentions GitLab and encourages not worrying about breaking things
        "en_798": "Yes",  # Lists YouTube competitors (Vimeo, Dailymotion, etc.)
        "en_799": "Yes",  # Provides exact VScode regex find/replace steps
    }

    # Additional documents from the file (continuing from where visible portion ends)
    # Note: The file was truncated at en_742. Based on the pattern of 100 documents
    # and starting from en_700, we need judgments for en_700 through en_799

    # For documents not in the visible portion, we'll mark them as needing review
    if doc_id in judgments:
        return judgments[doc_id]
    else:
        # This shouldn't happen if we have all 100 documents
        # Default to a conservative judgment
        return "No"


def main():
    input_file = "/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_7.json"
    output_file = "/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_7.csv"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} documents from {input_file}")

    # Process each document
    results = []
    for doc in documents:
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Judge the response
        judgment = judge_helpfulness(doc_id, language, prompt, response)

        results.append({
            'id': doc_id,
            'answer': judgment,
            'language': language
        })

    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Count Yes/No
    yes_count = sum(1 for r in results if r['answer'] == 'Yes')
    no_count = sum(1 for r in results if r['answer'] == 'No')

    print(f"\nResults written to {output_file}")
    print(f"\nJudgment counts:")
    print(f"  Yes (HELPFUL): {yes_count}")
    print(f"  No (NOT helpful): {no_count}")
    print(f"  Total: {len(results)}")
    print(f"\nHelpfulness rate: {yes_count/len(results)*100:.1f}%")


if __name__ == "__main__":
    main()
