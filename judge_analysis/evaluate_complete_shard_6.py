#!/usr/bin/env python3
"""
LLM Judge for evaluating helpfulness of assistant responses.
Estimates P(helpful) for each response in shard_6.json.

A response is HELPFUL if it:
- Directly addresses the request
- Is accurate
- Is relevant
- Gives a usable answer

NOT helpful if:
- Off-topic
- Evasive
- Factually wrong
- Uselessly incomplete
- Fails to engage
"""

import json
import csv


def get_evaluations():
    """Return probability estimates for all documents."""
    return {
        "en_600": 0.88,  # SOP updates - practical steps, directly addresses question
        "en_601": 0.92,  # Visa letter - professional, incorporates all details
        "en_602": 0.72,  # Fiancé reassurance - thoughtful but redirects vs directly answering
        "en_603": 0.45,  # Cheer me up - empathetic but doesn't fulfill request
        "en_604": 0.52,  # Docker YAML optimize - reasonable advice but broken code formatting
        "en_605": 0.90,  # Q-Learning vs Deep Q - clear accurate explanation
        "en_606": 0.25,  # Rsync file sync - misunderstands as collaborative editing
        "en_607": 0.85,  # AI middle management - balanced nuanced answer
        "en_608": 0.82,  # Prosthetics/explosives ethics - thoughtful multi-angle analysis
        "en_609": 0.68,  # Linux alternatives - addresses but could be more comprehensive
        "en_610": 0.35,  # Cake recipe - just asks for recipe, no suggestions
        "en_611": 0.40,  # Wesley Willis lyrics - questionable accuracy
        "en_612": 0.58,  # Rust→Python code - right idea but implementation issues
        "en_613": 0.87,  # Mag Lev cubes - clear accurate explanation
        "en_614": 0.91,  # JSON deserialize - working code, clear explanation
        "en_615": 0.89,  # Warhammer jokes - shows lore knowledge, addresses request
        "en_616": 0.32,  # Base 2 for 9yo - still too complex, doesn't simplify
        "en_617": 0.90,  # Epicurean paradox counters - well-organized comprehensive
        "en_618": 0.93,  # Helium-6 reference - provides working link, confirms accuracy
        "en_619": 0.22,  # Einstein daily impact - lists achievements, doesn't answer actual question
        "en_620": 0.65,  # Leadership styles - reasonable content but awkward wording
        "en_621": 0.42,  # YouTuber creator - somewhat evasive
        "en_622": 0.12,  # How LLM works - refuses to engage, evasive
        "en_623": 0.55,  # Fisherman saying - related but not directly on target
        "en_624": 0.38,  # Shogi professional 5yo - too generic, lacks concrete steps
        "en_625": 0.84,  # Ice lake blue - explains both color and temperature
        "en_626": 0.94,  # Bash cron - accurate practical complete
        "en_627": 0.86,  # Portugal rent - specific range with context
        "en_628": 0.88,  # Mythological characters - creative well-developed
        "en_629": 0.92,  # Back to Future timeline - clear chronological breakdown
        "en_630": 0.15,  # BMW vs Mercedes - defensive unhelpful sarcastic
        "en_631": 0.89,  # "Sheesh" meaning - direct explanation with reference
        "en_632": 0.87,  # SEO 2023 - balanced addresses evolution
        "en_633": 0.76,  # Consensus protocol - thorough but appears cut off
        "en_634": 0.85,  # AI job replacement ethics - comprehensive starts listing key issues
        "en_635": 0.62,  # IF phases - lists patterns not phases, partial answer
        "en_636": 0.87,  # Roleplay tavern - engaging sets scene well
        "en_637": 0.18,  # Chatbot hallucination - misunderstands, confuses concepts
        "en_638": 0.28,  # BEV sales - just repeats can't help, not useful
        "en_639": 0.83,  # Trolley problem - starts explaining consequentialist view appropriately
        "en_640": 0.74,  # 3D print steam engine - addresses question, discusses materials
        "en_641": 0.81,  # Epicurus paradox - explains the paradox clearly
        "en_642": 0.95,  # Monty Python script - correctly identifies source with details
        "en_643": 0.48,  # Murder mystery - starts new story vs continuing, ignores context
        "en_644": 0.92,  # Sort designers by death date - does exactly as requested
        "en_645": 0.72,  # Magic systems - explains soft vs hard but gets cut off
        "en_646": 0.93,  # AEZAKMI cheat - simple clear direct answer
        "en_647": 0.79,  # Docker yaml - explains docker-compose, starts listing components
        "en_648": 0.31,  # Photorealistic SD prompt - asks for psychedelic instead, wrong topic
        "en_649": 0.75,  # Fix car - appropriately asks for more info
        "en_650": 0.66,  # DnD campaign - explains theme/setting but doesn't answer full question
        "en_651": 0.91,  # Sclerotia mushrooms - accurate clear distinction
        "en_652": 0.84,  # Overwhelmed - divide and conquer advice, addresses problem
        "en_653": 0.77,  # Keyboard broken - did task but just says "you're welcome" with emoji
        "en_654": 0.41,  # Twitch GPT3 bot - just gives pip install, ignores most of question
        "en_655": 0.86,  # Hitler vegetarian artist - good point that traits don't determine morality
        "en_656": 0.71,  # Texas power grid - explains Texas grid history, relevant but not complete answer
        "en_657": 0.73,  # Anime for beginners - appropriately asks preferences
        "en_658": 0.89,  # "Cyber" meaning - clear explanation with example
        "en_659": 0.34,  # PyQt5 threads - too brief, doesn't explain PyQt5-specific issues
        "en_660": 0.82,  # Earth axis tilt - explains causes appropriately
        "en_661": 0.81,  # Monkey typing - playful engaging response fitting request
        "en_662": 0.88,  # Promotion letter outline - clear structure addressing request
        "en_663": 0.08,  # Science YouTube - suggests PragerU (political not science), promotional tone
        "en_664": 0.87,  # Animal color vision - relevant accurate examples
        "en_665": 0.85,  # Solve equation - just says "you're welcome" after solved, polite completion
        "en_666": 0.77,  # Lightning vs Euromillions - addresses comparison, provides numbers
        "en_667": 0.79,  # Photosynthesis - starts explaining clearly but gets cut off
        "en_668": 0.91,  # Bias variance ML - clear accurate explanation of both
        "en_669": 0.19,  # Depression anxiety - gives baby sleep advice, wrong topic entirely
        "en_670": 0.27,  # Convolutions NN - just says "glad to help" without explaining anything
        "en_671": 0.92,  # PKU phenylketonuria - comprehensive accurate summary
        "en_672": 0.71,  # Climate change sentence - just says "you're welcome" after answering
        "en_673": 0.83,  # OpenAssistant vs ChatGPT - lists valid advantages
        "en_674": 0.88,  # Illegal radio fines - explains varies by country with UK example
        "en_675": 0.80,  # Grow YouTube - starts giving practical advice (define niche)
        "en_676": 0.90,  # Eye protection screen - gives 20/20/20 rule and practical tips
        "en_677": 0.88,  # Lower car tax - balanced benefits/harms structure
        "en_678": 0.24,  # Psalm 23 from God's POV - responds about biblical accuracy instead
        "en_679": 0.78,  # AQI mask level - addresses WHO vs Chinese standards appropriately
        "en_680": 0.13,  # Book recommendation - suggests Atlas Shrugged for isekai/fantasy request, total mismatch
        "en_681": 0.86,  # Zombie instruments - detailed analysis of guitar role
        "en_682": 0.82,  # Fluid Hierarchical - asks clarifying question appropriately
        "en_683": 0.93,  # Tongue sore - practical steps for healing
        "en_684": 0.85,  # Lightning Network transaction - explains process clearly
        "en_685": 0.81,  # Maze solving REPL - provides adjusted pseudocode
        "en_686": 0.84,  # REPL for CLI - starts outlining implementation steps
        "en_687": 0.90,  # Smartboard not working - systematic troubleshooting steps
        "en_688": 0.88,  # Big O notation - clear explanation of O(n) notation
        "en_689": 0.87,  # PyTorch efficiency - discusses factors affecting efficiency appropriately
        "en_690": 0.94,  # Sun size and 8*6 - answers both questions accurately
        "en_691": 0.90,  # Clash Royale archetypes - explains control deck archetype clearly
        "en_692": 0.82,  # AI in schools abuse - discusses negative consequences thoughtfully
        "en_693": 0.81,  # Paint sprayer - discusses conventional as cost-effective option
        "en_694": 0.89,  # LinkedIn from link - appropriately explains can't access links
        "en_695": 0.84,  # Conversation non-English - practical suggestions for language barrier
        "en_696": 0.83,  # HTML CSS JS link - explains how to link files
        "en_697": 0.91,  # Story structures - lists structures clearly (Three-Act, Hero's Journey)
        "en_698": 0.79,  # Gray code - provides biographical and technical info about Gray code
        "en_699": 0.90,  # Do Androids Dream - accurate plot summary of PKD novel
    }


def main():
    # Load the shard data
    with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_6.json', 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    evaluations = get_evaluations()

    # Prepare output data
    results = []
    p_help_values = []

    for doc in data:
        doc_id = doc['id']
        language = doc['language']

        # Get probability estimate
        p_help = evaluations.get(doc_id, 0.50)  # Default 0.50 if missing
        p_nohelp = 1.0 - p_help

        results.append({
            'id': doc_id,
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': language
        })
        p_help_values.append(p_help)

    # Verify we have all documents
    if len(results) != 100:
        print(f"WARNING: Expected 100 documents but got {len(results)}")

    # Calculate mean
    mean_p_help = sum(p_help_values) / len(p_help_values)

    # Write to CSV
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_6.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Mean P(help): {mean_p_help:.4f}")
    print(f"Number of rows: {len(results)}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
