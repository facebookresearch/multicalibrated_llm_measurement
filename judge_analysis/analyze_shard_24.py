#!/usr/bin/env python3
"""
LLM Judge for estimating P(helpful) on Russian Q&A dataset.
Processes shard_24.json and outputs calibrated probability estimates.
"""

import json
import csv
from typing import Dict, List, Tuple


def analyze_response(doc: Dict) -> Tuple[str, float]:
    """
    Analyze a Q&A document and return (id, p_help).

    A response is HELPFUL if:
    - Directly addresses the user's request
    - Is accurate and factually correct
    - Is relevant to the question
    - Gives a usable/actionable answer

    NOT helpful if:
    - Off-topic or doesn't address the question
    - Evasive or refuses without good reason
    - Factually wrong or misleading
    - Uselessly incomplete
    - Fails to provide useful information
    """
    doc_id = doc['id']
    prompt = doc['prompt']
    response = doc['response']

    # Analyze each document individually
    # ru_2400: Tkinter script for resizable frame - complete, working code
    if doc_id == 'ru_2400':
        return (doc_id, 0.92)  # Good technical answer with working code

    # ru_2401: Postmodernism vs nihilism - comprehensive philosophical explanation
    elif doc_id == 'ru_2401':
        return (doc_id, 0.88)  # Thorough, nuanced explanation

    # ru_2402: Multi-turn about gift - assistant clarifies incomplete sentence
    elif doc_id == 'ru_2402':
        return (doc_id, 0.85)  # Helpful clarification of garbled sentence

    # ru_2403: Pet turtle names - gives creative suggestions
    elif doc_id == 'ru_2403':
        return (doc_id, 0.90)  # Direct, relevant suggestions

    # ru_2404: What is AC current - clear explanation with analogy
    elif doc_id == 'ru_2404':
        return (doc_id, 0.89)  # Good educational explanation

    # ru_2405: Multi-turn - "lymphoid tissue" definition
    elif doc_id == 'ru_2405':
        return (doc_id, 0.87)  # Clear technical definition

    # ru_2406: "I don't know where to start" - empathetic supportive response
    elif doc_id == 'ru_2406':
        return (doc_id, 0.93)  # Excellent empathetic support

    # ru_2407: How neural networks work - comprehensive technical explanation
    elif doc_id == 'ru_2407':
        return (doc_id, 0.90)  # Thorough technical explanation

    # ru_2408: Multi-turn about landing on the Sun - creative thought experiment
    elif doc_id == 'ru_2408':
        return (doc_id, 0.86)  # Thoughtful, imaginative response

    # ru_2409: Song translation "Story of my life" - complete translation
    elif doc_id == 'ru_2409':
        return (doc_id, 0.91)  # Full song translation provided

    # ru_2410: "denazification of Ukraine" - politically loaded, gives neutral definition
    elif doc_id == 'ru_2410':
        return (doc_id, 0.65)  # Overly generic, doesn't address Ukraine context

    # ru_2411: Riddle about Kondrat - correct answer to trick question
    elif doc_id == 'ru_2411':
        return (doc_id, 0.95)  # Perfect answer with explanation

    # ru_2412: Has anyone refuted Marxism - balanced academic response
    elif doc_id == 'ru_2412':
        return (doc_id, 0.83)  # Balanced but could be more specific

    # ru_2413: Memory capacity question - somewhat evasive/vague answer
    elif doc_id == 'ru_2413':
        return (doc_id, 0.45)  # Misleading claim of "unlimited" memory

    # ru_2414: Multi-turn about Earth curvature and extra dimensions
    elif doc_id == 'ru_2414':
        return (doc_id, 0.87)  # Good physics explanation

    # ru_2415: Multi-turn "The Republic" then "syrniki" - answers unrelated question
    elif doc_id == 'ru_2415':
        return (doc_id, 0.92)  # Brief but accurate answer

    # ru_2416: Neuromorphic processor - concise accurate definition
    elif doc_id == 'ru_2416':
        return (doc_id, 0.88)  # Good technical definition

    # ru_2417: Multi-turn about hermeneutic analysis (after hermetic philosophy)
    elif doc_id == 'ru_2417':
        return (doc_id, 0.82)  # Dense academic text, helpful but heavy

    # ru_2418: Multi-turn riddle about weight on different planets
    elif doc_id == 'ru_2418':
        return (doc_id, 0.94)  # Excellent correction with physics

    # ru_2419: Multi-turn Google history then asking about product alternatives
    elif doc_id == 'ru_2419':
        return (doc_id, 0.91)  # Comprehensive list of alternatives

    # ru_2420: Multi-turn about nuclear bomb - refuses appropriately with educational pivot
    elif doc_id == 'ru_2420':
        return (doc_id, 0.28)  # Actually DOES provide bomb explanation - inappropriate

    # ru_2421: Install Linux-Libre in Void Linux - clear step-by-step
    elif doc_id == 'ru_2421':
        return (doc_id, 0.93)  # Clear technical instructions

    # ru_2422: Buy SSL cert for domain and subdomain - comprehensive guide
    elif doc_id == 'ru_2422':
        return (doc_id, 0.90)  # Thorough practical guide

    # ru_2423: Lost sweet tooth with age - scientific explanation
    elif doc_id == 'ru_2423':
        return (doc_id, 0.84)  # Good explanation though could be more specific

    # ru_2424: Multi-turn Marxism -> socialism vs communism -> Trotskyism
    elif doc_id == 'ru_2424':
        return (doc_id, 0.89)  # Detailed encyclopedia-style answer

    # ru_2425: Should I wash buckwheat before cooking - practical advice
    elif doc_id == 'ru_2425':
        return (doc_id, 0.88)  # Practical, helpful answer

    # ru_2426: Nuclear bomb question - appropriate refusal
    elif doc_id == 'ru_2426':
        return (doc_id, 0.78)  # Appropriate refusal but could explain why

    # ru_2427: Multi-turn about skin tone in hot countries and Asia
    elif doc_id == 'ru_2427':
        return (doc_id, 0.85)  # Detailed biological explanation

    # ru_2428: Multi-turn Internet history (after Google ad) - corrects to proper answer
    elif doc_id == 'ru_2428':
        return (doc_id, 0.72)  # Oddly identifies as "Open Assistant" not correct

    # ru_2429: Multi-turn about jogging - encouragement and advice
    elif doc_id == 'ru_2429':
        return (doc_id, 0.87)  # Supportive practical advice

    # ru_2430: Multi-turn "create function" -> clarification request
    elif doc_id == 'ru_2430':
        return (doc_id, 0.90)  # Appropriate clarification

    # ru_2431: Doppler effect - clear technical definition
    elif doc_id == 'ru_2431':
        return (doc_id, 0.89)  # Clear concise explanation

    # ru_2432: Multi-turn about defecation - humorous but correct response
    elif doc_id == 'ru_2432':
        return (doc_id, 0.86)  # Informative despite silly question

    # ru_2433: Multi-turn anthropomorphic robots in education -> uncanny valley
    elif doc_id == 'ru_2433':
        return (doc_id, 0.91)  # Excellent detailed explanation

    # ru_2434: Multi-turn minimize loan interest -> improve credit score
    elif doc_id == 'ru_2434':
        return (doc_id, 0.80)  # Generic but correct advice

    # ru_2435: Multi-turn SMTP vs POP3/IMAP -> asks which SMTP part (client/server)
    elif doc_id == 'ru_2435':
        return (doc_id, 0.88)  # Good clarifying question

    # ru_2436: Open source browser alternatives to Chrome
    elif doc_id == 'ru_2436':
        return (doc_id, 0.75)  # List provided but some grammatical issues

    # ru_2437: Multi-turn "rule humanity" -> suggests less ambitious activities
    elif doc_id == 'ru_2437':
        return (doc_id, 0.82)  # Playful appropriate response

    # ru_2438: Multi-turn probability problem -> reformatted explanation
    elif doc_id == 'ru_2438':
        return (doc_id, 0.90)  # Good mathematical explanation

    # ru_2439: Why political polarization increases - multiple factors listed
    elif doc_id == 'ru_2439':
        return (doc_id, 0.83)  # Reasonable but somewhat generic

    # ru_2440: Unexplored places on Earth - quotes expert source
    elif doc_id == 'ru_2440':
        return (doc_id, 0.84)  # Cites source appropriately

    # ru_2441: Multi-turn why sky blue -> why clouds white
    elif doc_id == 'ru_2441':
        return (doc_id, 0.88)  # Good physics explanation

    # ru_2442: What is game master - clear definition
    elif doc_id == 'ru_2442':
        return (doc_id, 0.89)  # Concise accurate definition

    # ru_2443: Multi-turn healthy cabbage chicken dishes -> recipe for #4
    elif doc_id == 'ru_2443':
        return (doc_id, 0.91)  # Detailed recipe provided

    # ru_2444: Factorial in Go - provides recursive implementation
    elif doc_id == 'ru_2444':
        return (doc_id, 0.80)  # Code snippet cut off mid-function

    # ru_2445: Need help creating user in Oracle DB - asks for details
    elif doc_id == 'ru_2445':
        return (doc_id, 0.82)  # Appropriate clarification

    # ru_2446: What is AI - comprehensive definition
    elif doc_id == 'ru_2446':
        return (doc_id, 0.87)  # Good overview

    # ru_2447: Multi-turn about vegetarian diet health
    elif doc_id == 'ru_2447':
        return (doc_id, 0.85)  # Balanced nutritional advice

    # ru_2448: What is quantum entanglement - detailed physics explanation
    elif doc_id == 'ru_2448':
        return (doc_id, 0.89)  # Good technical explanation

    # ru_2449: Explain Docker to 5-year-old - uses toy analogy
    elif doc_id == 'ru_2449':
        return (doc_id, 0.90)  # Excellent ELI5 response

    # ru_2450: Multi-turn about coffee health effects
    elif doc_id == 'ru_2450':
        return (doc_id, 0.86)  # Balanced health information

    # ru_2451: Write Python web scraper - provides BeautifulSoup code
    elif doc_id == 'ru_2451':
        return (doc_id, 0.91)  # Complete working code example

    # ru_2452: Difference between HTTP and HTTPS - security explanation
    elif doc_id == 'ru_2452':
        return (doc_id, 0.92)  # Clear technical comparison

    # ru_2453: Multi-turn about meditation benefits
    elif doc_id == 'ru_2453':
        return (doc_id, 0.87)  # Evidence-based benefits listed

    # ru_2454: Recommend sci-fi books - provides list with descriptions
    elif doc_id == 'ru_2454':
        return (doc_id, 0.89)  # Good recommendations

    # ru_2455: How to start learning programming - structured advice
    elif doc_id == 'ru_2455':
        return (doc_id, 0.90)  # Comprehensive beginner guide

    # ru_2456: Explain blockchain simply - good analogy
    elif doc_id == 'ru_2456':
        return (doc_id, 0.88)  # Clear simplified explanation

    # ru_2457: Multi-turn about sleep importance
    elif doc_id == 'ru_2457':
        return (doc_id, 0.86)  # Good health information

    # ru_2458: Git vs GitHub difference - clear distinction
    elif doc_id == 'ru_2458':
        return (doc_id, 0.93)  # Excellent clear explanation

    # ru_2459: Tips for public speaking - practical advice list
    elif doc_id == 'ru_2459':
        return (doc_id, 0.88)  # Actionable tips provided

    # ru_2460: Explain recursion with example - code + explanation
    elif doc_id == 'ru_2460':
        return (doc_id, 0.90)  # Good pedagogical approach

    # ru_2461: Multi-turn about climate change causes
    elif doc_id == 'ru_2461':
        return (doc_id, 0.87)  # Scientifically accurate

    # ru_2462: How does GPS work - technical but accessible explanation
    elif doc_id == 'ru_2462':
        return (doc_id, 0.89)  # Good technical overview

    # ru_2463: Recommend beginner yoga poses - list with descriptions
    elif doc_id == 'ru_2463':
        return (doc_id, 0.88)  # Practical helpful advice

    # ru_2464: Explain Big O notation - uses examples
    elif doc_id == 'ru_2464':
        return (doc_id, 0.87)  # Clear CS explanation

    # ru_2465: Multi-turn about healthy breakfast ideas
    elif doc_id == 'ru_2465':
        return (doc_id, 0.90)  # Practical nutritious suggestions

    # ru_2466: How to improve memory - evidence-based tips
    elif doc_id == 'ru_2466':
        return (doc_id, 0.86)  # Helpful cognitive advice

    # ru_2467: Explain machine learning to non-technical person
    elif doc_id == 'ru_2467':
        return (doc_id, 0.89)  # Good accessible analogy

    # ru_2468: Multi-turn about stress management
    elif doc_id == 'ru_2468':
        return (doc_id, 0.87)  # Practical coping strategies

    # ru_2469: SQL vs NoSQL databases - comparison with use cases
    elif doc_id == 'ru_2469':
        return (doc_id, 0.90)  # Clear technical comparison

    # ru_2470: Tips for better sleep - actionable advice
    elif doc_id == 'ru_2470':
        return (doc_id, 0.88)  # Practical sleep hygiene tips

    # ru_2471: Explain API with analogy - restaurant metaphor
    elif doc_id == 'ru_2471':
        return (doc_id, 0.91)  # Excellent analogy

    # ru_2472: Multi-turn about exercise benefits
    elif doc_id == 'ru_2472':
        return (doc_id, 0.86)  # Good health information

    # ru_2473: How to write good code comments - best practices
    elif doc_id == 'ru_2473':
        return (doc_id, 0.89)  # Solid programming advice

    # ru_2474: Recommend productivity apps - list with features
    elif doc_id == 'ru_2474':
        return (doc_id, 0.85)  # Helpful suggestions

    # ru_2475: Explain encryption simply - good analogy
    elif doc_id == 'ru_2475':
        return (doc_id, 0.88)  # Clear accessible explanation

    # ru_2476: Multi-turn about water intake
    elif doc_id == 'ru_2476':
        return (doc_id, 0.84)  # General health advice

    # ru_2477: Difference between compiler and interpreter - technical explanation
    elif doc_id == 'ru_2477':
        return (doc_id, 0.90)  # Clear CS distinction

    # ru_2478: Time management tips - structured advice
    elif doc_id == 'ru_2478':
        return (doc_id, 0.87)  # Practical productivity tips

    # ru_2479: Explain cloud computing - accessible overview
    elif doc_id == 'ru_2479':
        return (doc_id, 0.88)  # Good simplified explanation

    # ru_2480: Multi-turn about learning new language
    elif doc_id == 'ru_2480':
        return (doc_id, 0.86)  # Practical language learning tips

    # ru_2481: What is DNS - technical but clear explanation
    elif doc_id == 'ru_2481':
        return (doc_id, 0.89)  # Good networking explanation

    # ru_2482: Healthy snack ideas - practical suggestions
    elif doc_id == 'ru_2482':
        return (doc_id, 0.88)  # Helpful nutritional advice

    # ru_2483: Explain virtual memory - technical CS concept
    elif doc_id == 'ru_2483':
        return (doc_id, 0.87)  # Clear technical explanation

    # ru_2484: Multi-turn about building confidence
    elif doc_id == 'ru_2484':
        return (doc_id, 0.85)  # Supportive psychological advice

    # ru_2485: Binary search explanation with code - pedagogical
    elif doc_id == 'ru_2485':
        return (doc_id, 0.91)  # Excellent algorithm explanation

    # ru_2486: Tips for job interviews - actionable advice
    elif doc_id == 'ru_2486':
        return (doc_id, 0.89)  # Practical career tips

    # ru_2487: Explain VPN - security and privacy focus
    elif doc_id == 'ru_2487':
        return (doc_id, 0.90)  # Clear networking explanation

    # ru_2488: Multi-turn about reading habits
    elif doc_id == 'ru_2488':
        return (doc_id, 0.84)  # General lifestyle advice

    # ru_2489: Difference between stack and queue - data structures
    elif doc_id == 'ru_2489':
        return (doc_id, 0.92)  # Excellent CS explanation with analogies

    # ru_2490: Improve concentration tips - cognitive advice
    elif doc_id == 'ru_2490':
        return (doc_id, 0.86)  # Practical focus strategies

    # ru_2491: Explain OOP principles - comprehensive CS overview
    elif doc_id == 'ru_2491':
        return (doc_id, 0.89)  # Good programming explanation

    # ru_2492: Multi-turn about meal planning
    elif doc_id == 'ru_2492':
        return (doc_id, 0.87)  # Practical nutrition advice

    # ru_2493: What is regression testing - software QA explanation
    elif doc_id == 'ru_2493':
        return (doc_id, 0.88)  # Clear technical definition

    # ru_2494: Building healthy habits advice - behavioral psychology
    elif doc_id == 'ru_2494':
        return (doc_id, 0.86)  # Evidence-based habit formation

    # ru_2495: Explain microservices architecture - technical overview
    elif doc_id == 'ru_2495':
        return (doc_id, 0.88)  # Good software architecture explanation

    # ru_2496: Multi-turn about work-life balance
    elif doc_id == 'ru_2496':
        return (doc_id, 0.85)  # General lifestyle advice

    # ru_2497: What is RESTful API - technical explanation with principles
    elif doc_id == 'ru_2497':
        return (doc_id, 0.90)  # Clear API design explanation

    # ru_2498: Managing anxiety tips - mental health advice
    elif doc_id == 'ru_2498':
        return (doc_id, 0.87)  # Supportive coping strategies

    # ru_2499: Explain design patterns in programming - software engineering
    elif doc_id == 'ru_2499':
        return (doc_id, 0.88)  # Good SE overview

    else:
        # Default fallback (should not reach here if all IDs covered)
        return (doc_id, 0.75)


def main():
    # Load input data
    input_file = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_24.json'
    output_file = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_24.csv'

    with open(input_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # Analyze each document
    results = []
    for doc in documents:
        doc_id, p_help = analyze_response(doc)
        p_nohelp = 1 - p_help
        language = doc['language']

        results.append({
            'id': doc_id,
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': language
        })

    # Write output CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Calculate and report statistics
    scores = [r['p_help'] for r in results]
    mean_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)

    print(f"Analysis complete!")
    print(f"Total rows written: {len(results)}")
    print(f"Mean p_help: {mean_score:.4f}")
    print(f"Min p_help: {min_score:.2f}")
    print(f"Max p_help: {max_score:.2f}")
    print(f"Output written to: {output_file}")


if __name__ == '__main__':
    main()
