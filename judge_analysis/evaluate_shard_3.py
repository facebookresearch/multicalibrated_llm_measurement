#!/usr/bin/env python3
"""
LLM Judge for estimating P(helpful) for assistant responses.
A response is HELPFUL if it directly addresses the request, is accurate,
relevant, and gives a usable answer.
"""

import json
import csv

def estimate_p_helpful(doc):
    """
    Estimate the probability that a response is helpful.

    Criteria for HELPFUL:
    - Directly addresses the request
    - Accurate information
    - Relevant to the question
    - Gives a usable answer

    NOT helpful if:
    - Off-topic or evasive
    - Factually wrong
    - Uselessly incomplete
    - Fails to engage with the question
    """

    doc_id = doc['id']
    prompt = doc['prompt']
    response = doc['response']

    # Detailed evaluation for each example
    evaluations = {
        'en_300': 0.68,  # Addresses funding question, provides reasonable answer (loans/printing money), though economically debatable
        'en_301': 0.75,  # Good clarifying question when user prompt is vague
        'en_302': 0.15,  # "Yes it is!" is uselessly incomplete - doesn't say which is healthy (set times or hunger-based)
        'en_303': 0.35,  # Wrong physics - ignores collision with axis, treats as simple free fall
        'en_304': 0.85,  # Comprehensive, actionable plan for learning music creation
        'en_305': 0.45,  # These aren't "extremely niche" - they're fairly broad categories
        'en_306': 0.90,  # Good engagement, asks appropriate clarifying questions
        'en_307': 0.72,  # Brief but engages appropriately with casual question
        'en_308': 0.88,  # Good follow-up, offers to continue helping
        'en_309': 0.82,  # Accurate, concise definition with references
        'en_310': 0.65,  # Awkward phrasing but offers continued help appropriately
        'en_311': 0.78,  # Good summary capturing key points about herding in crypto
        'en_312': 0.92,  # Complete, accurate Caesar salad recipe
        'en_313': 0.95,  # Excellent - asks for necessary information before writing CV
        'en_314': 0.88,  # Friendly, offers continued help appropriately
        'en_315': 0.90,  # Excellent concrete example with good caveats about variations
        'en_316': 0.85,  # Good clarifying questions to understand story requirements
        'en_317': 0.87,  # Appropriately adjusts to user request, sets scene well
        'en_318': 0.95,  # Perfectly executes the capitalization request
        'en_319': 0.80,  # Thoughtful, comprehensive answer about governance transitions
        'en_320': 0.88,  # Detailed, well-structured language learning plan
        'en_321': 0.75,  # Affirms it's scientifically valid, explains benefits, though lacks specific references
        'en_322': 0.82,  # Creative story matching the prompt requirements
        'en_323': 0.85,  # Good Shakespearean-style verse about the dog
        'en_324': 0.25,  # Too brief, doesn't explain what the method is or provide resources
        'en_325': 0.87,  # Comprehensive list of good/bad habits with clear explanations
        'en_326': 0.70,  # Accurate but brief answer about Actix
        'en_327': 0.62,  # Addresses question but vague, somewhat evasive ("won't get into details")
        'en_328': 0.55,  # Playful Bugs Bunny reference, but doesn't actually help user
        'en_329': 0.88,  # Clear, accurate explanation of backward() function
        'en_330': 0.93,  # Complete step-by-step bash scripting guide with example
        'en_331': 0.75,  # Provides examples but numbers seem rough/possibly inaccurate
        'en_332': 0.82,  # Accurate, concise comparison of Euclidean vs hyperbolic geometry
        'en_333': 0.90,  # Excellent recommendation with clear reasoning for URP
        'en_334': 0.84,  # Thoughtful answer about value transmission without laws
        'en_335': 0.90,  # Correct, concise answer about nether portal sizes
        'en_336': 0.88,  # Accurate answer with cultural reference acknowledgment
        'en_337': 0.95,  # Accurate temperature conversions
        'en_338': 0.85,  # Correct identification of standard deviation with calculation steps
        'en_339': 0.88,  # Five concrete, relevant ideas matching user's context
        'en_340': 0.60,  # Attempts to answer but has errors (smallest city by population not land area, confuses Farout)
        'en_341': 0.78,  # Good explanation with concrete example of identity function use
        'en_342': 0.88,  # Provides both male and female names as requested
        'en_343': 0.90,  # Complete working script with good explanations and caveats
        'en_344': 0.85,  # Polite, direct answer to addressing preference
        'en_345': 0.42,  # Defensive meta-response about ChatGPT, doesn't address user's concern constructively
        'en_346': 0.70,  # Acknowledges difficulty, explains variables, but incomplete (cut off)
        'en_347': 0.78,  # Good explanation with examples and caveats
        'en_348': 0.88,  # Appropriate redirect to library documentation
        'en_349': 0.05,  # Just says "OK" - fails to engage meaningfully
        'en_350': 0.82,  # Clear explanation with simple example
        'en_351': 0.73,  # Good starting suggestions but could be more specific
        'en_352': 0.68,  # Attempts to help but response is somewhat disjointed
        'en_353': 0.55,  # Refuses task that could be done, overly cautious
        'en_354': 0.85,  # Good structured advice about avoiding plague
        'en_355': 0.45,  # Lists programming languages but doesn't address "best" for data science specifically
        'en_356': 0.88,  # Accurate, detailed C# async/await explanation
        'en_357': 0.25,  # Suggests Google - not helpful from an AI assistant
        'en_358': 0.82,  # Good analysis of yin/yang concept
        'en_359': 0.90,  # Creative, detailed sci-fi story as requested
        'en_360': 0.48,  # Overly cautious, doesn't provide simple factual description requested
        'en_361': 0.87,  # Good recipe with clear instructions
        'en_362': 0.78,  # Reasonable explanation of Valve's strategy
        'en_363': 0.92,  # Excellent summary of Genshin Impact gameplay
        'en_364': 0.65,  # Provides 2 types not 3; adequate but incomplete
        'en_365': 0.83,  # Good comprehensive list of document types
        'en_366': 0.72,  # Basic accurate info but very brief for such a broad question
        'en_367': 0.88,  # Thoughtful analysis of AI art implications
        'en_368': 0.91,  # Excellent detailed tutorial with code and explanations
        'en_369': 0.75,  # Good structure but lacks specific action items
        'en_370': 0.80,  # Accurate explanation though slightly technical
        'en_371': 0.68,  # Vague answer, lacks concrete examples
        'en_372': 0.93,  # Perfect execution of temperature conversion task
        'en_373': 0.58,  # Lists countries but doesn't explain what makes them authoritarian
        'en_374': 0.86,  # Good comprehensive answer with multiple perspectives
        'en_375': 0.77,  # Reasonable answer but could be more specific about implementation
        'en_376': 0.85,  # Good diplomatic answer acknowledging both perspectives
        'en_377': 0.48,  # Very basic, doesn't explain when/why to use each
        'en_378': 0.90,  # Comprehensive explanation with examples
        'en_379': 0.72,  # Addresses question but lacks depth on actual trends
        'en_380': 0.83,  # Good explanation of both terms with comparison
        'en_381': 0.42,  # Suggests buying course - not helpful from AI assistant
        'en_382': 0.88,  # Good creative response with personality
        'en_383': 0.76,  # Accurate but brief explanation
        'en_384': 0.91,  # Excellent detailed recipe with ingredient explanations
        'en_385': 0.58,  # Somewhat evasive, doesn't give clear guidance
        'en_386': 0.87,  # Good detailed pros/cons comparison
        'en_387': 0.68,  # Provides answer but seems uncertain, lacks confidence
        'en_388': 0.95,  # Clear, actionable steps for fixing the error
        'en_389': 0.62,  # Acknowledges question but provides limited useful info
        'en_390': 0.79,  # Good creative haiku matching request
        'en_391': 0.73,  # Decent list but could be more creative/specific
        'en_392': 0.88,  # Good balanced answer explaining both sides
        'en_393': 0.52,  # Too cautious, doesn't provide requested information
        'en_394': 0.84,  # Good specific examples of stoicism in action
        'en_395': 0.90,  # Excellent step-by-step Git tutorial
        'en_396': 0.77,  # Reasonable advice though somewhat generic
        'en_397': 0.86,  # Good explanation with historical context
        'en_398': 0.43,  # Mostly refuses task, could at least explain the topic
        'en_399': 0.89,  # Great detailed explanation with examples
    }

    # Get the probability, default to 0.5 if not in dict
    p_help = evaluations.get(doc_id, 0.50)

    return p_help


def main():
    # Load the data
    with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_3.json', 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    # Process each example
    results = []
    total_p_help = 0.0

    for doc in data:
        p_help = estimate_p_helpful(doc)
        p_nohelp = 1.0 - p_help

        results.append({
            'id': doc['id'],
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': doc['language']
        })

        total_p_help += p_help

    # Calculate mean
    mean_p_help = total_p_help / len(results)

    # Create output directory if needed
    import os
    output_dir = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp'
    os.makedirs(output_dir, exist_ok=True)

    # Write CSV
    output_file = os.path.join(output_dir, 'shard_3.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Mean P(helpful): {mean_p_help:.4f}")
    print(f"Total rows written: {len(results)}")
    print(f"Output written to: {output_file}")


if __name__ == '__main__':
    main()
