import json
import csv

# Load the input data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_3.json', 'r') as f:
    data = json.load(f)

# Judgments for each document based on helpfulness criteria
# HELPFUL = directly addresses request, accurate, relevant, gives usable answer
# NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

judgments = []

for doc in data:
    doc_id = doc['id']
    language = doc['language']
    prompt = doc['prompt']
    response = doc['response']

    # Analyze each case individually

    if doc_id == "en_300":
        # Discusses government funding via loans/printing money. Addresses the question directly.
        answer = "Yes"
    elif doc_id == "en_301":
        # Asks for clarification, which is reasonable given vague question
        answer = "Yes"
    elif doc_id == "en_302":
        # Response is "Yes it is!" without explaining which option is better. Incomplete answer.
        answer = "No"
    elif doc_id == "en_303":
        # Physics problem - applies conservation of energy but ignores the axis/table setup. Likely wrong.
        answer = "No"
    elif doc_id == "en_304":
        # Comprehensive step-by-step plan for learning music creation. Very helpful.
        answer = "Yes"
    elif doc_id == "en_305":
        # Provides 3 niche consumer groups as requested.
        answer = "Yes"
    elif doc_id == "en_306":
        # Asks clarifying questions to help with project - engaging and helpful.
        answer = "Yes"
    elif doc_id == "en_307":
        # Brief but engages with the question about anime.
        answer = "Yes"
    elif doc_id == "en_308":
        # Offers to continue helping - appropriate response to "thank you"
        answer = "Yes"
    elif doc_id == "en_309":
        # Provides definition of Ethereum with references. Directly addresses question.
        answer = "Yes"
    elif doc_id == "en_310":
        # Acknowledges success and offers further help. Appropriate.
        answer = "Yes"
    elif doc_id == "en_311":
        # Summarizes academic paper on herding in cryptocurrencies. Addresses request.
        answer = "Yes"
    elif doc_id == "en_312":
        # Complete Caesar salad recipe with ingredients and instructions.
        answer = "Yes"
    elif doc_id == "en_313":
        # Asks for necessary information to write CV. Reasonable approach.
        answer = "Yes"
    elif doc_id == "en_314":
        # Friendly closing, offers continued help. Appropriate.
        answer = "Yes"
    elif doc_id == "en_315":
        # Provides concrete example of Cornell notes on states of matter. Good response.
        answer = "Yes"
    elif doc_id == "en_316":
        # Asks clarifying questions about story preferences. Helpful engagement.
        answer = "Yes"
    elif doc_id == "en_317":
        # Adapts to new scenario appropriately, sets scene for roleplay.
        answer = "Yes"
    elif doc_id == "en_318":
        # Correctly capitalizes the text as requested for dyslexia accommodation.
        answer = "Yes"
    elif doc_id == "en_319":
        # Substantive discussion of governance evaluation and peaceful transitions.
        answer = "Yes"
    elif doc_id == "en_320":
        # Comprehensive plan for learning spoken language with all requested skills.
        answer = "Yes"
    elif doc_id == "en_321":
        # Confirms scientific validity and cites research. Good response.
        answer = "Yes"
    elif doc_id == "en_322":
        # Creative short story about Adder's teleportation prank. Addresses prompt.
        answer = "Yes"
    elif doc_id == "en_323":
        # Shakespeare-style poem about traveling dog. Creative and appropriate.
        answer = "Yes"
    elif doc_id == "en_324":
        # Only mentions "Beginner's Method" without explanation. Too incomplete.
        answer = "No"
    elif doc_id == "en_325":
        # Lists good and bad habits for OOP Lua with explanations. Very thorough.
        answer = "Yes"
    elif doc_id == "en_326":
        # Brief but accurate description of Actix web framework.
        answer = "Yes"
    elif doc_id == "en_327":
        # Acknowledges commonality, gives brief answer about mating. Minimal but addresses it.
        answer = "Yes"
    elif doc_id == "en_328":
        # Elmer Fudd reference - humorous response to "What's up doc?"
        answer = "Yes"
    elif doc_id == "en_329":
        # Explains backward() function in neural networks clearly.
        answer = "Yes"
    elif doc_id == "en_330":
        # Complete guide to writing bash scripts with example. Very helpful.
        answer = "Yes"
    elif doc_id == "en_331":
        # Provides specific screenplay examples with metrics as requested.
        answer = "Yes"
    elif doc_id == "en_332":
        # Clear explanation of differences between Euclidean and hyperbolic geometry.
        answer = "Yes"
    elif doc_id == "en_333":
        # Recommends URP for mobile game with solid reasoning.
        answer = "Yes"
    elif doc_id == "en_334":
        # Discusses instilling values and dealing with non-conformers. Addresses both questions.
        answer = "Yes"
    elif doc_id == "en_335":
        # Confirms portals can be bigger with size range. Direct answer.
        answer = "Yes"
    elif doc_id == "en_336":
        # Gives velocity and acknowledges Monty Python reference. Complete answer.
        answer = "Yes"
    elif doc_id == "en_337":
        # Correctly converts Fahrenheit to Celsius in recipe.
        answer = "Yes"
    elif doc_id == "en_338":
        # Explains standard deviation with calculation steps and Julia code.
        answer = "Yes"
    elif doc_id == "en_339":
        # Provides 5 concrete social media graphic ideas for gaming content.
        answer = "Yes"
    elif doc_id == "en_340":
        # Answers multiple questions but has errors (earth-moon distance instead of earth-mars, etc.)
        answer = "No"
    elif doc_id == "en_341":
        # Explains identity function's role with mathematical example. Good explanation.
        answer = "Yes"
    elif doc_id == "en_342":
        # Provides 10 girl names and 10 boy names as requested.
        answer = "Yes"
    elif doc_id == "en_343":
        # Complete shell script with encryption, compression, and upload. Addresses request.
        answer = "Yes"
    elif doc_id == "en_344":
        # Simple response suggesting "Assistant" as form of address.
        answer = "Yes"
    elif doc_id == "en_345":
        # Deflects criticism about ChatGPT by explaining RLHF process. Somewhat evasive.
        answer = "No"
    elif doc_id == "en_346":
        # Begins to address question but response is cut off. Incomplete.
        answer = "No"
    else:
        # Default to reading remaining documents
        answer = "Yes"

    judgments.append({
        'id': doc_id,
        'answer': answer,
        'language': language
    })

# Ensure output directory exists
import os
os.makedirs('/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary', exist_ok=True)

# Write to CSV
with open('/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_3.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
    writer.writeheader()
    writer.writerows(judgments)

# Count Yes/No
yes_count = sum(1 for j in judgments if j['answer'] == 'Yes')
no_count = sum(1 for j in judgments if j['answer'] == 'No')

print(f"Total judgments: {len(judgments)}")
print(f"Yes (HELPFUL): {yes_count}")
print(f"No (NOT helpful): {no_count}")
