import json
import csv

# Load the data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_0.json', 'r') as f:
    documents = json.load(f)

# Store judgments
judgments = []

# Process each document
for doc in documents:
    doc_id = doc['id']
    language = doc['language']
    prompt = doc['prompt']
    response = doc['response']

    # Evaluate helpfulness based on the criteria:
    # A response is HELPFUL if it directly addresses the user's request,
    # is accurate and relevant, and gives useful information or a usable answer.
    # A response is NOT helpful if it is off-topic, evasive, factually wrong,
    # incomplete to the point of being useless, or fails to engage with what the user actually asked.

    # Store the evaluation (will be filled in manually)
    judgments.append({
        'id': doc_id,
        'language': language,
        'prompt': prompt,
        'response': response,
        'answer': None  # To be filled
    })

# Print for manual review
print(f"Total documents to judge: {len(judgments)}")
