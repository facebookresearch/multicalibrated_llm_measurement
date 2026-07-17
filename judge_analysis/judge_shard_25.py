import json
import csv

# Read input data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_25.json', 'r') as f:
    data = json.load(f)

# Judge each response
results = []
p_help_values = []

for doc in data:
    doc_id = doc['id']
    language = doc['language']
    prompt = doc['prompt']
    response = doc['response']

    # Judge P(helpful) for each response
    # I will assign probabilities based on careful assessment

    # Placeholder - will be filled in with actual judgments
    p_help = 0.5  # This will be replaced

    results.append({
        'id': doc_id,
        'language': language,
        'p_help': p_help
    })

# This is a template - actual judgments will be done manually
print(f"Template created for {len(results)} documents")
