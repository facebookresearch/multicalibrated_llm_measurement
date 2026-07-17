import json
import csv

# Read input data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_9.json', 'r') as f:
    data = json.load(f)

# Judge each response
results = []

for doc in data:
    doc_id = doc['id']
    language = doc['language']
    prompt = doc['prompt']
    response = doc['response']

    # Judge P(helpful) for this response
    # Will be filled in manually with calibrated probabilities
    p_help = None  # Placeholder

    results.append({
        'id': doc_id,
        'language': language,
        'prompt': prompt,
        'response': response,
        'p_help': p_help
    })

# Write output CSV
output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_9.csv'
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'score', 'p_help', 'p_nohelp', 'language'])

    for r in results:
        if r['p_help'] is not None:
            p_help = r['p_help']
            p_nohelp = 1 - p_help
            writer.writerow([r['id'], p_help, p_help, p_nohelp, r['language']])

print(f"Processed {len(results)} documents")
