#!/usr/bin/env python3
"""
LLM Judge to estimate P(helpful) for assistant responses.
"""

import json
import csv
from judgments_shard_30 import JUDGMENTS

# Load the data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_30.json', 'r') as f:
    data = json.load(f)

# Process all documents
results = []
for doc in data:
    doc_id = doc['id']
    language = doc['language']

    # Get judgment - should exist for all 100 docs
    if doc_id not in JUDGMENTS:
        raise ValueError(f"Missing judgment for {doc_id}")

    p_help = JUDGMENTS[doc_id]
    p_nohelp = 1.0 - p_help

    results.append({
        'id': doc_id,
        'score': p_help,
        'p_help': p_help,
        'p_nohelp': p_nohelp,
        'language': language
    })

# Write output CSV
output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_30.csv'
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
    writer.writeheader()
    writer.writerows(results)

# Report statistics
mean_p_help = sum(r['p_help'] for r in results) / len(results)
min_p_help = min(r['p_help'] for r in results)
max_p_help = max(r['p_help'] for r in results)

print(f"Processed {len(results)} documents")
print(f"Mean P(helpful): {mean_p_help:.4f}")
print(f"Range: [{min_p_help:.2f}, {max_p_help:.2f}]")
print(f"Output written to: {output_path}")
