#!/usr/bin/env python3
"""
Judge helpfulness of assistant responses in shard_120.json using Claude.
Processes in batches and writes judgments to CSV.
"""

import json
import csv
import sys


def load_data(filepath):
    """Load JSON data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def write_csv(output_path, judgments):
    """Write judgments to CSV with exact header format."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'answer', 'language'])
        for judgment in judgments:
            writer.writerow([judgment['id'], judgment['answer'], judgment['language']])


def main():
    input_path = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_120.json'
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_120.csv'

    # Load data
    data = load_data(input_path)
    print(f"Loaded {len(data)} documents")

    # Prepare for batch processing - will be filled by Claude
    judgments = []

    # For now, just create the structure
    for doc in data:
        judgments.append({
            'id': doc['id'],
            'answer': 'PLACEHOLDER',  # Will be replaced by actual judgments
            'language': doc['language']
        })

    # Write template
    write_csv(output_path, judgments)
    print(f"Wrote template CSV with {len(judgments)} rows to {output_path}")


if __name__ == "__main__":
    main()
