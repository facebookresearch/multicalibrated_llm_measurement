#!/usr/bin/env python3
"""
Process all 250 judgments and write to CSV.
This script will be populated with manual judgments based on reading each document.
"""

import json
import csv

# Manual judgments for all 250 documents
# Format: 'doc_id': 'Yes' or 'No'
JUDGMENTS = {}

def load_all_data():
    """Load all 250 documents from the main shard file."""
    with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_120.json', 'r') as f:
        return json.load(f)


def write_results(data, judgments):
    """Write judgments to CSV file."""
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_120.csv'

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'answer', 'language'])

        for doc in data:
            doc_id = doc['id']
            answer = judgments.get(doc_id, 'MISSING')
            language = doc['language']
            writer.writerow([doc_id, answer, language])

    return output_path


def count_answers(judgments):
    """Count Yes/No answers."""
    yes_count = sum(1 for v in judgments.values() if v == 'Yes')
    no_count = sum(1 for v in judgments.values() if v == 'No')
    return yes_count, no_count


def main():
    data = load_all_data()
    print(f"Loaded {len(data)} documents")

    if not JUDGMENTS:
        print("No judgments loaded yet - need to populate JUDGMENTS dict")
        return

    output_path = write_results(data, JUDGMENTS)
    yes_count, no_count = count_answers(JUDGMENTS)

    print(f"\nWrote {len(data)} judgments to {output_path}")
    print(f"Yes: {yes_count}")
    print(f"No: {no_count}")
    print(f"Total: {yes_count + no_count}")


if __name__ == "__main__":
    main()
