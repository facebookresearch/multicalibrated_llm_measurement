#!/usr/bin/env python3
"""
Prepare batches for manual LLM judging.
Splits the shard into smaller batches for processing.
"""

import json
from pathlib import Path

def main():
    input_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_110.json")
    batch_dir = Path("/Users/flinder/mc_measurement/judge_analysis/data/batches")
    batch_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Split into batches of 50 for easier processing
    batch_size = 50
    num_batches = (len(data) + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch = data[start_idx:end_idx]

        batch_file = batch_dir / f"shard_110_batch_{i+1}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)

        print(f"Batch {i+1}/{num_batches}: {len(batch)} documents -> {batch_file}")

    print(f"\nCreated {num_batches} batches in {batch_dir}")

if __name__ == "__main__":
    main()
