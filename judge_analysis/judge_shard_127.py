#!/usr/bin/env python3
"""
LLM judge for assistant response helpfulness on shard 127.
Judges whether responses directly address requests with accurate, relevant, usable answers.
"""

import json
import csv
from pathlib import Path
import anthropic
import os

def judge_helpfulness(prompt: str, response: str) -> str:
    """
    Judge if a response is helpful using Claude as a judge.

    HELPFUL = directly addresses the request, accurate, relevant, gives a usable answer
    NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage

    Returns: "Yes" or "No"
    """

    judge_prompt = f"""You are judging whether an assistant's response is HELPFUL to the user's request.

USER PROMPT:
{prompt}

ASSISTANT RESPONSE:
{response}

Is this response HELPFUL?

HELPFUL means:
- Directly addresses the user's request
- Accurate and factually correct
- Relevant to what was asked
- Gives a usable, actionable answer
- Engages substantively with the question

NOT HELPFUL means:
- Off-topic or doesn't address the request
- Evasive or avoids answering
- Factually wrong or misleading
- Uselessly incomplete (lacks key information needed)
- Fails to engage with the question

Judge ONLY based on the prompt and response shown. Be calibrated - do NOT default to "Yes". Many responses are not helpful.

Answer with a single word: Yes or No"""

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        temperature=0,
        messages=[{
            "role": "user",
            "content": judge_prompt
        }]
    )

    judgment = message.content[0].text.strip()

    # Normalize to Yes/No
    if judgment.lower().startswith("yes"):
        return "Yes"
    elif judgment.lower().startswith("no"):
        return "No"
    else:
        # Fallback - shouldn't happen with clear instructions
        print(f"Warning: Unexpected judgment '{judgment}', defaulting to No")
        return "No"


def main():
    # Load input data
    input_file = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_127.json")
    output_file = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_127.csv")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records")

    # Judge each record
    results = []
    yes_count = 0
    no_count = 0

    for i, record in enumerate(data, 1):
        doc_id = record['id']
        language = record['language']
        prompt = record['prompt']
        response = record['response']

        print(f"Judging {i}/{len(data)}: {doc_id}...", end=" ", flush=True)

        judgment = judge_helpfulness(prompt, response)

        if judgment == "Yes":
            yes_count += 1
        else:
            no_count += 1

        print(judgment)

        results.append({
            'id': doc_id,
            'answer': judgment,
            'language': language
        })

    # Write output CSV
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Report summary
    print(f"\n=== Summary ===")
    print(f"Total records: {len(results)}")
    print(f"Yes (helpful): {yes_count}")
    print(f"No (not helpful): {no_count}")
    print(f"Output written to: {output_file}")

    # Verify row count
    assert len(results) == 250, f"Expected 250 rows, got {len(results)}"
    print(f"✓ Confirmed 250 rows written")


if __name__ == "__main__":
    main()
