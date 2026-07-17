#!/usr/bin/env python3
"""
LLM judge for assistant response helpfulness using Claude API.
Judges each response as Yes (helpful) or No (not helpful).
"""

import json
import csv
from pathlib import Path
import anthropic
import os
from tqdm import tqdm

JUDGE_PROMPT = """You are an expert judge evaluating whether an assistant's response is HELPFUL.

HELPFUL means:
- Directly addresses the user's request
- Accurate and factually correct
- Relevant to what was asked
- Gives a usable, actionable answer
- Engages meaningfully with the question

NOT HELPFUL means:
- Off-topic or completely misses the point
- Evasive or refuses to engage without good reason
- Factually wrong or misleading
- Uselessly incomplete (acknowledges the question but provides no substance)
- Fails to attempt answering the actual question

Judge based ONLY on the prompt and response provided. Be calibrated - do NOT default to "Yes". Many responses will be unhelpful.

Respond with ONLY "Yes" or "No" on a single line."""


def judge_single(client: anthropic.Anthropic, prompt: str, response: str) -> str:
    """Judge a single response using Claude API."""

    conversation = f"""PROMPT:
{prompt}

RESPONSE:
{response}

Is this response HELPFUL? Answer only "Yes" or "No"."""

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=10,
        temperature=0,
        messages=[
            {"role": "user", "content": JUDGE_PROMPT + "\n\n" + conversation}
        ]
    )

    judgment = message.content[0].text.strip()

    # Ensure we only get Yes or No
    if judgment not in ["Yes", "No"]:
        # Try to extract from response
        if "yes" in judgment.lower():
            return "Yes"
        elif "no" in judgment.lower():
            return "No"
        else:
            # Default to No if unclear (be calibrated)
            return "No"

    return judgment


def main():
    input_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_110.json")
    output_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_110.csv")

    # Initialize Claude client
    # In Meta environment, use AnthropicVertex or direct Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        # Try to use a placeholder - the Meta environment may handle auth differently
        api_key = "placeholder"

    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        print(f"Warning: Failed to initialize client with standard method: {e}")
        print("Attempting to use AnthropicVertex...")
        # Try Vertex AI endpoint
        vertex_project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "devai-mea-egeit")
        vertex_base_url = os.environ.get("ANTHROPIC_VERTEX_BASE_URL", "http://vertex.ai-gateway.x2p.facebook.net:443/v1")
        client = anthropic.AnthropicVertex(
            project_id=vertex_project,
            region="us-east5"
        )

    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")
    print(f"Using model: claude-sonnet-4-5-20250929")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    for doc in tqdm(data, desc="Judging responses"):
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Judge the response
        answer = judge_single(client, prompt, response)

        if answer == "Yes":
            yes_count += 1
        else:
            no_count += 1

        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

    # Write output
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCompleted judgment:")
    print(f"  Yes (helpful): {yes_count} ({yes_count/len(results)*100:.1f}%)")
    print(f"  No (not helpful): {no_count} ({no_count/len(results)*100:.1f}%)")
    print(f"  Total rows: {len(results)}")
    print(f"\nOutput written to: {output_path}")

    # Verify
    assert len(results) == len(data), f"Row count mismatch: {len(results)} != {len(data)}"
    print(f"✓ Confirmed {len(results)} rows match input")

if __name__ == "__main__":
    main()
