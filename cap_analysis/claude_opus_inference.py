"""
Claude Opus inference for CAP law/crime classification.

Each shard is processed by a Claude Code sub-agent that calls this script.
The sub-agent passes its own classifications via stdin as a JSON array.

Usage (called by sub-agent):
    python3 claude_opus_inference.py --shard 42

The script reads the shard JSON, prints the prompt + documents to stdout
for the calling agent to classify, then the agent writes results back.

Alternatively, used in "write" mode to save agent results:
    python3 claude_opus_inference.py --shard 42 --write-results '<json>'
"""

import argparse
import csv
import json
import os
import sys


SHARD_DIR = "cap_analysis/data/claude_shards_30k"
OUTPUT_DIR = "cap_analysis/data/inference_output/claude-opus-30k"

PROMPT_TEMPLATE = (
    'The following text is in {language}. '
    'Does it primarily fall under the policy topic '
    '"Law and Crime" as defined by the Comparative Agendas Project? '
    'This topic includes: general law, crime, and family issues; law '
    'enforcement agencies including border, customs, and specialized '
    'enforcement agencies; white collar crime, organized crime, '
    'counterfeiting, fraud, cyber-crime, and money laundering; illegal drug '
    'crime and enforcement, criminal penalties for drug crimes, and '
    'international efforts to combat drug trafficking; court administration, '
    'bail, pre-release, fines, and legal representation; prisons, jails, '
    'and parole systems; juvenile crime and justice, and efforts to reduce '
    'juvenile crime and recidivism; child abuse, child pornography, sexual '
    'exploitation of children, and parental kidnapping; family issues, '
    'domestic violence, child welfare, and family law; domestic criminal and '
    'civil codes; crime control, prevention, and impact of crime; and '
    'police and domestic security responses to terrorism.\n\n'
    'Text: {text}\n\n'
    'Step 1: Answer Yes or No.\n'
    'Step 2: Setting aside your answer above, estimate P(Yes) and P(No) '
    'as your true belief probabilities, summing to 1.0.\n\n'
    'Respond in exactly this format:\n'
    'Answer: [Yes/No]\n'
    'P(Yes): [0.XX]\n'
    'P(No): [0.XX]'
)


def load_shard(shard_id):
    path = os.path.join(SHARD_DIR, f"shard_{shard_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def shard_output_path(shard_id):
    return os.path.join(OUTPUT_DIR, f"shard_{shard_id}.csv")


def shard_is_done(shard_id):
    path = shard_output_path(shard_id)
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return len(rows) >= 100  # full shard


def write_results(shard_id, results):
    """Write classification results to CSV."""
    path = shard_output_path(shard_id)
    fieldnames = ["id", "score", "answer", "p_yes", "p_no", "language", "label"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    return path


def format_prompt(doc):
    text = doc["text"]
    if len(text) > 4000:
        text = text[:4000]
    return PROMPT_TEMPLATE.format(language=doc["language"], text=text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--write-results", type=str, default=None,
                        help="JSON string of results to write")
    parser.add_argument("--check", action="store_true",
                        help="Check which shards are done")
    args = parser.parse_args()

    if args.check:
        done = []
        todo = []
        for i in range(300):
            if shard_is_done(i):
                done.append(i)
            else:
                todo.append(i)
        print(f"Done: {len(done)}, Todo: {len(todo)}")
        if todo:
            print(f"Next batch: {todo[:50]}")
        return

    if args.write_results:
        results = json.loads(args.write_results)
        path = write_results(args.shard, results)
        print(f"Wrote {len(results)} results to {path}")
        return

    # Default: print shard info for the agent
    docs = load_shard(args.shard)
    print(f"Shard {args.shard}: {len(docs)} documents")
    print(f"Output: {shard_output_path(args.shard)}")
    print(f"\nPrompt template preview:")
    print(format_prompt(docs[0])[:200] + "...")


if __name__ == "__main__":
    main()
