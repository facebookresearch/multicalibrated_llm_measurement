"""
Claude Opus 4.6 inference for CAP law/crime classification.

Two separate campaigns were run on 30,000 documents (300 shards x 100 docs):

Campaign 1 (Binary): Simple Yes/No classification.
Campaign 2 (P(Y/N)): Direct probability elicitation without first committing
    to an answer, to avoid anchoring the probability toward 0 or 1.

The campaigns were run independently via Claude Code sub-agents (50 concurrent
agents per round, 6 rounds per campaign). Each agent received a shard of
documents and classified them according to the prompt templates below.

This script provides utilities for shard management and result writing used
by the sub-agents.

Usage:
    # Check progress
    python3 claude_opus_inference.py --campaign binary --check
    python3 claude_opus_inference.py --campaign pyn --check

    # Print shard info (used by sub-agents)
    python3 claude_opus_inference.py --campaign binary --shard 42

    # Write results (called by sub-agents)
    python3 claude_opus_inference.py --campaign binary --shard 42 --write-results '<json>'
"""

import argparse
import csv
import json
import os
import sys


SHARD_DIR = "cap_analysis/data/claude_shards_30k"

OUTPUT_DIRS = {
    "binary": "cap_analysis/data/inference_output/claude-opus-30k-binary",
    "pyn": "cap_analysis/data/inference_output/claude-opus-30k-pyn",
}

CAP_CODEBOOK_DEFINITION = (
    "Law and Crime includes: general law, crime, and family issues; law "
    "enforcement agencies including border, customs, and specialized "
    "enforcement agencies; white collar crime, organized crime, "
    "counterfeiting, fraud, cyber-crime, and money laundering; illegal drug "
    "crime and enforcement, criminal penalties for drug crimes, and "
    "international efforts to combat drug trafficking; court administration, "
    "bail, pre-release, fines, and legal representation; prisons, jails, "
    "and parole systems; juvenile crime and justice, and efforts to reduce "
    "juvenile crime and recidivism; child abuse, child pornography, sexual "
    "exploitation of children, and parental kidnapping; family issues, "
    "domestic violence, child welfare, and family law; domestic criminal and "
    "civil codes; crime control, prevention, and impact of crime; and "
    "police and domestic security responses to terrorism."
)

# --- Campaign 1: Binary Yes/No ---
# The model classifies each document with a simple Yes or No.
BINARY_PROMPT_TEMPLATE = (
    'The following text is in {language}. '
    'Does it primarily fall under the policy topic "Law and Crime" '
    'as defined by the Comparative Agendas Project? '
    '{codebook}\n\n'
    'Text: {text}\n\n'
    'Answer Yes or No.'
)

# --- Campaign 2: Direct P(Y/N) probability elicitation ---
# The model estimates P(Yes) and P(No) directly, WITHOUT first committing
# to a Yes/No answer. This avoids the anchoring effect where committing to
# an answer biases the subsequent probability toward 0 or 1.
PYN_PROMPT_TEMPLATE = (
    'The following text is in {language}. '
    'Consider whether it primarily falls under the policy topic "Law and Crime" '
    'as defined by the Comparative Agendas Project. '
    '{codebook}\n\n'
    'Text: {text}\n\n'
    'Without first deciding Yes or No, directly estimate the probability that '
    'this text is about Law and Crime. Provide P(Yes) and P(No) as your true '
    'belief probabilities, summing to 1.0.\n\n'
    'Respond in exactly this format:\n'
    'P(Yes): [0.XX]\n'
    'P(No): [0.XX]'
)

PROMPT_TEMPLATES = {
    "binary": BINARY_PROMPT_TEMPLATE,
    "pyn": PYN_PROMPT_TEMPLATE,
}

FIELDNAMES = {
    "binary": ["id", "answer", "language", "label"],
    "pyn": ["id", "score", "p_yes", "p_no", "language", "label"],
}


def load_shard(shard_id):
    path = os.path.join(SHARD_DIR, f"shard_{shard_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def shard_output_path(campaign, shard_id):
    return os.path.join(OUTPUT_DIRS[campaign], f"shard_{shard_id}.csv")


def shard_is_done(campaign, shard_id):
    path = shard_output_path(campaign, shard_id)
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return len(rows) >= 100


def write_results(campaign, shard_id, results):
    """Write classification results to CSV."""
    path = shard_output_path(campaign, shard_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES[campaign])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    return path


def format_prompt(campaign, doc):
    text = doc["text"]
    if len(text) > 4000:
        text = text[:4000]
    return PROMPT_TEMPLATES[campaign].format(
        language=doc["language"],
        text=text,
        codebook=CAP_CODEBOOK_DEFINITION,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Claude Opus inference utilities for CAP classification"
    )
    parser.add_argument("--campaign", required=True, choices=["binary", "pyn"],
                        help="Which campaign: 'binary' (Yes/No) or 'pyn' (probability)")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--write-results", type=str, default=None,
                        help="JSON string of results to write")
    parser.add_argument("--check", action="store_true",
                        help="Check which shards are done")
    args = parser.parse_args()

    if args.check:
        done, todo = [], []
        for i in range(300):
            (done if shard_is_done(args.campaign, i) else todo).append(i)
        print(f"Campaign '{args.campaign}': {len(done)} done, {len(todo)} todo")
        if todo:
            print(f"Next batch: {todo[:50]}")
        return

    if args.shard is None:
        parser.error("--shard is required unless --check is used")

    if args.write_results:
        results = json.loads(args.write_results)
        path = write_results(args.campaign, args.shard, results)
        print(f"Wrote {len(results)} results to {path}")
        return

    docs = load_shard(args.shard)
    print(f"Campaign '{args.campaign}', shard {args.shard}: {len(docs)} documents")
    print(f"Output: {shard_output_path(args.campaign, args.shard)}")
    print(f"\nPrompt preview:\n{format_prompt(args.campaign, docs[0])[:300]}...")


if __name__ == "__main__":
    main()
