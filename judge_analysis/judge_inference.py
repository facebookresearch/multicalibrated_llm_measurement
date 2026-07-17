# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""LLM-as-judge inference utilities for the OASST helpfulness experiment.

Two campaigns, mirroring the CAP design, run by Claude Code sub-agents
(the judge is Claude Sonnet):

  binary : Yes/No -- is the assistant's final response helpful?
  phelp  : Direct probability elicitation of P(helpful), no prior commitment.

The judge assesses the SAME concept the human gold measures (OASST
"helpfulness"), so judge output is comparable to the gold label.

Usage:
    # Progress check
    python judge_analysis/judge_inference.py --campaign binary --check

    # Show a shard + prompt preview (used by sub-agents)
    python judge_analysis/judge_inference.py --campaign binary --shard 0

    # Write results (called by sub-agents)
    python judge_analysis/judge_inference.py --campaign binary --shard 0 \
        --write-results '<json list of row dicts>'
"""

import argparse
import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARD_DIR = os.path.join(SCRIPT_DIR, "data", "shards")
OUTPUT_DIRS = {
    "binary": os.path.join(SCRIPT_DIR, "data", "inference_output", "sonnet-binary"),
    "phelp": os.path.join(SCRIPT_DIR, "data", "inference_output", "sonnet-phelp"),
}

HELPFUL_DEFINITION = (
    "A response is HELPFUL if it directly addresses the user's request, is "
    "accurate and relevant, and gives the user useful information or a usable "
    "answer. A response is NOT helpful if it is off-topic, evasive, factually "
    "wrong, incomplete to the point of being useless, or fails to engage with "
    "what the user actually asked."
)

BINARY_PROMPT_TEMPLATE = (
    "You are evaluating an AI assistant's response for helpfulness. "
    "The conversation is in {language}.\n\n"
    "{definition}\n\n"
    "Conversation so far:\n{prompt}\n\n"
    "Assistant's response to evaluate:\n{response}\n\n"
    "Is the assistant's response helpful? Answer Yes or No."
)

PHELP_PROMPT_TEMPLATE = (
    "You are evaluating an AI assistant's response for helpfulness. "
    "The conversation is in {language}.\n\n"
    "{definition}\n\n"
    "Conversation so far:\n{prompt}\n\n"
    "Assistant's response to evaluate:\n{response}\n\n"
    "Without first deciding Yes or No, directly estimate the probability that "
    "this response is helpful. Provide P(helpful) and P(not helpful) as your "
    "true belief probabilities, summing to 1.0.\n\n"
    "Respond in exactly this format:\n"
    "P(helpful): [0.XX]\n"
    "P(not helpful): [0.XX]"
)

PROMPT_TEMPLATES = {"binary": BINARY_PROMPT_TEMPLATE, "phelp": PHELP_PROMPT_TEMPLATE}

FIELDNAMES = {
    "binary": ["id", "answer", "language"],
    "phelp": ["id", "score", "p_help", "p_nohelp", "language"],
}


def load_shard(shard_id):
    with open(os.path.join(SHARD_DIR, f"shard_{shard_id}.json"), encoding="utf-8") as f:
        return json.load(f)


def shard_output_path(campaign, shard_id):
    return os.path.join(OUTPUT_DIRS[campaign], f"shard_{shard_id}.csv")


def shard_is_done(campaign, shard_id, expected):
    path = shard_output_path(campaign, shard_id)
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return len(list(csv.DictReader(f))) >= expected


def write_results(campaign, shard_id, results):
    path = shard_output_path(campaign, shard_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES[campaign])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    return path


def format_prompt(campaign, doc):
    return PROMPT_TEMPLATES[campaign].format(
        language=doc["language"],
        definition=HELPFUL_DEFINITION,
        prompt=doc["prompt"],
        response=doc["response"],
    )


def n_shards():
    return len([f for f in os.listdir(SHARD_DIR) if f.startswith("shard_")])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True, choices=["binary", "phelp"])
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--write-results", type=str, default=None)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    total = n_shards()

    if args.check:
        done, todo = [], []
        for i in range(total):
            expected = len(load_shard(i))
            (done if shard_is_done(args.campaign, i, expected) else todo).append(i)
        print(f"Campaign '{args.campaign}': {len(done)}/{total} shards done")
        if todo:
            print(f"TODO: {todo}")
        return

    if args.shard is None:
        p.error("--shard required unless --check")

    if args.write_results:
        results = json.loads(args.write_results)
        path = write_results(args.campaign, args.shard, results)
        print(f"Wrote {len(results)} rows to {path}")
        return

    docs = load_shard(args.shard)
    print(f"Campaign '{args.campaign}', shard {args.shard}: {len(docs)} docs")
    print(f"Output: {shard_output_path(args.campaign, args.shard)}")
    print(f"\n--- PROMPT PREVIEW (doc 0) ---\n{format_prompt(args.campaign, docs[0])}")


if __name__ == "__main__":
    main()
