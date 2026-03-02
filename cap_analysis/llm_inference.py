"""
LLM inference pipeline for CAP parliamentary text classification.

Classifies parliamentary questions as law/crime-related (CAP topic 12)
using Llama 3.1 8B (4-bit) via MLX on Apple Silicon.

Extracts log-probabilities for Yes/No tokens to produce a continuous
score in [0, 1] for each document.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from tqdm import tqdm


MODEL_ID = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

PROMPT_TEMPLATE = (
    "The following text is in {language}. "
    "Does it primarily discuss law, crime, or criminal justice? "
    "Respond Yes or No.\n\n"
    "Text: {text}\n\n"
    "Answer:"
)

LANGUAGE_MAP = {
    "danish": "Danish",
    "spanish": "Spanish",
    "dutch": "Dutch",
    "english": "English",
    "da": "Danish",
    "es": "Spanish",
    "nl": "Dutch",
    "en": "English",
}

SAVE_INTERVAL = 1000


def get_yes_no_token_ids(tokenizer):
    """Find token IDs for 'Yes' and 'No' (and common variants)."""
    yes_candidates = ["Yes", "yes", " Yes", " yes"]
    no_candidates = ["No", "no", " No", " no"]

    yes_ids = set()
    no_ids = set()
    for w in yes_candidates:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.add(ids[0])
    for w in no_candidates:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            no_ids.add(ids[0])

    if not yes_ids or not no_ids:
        raise ValueError(
            f"Could not find single-token encodings for Yes/No. "
            f"Yes IDs: {yes_ids}, No IDs: {no_ids}"
        )

    return list(yes_ids), list(no_ids)


def compute_score(logprobs, yes_ids, no_ids):
    """
    Compute P(Yes) / (P(Yes) + P(No)) from the log-probability vector.

    logprobs: mx.array of shape (vocab_size,) — log-probabilities over the
              full vocabulary for the first generated token.
    yes_ids: list of token IDs corresponding to "Yes"
    no_ids:  list of token IDs corresponding to "No"

    Returns a float in [0, 1].
    """
    # Gather log-probs for yes/no token IDs, take the max (most likely variant)
    yes_logprobs = [logprobs[tid].item() for tid in yes_ids]
    no_logprobs = [logprobs[tid].item() for tid in no_ids]

    yes_lp = max(yes_logprobs)
    no_lp = max(no_logprobs)

    # Convert to probabilities via logsumexp for numerical stability
    max_lp = max(yes_lp, no_lp)
    p_yes = math.exp(yes_lp - max_lp)
    p_no = math.exp(no_lp - max_lp)

    score = p_yes / (p_yes + p_no)
    return score


def format_prompt(text, language, tokenizer):
    """Format the classification prompt using the chat template."""
    user_message = PROMPT_TEMPLATE.format(language=language, text=text)
    messages = [{"role": "user", "content": user_message}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt_str


def classify_single(text, language, model, tokenizer, yes_ids, no_ids):
    """
    Classify a single document and return the law/crime score.

    Returns (score, generated_token_text) or raises on failure.
    """
    prompt_str = format_prompt(text, language, tokenizer)

    # We only need the first token's logprobs
    for response in mlx_lm.stream_generate(
        model, tokenizer, prompt_str, max_tokens=1
    ):
        logprobs = response.logprobs
        token_text = response.text
        break

    score = compute_score(logprobs, yes_ids, no_ids)
    return score, token_text


def load_input_data(input_path):
    """Load input CSV. Expects columns: id, text, and optionally language."""
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_existing_results(output_path):
    """Load already-processed document IDs from output file for resumption."""
    done_ids = set()
    if not os.path.exists(output_path):
        return done_ids
    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done_ids.add(row["id"])
    return done_ids


def save_results(results, output_path, mode="a"):
    """Append results to the output CSV."""
    if not results:
        return
    write_header = mode == "w" or not os.path.exists(output_path)
    fieldnames = ["id", "score", "token", "language", "error"]
    with open(output_path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(
        description="Classify parliamentary texts as law/crime-related using Llama 3.1 8B via MLX."
    )
    parser.add_argument(
        "--input", required=True, help="Path to input CSV with columns: id, text"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output CSV for scores"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, only process a random sample of N documents",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language of the texts (danish/spanish/dutch/english). "
        "If not set, reads from 'language' column in input CSV.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help=f"Model ID to load (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from existing output",
    )
    args = parser.parse_args()

    # Resolve language
    if args.language:
        lang_key = args.language.lower()
        if lang_key not in LANGUAGE_MAP:
            print(f"Error: unknown language '{args.language}'. "
                  f"Choose from: {list(LANGUAGE_MAP.keys())}")
            sys.exit(1)
        default_language = LANGUAGE_MAP[lang_key]
    else:
        default_language = None

    # Load input data
    print(f"Loading input data from {args.input}...")
    rows = load_input_data(args.input)
    print(f"  Loaded {len(rows)} documents.")

    # Sample if requested
    if args.sample is not None and args.sample < len(rows):
        import random
        random.seed(42)
        rows = random.sample(rows, args.sample)
        print(f"  Sampled {len(rows)} documents.")

    # Check for resume
    if not args.no_resume:
        done_ids = load_existing_results(args.output)
        if done_ids:
            rows = [r for r in rows if r["id"] not in done_ids]
            print(f"  Resuming: {len(done_ids)} already done, {len(rows)} remaining.")
    else:
        # Start fresh — write header
        if os.path.exists(args.output):
            os.remove(args.output)

    if not rows:
        print("No documents to process. Done.")
        return

    # Load model
    print(f"Loading model {args.model}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.model)
    print(f"  Model loaded in {time.time() - t0:.1f}s.")

    # Find Yes/No token IDs
    yes_ids, no_ids = get_yes_no_token_ids(tokenizer)
    print(f"  Yes token IDs: {yes_ids}")
    print(f"  No token IDs: {no_ids}")

    # Process documents
    results_buffer = []
    n_errors = 0
    t_start = time.time()

    for i, row in enumerate(tqdm(rows, desc="Classifying")):
        doc_id = row["id"]
        text = row["text"]

        # Determine language
        if default_language:
            language = default_language
        elif "language" in row:
            lang_key = row["language"].lower()
            language = LANGUAGE_MAP.get(lang_key, row["language"])
        else:
            print(f"Error: no language specified for document {doc_id} "
                  "and no --language flag set.")
            sys.exit(1)

        # Truncate very long texts to avoid context overflow
        # Llama 3.1 has 128K context but we keep it short for speed
        max_chars = 4000
        if len(text) > max_chars:
            text = text[:max_chars]

        try:
            score, token_text = classify_single(
                text, language, model, tokenizer, yes_ids, no_ids
            )
            results_buffer.append({
                "id": doc_id,
                "score": f"{score:.6f}",
                "token": token_text.strip(),
                "language": language,
                "error": "",
            })
        except Exception as e:
            n_errors += 1
            results_buffer.append({
                "id": doc_id,
                "score": "",
                "token": "",
                "language": language,
                "error": str(e),
            })

        # Periodic save
        if len(results_buffer) >= SAVE_INTERVAL:
            save_results(results_buffer, args.output)
            results_buffer = []
            elapsed = time.time() - t_start
            docs_done = i + 1
            rate = docs_done / elapsed
            remaining = (len(rows) - docs_done) / rate if rate > 0 else 0
            tqdm.write(
                f"  Saved checkpoint at {docs_done}/{len(rows)} docs "
                f"({rate:.1f} docs/sec, ~{remaining/3600:.1f}h remaining)"
            )

    # Final save
    save_results(results_buffer, args.output)

    elapsed = time.time() - t_start
    print(f"\nDone. Processed {len(rows)} documents in {elapsed:.0f}s "
          f"({len(rows)/elapsed:.1f} docs/sec).")
    print(f"  Errors: {n_errors}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
