"""
LLM inference pipeline for CAP text classification — verbalized confidence.

MLX version for Apple Silicon. Uses Llama 3.1 8B Instruct (4-bit) to classify
parliamentary questions as law/crime-related (CAP topic 12).

Instead of extracting Yes/No token logprobs, asks the model to state its
confidence as a number (0-100), producing less bimodal score distributions.

Output format is identical to llm_inference.py for downstream compatibility.
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import mlx_lm
from tqdm import tqdm


MODEL_ID = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

PROMPT_TEMPLATE = (
    "The following text is in {language}. "
    "Does it primarily fall under the policy topic "
    "\"Law and Crime\" as defined by the Comparative Agendas Project? "
    "This topic includes: general law, crime, and family issues; law "
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
    "police and domestic security responses to terrorism. "
    "Rate your confidence from 0 (definitely not) to 100 (definitely yes). "
    "Respond with only a number.\n\n"
    "Text: {text}\n\n"
    "Confidence:"
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


def format_prompt(text, language, tokenizer):
    """Format the classification prompt using the chat template."""
    user_message = PROMPT_TEMPLATE.format(language=language, text=text)
    messages = [{"role": "user", "content": user_message}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt_str


def parse_confidence(text):
    """
    Parse a confidence number (0-100) from generated text.

    Handles common model outputs like "85", "85%", "85.", " 85\n", etc.
    Returns a float in [0, 1] or None if parsing fails.
    """
    text = text.strip()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match is None:
        return None
    value = float(match.group(1))
    if value > 100:
        return None
    return value / 100.0


def classify_single(text, language, model, tokenizer):
    """
    Classify a single document via verbalized confidence.

    Returns (score, raw_text) or raises on failure.
    """
    prompt_str = format_prompt(text, language, tokenizer)

    # Generate up to 10 tokens (enough for "85%" or "100")
    raw_text = ""
    for response in mlx_lm.stream_generate(
        model, tokenizer, prompt_str, max_tokens=10
    ):
        raw_text += response.text

    raw_text = raw_text.strip()
    score = parse_confidence(raw_text)
    if score is None:
        raise ValueError(f"Could not parse confidence from: {raw_text!r}")

    return score, raw_text


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
        description="Classify parliamentary texts using verbalized confidence "
        "(0-100 scale) via Llama 3.1 8B on MLX."
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

    # Process documents
    results_buffer = []
    n_errors = 0
    n_parse_failures = 0
    t_start = time.time()

    for i, row in enumerate(tqdm(rows, desc="Classifying (verbalized)")):
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

        # Truncate very long texts
        max_chars = 4000
        if len(text) > max_chars:
            text = text[:max_chars]

        try:
            score, raw_text = classify_single(
                text, language, model, tokenizer
            )
            results_buffer.append({
                "id": doc_id,
                "score": f"{score:.6f}",
                "token": raw_text,
                "language": language,
                "error": "",
            })
        except ValueError as e:
            n_parse_failures += 1
            results_buffer.append({
                "id": doc_id,
                "score": "",
                "token": str(e),
                "language": language,
                "error": f"parse_failure: {e}",
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
                f"({rate:.1f} docs/sec, ~{remaining/3600:.1f}h remaining) "
                f"[{n_parse_failures} parse failures, {n_errors} errors]"
            )

    # Final save
    save_results(results_buffer, args.output)

    elapsed = time.time() - t_start
    print(f"\nDone. Processed {len(rows)} documents in {elapsed:.0f}s "
          f"({len(rows)/elapsed:.1f} docs/sec).")
    print(f"  Parse failures: {n_parse_failures}")
    print(f"  Other errors: {n_errors}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
