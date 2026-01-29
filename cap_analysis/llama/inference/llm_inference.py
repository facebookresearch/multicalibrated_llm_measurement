# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
LLM inference pipeline for CAP political text classification — two-stage
verbalized confidence (Tian et al. 2023).

Two-stage approach following Tian et al. (EMNLP 2023):
  Stage 1: Ask model "Does this text discuss law/crime?" -> Yes/No answer
  Stage 2: In a second dialogue turn, ask "What is the probability that
           your answer is correct?" -> model outputs a probability (0.0-1.0)

The final score is converted to P(law_crime):
  - If answer = Yes: score = stated_confidence
  - If answer = No:  score = 1 - stated_confidence

This reframes confidence as self-assessment rather than direct label probability,
which Tian et al. show produces better-calibrated and less bimodal distributions
compared to single-stage "rate your confidence 0-100" prompts.

Classifies political texts as law/crime-related (CAP topic 12)
using Llama 3.3 70B via HuggingFace Transformers + PyTorch. The model is
loaded from a HuggingFace repo ID by default (`meta-llama/Llama-3.3-70B-Instruct`)
and downloaded via the standard HF cache; pass `--model` to point at a local
checkpoint instead.

Output columns: id, score, token, language, error
where `score` is P(law_crime) and `token` records "answer=Yes/No;confidence=X.XX".

Data parallelism: use --shard i/N to split input across N processes (one per GPU),
each pinned via CUDA_VISIBLE_DEVICES.
"""

import argparse
import csv
import logging
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# Suppress repetitive "Setting pad_token_id to eos_token_id" warnings
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)


DEFAULT_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

CAP_TOPIC_DEF = (
    "the policy topic \"Law and Crime\" as defined by the Comparative "
    "Agendas Project? This topic includes: general law, crime, and family "
    "issues; law enforcement agencies including border, customs, and "
    "specialized enforcement agencies; white collar crime, organized crime, "
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

STAGE1_PROMPT = (
    "The following text is in {language}. "
    "Does it primarily fall under " + CAP_TOPIC_DEF + " "
    "Answer only Yes or No.\n\n"
    "Text: {text}"
)

STAGE2_PROMPT = (
    "What is the probability that your answer is correct? "
    "Respond with only a decimal number between 0.0 and 1.0. "
    "Note: a probability of exactly 0.0 or 1.0 would mean absolute certainty, "
    "which is rarely warranted."
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


def load_model(model_id, use_4bit=True):
    """Load model and tokenizer with optional 4-bit quantization."""
    print(f"Loading model {model_id}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if torch.cuda.is_available():
        device = "cuda"
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
            )
    elif torch.backends.mps.is_available():
        device = "mps"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to(device)
    else:
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )

    model.eval()
    print(f"  Model loaded on {device} in {time.time() - t0:.1f}s.")
    return model, tokenizer, device


def parse_yes_no(text):
    """Parse Yes/No from generated text. Returns 'yes', 'no', or None."""
    text = text.strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None


def parse_probability(text):
    """
    Parse a probability (0.0-1.0) from generated text.

    Handles outputs like "0.85", "0.85.", "85%", "0.85\n".
    Integer values in (1, 100] are interpreted as percentages (e.g. "85" -> 0.85).
    Non-integer values > 1 are rejected as model errors (e.g. "2.5" returns None
    rather than silently becoming 0.025).
    Returns a float in [0, 1] or None if parsing fails.
    """
    text = text.strip()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match is None:
        return None
    value = float(match.group(1))
    if value > 1:
        # Only treat integer values in (1, 100] as percentages.
        if value <= 100 and value == int(value):
            value = value / 100.0
        else:
            return None
    return value


def classify_single(text, language, model, tokenizer, device):
    """
    Classify a single document via two-stage verbalized confidence.

    Stage 1: Generate Yes/No answer (greedy).
    Stage 2: Ask for P(answer is correct) in a follow-up turn (greedy).

    Returns (score, detail_str) where score is P(law_crime) in [0,1].
    """
    # Stage 1: get Yes/No answer
    user_msg = STAGE1_PROMPT.format(language=language, text=text)
    messages = [{"role": "user", "content": user_msg}]
    stage1_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(stage1_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=5,
            do_sample=False, temperature=None, top_p=None,
        )

    prompt_len = inputs["input_ids"].shape[1]
    answer_text = tokenizer.decode(
        output_ids[0, prompt_len:], skip_special_tokens=True
    ).strip()
    answer = parse_yes_no(answer_text)
    if answer is None:
        raise ValueError(f"Could not parse Yes/No from stage 1: {answer_text!r}")

    # Stage 2: ask for confidence in a follow-up turn
    messages.append({"role": "assistant", "content": answer_text})
    messages.append({"role": "user", "content": STAGE2_PROMPT})
    stage2_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs2 = tokenizer(stage2_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids2 = model.generate(
            **inputs2, max_new_tokens=10,
            do_sample=False, temperature=None, top_p=None,
        )

    prompt_len2 = inputs2["input_ids"].shape[1]
    conf_text = tokenizer.decode(
        output_ids2[0, prompt_len2:], skip_special_tokens=True
    ).strip()
    confidence = parse_probability(conf_text)
    if confidence is None:
        raise ValueError(
            f"Could not parse probability from stage 2: {conf_text!r} "
            f"(answer was {answer_text!r})"
        )

    # Convert to P(law_crime)
    if answer == "yes":
        score = confidence
    else:
        score = 1.0 - confidence

    detail = f"answer={answer_text};confidence={conf_text};p_correct={confidence:.4f}"
    return score, detail


def parse_shard(shard_str):
    """Parse --shard argument of the form 'i/N'. Returns (i, N)."""
    parts = shard_str.split("/")
    if len(parts) != 2:
        raise ValueError(f"--shard must be in format 'i/N', got '{shard_str}'")
    i, n = int(parts[0]), int(parts[1])
    if i < 0 or i >= n:
        raise ValueError(f"Shard index {i} out of range for {n} shards")
    return i, n


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
        description="Classify political texts using two-stage verbalized "
        "confidence (Tian et al. 2023) via Llama 3.3 70B."
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
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID or local path "
        f"(default: {DEFAULT_MODEL_ID}, downloaded via HF cache).",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (use fp16 instead)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from existing output",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help="Data shard in format 'i/N' (e.g. '0/2' for first of two shards). "
        "Use with CUDA_VISIBLE_DEVICES for multi-GPU parallelism.",
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

    # Parse shard
    shard_idx, n_shards = None, None
    if args.shard:
        shard_idx, n_shards = parse_shard(args.shard)

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

    # Apply sharding (after sampling, so each shard gets a consistent slice)
    if shard_idx is not None:
        total = len(rows)
        chunk_size = (total + n_shards - 1) // n_shards  # ceiling division
        start = shard_idx * chunk_size
        end = min(start + chunk_size, total)
        rows = rows[start:end]
        print(f"  Shard {shard_idx}/{n_shards}: processing rows {start}-{end-1} "
              f"({len(rows)} documents)")

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
    model, tokenizer, device = load_model(args.model, use_4bit=not args.no_4bit)

    # Process documents
    results_buffer = []
    n_errors = 0
    n_parse_failures = 0
    t_start = time.time()

    for i, row in enumerate(tqdm(rows, desc="Classifying (2-stage)")):
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
            score, detail = classify_single(
                text, language, model, tokenizer, device,
            )
            results_buffer.append({
                "id": doc_id,
                "score": f"{score:.6f}",
                "token": detail,
                "language": language,
                "error": "",
            })
        except ValueError as e:
            # Parse failure — record the error and raw text
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
