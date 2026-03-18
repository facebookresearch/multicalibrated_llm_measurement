"""
LLM inference pipeline for CAP political text classification — verbalized confidence.

Instead of extracting Yes/No token logprobs, this script asks the model to state
its confidence as a number (0-100), producing less bimodal score distributions.

Supports temperature sampling with K completions per document to produce
continuous averaged scores (useful when greedy decoding yields bimodal outputs).

Classifies political texts as law/crime-related (CAP topic 12)
using Llama 3.3 70B via HuggingFace Transformers + PyTorch.

Output format is identical to llm_inference_cuda.py for downstream compatibility:
    id, score, token, language, error

where `score` is the mean verbalized confidence across K samples (mapped to [0, 1]),
and `token` records the K individual parsed scores (semicolon-separated).

Data parallelism: use --shard i/N to split input across N processes (one per GPU),
each pinned via CUDA_VISIBLE_DEVICES.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# Suppress repetitive "Setting pad_token_id to eos_token_id" warnings
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)


MODEL_LOCAL_PATH = os.path.expanduser("~/models/Llama-3.3-70B-Instruct")
MANIFOLD_PATH = "asa/tree/huggingface/model--meta-llama--Llama-3.3-70B-Instruct"

PROMPT_TEMPLATE = (
    "The following text is in {language}. "
    "Does it primarily discuss law, crime, or criminal justice? "
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


def ensure_model_local(local_path, manifold_path):
    """Download model weights from manifold if not available locally."""
    if os.path.exists(local_path) and os.path.isdir(local_path):
        safetensors = [f for f in os.listdir(local_path) if f.endswith(".safetensors")]
        if safetensors:
            print(f"Model found locally at {local_path}")
            return local_path

    print(f"Model not found locally at {local_path}")
    print(f"Downloading from manifold: {manifold_path} ...")
    os.makedirs(local_path, exist_ok=True)
    result = subprocess.run(
        ["manifold", "--prod-use-cython-client", "getr",
         manifold_path, local_path, "--threads", "20", "--jobs", "10"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Manifold download failed (exit {result.returncode}):")
        print(result.stderr)
        sys.exit(1)
    print("Download complete.")
    return local_path


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


def classify_single(text, language, model, tokenizer, device,
                     temperature=0.0, num_samples=1):
    """
    Classify a single document via verbalized confidence.

    When num_samples > 1, generates K completions at the given temperature
    and returns the mean score. The prompt is encoded once and K completions
    are sampled in a single forward pass via num_return_sequences.

    Returns (mean_score, individual_scores_list) or raises on failure.
    """
    prompt_str = format_prompt(text, language, tokenizer)
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)

    do_sample = temperature > 0 and num_samples > 1
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=10,
        do_sample=do_sample,
        num_return_sequences=num_samples if do_sample else 1,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95
    else:
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    # Decode each of the K generated sequences
    prompt_len = inputs["input_ids"].shape[1]
    n_seqs = output_ids.shape[0]
    parsed_scores = []
    raw_texts = []

    for k in range(n_seqs):
        generated_ids = output_ids[k, prompt_len:]
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        raw_texts.append(raw_text)
        score = parse_confidence(raw_text)
        if score is not None:
            parsed_scores.append(score)

    if not parsed_scores:
        raise ValueError(
            f"Could not parse confidence from any of {n_seqs} samples: "
            f"{raw_texts!r}"
        )

    mean_score = sum(parsed_scores) / len(parsed_scores)
    return mean_score, parsed_scores


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
        description="Classify political texts using verbalized confidence "
        "(0-100 scale) via Llama 3.3 70B."
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
        default=None,
        help="Model path to load (default: auto-download to ~/models/)",
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
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7). Only used when --num-samples > 1.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of sampled completions per document (default: 1 = greedy). "
        "The final score is the mean across K parsed samples.",
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

    # Resolve model path (download from manifold if needed)
    model_path = args.model or ensure_model_local(MODEL_LOCAL_PATH, MANIFOLD_PATH)

    # Log sampling config
    if args.num_samples > 1:
        print(f"  Sampling: K={args.num_samples}, temperature={args.temperature}")
    else:
        print("  Mode: greedy (single completion per document)")

    # Load model
    model, tokenizer, device = load_model(model_path, use_4bit=not args.no_4bit)

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
            mean_score, individual_scores = classify_single(
                text, language, model, tokenizer, device,
                temperature=args.temperature,
                num_samples=args.num_samples,
            )
            # Store individual scores as semicolon-separated for transparency
            scores_str = ";".join(f"{s:.4f}" for s in individual_scores)
            results_buffer.append({
                "id": doc_id,
                "score": f"{mean_score:.6f}",
                "token": scores_str,
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
