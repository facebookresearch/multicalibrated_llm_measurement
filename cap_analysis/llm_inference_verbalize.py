"""
LLM inference pipeline for CAP classification with verbalized reasoning.

Like llm_inference_cuda.py but generates full text responses instead of
only extracting Yes/No log-probabilities. Produces both a continuous score
(from log-probs) and the model's verbalized explanation.

Supports sharding for parallel execution across multiple processes/GPUs:
    --num-shards K  --shard-id I  (0-indexed, processes shard I of K)
"""

import argparse
import csv
import math
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"

PROMPT_TEMPLATE = (
    "The following text is in {language}. "
    "Does it primarily discuss law, crime, or criminal justice? "
    "First explain your reasoning in 1-2 sentences, then answer Yes or No.\n\n"
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

SAVE_INTERVAL = 500
MAX_NEW_TOKENS = 150


def load_model(model_id, use_4bit=True):
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
                dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                dtype=torch.float16,
            )
    else:
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
        )

    model.eval()
    print(f"  Model loaded on {device} in {time.time() - t0:.1f}s.")
    return model, tokenizer, device


def get_yes_no_token_ids(tokenizer):
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


def compute_score(logits, yes_ids, no_ids):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    yes_logprobs = [log_probs[tid].item() for tid in yes_ids]
    no_logprobs = [log_probs[tid].item() for tid in no_ids]

    yes_lp = max(yes_logprobs)
    no_lp = max(no_logprobs)

    max_lp = max(yes_lp, no_lp)
    p_yes = math.exp(yes_lp - max_lp)
    p_no = math.exp(no_lp - max_lp)

    return p_yes / (p_yes + p_no)


def format_prompt(text, language, tokenizer):
    user_message = PROMPT_TEMPLATE.format(language=language, text=text)
    messages = [{"role": "user", "content": user_message}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt_str


def classify_verbalized(text, language, model, tokenizer, device, yes_ids, no_ids,
                        max_new_tokens=150):
    """Classify a document, returning both log-prob score and verbalized response."""
    prompt_str = format_prompt(text, language, tokenizer)
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        # First get the log-prob score from the first token position
        outputs = model(**inputs)
        first_logits = outputs.logits[0, -1, :]
        score = compute_score(first_logits, yes_ids, no_ids)

        # Then generate the full verbalized response
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_ids = gen_outputs[0, input_len:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return score, response_text


def load_input_data(input_path):
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def shard_rows(rows, num_shards, shard_id):
    return [row for i, row in enumerate(rows) if i % num_shards == shard_id]


def load_existing_results(output_path):
    done_ids = set()
    if not os.path.exists(output_path):
        return done_ids
    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done_ids.add(row["id"])
    return done_ids


def save_results(results, output_path, mode="a"):
    if not results:
        return
    write_header = mode == "w" or not os.path.exists(output_path)
    fieldnames = ["id", "score", "response", "language", "error"]
    with open(output_path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(
        description="Classify political texts with verbalized reasoning "
        "using Llama via HuggingFace Transformers."
    )
    parser.add_argument(
        "--input", required=True, help="Path to input CSV with columns: id, text"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output CSV for scores and responses"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="If set, only process a random sample of N documents",
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Language of the texts (danish/spanish/dutch/english). "
        "If not set, reads from 'language' column in input CSV.",
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_ID,
        help=f"Model ID to load (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Disable 4-bit quantization (use fp16 instead)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh instead of resuming from existing output",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of shards to split the dataset into (default: 1)",
    )
    parser.add_argument(
        "--shard-id", type=int, default=0,
        help="0-indexed shard to process (default: 0)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_NEW_TOKENS,
        help=f"Max tokens to generate per response (default: {MAX_NEW_TOKENS})",
    )
    args = parser.parse_args()

    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        print(f"Error: --shard-id {args.shard_id} out of range "
              f"for --num-shards {args.num_shards}")
        sys.exit(1)

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

    # Apply sharding
    if args.num_shards > 1:
        rows = shard_rows(rows, args.num_shards, args.shard_id)
        print(f"  Shard {args.shard_id}/{args.num_shards}: {len(rows)} documents.")

    # Determine output path (append shard suffix if sharded)
    output_path = args.output
    if args.num_shards > 1:
        base, ext = os.path.splitext(args.output)
        output_path = f"{base}_shard{args.shard_id}{ext}"

    # Check for resume
    if not args.no_resume:
        done_ids = load_existing_results(output_path)
        if done_ids:
            rows = [r for r in rows if r["id"] not in done_ids]
            print(f"  Resuming: {len(done_ids)} already done, {len(rows)} remaining.")
    else:
        if os.path.exists(output_path):
            os.remove(output_path)

    if not rows:
        print("No documents to process. Done.")
        return

    # Load model
    max_new_tokens = args.max_tokens
    model, tokenizer, device = load_model(args.model, use_4bit=not args.no_4bit)

    # Find Yes/No token IDs
    yes_ids, no_ids = get_yes_no_token_ids(tokenizer)
    print(f"  Yes token IDs: {yes_ids}")
    print(f"  No token IDs: {no_ids}")

    # Process documents
    results_buffer = []
    n_errors = 0
    t_start = time.time()

    for i, row in enumerate(tqdm(rows, desc=f"Classifying (shard {args.shard_id})")):
        doc_id = row["id"]
        text = row["text"]

        if default_language:
            language = default_language
        elif "language" in row:
            lang_key = row["language"].lower()
            language = LANGUAGE_MAP.get(lang_key, row["language"])
        else:
            print(f"Error: no language specified for document {doc_id} "
                  "and no --language flag set.")
            sys.exit(1)

        max_chars = 4000
        if len(text) > max_chars:
            text = text[:max_chars]

        try:
            score, response_text = classify_verbalized(
                text, language, model, tokenizer, device, yes_ids, no_ids,
                max_new_tokens=max_new_tokens,
            )
            results_buffer.append({
                "id": doc_id,
                "score": f"{score:.6f}",
                "response": response_text,
                "language": language,
                "error": "",
            })
        except Exception as e:
            n_errors += 1
            results_buffer.append({
                "id": doc_id,
                "score": "",
                "response": "",
                "language": language,
                "error": str(e),
            })

        if len(results_buffer) >= SAVE_INTERVAL:
            save_results(results_buffer, output_path)
            results_buffer = []
            elapsed = time.time() - t_start
            docs_done = i + 1
            rate = docs_done / elapsed
            remaining = (len(rows) - docs_done) / rate if rate > 0 else 0
            tqdm.write(
                f"  Saved checkpoint at {docs_done}/{len(rows)} docs "
                f"({rate:.1f} docs/sec, ~{remaining/3600:.1f}h remaining)"
            )

    save_results(results_buffer, output_path)

    elapsed = time.time() - t_start
    print(f"\nDone. Processed {len(rows)} documents in {elapsed:.0f}s "
          f"({len(rows)/elapsed:.1f} docs/sec).")
    print(f"  Errors: {n_errors}")
    print(f"  Output: {output_path}")

    if args.num_shards > 1:
        print(f"\n  This was shard {args.shard_id} of {args.num_shards}.")
        print(f"  To merge all shards, concatenate the shard CSVs:")
        base, ext = os.path.splitext(args.output)
        print(f"    head -1 {base}_shard0{ext} > {args.output}")
        print(f"    for i in $(seq 0 {args.num_shards - 1}); do "
              f"tail -n+2 {base}_shard$i{ext} >> {args.output}; done")


if __name__ == "__main__":
    main()
