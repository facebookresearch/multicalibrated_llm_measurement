"""
Extract last-token hidden-state embeddings from Llama for CAP documents.

Runs batched forward passes (no generation) with output_hidden_states=True
and saves the final-layer last-token hidden state for each document.

Output:
    - embeddings.npy: float16 array of shape (N, hidden_dim)
    - embedding_ids.csv: row-index-to-document-id mapping

Usage:
    cd cap_analysis
    python extract_embeddings.py \
        --input data/full_sample.csv \
        --output-dir data/inference_output/llama-70b/embeddings \
        --batch-size 8
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


MODEL_ID = (
    "/home/flinder/models/Llama-3.3-70B-Instruct/"
    "model--meta-llama--Llama-3.3-70B-Instruct"
)

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

SAVE_INTERVAL = 500  # batches between checkpoints


def load_model(model_id, use_4bit=True):
    """Load model and tokenizer with optional 4-bit quantization."""
    print(f"Loading model {model_id}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Left-padding for batched forward pass on decoder-only models
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_input_data(input_path):
    """Load input CSV."""
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_batch_embeddings(
    batch_texts,
    batch_languages,
    model,
    tokenizer,
    device,
    max_chars=4000,
):
    """
    Extract last-token hidden states for a batch of documents.

    Returns: numpy array of shape (batch_size, hidden_dim), dtype float16.
    """
    # Format prompts
    prompts = []
    for text, lang in zip(batch_texts, batch_languages):
        truncated = text[:max_chars] if len(text) > max_chars else text
        prompts.append(format_prompt(truncated, lang, tokenizer))

    # Tokenize with left-padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )

    # Last layer hidden states: (batch, seq_len, hidden_dim)
    last_hidden = outputs.hidden_states[-1]

    # Find last real (non-padding) token position per sequence.
    # With left-padding, the last non-pad token is at the rightmost
    # attended position.
    attention_mask = inputs["attention_mask"]  # (batch, seq_len)
    seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)

    # Gather last-token hidden states
    batch_size = last_hidden.size(0)
    embeddings = last_hidden[
        torch.arange(batch_size, device=device), seq_lengths
    ]  # (batch, hidden_dim)

    return embeddings.cpu().to(torch.float16).numpy()


def save_checkpoint(embeddings_list, existing_embeddings, all_ids, paths):
    """Save checkpoint: concatenate all embeddings and write to disk."""
    new_emb = np.concatenate(embeddings_list, axis=0)
    if existing_embeddings is not None:
        combined = np.concatenate([existing_embeddings, new_emb], axis=0)
    else:
        combined = new_emb
    np.savez_compressed(paths["checkpoint"], embeddings=combined)
    with open(paths["ids"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_idx", "id"])
        for i, doc_id in enumerate(all_ids):
            writer.writerow([i, doc_id])
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from Llama for CAP documents."
    )
    parser.add_argument(
        "--input", required=True, help="Path to input CSV (full_sample.csv)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save embeddings.npy and embedding_ids.csv",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help=f"Model path or HF ID (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for forward passes (default: 8)",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from checkpoint",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    paths = {
        "embeddings": os.path.join(args.output_dir, "embeddings.npy"),
        "ids": os.path.join(args.output_dir, "embedding_ids.csv"),
        "checkpoint": os.path.join(args.output_dir, "checkpoint.npz"),
    }

    # Load input data
    print(f"Loading input data from {args.input}...")
    rows = load_input_data(args.input)
    print(f"  Loaded {len(rows)} documents.")

    # Check for resume
    start_idx = 0
    existing_embeddings = None
    existing_ids = []
    if not args.no_resume and os.path.exists(paths["checkpoint"]):
        print("  Found checkpoint, resuming...")
        ckpt = np.load(paths["checkpoint"])
        existing_embeddings = ckpt["embeddings"]
        start_idx = len(existing_embeddings)
        with open(paths["ids"], "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            existing_ids = [r[1] for r in reader]
        print(f"  Resuming from document {start_idx}/{len(rows)}")

    if start_idx >= len(rows):
        print("All documents already processed.")
        return

    # Load model
    model, tokenizer, device = load_model(args.model, use_4bit=not args.no_4bit)

    # Process in batches
    remaining_rows = rows[start_idx:]
    n_batches = (len(remaining_rows) + args.batch_size - 1) // args.batch_size
    total_docs = len(rows)

    all_embeddings = []
    all_ids = list(existing_ids)
    n_errors = 0
    docs_processed = 0
    t_start = time.time()

    print(f"\nExtracting embeddings for {len(remaining_rows)} documents "
          f"(batch_size={args.batch_size}, {n_batches} batches)...")

    for batch_idx in range(n_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(remaining_rows))
        batch_rows = remaining_rows[batch_start:batch_end]

        batch_texts = []
        batch_languages = []
        batch_ids = []

        for row in batch_rows:
            text = row["text"]
            lang_key = row.get("language", "english").lower()
            language = LANGUAGE_MAP.get(lang_key, lang_key)
            batch_texts.append(text)
            batch_languages.append(language)
            batch_ids.append(row["id"])

        try:
            emb = extract_batch_embeddings(
                batch_texts, batch_languages, model, tokenizer, device
            )
            all_embeddings.append(emb)
            all_ids.extend(batch_ids)
        except Exception as e:
            # Fall back to one-at-a-time for this batch
            print(f"  Batch {batch_idx} error: {e}. Processing individually...")
            for text, lang, doc_id in zip(batch_texts, batch_languages, batch_ids):
                try:
                    emb = extract_batch_embeddings(
                        [text], [lang], model, tokenizer, device
                    )
                    all_embeddings.append(emb)
                    all_ids.append(doc_id)
                except Exception as e2:
                    n_errors += 1
                    print(f"  Error on doc {doc_id}: {e2}")
                    hidden_dim = model.config.hidden_size
                    all_embeddings.append(
                        np.zeros((1, hidden_dim), dtype=np.float16)
                    )
                    all_ids.append(doc_id)

        docs_processed += len(batch_rows)

        # Progress report every 100 batches
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == n_batches:
            elapsed = time.time() - t_start
            rate = docs_processed / elapsed
            remaining_docs = len(remaining_rows) - docs_processed
            eta_sec = remaining_docs / rate if rate > 0 else 0
            eta_h = eta_sec / 3600
            global_done = start_idx + docs_processed
            print(
                f"  [{global_done:,}/{total_docs:,}] "
                f"{docs_processed:,} done in {elapsed:.0f}s "
                f"({rate:.1f} docs/sec) | "
                f"ETA: {eta_h:.1f}h | "
                f"Errors: {n_errors}"
            )

        # Periodic checkpoint
        if (batch_idx + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(all_embeddings, existing_embeddings, all_ids, paths)
            print(f"  Checkpoint saved ({len(all_ids):,} docs total)")

    # Final save
    new_emb = np.concatenate(all_embeddings, axis=0)
    if existing_embeddings is not None:
        combined = np.concatenate([existing_embeddings, new_emb], axis=0)
    else:
        combined = new_emb

    np.save(paths["embeddings"], combined)
    print(f"\nSaved embeddings: {combined.shape} to {paths['embeddings']}")

    with open(paths["ids"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_idx", "id"])
        for i, doc_id in enumerate(all_ids):
            writer.writerow([i, doc_id])
    print(f"Saved ID mapping to {paths['ids']}")

    # Clean up checkpoint
    if os.path.exists(paths["checkpoint"]):
        os.remove(paths["checkpoint"])

    elapsed = time.time() - t_start
    mem_mb = combined.nbytes / 1024**2
    print(
        f"\nDone. Processed {len(remaining_rows):,} documents in {elapsed:.0f}s "
        f"({len(remaining_rows) / elapsed:.1f} docs/sec)."
    )
    print(f"  Errors: {n_errors}")
    print(f"  Embedding shape: {combined.shape}")
    print(f"  Dtype: {combined.dtype}")
    print(f"  File size: {mem_mb:.1f} MB")


if __name__ == "__main__":
    main()
