"""
Prototype: DeepSeek-R1-Distill-Qwen-32B for verbalized confidence.

Reasoning models produce <think>...</think> tokens before answering.
This may yield better-calibrated confidence scores than standard LLMs.

Uses the two-stage approach (Tian et al. 2023):
  Stage 1: classify Yes/No
  Stage 2: ask P(correct)
  Score = P(correct) if Yes, else 1 - P(correct)
"""

import csv
import math
import random
import re
import time

import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

MODEL_PATH = "/home/flinder/models/DeepSeek-R1-Distill-Qwen-32B/DeepSeek-R1-Distill-Qwen-32B"

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
    "Respond with only a decimal number between 0.0 and 1.0."
)

LANGUAGE_MAP = {
    "Danish": "Danish", "Spanish": "Spanish",
    "Dutch": "Dutch", "English": "English",
}


def strip_think_tags(text):
    """Remove <think>...</think> reasoning blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_yes_no(text):
    text = strip_think_tags(text).strip().lower()
    if text.startswith("yes"):
        return "Yes"
    if text.startswith("no"):
        return "No"
    match = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None


def parse_probability(text):
    text = strip_think_tags(text)
    match = re.search(r"(0?\.\d+|1\.0|0\.0|[01])", text)
    if match:
        val = float(match.group(1))
        if 0.0 <= val <= 1.0:
            return val
    return None


def load_model(model_path):
    print(f"Loading {model_path}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # fp16 across both GPUs — 32B model fits in ~65GB
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def generate(messages, model, tokenizer, max_new_tokens=512):
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()


def classify_two_stage(text, language, model, tokenizer):
    """Two-stage verbalized confidence. Returns (score, answer, confidence, reasoning)."""
    # Stage 1
    user_msg = STAGE1_PROMPT.format(language=language, text=text)
    messages = [{"role": "user", "content": user_msg}]
    raw_s1 = generate(messages, model, tokenizer, max_new_tokens=512)

    answer = parse_yes_no(raw_s1)
    if answer is None:
        raise ValueError(f"Could not parse Yes/No: {raw_s1[:200]!r}")

    # Stage 2
    messages.append({"role": "assistant", "content": raw_s1})
    messages.append({"role": "user", "content": STAGE2_PROMPT})
    raw_s2 = generate(messages, model, tokenizer, max_new_tokens=256)

    confidence = parse_probability(raw_s2)
    if confidence is None:
        raise ValueError(f"Could not parse probability: {raw_s2[:200]!r}")

    score = confidence if answer == "Yes" else 1.0 - confidence

    # Extract reasoning if present
    think_match = re.search(r"<think>(.*?)</think>", raw_s1, re.DOTALL)
    reasoning = think_match.group(1).strip()[:200] if think_match else ""

    return score, answer, confidence, reasoning


def sample_data(path, n_per_group=25, seed=42):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    random.seed(seed)
    by_group = {}
    for r in rows:
        key = (r["country"], int(r["law_crime"]))
        by_group.setdefault(key, []).append(r)
    sample = []
    for key, group in sorted(by_group.items()):
        n = min(n_per_group, len(group))
        sample.extend(random.sample(group, n))
    random.shuffle(sample)
    return sample


def main():
    data_path = "cap_analysis/data/full_sample.csv"
    sample = sample_data(data_path, n_per_group=25)
    print(f"Sample: {len(sample)} docs")

    for c in sorted(set(r["country"] for r in sample)):
        pos = sum(1 for r in sample if r["country"] == c and r["law_crime"] == "1")
        neg = sum(1 for r in sample if r["country"] == c and r["law_crime"] == "0")
        print(f"  {c}: {pos}+ {neg}-")

    model, tokenizer = load_model(MODEL_PATH)

    results = []
    errors = 0
    for row in tqdm(sample, desc="DeepSeek-R1-Distill"):
        text = row["text"][:4000]
        lang = LANGUAGE_MAP[row["language"]]
        label = int(row["law_crime"])

        try:
            score, answer, conf, reasoning = classify_two_stage(
                text, lang, model, tokenizer
            )
            results.append({
                "id": row["id"], "country": row["country"],
                "label": label, "score": score,
                "answer": answer, "confidence": conf,
                "reasoning_preview": reasoning,
            })
        except Exception as e:
            errors += 1
            tqdm.write(f"  Error id={row['id']}: {e}")
            results.append({
                "id": row["id"], "country": row["country"],
                "label": label, "score": None,
                "answer": None, "confidence": None,
                "reasoning_preview": str(e)[:200],
            })

    valid = [r for r in results if r["score"] is not None]
    labels = [r["label"] for r in valid]
    scores = [r["score"] for r in valid]

    print(f"\n=== RESULTS ({len(valid)} valid, {errors} errors) ===")
    if len(set(labels)) >= 2:
        print(f"  AUC: {roc_auc_score(labels, scores):.3f}")

    # Score distribution
    above99 = sum(1 for s in scores if s > 0.99) / len(scores)
    below01 = sum(1 for s in scores if s < 0.01) / len(scores)
    mid = sum(1 for s in scores if 0.01 <= s <= 0.99) / len(scores)
    print(f"  < 0.01: {below01:.1%} | [0.01, 0.99]: {mid:.1%} | > 0.99: {above99:.1%}")

    # Per-country AUC
    print("\n=== PER COUNTRY ===")
    for c in sorted(set(r["country"] for r in valid)):
        c_v = [r for r in valid if r["country"] == c]
        c_labels = [r["label"] for r in c_v]
        if len(set(c_labels)) < 2:
            continue
        auc = roc_auc_score(c_labels, [r["score"] for r in c_v])
        print(f"  {c:20s}  AUC: {auc:.3f}  n={len(c_v)}")

    # Show some reasoning examples
    print("\n=== SAMPLE REASONING ===")
    examples = [r for r in valid if r["reasoning_preview"]][:5]
    for r in examples:
        print(f"  id={r['id']} label={r['label']} answer={r['answer']} "
              f"conf={r['confidence']:.2f} score={r['score']:.3f}")
        print(f"    thinking: {r['reasoning_preview'][:150]}")

    # Save
    out_path = "cap_analysis/data/inference_output/deepseek_r1_distill_prototype.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
