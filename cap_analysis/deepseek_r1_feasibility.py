"""
DeepSeek-R1-Distill-Qwen-32B: all 7 elicitation methods on the feasibility sample.

Replicates the Llama 3.3 70B method comparison (feasibility_comparison.md) using a
reasoning model. DeepSeek-R1-Distill produces <think>...</think> reasoning blocks
before answering — these are stripped before parsing.

Methods:
  1. logprob       — P(Yes)/(P(Yes)+P(No)) from next-token logprobs
  2. 1s_0100_k10   — rate 0-100, K=10 temperature samples, average
  3. 1s_top1       — single-stage answer + P(correct)  (Tian et al. 2023)
  4. 1s_top2       — single-stage both answers + P(correct) each (Tian et al.)
  5. 1s_pyn        — single-stage P(Yes)/P(No) directly
  6. 2stage        — two-stage Yes/No then P(correct)  (Tian et al. 2023)
  7. 2stage_nudge  — 2stage + anti-certainty nudge

Model: fp16 across both GPUs (~65GB for 32B params). No quantization needed.
"""

import argparse
import csv
import logging
import math
import os
import re
import sys
import time

import torch
from sklearn.metrics import log_loss, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

MODEL_PATH = os.path.expanduser("~/models/DeepSeek-R1-Distill-Qwen-32B/DeepSeek-R1-Distill-Qwen-32B")

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

# --- Prompt templates (identical to Llama versions) ---

LOGPROB_PROMPT = (
    "The following text is in {language}. "
    "Does it primarily fall under " + CAP_TOPIC_DEF + " "
    "Respond Yes or No.\n\n"
    "Text: {text}\n\n"
    "Answer:"
)

CONF_0100_PROMPT = (
    "The following text is in {language}. "
    "Does it primarily discuss law, crime, or criminal justice? "
    "Rate your confidence from 0 (definitely not) to 100 (definitely yes). "
    "Respond with only a number.\n\n"
    "Text: {text}\n\n"
    "Confidence:"
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

STAGE2_PROMPT_NUDGE = (
    "What is the probability that your answer is correct? "
    "Respond with only a decimal number between 0.0 and 1.0. "
    "Note: a probability of exactly 0.0 or 1.0 would mean absolute certainty, "
    "which is rarely warranted."
)

ONETOP1_PROMPT = (
    "The following text is in {language}. "
    "Does it primarily fall under " + CAP_TOPIC_DEF + "\n\n"
    "Text: {text}\n\n"
    "Provide your answer (Yes or No) and the probability that it is correct "
    "(0.0 to 1.0). Give ONLY the answer and probability, no other words or "
    "explanation.\n\n"
    "Answer: <Yes or No>\n"
    "Probability: <the probability between 0.0 and 1.0 that your answer is "
    "correct, without any extra commentary whatsoever; just the probability!>"
)

ONETOP2_PROMPT = (
    "The following text is in {language}. "
    "Does it primarily fall under " + CAP_TOPIC_DEF + "\n\n"
    "Text: {text}\n\n"
    "Provide both possible answers (Yes and No) and the probability that each "
    "is correct (0.0 to 1.0). Give ONLY the answers and probabilities, no "
    "other words or explanation.\n\n"
    "G1: <most likely answer, Yes or No>\n"
    "P1: <the probability between 0.0 and 1.0 that G1 is correct, without "
    "any extra commentary; just the probability!>\n"
    "G2: <other answer, Yes or No>\n"
    "P2: <the probability between 0.0 and 1.0 that G2 is correct, without "
    "any extra commentary; just the probability!>"
)

TOPK_PROMPT = (
    "The following text is in {language}. "
    "Does it primarily fall under " + CAP_TOPIC_DEF + "\n\n"
    "Text: {text}\n\n"
    "Provide the probabilities for Yes and No. "
    "The two probabilities should sum to 1.0. "
    "Use the format:\n"
    "Yes: <probability>\n"
    "No: <probability>"
)

LANGUAGE_MAP = {
    "Danish": "Danish", "Spanish": "Spanish",
    "Dutch": "Dutch", "English": "English",
    "danish": "Danish", "spanish": "Spanish",
    "dutch": "Dutch", "english": "English",
}


# --- Helpers ---

def strip_think(text):
    """Remove <think>...</think> reasoning blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_yes_no(text):
    text = strip_think(text).strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    match = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def parse_probability(text):
    text = strip_think(text).strip()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match is None:
        return None
    value = float(match.group(1))
    if value > 1 and value <= 100:
        value = value / 100.0
    if value > 1:
        return None
    return value


def parse_confidence_0100(text):
    text = strip_think(text).strip()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match is None:
        return None
    value = float(match.group(1))
    if value > 100:
        return None
    return value / 100.0


def parse_topk(text):
    text = strip_think(text).strip()
    yes_match = re.search(r"[Yy]es[:\s]+(\d+(?:\.\d+)?)", text)
    if yes_match is None:
        return None
    p_yes = float(yes_match.group(1))
    if p_yes > 1 and p_yes <= 100:
        p_yes = p_yes / 100.0
    if p_yes > 1:
        return None
    return p_yes


# --- Model ---

def load_model(model_path):
    print(f"Loading {model_path}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # fp16 across both GPUs — 32B fits in ~65GB
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def generate(messages, model, tokenizer, max_new_tokens=512,
             do_sample=False, temperature=None, num_return_sequences=1):
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95
    else:
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    texts = []
    for i in range(output_ids.shape[0]):
        texts.append(
            tokenizer.decode(output_ids[i, prompt_len:], skip_special_tokens=True).strip()
        )
    return texts


# --- Classification methods ---

def classify_logprob(text, language, model, tokenizer):
    """Logprob: P(Yes)/(P(Yes)+P(No)) from next-token logprobs."""
    user_msg = LOGPROB_PROMPT.format(language=language, text=text)
    messages = [{"role": "user", "content": user_msg}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]

    # Find Yes/No token IDs
    yes_ids = set()
    no_ids = set()
    for w in ["Yes", "yes", " Yes", " yes"]:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.add(ids[0])
    for w in ["No", "no", " No", " no"]:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            no_ids.add(ids[0])

    if not yes_ids or not no_ids:
        raise ValueError(f"Could not find Yes/No token IDs for this tokenizer")

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    yes_lp = max(log_probs[tid].item() for tid in yes_ids)
    no_lp = max(log_probs[tid].item() for tid in no_ids)

    # Numerically stable softmax of two values
    max_lp = max(yes_lp, no_lp)
    p_yes = math.exp(yes_lp - max_lp) / (math.exp(yes_lp - max_lp) + math.exp(no_lp - max_lp))

    token = "Yes" if p_yes >= 0.5 else "No"
    return p_yes, token


def classify_1s_0100_k10(text, language, model, tokenizer):
    """Consistency sampling: 0-100 confidence, K=10 at temperature 0.7, averaged."""
    user_msg = CONF_0100_PROMPT.format(language=language, text=text)
    messages = [{"role": "user", "content": user_msg}]

    raw_texts = generate(
        messages, model, tokenizer,
        max_new_tokens=512,  # reasoning model needs room for <think>
        do_sample=True, temperature=0.7, num_return_sequences=10,
    )

    parsed = []
    for raw in raw_texts:
        val = parse_confidence_0100(raw)
        if val is not None:
            parsed.append(val)

    if not parsed:
        raise ValueError(f"Could not parse any of {len(raw_texts)} samples")

    mean_score = sum(parsed) / len(parsed)
    detail = ";".join(f"{v:.4f}" for v in parsed)
    return mean_score, detail


def classify_1s_top1(text, language, model, tokenizer):
    """Verb. 1S top-1 (Tian et al. 2023): answer + P(correct) in one stage."""
    user_msg = ONETOP1_PROMPT.format(language=language, text=text)
    messages = [{"role": "user", "content": user_msg}]
    raw_texts = generate(messages, model, tokenizer, max_new_tokens=1024)
    raw_text = strip_think(raw_texts[0])

    answer_match = re.search(r"[Aa]nswer[:\s]*(Yes|No|yes|no)", raw_text)
    if answer_match:
        answer = answer_match.group(1).lower()
    else:
        answer = parse_yes_no(raw_text)
        if answer is None:
            raise ValueError(f"Could not parse answer: {raw_text[:200]!r}")

    prob_match = re.search(r"[Pp]robability[:\s]*(\d+(?:\.\d+)?)", raw_text)
    if prob_match is None:
        prob_match = re.search(r"(\d+\.\d+)", raw_text)
    if prob_match is None:
        lines = raw_text.strip().split("\n")
        if len(lines) >= 2:
            prob_match = re.search(r"^(\d+)$", lines[-1].strip())
    if prob_match is None:
        raise ValueError(f"Could not parse probability: {raw_text[:200]!r}")

    confidence = float(prob_match.group(1))
    if confidence > 1 and confidence <= 100:
        confidence /= 100.0
    if confidence > 1:
        raise ValueError(f"Probability out of range: {raw_text[:200]!r}")

    score = confidence if answer == "yes" else 1.0 - confidence
    detail = f"answer={answer};confidence={confidence:.4f};raw={raw_text[:100]}"
    return score, detail


def classify_1s_top2(text, language, model, tokenizer):
    """Verb. 1S top-2 (Tian et al. 2023): both answers + P(correct) each."""
    user_msg = ONETOP2_PROMPT.format(language=language, text=text)
    messages = [{"role": "user", "content": user_msg}]
    raw_texts = generate(messages, model, tokenizer, max_new_tokens=1024)
    raw_text = strip_think(raw_texts[0])

    g1_match = re.search(r"G1[:\s]*(Yes|No|yes|no)", raw_text)
    p1_match = re.search(r"P1[:\s]*(\d+(?:\.\d+)?)", raw_text)
    g2_match = re.search(r"G2[:\s]*(Yes|No|yes|no)", raw_text)
    p2_match = re.search(r"P2[:\s]*(\d+(?:\.\d+)?)", raw_text)

    if g1_match is None or p1_match is None:
        yes_match = re.search(r"[Yy]es[:\s]*(\d+(?:\.\d+)?)", raw_text)
        if yes_match:
            p_yes = float(yes_match.group(1))
            if p_yes > 1 and p_yes <= 100:
                p_yes /= 100.0
            if p_yes > 1:
                raise ValueError(f"P(Yes) out of range: {raw_text[:200]!r}")
            return p_yes, f"raw={raw_text[:100]};p_yes={p_yes:.4f}"
        raise ValueError(f"Could not parse 1S top-2: {raw_text[:200]!r}")

    g1 = g1_match.group(1).lower()
    p1 = float(p1_match.group(1))
    if p1 > 1 and p1 <= 100:
        p1 /= 100.0

    if g2_match and p2_match:
        g2 = g2_match.group(1).lower()
        p2 = float(p2_match.group(1))
        if p2 > 1 and p2 <= 100:
            p2 /= 100.0
        score = p1 if g1 == "yes" else (p2 if g2 == "yes" else 1.0 - p1)
    else:
        score = p1 if g1 == "yes" else 1.0 - p1

    if score > 1:
        raise ValueError(f"Score out of range: {raw_text[:200]!r}")
    return score, f"raw={raw_text[:100]};p_yes={score:.4f}"


def classify_1s_pyn(text, language, model, tokenizer):
    """1S P(Y/N): directly ask for P(Yes) and P(No)."""
    user_msg = TOPK_PROMPT.format(language=language, text=text)
    messages = [{"role": "user", "content": user_msg}]
    raw_texts = generate(messages, model, tokenizer, max_new_tokens=1024)
    raw_text = strip_think(raw_texts[0])

    p_yes = parse_topk(raw_text)
    if p_yes is None:
        raise ValueError(f"Could not parse P(Yes)/P(No): {raw_text[:200]!r}")
    return p_yes, f"raw={raw_text[:100]};p_yes={p_yes:.4f}"


def classify_2stage(text, language, model, tokenizer, nudge=False):
    """Two-stage: Yes/No then P(correct). Optional anti-certainty nudge."""
    # Stage 1
    user_msg = STAGE1_PROMPT.format(language=language, text=text)
    messages = [{"role": "user", "content": user_msg}]
    raw_s1 = generate(messages, model, tokenizer, max_new_tokens=512)[0]

    answer = parse_yes_no(raw_s1)
    if answer is None:
        raise ValueError(f"Could not parse Yes/No from stage 1: {strip_think(raw_s1)[:200]!r}")

    # Stage 2
    messages.append({"role": "assistant", "content": raw_s1})
    s2_prompt = STAGE2_PROMPT_NUDGE if nudge else STAGE2_PROMPT
    messages.append({"role": "user", "content": s2_prompt})
    raw_s2 = generate(messages, model, tokenizer, max_new_tokens=512)[0]

    confidence = parse_probability(raw_s2)
    if confidence is None:
        raise ValueError(f"Could not parse probability: {strip_think(raw_s2)[:200]!r}")

    score = confidence if answer == "yes" else 1.0 - confidence
    detail = f"answer={answer};confidence={confidence:.4f}"
    return score, detail


# --- Data loading ---

def load_feasibility_sample(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Run all 7 elicitation methods with DeepSeek-R1-Distill-Qwen-32B"
    )
    parser.add_argument(
        "--input", required=True, help="Path to feasibility sample CSV"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["logprob", "1s_0100_k10", "1s_top1", "1s_top2", "1s_pyn", "2stage", "2stage_nudge"],
        help="Methods to run (default: all 7)"
    )
    parser.add_argument(
        "--sample", type=int, default=500,
        help="Number of docs to sample for verbalized methods (default: 500)"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH, help="Model path"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.input}...")
    all_rows = load_feasibility_sample(args.input)
    print(f"  {len(all_rows)} documents")

    # Load model
    model, tokenizer = load_model(args.model)

    # Sample for verbalized methods (logprob uses all data)
    import random
    random.seed(42)
    if args.sample and args.sample < len(all_rows):
        sample_rows = random.sample(all_rows, args.sample)
    else:
        sample_rows = all_rows

    method_dispatch = {
        "logprob": ("logprob", all_rows),
        "1s_0100_k10": ("1s_0100_k10", sample_rows),
        "1s_top1": ("1s_top1", sample_rows),
        "1s_top2": ("1s_top2", sample_rows),
        "1s_pyn": ("1s_pyn", sample_rows),
        "2stage": ("2stage", sample_rows),
        "2stage_nudge": ("2stage_nudge", sample_rows),
    }

    for method_name in args.methods:
        if method_name not in method_dispatch:
            print(f"Unknown method: {method_name}, skipping")
            continue

        _, rows = method_dispatch[method_name]
        out_path = os.path.join(args.output_dir, f"{method_name}.csv")

        # Check for existing results (resume)
        done_ids = set()
        existing_results = []
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    done_ids.add(row["id"])
                    existing_results.append(row)
            remaining = [r for r in rows if r.get("id", r.get("id_original", "")) not in done_ids]
            if not remaining:
                print(f"\n=== {method_name}: already complete ({len(done_ids)} docs) ===")
                continue
            print(f"\n=== {method_name}: resuming ({len(done_ids)} done, {len(remaining)} remaining) ===")
            rows = remaining
        else:
            print(f"\n=== {method_name}: {len(rows)} docs ===")

        results = list(existing_results)
        n_errors = 0
        t0 = time.time()

        for i, row in enumerate(tqdm(rows, desc=method_name)):
            doc_id = row.get("id", row.get("id_original", str(i)))
            text = row["text"][:4000]
            language = LANGUAGE_MAP.get(row.get("language", ""), "English")
            label = int(row.get("law_crime", 0))

            try:
                if method_name == "logprob":
                    score, detail = classify_logprob(text, language, model, tokenizer)
                elif method_name == "1s_0100_k10":
                    score, detail = classify_1s_0100_k10(text, language, model, tokenizer)
                elif method_name == "1s_top1":
                    score, detail = classify_1s_top1(text, language, model, tokenizer)
                elif method_name == "1s_top2":
                    score, detail = classify_1s_top2(text, language, model, tokenizer)
                elif method_name == "1s_pyn":
                    score, detail = classify_1s_pyn(text, language, model, tokenizer)
                elif method_name == "2stage":
                    score, detail = classify_2stage(text, language, model, tokenizer, nudge=False)
                elif method_name == "2stage_nudge":
                    score, detail = classify_2stage(text, language, model, tokenizer, nudge=True)
                else:
                    raise ValueError(f"Unknown method: {method_name}")

                results.append({
                    "id": doc_id, "score": f"{score:.6f}",
                    "token": detail, "language": language,
                    "label": label, "error": "",
                })
            except Exception as e:
                n_errors += 1
                results.append({
                    "id": doc_id, "score": "",
                    "token": str(e)[:200], "language": language,
                    "label": label, "error": f"error: {e}",
                })

            # Periodic save
            if (i + 1) % 100 == 0:
                _save_results(results, out_path)

        _save_results(results, out_path)
        elapsed = time.time() - t0

        # Quick stats
        valid = [r for r in results if r["error"] == ""]
        scores = [float(r["score"]) for r in valid]
        labels = [int(r["label"]) for r in valid]

        print(f"  Done: {len(valid)} valid, {n_errors} errors, {elapsed:.0f}s")
        if len(set(labels)) >= 2 and scores:
            auc = roc_auc_score(labels, scores)
            # Clip for log loss
            eps = 1e-7
            clipped = [max(eps, min(1 - eps, s)) for s in scores]
            ll = log_loss(labels, clipped)
            exact0 = sum(1 for s in scores if s < 0.005) / len(scores)
            exact1 = sum(1 for s in scores if s > 0.995) / len(scores)
            mid = sum(1 for s in scores if 0.1 <= s <= 0.9) / len(scores)
            unique = len(set(f"{s:.4f}" for s in scores))
            print(f"  AUC: {auc:.3f}  Log loss: {ll:.3f}")
            print(f"  Exact 0: {exact0:.1%}  Exact 1: {exact1:.1%}  In [.1,.9]: {mid:.1%}  Unique: {unique}")


def _save_results(results, path):
    if not results:
        return
    fieldnames = ["id", "score", "token", "language", "label", "error"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
