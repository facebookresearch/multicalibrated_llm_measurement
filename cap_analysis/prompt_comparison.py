"""
Compare current prompt vs CAP codebook definition prompt.

Runs both prompts on a small stratified sample and compares
AUC, score distributions, and agreement.
"""

import csv
import math
import random
import time

import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

MODEL_ID = "/home/flinder/models/Llama-3.3-70B-Instruct"

PROMPT_CURRENT = (
    "The following text is in {language}. "
    "Does it primarily discuss law, crime, or criminal justice? "
    "Respond Yes or No.\n\n"
    "Text: {text}\n\n"
    "Answer:"
)

PROMPT_CODEBOOK = (
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
    "Respond Yes or No.\n\n"
    "Text: {text}\n\n"
    "Answer:"
)

LANGUAGE_MAP = {
    "Danish": "Danish", "Spanish": "Spanish",
    "Dutch": "Dutch", "English": "English",
}


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def get_yes_no_ids(tokenizer):
    yes_ids, no_ids = [], []
    for w in ["Yes", "yes", " Yes", " yes"]:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.append(ids[0])
    for w in ["No", "no", " No", " no"]:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            no_ids.append(ids[0])
    return yes_ids, no_ids


def score_prompt(prompt_str, model, tokenizer, device, yes_ids, no_ids):
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    yes_lp = max(log_probs[t].item() for t in yes_ids)
    no_lp = max(log_probs[t].item() for t in no_ids)
    mx = max(yes_lp, no_lp)
    p_yes = math.exp(yes_lp - mx)
    p_no = math.exp(no_lp - mx)
    return p_yes / (p_yes + p_no)


def sample_data(path, n_per_group=50, seed=42):
    """Stratified sample: n_per_group positive + n_per_group negative per country."""
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
    sample = sample_data(data_path, n_per_group=50)
    print(f"Sample size: {len(sample)}")

    countries = sorted(set(r["country"] for r in sample))
    for c in countries:
        pos = sum(1 for r in sample if r["country"] == c and r["law_crime"] == "1")
        neg = sum(1 for r in sample if r["country"] == c and r["law_crime"] == "0")
        print(f"  {c}: {pos} pos, {neg} neg")

    print("Loading model...")
    model, tokenizer = load_model(MODEL_ID)
    device = next(model.parameters()).device
    yes_ids, no_ids = get_yes_no_ids(tokenizer)

    results = []
    for row in tqdm(sample, desc="Scoring"):
        text = row["text"][:4000]
        lang = LANGUAGE_MAP[row["language"]]
        label = int(row["law_crime"])

        msg_cur = [{"role": "user", "content": PROMPT_CURRENT.format(language=lang, text=text)}]
        msg_cb = [{"role": "user", "content": PROMPT_CODEBOOK.format(language=lang, text=text)}]

        p_cur = tokenizer.apply_chat_template(msg_cur, tokenize=False, add_generation_prompt=True)
        p_cb = tokenizer.apply_chat_template(msg_cb, tokenize=False, add_generation_prompt=True)

        s_cur = score_prompt(p_cur, model, tokenizer, device, yes_ids, no_ids)
        s_cb = score_prompt(p_cb, model, tokenizer, device, yes_ids, no_ids)

        results.append({
            "id": row["id"], "country": row["country"],
            "label": label, "score_current": s_cur, "score_codebook": s_cb,
        })

    # Overall metrics
    labels = [r["label"] for r in results]
    s_cur = [r["score_current"] for r in results]
    s_cb = [r["score_codebook"] for r in results]

    print("\n=== OVERALL ===")
    print(f"  AUC current:  {roc_auc_score(labels, s_cur):.3f}")
    print(f"  AUC codebook: {roc_auc_score(labels, s_cb):.3f}")

    # Score distribution summary
    for name, scores in [("current", s_cur), ("codebook", s_cb)]:
        above99 = sum(1 for s in scores if s > 0.99) / len(scores)
        below01 = sum(1 for s in scores if s < 0.01) / len(scores)
        mid = sum(1 for s in scores if 0.01 <= s <= 0.99) / len(scores)
        print(f"  {name}: {below01:.1%} < 0.01 | {mid:.1%} in [0.01,0.99] | {above99:.1%} > 0.99")

    # Per-country AUC
    print("\n=== PER COUNTRY ===")
    for c in countries:
        c_res = [r for r in results if r["country"] == c]
        c_labels = [r["label"] for r in c_res]
        if len(set(c_labels)) < 2:
            continue
        auc_cur = roc_auc_score(c_labels, [r["score_current"] for r in c_res])
        auc_cb = roc_auc_score(c_labels, [r["score_codebook"] for r in c_res])
        print(f"  {c:20s}  AUC current: {auc_cur:.3f}  codebook: {auc_cb:.3f}")

    # Disagreements (different binary classification at 0.5 threshold)
    disagree = [(r, "cur>cb" if r["score_current"] > 0.5 else "cb>cur")
                for r in results
                if (r["score_current"] > 0.5) != (r["score_codebook"] > 0.5)]
    print(f"\n=== DISAGREEMENTS (threshold=0.5): {len(disagree)} / {len(results)} ===")
    for r, direction in disagree[:20]:
        correct_prompt = "current" if (r["score_current"] > 0.5) == r["label"] else "codebook"
        print(f"  id={r['id']} country={r['country']} label={r['label']} "
              f"cur={r['score_current']:.3f} cb={r['score_codebook']:.3f} "
              f"correct={correct_prompt}")

    # Save full results
    out_path = "cap_analysis/data/inference_output/prompt_comparison.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
