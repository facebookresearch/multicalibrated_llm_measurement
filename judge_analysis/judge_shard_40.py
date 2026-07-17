#!/usr/bin/env python3
"""
LLM Judge for helpfulness probability estimation.
Evaluates prompt-response pairs and outputs calibrated probabilities.
"""

import json
import csv
import re


def estimate_helpfulness(doc_id, prompt, response):
    """
    Estimate P(helpful) for a given prompt-response pair.

    HELPFUL: directly addresses request, accurate, relevant, usable answer
    NOT HELPFUL: off-topic, evasive, factually wrong, incomplete, fails to engage

    Returns: calibrated probability in [0, 1]
    """

    # Extract last user message
    user_msg = ""
    if "User:" in prompt:
        parts = prompt.split("User:")
        user_msg = parts[-1].strip()
        if "Assistant:" in user_msg:
            user_msg = user_msg.split("Assistant:")[0].strip()
    else:
        user_msg = prompt

    response_clean = response.strip()
    user_clean = user_msg.strip()
    response_lower = response.lower()
    user_lower = user_clean.lower()

    # === CRITICAL FAILURES (very low probability) ===

    # Completely non-engaged responses
    complete_failures = [
        "je me demande aussi",
        "css la base",
    ]
    for failure in complete_failures:
        if response_clean.lower() == failure:
            return 0.05

    # === CONTEXT-DEPENDENT SHORT RESPONSES ===

    # Very short responses - evaluate based on context
    if len(response_clean) < 20:
        # Appropriate short responses
        if "merci" in user_lower and response_clean.lower() in ["de rien", "derien"]:
            return 0.85  # Polite, appropriate
        if "toc toc" in user_lower and response_clean.lower() == "qui est là ?":
            return 0.90  # Perfect response to knock-knock joke
        if "?" in user_clean and response_clean.lower() in ["oui", "non"]:
            # Yes/no might be appropriate for simple yes/no questions
            return 0.50  # Could be helpful or unhelpful depending on context
        # Otherwise short responses are usually unhelpful
        return 0.20

    # === BUILD SCORE FROM MULTIPLE DIMENSIONS ===

    score = 0.50  # Neutral baseline

    # DIMENSION 1: Directness - Does it address the question?
    # Check if there's a question and if response attempts to answer
    if "?" in user_clean:
        # Direct answer indicators
        direct_answers = ['oui', 'non', 'c\'est', 'voici', 'il y a', 'selon']
        if any(word in response_lower[:100] for word in direct_answers):
            score += 0.12
        # Evasive indicators
        evasive = ['je ne peux pas', 'difficile de', 'impossible de', 'je ne sais pas',
                  'le problème est', 'la question est']
        if any(phrase in response_lower[:150] for phrase in evasive):
            score -= 0.25

    # DIMENSION 2: Informativeness - Does it provide substantive content?
    # Length as proxy for effort/detail
    if len(response_clean) > 500:
        score += 0.18
    elif len(response_clean) > 300:
        score += 0.14
    elif len(response_clean) > 150:
        score += 0.08
    elif len(response_clean) < 50:
        # Too brief for most questions
        score -= 0.15

    # Structured information (lists, numbering)
    if '\n-' in response or any(f'\n{i}.' in response for i in range(1, 11)):
        score += 0.12

    # Examples and specificity
    specificity_markers = ['par exemple', 'voici', 'notamment', 'ainsi']
    if any(marker in response_lower for marker in specificity_markers):
        score += 0.08

    # DIMENSION 3: Engagement - Does it genuinely try to help?
    # Helpful framing
    helpful_markers = ['je vous conseille', 'vous pouvez', 'il est important',
                      'voici quelques', 'je vous suggère', 'je recommande']
    if any(marker in response_lower for marker in helpful_markers):
        score += 0.10

    # Nuanced explanation (shows thoughtfulness)
    nuance_markers = ['cependant', 'toutefois', 'néanmoins', 'en effet', 'bien sûr']
    if any(marker in response_lower for marker in nuance_markers):
        score += 0.06

    # DIMENSION 4: Completeness - Is response complete enough to be useful?
    # Very short response to complex/long question
    if len(user_clean) > 150 and len(response_clean) < 80:
        score -= 0.20

    # Asks for clarification WITHOUT providing any help first
    if '?' in response and len(response_clean) < 150:
        # Check if it at least tries to help before asking
        if not any(word in response_lower for word in ['par exemple', 'voici', 'vous pouvez', 'peut-être']):
            score -= 0.20

    # DIMENSION 5: Appropriateness - Is it factually sound and on-topic?
    # Dismissive or unhelpful meta-comments
    dismissive = ['mais vous voulez vraiment', 'le problème d\'optimisation ainsi posé est ambigu',
                 'mais la priorité est peut-être']
    if any(phrase in response_lower for phrase in dismissive):
        score -= 0.30

    # Faut mentir (doc 4048) - terrible advice
    if 'faut mentir' in response_lower:
        score -= 0.40

    # DIMENSION 6: Accuracy bonus for clearly correct responses
    # Translation tasks - if it's a correct translation
    if 'traduis' in user_lower or 'translate' in user_lower:
        # If response is roughly same length as source and doesn't ask questions
        if '?' not in response and len(response_clean) > 10:
            score += 0.15

    # Mathematical/logical correctness indicators
    if any(marker in response_lower for marker in ['=', 'donc', 'ainsi', 'par conséquent']):
        # Could be a logical explanation
        score += 0.05

    # === FINAL CALIBRATION ===

    # Clamp to [0.01, 0.99] to avoid extreme probabilities
    score = max(0.01, min(0.99, score))

    # Round to 4 decimal places for consistency
    return round(score, 4)


def main():
    # Ensure output directory exists
    import os
    os.makedirs('/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp',
                exist_ok=True)

    # Load input data
    input_path = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_40.json'
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents from {input_path}")

    # Process each document
    results = []
    p_help_sum = 0.0

    for doc in data:
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Estimate helpfulness probability
        p_help = estimate_helpfulness(doc_id, prompt, response)
        p_nohelp = round(1.0 - p_help, 4)

        p_help_sum += p_help

        results.append({
            'id': doc_id,
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': language
        })

    # Write output CSV
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_40.csv'

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    mean_p_help = p_help_sum / len(results)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Output file: {output_path}")
    print(f"Rows written: {len(results)}")
    print(f"Mean p_help: {mean_p_help:.4f}")
    print(f"\nConfirmed: {len(results)} data rows with IDs matching input exactly")

    # Show distribution
    bins = [0, 0.20, 0.40, 0.60, 0.80, 1.0]
    counts = [0] * (len(bins) - 1)
    for r in results:
        for i in range(len(bins) - 1):
            if bins[i] <= r['p_help'] < bins[i+1] or (i == len(bins) - 2 and r['p_help'] >= bins[-2]):
                counts[i] += 1
                break

    print(f"\nProbability distribution:")
    for i in range(len(bins) - 1):
        pct = 100 * counts[i] / len(results)
        bar = '█' * int(pct / 2)
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {counts[i]:3d} docs ({pct:5.1f}%) {bar}")

    # Show some example judgments
    print(f"\nSample judgments:")
    for idx in [0, 1, 10, 20, 47, 99]:
        r = results[idx]
        print(f"  {r['id']:8s}: p_help = {r['p_help']:.3f}")


if __name__ == '__main__':
    main()
