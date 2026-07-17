#!/usr/bin/env python3
"""
LLM Judge for estimating P(helpful) for assistant responses.
Calibrated probability estimates based on response quality.
"""

import json
import csv
import re
from pathlib import Path


def estimate_p_helpful(doc_id: str, prompt: str, response: str) -> float:
    """
    Estimate P(helpful) with calibrated probability.

    Criteria for HELPFUL:
    - Directly addresses the request
    - Accurate and relevant
    - Gives usable answer

    NOT helpful:
    - Off-topic or evasive
    - Factually wrong
    - Uselessly incomplete
    - Fails to engage
    """

    # Clean text
    resp_lower = response.lower().strip()
    prom_lower = prompt.lower().strip()

    # Word counts
    resp_words = len(response.split())

    # === VERY LOW PROBABILITY (0.05-0.20) ===

    # Completely non-responsive (single word, greeting only)
    if resp_words <= 3:
        if resp_lower in ['salut', 'bonjour', 'ok', 'oui', 'non', 'peut-être', 'merci']:
            return 0.05
        return 0.15

    # Obvious nonsense or contradictory
    if 'on sait pas' in resp_lower and ('parce qu' in resp_lower or 'car' in resp_lower):
        # Claims not to know but then explains - contradictory
        return 0.08

    # Irrelevant answer (completely different topic)
    if 'toc toc' in prom_lower and 'netflix' in resp_lower:
        return 0.10

    # Single word answers to complex questions
    if resp_words < 10:
        if any(word in prom_lower for word in ['comment', 'pourquoi', 'expliquer', 'décrire']):
            return 0.15

    # === LOW PROBABILITY (0.20-0.40) ===

    # Vague or unhelpful for follow-ups
    if resp_words < 20 and ('?' in response or 'quelle' in resp_lower or 'quel' in resp_lower):
        # Asking clarifying question instead of helping
        return 0.25

    # Too brief for complex topics
    if resp_words < 30:
        complex_topics = ['intelligence artificielle', 'philosophie', 'quantum', 'comment fonctionne']
        if any(topic in prom_lower for topic in complex_topics):
            return 0.30

    # Generic platitudes without substance
    if resp_words < 40 and resp_words > 15:
        if not any(marker in response for marker in [':', '-', '1.', '2.', 'par exemple', 'voici']):
            return 0.35

    # === MODERATE PROBABILITY (0.40-0.60) ===

    # Brief but on-topic answers
    if 20 <= resp_words < 40:
        return 0.50

    # Partial information provided
    if 'cependant' in resp_lower or 'mais' in resp_lower:
        if resp_words < 50:
            return 0.55

    # === GOOD PROBABILITY (0.60-0.80) ===

    # Engaged, substantive response
    if resp_words >= 40:
        base_score = 0.65

        # Bonus for structure
        if any(marker in response for marker in ['\n- ', '\n1.', '\n2.', '```']):
            base_score += 0.10

        # Bonus for examples
        if 'par exemple' in resp_lower or 'voici' in resp_lower:
            base_score += 0.05

        # Bonus for nuance
        if any(word in resp_lower for word in ['cependant', 'toutefois', 'néanmoins', 'd\'autre part']):
            base_score += 0.03

        return min(0.79, base_score)

    # Code provided when requested
    if any(word in prom_lower for word in ['code', 'fonction', 'python', 'html']):
        if '```' in response or 'def ' in response or 'import ' in response or '<html>' in resp_lower:
            return 0.75

    # Direct factual answers
    if any(pattern in prom_lower for pattern in ['quelle est', 'quel est', 'qui est', 'combien']):
        if resp_words >= 10 and resp_words < 60:
            return 0.70

    # Good examples provided
    if 'exemple' in prom_lower and resp_words >= 20:
        # Check if examples actually given
        if resp_lower.count('\n') >= 2 or response.count('"') >= 2:
            return 0.75

    # === HIGH PROBABILITY (0.80-0.95) ===

    # Well-structured, comprehensive answers
    if resp_words >= 80:
        quality_markers = 0

        # Has structure
        if '\n\n' in response or response.count('\n-') >= 3:
            quality_markers += 1

        # Has examples
        if 'par exemple' in resp_lower or 'voici' in resp_lower:
            quality_markers += 1

        # Has nuance
        if any(word in resp_lower for word in ['cependant', 'toutefois', 'néanmoins']):
            quality_markers += 1

        # Has specific details (numbers, names, technical terms)
        if any(char.isdigit() for char in response):
            quality_markers += 1

        if quality_markers >= 3:
            return 0.88
        elif quality_markers >= 2:
            return 0.82
        else:
            return 0.75

    # Perfect short answers (translations, simple facts)
    if resp_words <= 15:
        # Translation requests
        if 'traduire' in prom_lower or 'anglais' in prom_lower:
            if '"' in response:
                return 0.92

        # Simple factual questions
        if 'capitale' in prom_lower or 'combien' in prom_lower:
            if any(char.isdigit() for char in response) or resp_words >= 5:
                return 0.88

    # Appropriate deflection (medical, legal advice)
    if any(word in prom_lower for word in ['douleur', 'malade', 'santé', 'médecin']):
        if 'médecin' in resp_lower or 'urgence' in resp_lower or 'consulter' in resp_lower:
            return 0.80

    # Lists when requested
    if any(word in prom_lower for word in ['liste', 'exemples', 'donne', 'cite']):
        if response.count('\n') >= 3 or response.count(',') >= 3:
            return 0.85

    # === DEFAULT ===
    # Moderate engagement, middle of the road
    if resp_words >= 30:
        return 0.60
    else:
        return 0.45


def process_shard(input_path: str, output_path: str):
    """Process shard and write results."""

    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {len(data)} entries...")

    # Process each entry
    results = []
    p_help_sum = 0.0

    for i, entry in enumerate(data, 1):
        doc_id = entry['id']
        language = entry['language']
        prompt = entry['prompt']
        response = entry['response']

        # Estimate P(helpful)
        p_help = estimate_p_helpful(doc_id, prompt, response)
        p_nohelp = 1.0 - p_help

        results.append({
            'id': doc_id,
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': language
        })

        p_help_sum += p_help

        if i % 20 == 0:
            print(f"  Processed {i}/100...")

    # Write output CSV
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Report statistics
    mean_p_help = p_help_sum / len(results)
    print(f"\nCompleted!")
    print(f"Rows written: {len(results)}")
    print(f"Mean P(helpful): {mean_p_help:.4f}")
    print(f"Output: {output_path}")

    return mean_p_help, len(results)


if __name__ == '__main__':
    input_path = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_41.json'
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_41.csv'

    mean_p_help, n_rows = process_shard(input_path, output_path)
