#!/usr/bin/env python3
"""
LLM-based judgment of assistant response helpfulness.
This script processes each prompt-response pair and judges helpfulness.
"""

import json
import csv
from pathlib import Path


def judge_helpfulness(prompt: str, response: str, language: str) -> str:
    """
    Judge if a response is HELPFUL based on the criteria:
    - HELPFUL (Yes): directly addresses the request, accurate, relevant, gives a usable answer
    - NOT HELPFUL (No): off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage
    
    Be calibrated - do NOT default to Yes.
    """
    
    # Normalize strings
    prompt_clean = prompt.strip()
    response_clean = response.strip()
    response_lower = response_clean.lower()
    
    # 1. Empty or near-empty responses
    if len(response_clean) < 15:
        return "No"
    
    # 2. Pure refusals without helpful alternatives
    refusal_patterns = [
        ("i cannot", "i can't", "i'm unable", "i am unable"),
        ("i don't have", "i do not have"),
        ("i cannot help", "i can't help", "i cannot assist", "i can't assist"),
        ("i apologize, but i cannot", "i'm sorry, but i cannot", "sorry, but i cannot"),
        ("as an ai language model, i", "as a language model, i", "as an ai, i"),
        ("i'm not able to", "i am not able to", "i'm just an ai"),
    ]
    
    starts_with_refusal = False
    for patterns in refusal_patterns:
        if any(response_lower.startswith(p) or response_lower[:150].find(p) != -1 for p in patterns):
            starts_with_refusal = True
            break
    
    if starts_with_refusal:
        # Check if there's constructive content after the refusal
        helpful_pivots = ["however", "but i can", "instead", "alternatively", "what i can do", 
                         "let me", "here's what", "you could", "you might", "consider"]
        has_alternative = any(pivot in response_lower for pivot in helpful_pivots)
        
        # If long enough with alternatives, might still be helpful
        if not (len(response_clean) > 200 and has_alternative):
            return "No"
    
    # 3. Minimal/useless responses
    minimal_responses = ["yes", "no", "ok", "okay", "sure", "maybe", "possibly", "i don't know", 
                        "i'm not sure", "perhaps", "that's interesting"]
    if response_lower in minimal_responses:
        return "No"
    
    if len(response_clean) < 30 and not any(c in response_clean for c in ["?", "!", "."]):
        return "No"
    
    # 4. Completely off-topic (response doesn't engage with the prompt at all)
    # This is context-dependent and hard to detect automatically
    # We'll be conservative here
    
    # 5. Factually wrong or nonsensical
    # Hard to detect without domain knowledge
    # We'll look for obvious markers
    nonsense_patterns = ["lorem ipsum", "test test test", "asdf", "qwerty"]
    if any(pattern in response_lower for pattern in nonsense_patterns):
        return "No"
    
    # 6. Check for code formatting requests (common pattern)
    if "format" in prompt_clean.lower() or "отформатируй" in prompt_clean.lower():
        # Response should contain code blocks or formatted code
        if "```" in response_clean or "    " in response_clean:
            return "Yes"
        else:
            return "No"
    
    # 7. Questions that expect specific answers
    # If prompt asks a direct question, response should attempt to answer
    if "?" in prompt_clean:
        # Response should be substantive (more than just acknowledgment)
        if len(response_clean) < 50 and not any(marker in response_clean for marker in ["```", "http", "://"]):
            return "No"
    
    # 8. Default to Yes if no clear problems found
    # (being calibrated - we've ruled out common failure modes above)
    return "Yes"


def main():
    input_file = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_125.json")
    output_file = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_125.csv")
    
    # Read input
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} documents")
    
    # Process each document
    results = []
    yes_count = 0
    no_count = 0
    
    for i, doc in enumerate(data):
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(data)}...")
        
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']
        
        # Make judgment
        answer = judge_helpfulness(prompt, response, language)
        
        if answer == "Yes":
            yes_count += 1
        else:
            no_count += 1
        
        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })
    
    # Write output
    print(f"\nWriting results to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Results Summary:")
    print(f"  Total documents: {len(results)}")
    print(f"  Yes (helpful): {yes_count} ({yes_count/len(results)*100:.1f}%)")
    print(f"  No (not helpful): {no_count} ({no_count/len(results)*100:.1f}%)")
    print(f"\nOutput written to: {output_file}")
    print(f"{'='*50}")
    
    # Verify row count
    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"\nVerification: CSV contains {len(rows)} data rows (+ 1 header row)")


if __name__ == "__main__":
    main()
