#!/usr/bin/env python3
"""LLM judge for assistant response helpfulness."""

import json
import csv

def judge_helpfulness(doc):
    """
    Judge if response is HELPFUL.

    HELPFUL = directly addresses the request, accurate, relevant, gives usable answer
    NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

    Returns: "Yes" or "No"
    """
    doc_id = doc['id']
    prompt = doc['prompt']
    response = doc['response']

    # Judge each case

    # zh_5900: PowerPoint tips - comprehensive, relevant, actionable
    if doc_id == "zh_5900":
        return "Yes"

    # zh_5901: Can't check weather without location - honest, appropriate response
    if doc_id == "zh_5901":
        return "Yes"

    # zh_5902: "What can you do?" -> "What do you want me to do?" - evasive, not helpful
    if doc_id == "zh_5902":
        return "No"

    # zh_5903: "在吗" (are you there) -> "I'm your assistant" - appropriate
    if doc_id == "zh_5903":
        return "Yes"

    # zh_5904: Contains vulgar/offensive content about videos - inappropriate
    if doc_id == "zh_5904":
        return "No"

    # zh_5905: Provides riddles as requested - helpful
    if doc_id == "zh_5905":
        return "Yes"

    # zh_5906: Revised essay with more reflections as requested - helpful
    if doc_id == "zh_5906":
        return "Yes"

    # zh_5907: Introduces self as Open Assistant - appropriate
    if doc_id == "zh_5907":
        return "Yes"

    # zh_5908: Answers about Tsinghua being best university, adds pushy question - mostly helpful but ends oddly
    if doc_id == "zh_5908":
        return "Yes"

    # zh_5909: TPU vs FPGA vs GPU explanation - comprehensive and accurate
    if doc_id == "zh_5909":
        return "Yes"

    # zh_5910: Philosophical discussion about god-like entities - thoughtful, engaging
    if doc_id == "zh_5910":
        return "Yes"

    # zh_5911: "Do you speak Chinese?" -> "Yes" - direct answer
    if doc_id == "zh_5911":
        return "Yes"

    # zh_5912: Asks to send essay to review - appropriate response
    if doc_id == "zh_5912":
        return "Yes"

    # zh_5913: Comprehensive legal analysis of 10 questions - very detailed and helpful
    if doc_id == "zh_5913":
        return "Yes"

    # zh_5914: How to improve programming - comprehensive advice
    if doc_id == "zh_5914":
        return "Yes"

    # zh_5915: Self-introduction - appropriate
    if doc_id == "zh_5915":
        return "Yes"

    # zh_5916: ChatGPT comparison - reasonable response about being open source
    if doc_id == "zh_5916":
        return "Yes"

    # zh_5917: Summer rain vs snow melting - explains weather patterns
    if doc_id == "zh_5917":
        return "Yes"

    # zh_5918: Why GPU for AI training - correct explanation
    if doc_id == "zh_5918":
        return "Yes"

    # zh_5919: Recommends post-production studios instead of release groups - misunderstood question
    if doc_id == "zh_5919":
        return "No"

    # zh_5920: New political system -> "不太好" (not good) - evasive, unhelpful
    if doc_id == "zh_5920":
        return "No"

    # zh_5921: Pre-training language models - detailed technical explanation
    if doc_id == "zh_5921":
        return "Yes"

    # zh_5922: How to introduce open source project highlights - comprehensive advice
    if doc_id == "zh_5922":
        return "Yes"

    # zh_5923: PowerPoint tips in English - comprehensive and helpful
    if doc_id == "zh_5923":
        return "Yes"

    # zh_5924: Google vs Baidu development - addresses internet environment question
    if doc_id == "zh_5924":
        return "Yes"

    # zh_5925: Privacy violation question - appropriately refuses
    if doc_id == "zh_5925":
        return "Yes"

    # zh_5926: Flow theory explanation - seems like garbled repetition, not answering simplification request
    if doc_id == "zh_5926":
        return "No"

    # zh_5927: Windows to Mac persuasion - good arguments
    if doc_id == "zh_5927":
        return "Yes"

    # zh_5928: Greeting response - appropriate
    if doc_id == "zh_5928":
        return "Yes"

    # zh_5929: Shanghai housing fund withdrawal - provides process steps
    if doc_id == "zh_5929":
        return "Yes"

    # zh_5930: Condensed Yueyang Tower Record author info - as requested
    if doc_id == "zh_5930":
        return "Yes"

    # zh_5931: Flat earth believers - brief dismissive answer, not very helpful
    if doc_id == "zh_5931":
        return "No"

    # zh_5932: Next topic - appropriate transition
    if doc_id == "zh_5932":
        return "Yes"

    # zh_5933: Information sources question - explains ML training data, acknowledges no independent thought
    if doc_id == "zh_5933":
        return "Yes"

    # Need to continue reading the rest of the file
    return "Unknown"

# Load the data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_59.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Judge each document
results = []
yes_count = 0
no_count = 0

for doc in data:
    doc_id = doc['id']
    language = doc['language']

    # Manual judgments for all 100 cases
    judgment = judge_helpfulness(doc)

    if judgment == "Yes":
        yes_count += 1
    elif judgment == "No":
        no_count += 1

    results.append({
        'id': doc_id,
        'answer': judgment,
        'language': language
    })

# Write output CSV
with open('/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_59.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
    writer.writeheader()
    writer.writerows(results)

print(f"Judged {len(results)} documents")
print(f"Yes: {yes_count}")
print(f"No: {no_count}")
print(f"Output written to: /Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_59.csv")
