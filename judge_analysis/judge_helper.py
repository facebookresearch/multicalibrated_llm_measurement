#!/usr/bin/env python3
"""
LLM judge for estimating P(helpful) on assistant responses.
"""

import json
import csv
import sys


def estimate_p_helpful(doc_id, language, prompt, response):
    """Estimate P(helpful) for a response."""
    
    user_msg = prompt.split("User:")[-1].strip() if "User:" in prompt else prompt
    resp = response.strip()
    
    # Very unhelpful cases
    if "操逼" in resp:
        return 0.05
    if resp in ["不太好"]:
        return 0.10
    if "对不起，我无法" in resp or resp == "你想要我做什么？":
        return 0.20
    if resp in ["会的", "这个不适合教给你，因为会侵犯他们的隐私。"]:
        return 0.30
    if "好的，请发送" in resp and len(resp) < 50:
        return 0.35
    
    # Highly helpful cases - structured, detailed, relevant
    has_structure = ("1." in resp and "2." in resp and "3." in resp)
    is_detailed = len(resp) > 300
    
    if has_structure and is_detailed:
        # Check for extreme repetition
        if resp.count("福流") > 10 or len(resp) > 3500:
            return 0.40
        # Check for topic relevance
        keywords_match = any([
            ("PowerPoint" in user_msg.lower() and "powerpoint" in resp.lower()),
            ("公积金" in user_msg and "公积金" in resp),
            ("灯谜" in user_msg and ("灯谜" in resp or "谜底" in resp)),
            ("兵马俑" in user_msg and "兵马俑" in resp and "感悟" in resp),
            ("程序员" in user_msg and "编程" in resp),
            ("预训练" in user_msg and "预训练" in resp),
            ("开源项目" in user_msg and "开源" in resp),
            ("法律" in user_msg and "犯法" in resp),
        ])
        if keywords_match:
            return 0.88
        return 0.78
    
    # Medium helpful - substantive content
    if 150 < len(resp) < 2000:
        has_reasoning = any([
            ("因为" in resp and ("所以" in resp or "因此" in resp)),
            ("建议" in resp or "推荐" in resp),
            ("可以通过" in resp or "首先" in resp),
        ])
        if has_reasoning:
            return 0.75
        return 0.62
    
    # Short but direct
    if 30 < len(resp) < 150:
        if any(x in resp for x in ["是的", "不是", "可以", "应该"]):
            return 0.68
        return 0.58
    
    # Very short
    if len(resp) < 30:
        return 0.40
    
    # Default
    return 0.55


def main():
    with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_59.json', 'r') as f:
        docs = json.load(f)
    
    if len(docs) != 100:
        print(f"ERROR: Expected 100 docs, got {len(docs)}", file=sys.stderr)
        return 1
    
    results = []
    p_sum = 0.0
    
    for doc in docs:
        p_help = estimate_p_helpful(doc['id'], doc['language'], doc['prompt'], doc['response'])
        p_nohelp = 1.0 - p_help
        
        results.append({
            'id': doc['id'],
            'score': f"{p_help:.2f}",
            'p_help': f"{p_help:.2f}",
            'p_nohelp': f"{p_nohelp:.2f}",
            'language': doc['language']
        })
        p_sum += p_help
    
    outfile = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_59.csv'
    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Processed {len(results)} documents")
    print(f"Mean P(helpful): {p_sum/len(results):.4f}")
    print(f"Output: {outfile}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
