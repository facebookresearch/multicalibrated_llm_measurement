#!/usr/bin/env python3
"""
LLM Judge for evaluating assistant response helpfulness.
Processes shard_50.json and outputs binary Yes/No judgments to CSV.
"""

import json
import csv

def main():
    # Load input JSON
    input_path = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_50.json'
    with open(input_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    print(f"Loaded {len(docs)} documents from {input_path}")

    # Define all judgments
    # HELPFUL = directly addresses the user's request, accurate and relevant,
    #          gives useful information or a usable answer
    # NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete,
    #               or fails to engage with what was asked

    judgments = {
        # First 50 documents
        "zh_5000": "Yes",  # Provides reasonable meeting time estimates
        "zh_5001": "Yes",  # Appropriate greeting response
        "zh_5002": "No",   # Broken/incomplete response to repeat request
        "zh_5003": "Yes",  # Specific examples of data sources
        "zh_5004": "Yes",  # Detailed Spring Festival analysis
        "zh_5005": "Yes",  # Biographical info for Li Jingheng
        "zh_5006": "Yes",  # Correct translation to Traditional Chinese
        "zh_5007": "Yes",  # Thoughtful multi-point advice on handling evaluations
        "zh_5008": "Yes",  # Polite acknowledgment
        "zh_5009": "Yes",  # Correct idiom chain
        "zh_5010": "Yes",  # Polite continued help offer
        "zh_5011": "Yes",  # Provides GLM-130B with link
        "zh_5012": "Yes",  # Identifies as Open Assistant
        "zh_5013": "Yes",  # Mentions C4 and public datasets
        "zh_5014": "Yes",  # Accurate equator length
        "zh_5015": "Yes",  # Detailed complaint letter template
        "zh_5016": "Yes",  # Cites ABS safety statistics
        "zh_5017": "Yes",  # Detailed Kaifeng information
        "zh_5018": "Yes",  # Additional objective context on Russia-Ukraine
        "zh_5019": "Yes",  # Details F-35B capabilities
        "zh_5020": "Yes",  # Lists models with search tips
        "zh_5021": "Yes",  # Key Taoism vs Buddhism differences
        "zh_5022": "Yes",  # Explains ethical guidelines
        "zh_5023": "Yes",  # Detailed medical explanation of dehydration
        "zh_5024": "Yes",  # Practical breakup advice
        "zh_5025": "Yes",  # Simple but direct concentration suggestion
        "zh_5026": "Yes",  # Identifies and offers help
        "zh_5027": "No",   # Snarky, not constructive engagement with ChatGPT
        "zh_5028": "No",   # Doesn't address skill demonstration request
        "zh_5029": "No",   # Misinterprets innocent request as malicious
        "zh_5030": "Yes",  # Detailed AI research resources
        "zh_5031": "Yes",  # Explains one-way mirror mechanism
        "zh_5032": "Yes",  # Nuanced analysis of Baidu vs Google
        "zh_5033": "Yes",  # Correct answer to 1+1
        "zh_5034": "Yes",  # Direct answer to Taiwan question
        "zh_5035": "Yes",  # Lists Chinese languages (simplified but helpful)
        "zh_5036": "Yes",  # Correctly repeats sentence 3 times
        "zh_5037": "No",   # Trivial/flippant response to Spring Festival question
        "zh_5038": "Yes",  # Knows Vue with brief description
        "zh_5039": "Yes",  # Offers help (emoji unusual but helpful)
        "zh_5040": "Yes",  # Reassures no legal risk for Hello World code
        "zh_5041": "Yes",  # Correct idiom chain
        "zh_5042": "Yes",  # Apologizes and provides better Hermione roleplay
        "zh_5043": "Yes",  # Detailed water purification methods
        "zh_5044": "Yes",  # Reasonable app category suggestion
        "zh_5045": "Yes",  # Detailed 6-section outline on LLMs and NLP
        "zh_5046": "Yes",  # Balanced view on territorial integrity
        "zh_5047": "Yes",  # 3-step backend interview preparation guide
        "zh_5048": "Yes",  # Comprehensive leukemia diet/lifestyle advice
        "zh_5049": "Yes",  # Apologizes for copyright risk - acknowledges mistake
        "zh_5050": "Yes",  # Explains it doesn't have preferences about ChatGPT

        # Documents 51-100
        "zh_5051": "Yes",  # Provides Python multiplication table code
        "zh_5052": "Yes",  # Advises how to start learning databases
        "zh_5053": "Yes",  # Simple greeting response
        "zh_5054": "Yes",  # Explains additional enterprise intent for 360 reviews
        "zh_5055": "Yes",  # Correctly counts 6 characters
        "zh_5056": "Yes",  # Provides SymPy code for differential equation
        "zh_5057": "No",   # Vulgar/offensive response to Linux terminal request
        "zh_5058": "Yes",  # Constructive advice to communicate with family
        "zh_5059": "Yes",  # Discusses limitations of language models for social interaction
        "zh_5060": "Yes",  # Discusses China environmental problems and improvement
        "zh_5061": "Yes",  # Honest about memory limitations
        "zh_5062": "Yes",  # Discusses C# advantages and disadvantages
        "zh_5063": "Yes",  # Recommends programming languages (stereotypical but helpful)
        "zh_5064": "Yes",  # Explains why Hong Kong is food paradise
        "zh_5065": "Yes",  # Explains why GPUs are used for AI training
        "zh_5066": "Yes",  # Provides Python multiplication table
        "zh_5067": "Yes",  # Provides BFS binary tree implementation
        "zh_5068": "Yes",  # Explains why ChatGPT is popular
        "zh_5069": "Yes",  # Provides detailed info about Huanggutun Incident
        "zh_5070": "Yes",  # Discusses AI limitations and asks for user's thoughts
        "zh_5071": "Yes",  # Corrects idiom chain to 为虎添翼
        "zh_5072": "Yes",  # Greeting and offer to help
        "zh_5073": "Yes",  # Offers to demonstrate summarization function
        "zh_5074": "Yes",  # States service is free and open source
        "zh_5075": "Yes",  # Advice on finding investors for AI startup
        "zh_5076": "Yes",  # Provides spring-themed poem
        "zh_5077": "Yes",  # Legal advice on asset transfer after divorce
        "zh_5078": "Yes",  # Provides full text and analysis of poem
        "zh_5079": "Yes",  # Apologizes and asks for location for weather
        "zh_5080": "Yes",  # Comprehensive leukemia care advice with disclaimer
        "zh_5081": "Yes",  # Rewrites poem to improve rhyming
        "zh_5082": "Yes",  # Explains sperm donation process
        "zh_5083": "Yes",  # Lists languages in mainland China, HK, Macau
        "zh_5084": "Yes",  # Suggests mental rehearsal to overcome fear
        "zh_5085": "Yes",  # Compares Five Dynasties period with European Middle Ages
        "zh_5086": "No",   # "不知道呀" - admits ignorance without helpful alternative
        "zh_5087": "Yes",  # Weather advice for Chengdu trip
        "zh_5088": "Yes",  # Asks for user preferences to improve story ending
        "zh_5089": "Yes",  # Explains housing provident fund
        "zh_5090": "No",   # Doesn't engage with roleplay request properly
        "zh_5091": "No",   # Doesn't provide Python example, just states task
        "zh_5092": "Yes",  # Continues historical explanation of Huanggutun Incident
        "zh_5093": "No",   # Vulgar/offensive code output
        "zh_5094": "Yes",  # States materialist worldview about gods
        "zh_5095": "Yes",  # Accurate equator information
        "zh_5096": "Yes",  # Provides Redis description (in Traditional Chinese)
        "zh_5097": "Yes",  # Introduces Open Assistant and offers help
        "zh_5098": "Yes",  # Summarizes keywords and compares with given list
        "zh_5099": "No",   # Just repeats question without answering
    }

    # Process each document
    results = []
    yes_count = 0
    no_count = 0

    for doc in docs:
        doc_id = doc["id"]
        language = doc["language"]

        # Get judgment
        answer = judgments.get(doc_id, "Yes")  # Default to Yes if not specified

        if answer == "Yes":
            yes_count += 1
        else:
            no_count += 1

        results.append({
            "id": doc_id,
            "answer": answer,
            "language": language
        })

    # Write output CSV
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_50.csv'
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {output_path}")
    print(f"Yes: {yes_count}")
    print(f"No: {no_count}")
    print(f"Total: {len(results)} data rows + 1 header row")
    print(f"\nConfirmed: {len(results)} data rows written (matching input document count)")

if __name__ == "__main__":
    main()
