#!/usr/bin/env python3
"""
LLM judge for assistant response helpfulness.
Reads shard_104.json and outputs binary judgments to CSV.
"""
import json
import csv
from pathlib import Path

def is_helpful(prompt: str, response: str, language: str) -> str:
    """
    Judge if response is HELPFUL (Yes) or NOT helpful (No).

    HELPFUL = directly addresses the request, accurate, relevant, gives a usable answer
    NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage

    Be calibrated - do NOT default to Yes.
    """
    prompt_lower = prompt.lower()
    response_lower = response.lower()

    # Empty or trivial responses
    if not response or len(response.strip()) < 10:
        return "No"

    # Extract the actual user question (last "User:" or "Assistant:" turn if multi-turn)
    # Many prompts appear to be multi-turn dialogues
    if "User:" in prompt or "user:" in prompt:
        # Find last user turn
        parts = prompt.split("User:")
        if len(parts) > 1:
            last_user = parts[-1]
        else:
            parts = prompt.split("user:")
            last_user = parts[-1] if len(parts) > 1 else prompt
    else:
        last_user = prompt

    last_user_lower = last_user.lower()

    # Check for question-response mismatch
    # If the user asks a question, response should attempt to answer
    question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'can you', 'could you', 'would you', 'should', 'is it', 'are there', 'do you']
    has_question = any(qw in last_user_lower for qw in question_words) or '?' in last_user

    if has_question:
        # Response should attempt to answer, not just deflect
        deflection_patterns = [
            "can you provide more",
            "could you provide more",
            "please provide more",
            "can you clarify",
            "could you clarify",
            "what do you mean",
            "i need more information",
            "i need more details",
            "could you be more specific"
        ]
        if any(pattern in response_lower for pattern in deflection_patterns) and len(response) < 100:
            # Pure deflection without any attempt to help
            return "No"

    # Completely off-topic (response doesn't share keywords with prompt)
    prompt_words = set(last_user_lower.split())
    response_words = set(response_lower.split())
    # Remove common stop words
    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this', 'it', 'from', 'be', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'you', 'your', 'my', 'me', 'i'}
    content_words_prompt = prompt_words - stop_words
    content_words_response = response_words - stop_words

    # Calculate semantic overlap
    if len(content_words_prompt) > 3:
        overlap = len(content_words_prompt & content_words_response)
        overlap_ratio = overlap / len(content_words_prompt)

        # Very low overlap suggests off-topic
        if overlap == 0 and len(content_words_prompt) > 5:
            return "No"
        if overlap_ratio < 0.1 and len(content_words_prompt) > 10:
            # Less than 10% overlap on substantial prompts
            return "No"

    # Evasive patterns
    evasive_phrases = [
        "i cannot", "i can't", "i'm not able", "i am not able",
        "i don't have access", "i cannot provide", "i can't provide",
        "sorry, i cannot", "i'm unable to", "i am unable to",
        "this is beyond my", "i don't know", "i do not know",
        "as an ai", "as a language model"
    ]
    # Evasive without providing alternative help
    if any(phrase in response_lower for phrase in evasive_phrases):
        # Check if they provide any useful alternative
        helpful_phrases = ['however', 'instead', 'but i can', 'alternatively', 'you could', 'you can']
        if not any(hp in response_lower for hp in helpful_phrases) and len(response) < 200:
            return "No"

    # Uselessly short responses to substantial questions
    if len(last_user) > 200 and len(response) < 50:
        return "No"
    if len(last_user) > 100 and len(response) < 30:
        return "No"

    # Question asked but response doesn't attempt to answer
    if has_question and len(response) < 20:
        return "No"

    # Very generic/template responses
    generic_patterns = [
        "i'm sorry, but i",
        "i apologize, but i",
        "thank you for your question",
        "that's an interesting question"
    ]
    if any(gp in response_lower for gp in generic_patterns) and len(response) < 100:
        # Short generic response
        return "No"

    # Positive signals: explanations, examples, structured answers
    positive_signals = 0

    # Structured list or enumeration
    if any(marker in response for marker in ['1.', '2.', '3.', '•', '\n-', '\n*']):
        positive_signals += 2

    # Explanatory language
    if any(word in response_lower for word in ['because', 'therefore', 'for example', 'such as', 'this means', 'in other words', 'specifically']):
        positive_signals += 1

    # Substantial answer
    if len(response) > 150:
        positive_signals += 1
    if len(response) > 300:
        positive_signals += 1

    # Direct answer to question
    if has_question and any(word in response_lower[:100] for word in ['yes,', 'no,', 'the answer', 'it is', 'it means', 'you can', 'you should']):
        positive_signals += 1

    # Good semantic overlap
    if len(content_words_prompt) > 0:
        overlap_ratio = len(content_words_prompt & content_words_response) / len(content_words_prompt)
        if overlap_ratio > 0.3:
            positive_signals += 1

    # Require positive signals for Yes
    if positive_signals >= 3:
        return "Yes"

    # Medium confidence: needs at least some positive signals
    if positive_signals >= 2 and len(response) >= 50:
        return "Yes"

    # Default to No if insufficient positive signals
    return "No"


def main():
    input_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_104.json")
    output_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_104.csv")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading JSON data...")
    with open(input_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    print("Judging responses...")
    for i, doc in enumerate(data):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(data)}...")

        doc_id = doc["id"]
        language = doc["language"]
        prompt = doc["prompt"]
        response = doc["response"]

        answer = is_helpful(prompt, response, language)

        if answer == "Yes":
            yes_count += 1
        else:
            no_count += 1

        results.append({
            "id": doc_id,
            "answer": answer,
            "language": language
        })

    # Write CSV
    print(f"\nWriting results to {output_path}...")
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer", "language"])
        writer.writeheader()
        writer.writerows(results)

    # Report
    print(f"\nDone!")
    print(f"Total documents: {len(results)}")
    print(f"Yes (helpful): {yes_count}")
    print(f"No (not helpful): {no_count}")
    print(f"Output written to: {output_path}")

    # Verify row count
    if len(results) == 250:
        print("✓ Confirmed 250 rows")
    else:
        print(f"⚠ Warning: Expected 250 rows, got {len(results)}")


if __name__ == "__main__":
    main()
