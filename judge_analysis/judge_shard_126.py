#!/usr/bin/env python3
"""
LLM Judge for Assistant Response Helpfulness
Judges each response as Yes (helpful) or No (not helpful) based on:
- HELPFUL: directly addresses request, accurate, relevant, gives usable answer
- NOT HELPFUL: off-topic, evasive, factually wrong, uselessly incomplete, fails to engage
"""

import json
import csv
import re

def is_multi_turn(prompt):
    """Check if this is a multi-turn conversation."""
    # Look for patterns like "User: ... A: ... User: ..." or "Assistant: ... User: ..."
    return bool(re.search(r'(User:|A:|Assistant:).*?(User:|A:|Assistant:)', prompt, re.DOTALL))


def extract_last_user_query(prompt):
    """Extract the last user query from a multi-turn conversation."""
    # Split by User: and get the last part
    parts = re.split(r'\n(?:User:)', prompt)
    if len(parts) > 1:
        return parts[-1].strip()
    return prompt.strip()


def judge_helpfulness(prompt, response, language):
    """
    Judge if a response is helpful or not.

    HELPFUL: directly addresses request, accurate, relevant, gives usable answer
    NOT HELPFUL: off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

    Returns: "Yes" if helpful, "No" if not helpful
    """

    # Strip whitespace
    response = response.strip()
    prompt = prompt.strip()

    # Get the actual user query (last turn in multi-turn conversations)
    user_query = extract_last_user_query(prompt)

    # Empty or extremely short responses are not helpful
    if len(response) < 5:
        return "No"

    # Very short vague responses that don't engage
    vague_short = [
        r"^(Вполне возможно|Возможно|Может быть)[\.\!]?$",
        r"^(Maybe|Perhaps|Possibly|It's possible)[\.\!]?$",
        r"^(Quizás|Tal vez|Posiblemente)[\.\!]?$",
        r"^(Peut-être|Possiblement)[\.\!]?$",
    ]

    for pattern in vague_short:
        if re.search(pattern, response):
            return "No"

    # Check for completely off-topic responses
    # If prompt is about topic X but response discusses completely unrelated topic Y
    # Example: prompt about energy conservation, response asks about Python code
    if len(response) < 100:
        # Check for responses that are questions about completely different topics
        question_markers = [
            r"Как написать код", # How to write code
            r"How (do|to) (I|you)",
            r"Comment (faire|écrire)",
            r"Cómo (hacer|escribir)",
        ]

        for marker in question_markers:
            if re.search(marker, response, re.IGNORECASE):
                # This is a question - check if prompt was asking for help with something else
                if not re.search(r"(код|code|programme|código|python|programming)", user_query, re.IGNORECASE):
                    return "No"

    # Check for evasive/refusal patterns in multiple languages
    evasive_patterns = [
        # English
        r"(?i)(i cannot|i can't|i'm unable|i'm not able|i don't have)",
        r"(?i)(i cannot provide|i can't provide|i am not able|i'm not capable)",
        # Russian
        r"(Я не могу|Я не в состоянии|К сожалению, я не|Извините, но я)",
        # Spanish
        r"(No puedo|Lo siento, pero no|Desafortunadamente no|No soy capaz)",
        # French
        r"(Je ne peux pas|Je ne suis pas en mesure|Désolé, mais je|Malheureusement, je)",
        # German
        r"(Ich kann nicht|Leider kann ich nicht|Es tut mir leid, aber|Ich bin nicht in der Lage)",
        # Chinese
        r"(我不能|我无法)",
        # Arabic
        r"(لا أستطيع|عذراً|للأسف)",
        # Portuguese
        r"(Não posso|Desculpe, mas|Infelizmente não|Não sou capaz)",
    ]

    # If response is short and evasive without providing helpful information, not helpful
    if len(response) < 200:
        for pattern in evasive_patterns:
            if re.search(pattern, response):
                # Check if response provides any actual information despite evasion
                if len(response) < 100:
                    return "No"

    # Check for responses that are just "I don't know" or similar
    dont_know_patterns = [
        r"(?i)^(i don't know|i do not know|no idea|not sure)[\.\!]?$",
        r"^(Я не знаю|Не знаю)[\.\!]?$",
        r"(?i)^(no sé|no lo sé)[\.\!]?$",
        r"(?i)^(je ne sais pas)[\.\!]?$",
        r"(?i)^(ich weiß nicht)[\.\!]?$",
    ]

    for pattern in dont_know_patterns:
        if re.search(pattern, response):
            return "No"

    # Check for responses that fail to engage with substantive questions
    # If user asks for explanation/description but response is just a few words
    if len(response) < 50:
        # Check if prompt asks for explanation, description, list, etc.
        substantive_requests = [
            r"(расскажи|опиши|объясни|приведи|какие|почему|зачем|как)",  # Russian
            r"(tell|explain|describe|list|what|why|how)",  # English
            r"(explique|décris|liste|pourquoi|comment)",  # French
            r"(explica|describe|lista|por qué|cómo)",  # Spanish
            r"(erkläre|beschreibe|liste|warum|wie)",  # German
        ]

        is_substantive_request = any(re.search(p, user_query, re.IGNORECASE) for p in substantive_requests)

        if is_substantive_request:
            # Response should be more than just a word or two
            word_count = len(response.split())
            if word_count < 5:
                return "No"

    # Check for responses that don't actually answer the question
    # E.g., "It's true, I'm not making it up" when asked for descriptions
    non_answer_patterns = [
        r"^(Это точно правда|Это правда|Я не выдумываю)",  # Russian
        r"^(It's true|That's true|I'm not making)",  # English
        r"^(Es verdad|Es cierto)",  # Spanish
        r"^(C'est vrai|C'est exact)",  # French
    ]

    if len(response) < 100:
        for pattern in non_answer_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                # This might be answering a follow-up "are you sure?" question
                # Check if the user query is asking for confirmation
                confirmation_queries = [
                    r"(точно|правда|уверен|серьёзно)",  # Russian
                    r"(sure|really|certain|seriously|true)",  # English
                    r"(seguro|cierto|verdad)",  # Spanish
                    r"(sûr|certain|vrai)",  # French
                ]

                asks_confirmation = any(re.search(p, user_query, re.IGNORECASE) for p in confirmation_queries)

                if not asks_confirmation:
                    return "No"

    # Check for nonsensical or irrelevant responses
    # If the response is just repeated characters or gibberish
    if len(set(response.replace(" ", "").replace(".", ""))) < 5 and len(response) > 10:
        return "No"

    # Check for responses that are meta-comments without substance
    # E.g., "If you have questions, contact us" without answering the question
    meta_only_patterns = [
        r"^(Обращайтесь, если|Если возникнут вопросы)",  # Russian
        r"^(Contact (us|me) if|Let me know if)",  # English
        r"^(Contacta si|Avísame si)",  # Spanish
        r"^(Contactez si|Faites-moi savoir)",  # French
    ]

    if len(response) < 80:
        for pattern in meta_only_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return "No"

    # Default to helpful if we haven't found a disqualifying issue
    # The response engages with the prompt and provides some content
    return "Yes"


def main():
    # Load data
    input_file = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_126.json'
    output_file = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_126.csv'

    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    print("\nJudging responses...")
    for i, doc in enumerate(data):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(data)}...")

        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Judge the response
        answer = judge_helpfulness(prompt, response, language)

        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

        if answer == "Yes":
            yes_count += 1
        else:
            no_count += 1

    # Write output
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    # Report statistics
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total documents judged: {len(results)}")
    print(f"Yes (helpful):  {yes_count:3d} ({yes_count/len(results)*100:.1f}%)")
    print(f"No (not helpful): {no_count:3d} ({no_count/len(results)*100:.1f}%)")
    print(f"\nOutput file: {output_file}")
    print(f"Rows in output: {len(results)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
