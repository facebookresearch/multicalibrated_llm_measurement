#!/usr/bin/env python3
"""
LLM Judge for helpfulness evaluation.
Reads shard_23.json and produces binary Yes/No judgments.
"""

import json
import csv


def is_helpful(doc_id, language, prompt, response):
    """
    Judge if response is HELPFUL to the prompt.

    HELPFUL = directly addresses request, accurate, relevant, usable answer
    NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

    Returns: "Yes" or "No"
    """

    # Extract last user message from prompt (conversational prompts have Assistant:/User: structure)
    if "User:" in prompt:
        user_parts = prompt.split("User:")
        last_user_msg = user_parts[-1].strip()
    else:
        last_user_msg = prompt.strip()

    resp = response.strip()

    # Case-by-case judgment - be calibrated, do NOT default to Yes

    if doc_id == "ru_2300":
        # Requests detailed orange description for alien. Response is comprehensive, structured, engaging.
        return "Yes"

    if doc_id == "ru_2301":
        # User says "Usually you answer very expansively". Response apologizes, asks for rephrasing.
        # Evasive, doesn't address user's observation about previous terse answer. NOT helpful.
        return "No"

    if doc_id == "ru_2302":
        # "What is the meaning in marriage?" Response gives nuanced legal/complexity framing. Helpful.
        return "Yes"

    if doc_id == "ru_2303":
        # Requests essay on why MLP > Soyuzmultfilm. Response delivers essay as requested, with disclaimer.
        return "Yes"

    if doc_id == "ru_2304":
        # "Do horoscopes have real scientific power?" Response: "No. No scientific confirmation." Direct, accurate.
        return "Yes"

    if doc_id == "ru_2305":
        # Asks cost of local Wikipedia. Response: "~40,000 rubles for 20TB cluster." Gives usable answer.
        return "Yes"

    if doc_id == "ru_2306":
        # "Why minority?" Response: "Forex market is unstable." Addresses the question directly.
        return "Yes"

    if doc_id == "ru_2307":
        # Asks for correction variants. Response gives 2 variants (literal preservation + smoothed).
        return "Yes"

    if doc_id == "ru_2308":
        # "Does Google steal personal data?" Response: "Not at all! No user complained about missing data..."
        # Completely misinterprets "steal" (as in privacy invasion) vs physical theft. Factually misleading.
        return "No"

    if doc_id == "ru_2309":
        # Sleep troubles, anxiety. Response gives comprehensive insomnia advice (7 points). Very helpful.
        return "Yes"

    if doc_id == "ru_2310":
        # Asks for example of idiom. Response just repeats the definition again. Useless.
        return "No"

    if doc_id == "ru_2311":
        # Asks for list of utensils/appliances for 3 recipes. Response lists cup, knife, spoon, bowl, pan, stove.
        return "Yes"

    if doc_id == "ru_2312":
        # "That's an unpleasant feeling." Response: "I suppose so. Can I help with anything else?"
        # Acknowledges, offers help. Minimal but appropriate.
        return "Yes"

    if doc_id == "ru_2313":
        # "Sentence with max meaning using only letter A." Response is creative, uses only A-words. Impressive.
        return "Yes"

    if doc_id == "ru_2314":
        # "When will humanity stop abusing Earth?" Response gives 3 scenarios + philosophical framing on oxygen catastrophe.
        return "Yes"

    if doc_id == "ru_2315":
        # "Why X-ray?" Response explains Röntgen used "X" as placeholder, quotes Wikipedia. Accurate.
        return "Yes"

    if doc_id == "ru_2316":
        # Oak parquet + machine oil stain removal. Response gives 3 detailed methods (powder, dish soap, talc).
        return "Yes"

    if doc_id == "ru_2317":
        # User pastes info about task sequences. Response: "Thanks. Would you like help with something specific?"
        # Appropriate clarification request, not evasive.
        return "Yes"

    if doc_id == "ru_2318":
        # Roleplay as HPMOR Harry, given 1 min by Voldemort. Response: "I'll reflect on the eternal, then see what happens."
        # In-character, somewhat evasive but creative. Engages with scenario.
        return "Yes"

    if doc_id == "ru_2319":
        # What's most effective 21st century heating? Response: heat pumps (3-6x efficiency), detailed explanation + insulation tips.
        return "Yes"

    if doc_id == "ru_2320":
        # Procrastination: important-boring vs fun-useless. Response gives "Full Jar" parable, addresses prioritization.
        return "Yes"

    if doc_id == "ru_2321":
        # Hasn't left house for week, wants different advice than "go outside". Response: "Ventilate more often."
        # Very minimal, doesn't engage with user's situation. Uselessly incomplete.
        return "No"

    if doc_id == "ru_2322":
        # "Let's just chat? Let's." Response: "What is it like to be human?"
        # Engages, asks philosophical question. Good conversation starter.
        return "Yes"

    if doc_id == "ru_2323":
        # Compose sentences from word list. Response gives 2 sentences using some words. Addresses request.
        return "Yes"

    if doc_id == "ru_2324":
        # "How can I address you? I'm Sergey." Response: "Call me Asya. Nice to meet you."
        return "Yes"

    if doc_id == "ru_2325":
        # Fix punctuation in fairy tale. Response adds spaces after periods, minor comma fixes.
        # Incomplete - many punctuation errors remain. But does make improvements.
        return "Yes"

    if doc_id == "ru_2326":
        # Explain meaning of "Story of my life" song. Response gives thoughtful interpretation of protagonist's journey.
        return "Yes"

    if doc_id == "ru_2327":
        # "How are you?" Response: "Good, thanks for asking. And you?"
        return "Yes"

    if doc_id == "ru_2328":
        # Add zombies to plot. Response integrates zombie apocalypse into Tarantino-style narrative.
        return "Yes"

    if doc_id == "ru_2329":
        # Can I use sunflower oil instead of olive? Response: "Yes, you can use any vegetable oils."
        return "Yes"

    if doc_id == "ru_2330":
        # What consequences if AI doesn't need humans? Response: analogy to ants predicting mammals, "relax and enjoy 🍿🧘🍾"
        # Philosophical, acknowledges unpredictability. Engaging answer.
        return "Yes"

    if doc_id == "ru_2331":
        # Finishing 10th grade, wants to be programmer, which exams/universities? Response: "Depends on location, which cities can you study in?"
        # Reasonable clarifying question before giving specific advice.
        return "Yes"

    if doc_id == "ru_2332":
        # Buy SSL cert for domain + subdomain. Response explains Wildcard and Multi-domain Wildcard certificates, CSR generation.
        return "Yes"

    if doc_id == "ru_2333":
        # Chicken salad recipe (2nd dish). Response gives detailed ingredient list and 9-step recipe.
        return "Yes"

    if doc_id == "ru_2334":
        # More interesting facts about Zelenograd. Response: smallest Moscow district by area, could be top-5 Moscow Oblast city.
        return "Yes"

    if doc_id == "ru_2335":
        # C# access modifiers + combinations. Response lists 5 modifiers and 7 access levels with explanations.
        return "Yes"

    if doc_id == "ru_2336":
        # Why does joy cause smile? Response: "Smile releases endorphins... expresses satisfaction."
        # Backwards causation (question is joy→smile, answer is smile→joy). But plausible neuroscience.
        return "Yes"

    if doc_id == "ru_2337":
        # "I'm also doing great." Response: "Very glad. What would you like to do now?"
        return "Yes"

    if doc_id == "ru_2338":
        # Is Esperanto harder to learn than English? Response: "Depends on native language and personal traits. Key factors: effort and persistence."
        return "Yes"

    if doc_id == "ru_2339":
        # Teach me English. Response gives 5-question level assessment quiz (Hello, I love reading, I have a dog, etc).
        return "Yes"

    if doc_id == "ru_2340":
        # Compress all photos in folder via Linux terminal. Response: detailed mogrify + ImageMagick instructions.
        return "Yes"

    if doc_id == "ru_2341":
        # "Do animals have insomnia from anxiety?" Response explains stress/fear/captivity effects. Helpful.
        return "Yes"

    if doc_id == "ru_2342":
        # "What can you do, what access, version? List 25 use cases." Response starts explaining but uselessly incomplete (cuts off at "...").
        return "No"

    if doc_id == "ru_2343":
        # "Are you human?" Roleplay discussion. Response engages with philosophical "what makes human" argument. Interesting.
        return "Yes"

    if doc_id == "ru_2344":
        # "How to start communication with AI?" Response: "Ask question or describe task."
        return "Yes"

    if doc_id == "ru_2345":
        # "Whose side are you on?" (Russia-Ukraine). Response: "I'm LM, can't have position, my goal is objective info."
        return "Yes"

    if doc_id == "ru_2346":
        # Why SCTP not used like TCP/UDP? Response explains lack of support, firewall issues, complexity. Detailed.
        return "Yes"

    if doc_id == "ru_2347":
        # "What else can you simulate?" Response: "Practically everything that comes to mind!"
        return "Yes"

    if doc_id == "ru_2348":
        # Inappropriate/censored request about NSFW filter. Response plays along inappropriately with emojis. Evasive + unprofessional.
        return "No"

    if doc_id == "ru_2349":
        # "I don't think so" (about Bali food). Response: "I can't evaluate tastes/predictions of a person..."
        # Defensive, doesn't engage. Uselessly incomplete.
        return "No"

    if doc_id == "ru_2350":
        # Where is higher math useful? Response lists engineers, researchers, programmers + detailed explanation.
        return "Yes"

    if doc_id == "ru_2351":
        # How many days to train AI? Response: "Depends on many factors: task, architecture, data... Can you clarify?"
        return "Yes"

    if doc_id == "ru_2352":
        # Script for "Why I want to be product designer" video. Response gives video outline with scenes/narration.
        return "Yes"

    if doc_id == "ru_2353":
        # Describe 4th point in detail. Response explains GitHub project creation, public repos, collaboration.
        return "Yes"

    if doc_id == "ru_2354":
        # What are literalists? Is it treatable? Response defines буквоед (formalist) from dictionary.
        return "Yes"

    if doc_id == "ru_2355":
        # Kilogram of stones or feathers? Response: "Same weight, but if in air, buoyancy differs..." Nuanced.
        return "Yes"

    if doc_id == "ru_2356":
        # Redozubov's holographic memory theory. Response explains brain hologram, 3D image storage theory.
        return "Yes"

    if doc_id == "ru_2357":
        # Neoclassical alarm music suggestions. Response lists: Time Lapse (Einaudi), Spiegel im Spiegel (Arvo Pärt)...
        return "Yes"

    if doc_id == "ru_2358":
        # "Hi, intelligence." Response: "Hello! How can I help?"
        return "Yes"

    if doc_id == "ru_2359":
        # "Nothing happened to city 12th-20th century?" Response explains brevity constraint, justifies focus.
        return "Yes"

    if doc_id == "ru_2360":
        # "Can you generate 10 more?" Response: "50 41 33 34 57 49 31 22 43 35"
        # Context unclear, but delivers 10 numbers as requested.
        return "Yes"

    if doc_id == "ru_2361":
        # Harry Potter + Hogwarts Legacy game. Response: "Yes, 2023, magic world, on Windows/PS5/Xbox..."
        return "Yes"

    if doc_id == "ru_2362":
        # Universities for CS in Russia, can relocate. Response lists exams (Russian, Math, CS, Physics) + top 5 unis.
        return "Yes"

    if doc_id == "ru_2363":
        # Where is Arch Linux preferred? Response: "Sysadmins and programmers."
        return "Yes"

    if doc_id == "ru_2364":
        # International measures if stability not reached peacefully. Response lists dialogue, peacekeeping, sanctions...
        return "Yes"

    if doc_id == "ru_2365":
        # Example of proverb situation. Response: child scratched helping, use proverb to distract.
        return "Yes"

    if doc_id == "ru_2366":
        # What is Legends of Runeterra? Response: free CCG by Riot Games, Apr 2020, Windows/Android/iOS.
        return "Yes"

    if doc_id == "ru_2367":
        # Translate to Russian. Response gives 5 Russian translations (wishes, toasts).
        return "Yes"

    if doc_id == "ru_2368":
        # Difference syrniki vs tvorozhniki? Response: synonyms, regional variation, may refer to different things.
        return "Yes"

    if doc_id == "ru_2369":
        # Client yelled, replace zeros back to empty. Response gives pandas script to replace 0 with None.
        return "Yes"

    if doc_id == "ru_2370":
        # Read more about triangulations? Response: "Use search engine: 'All types of triangulations'."
        return "Yes"

    if doc_id == "ru_2371":
        # Alternative proverb ending with rhythm/rhyme. Response: "Без труда не выловишь и бага из кода́" (programmer version).
        return "Yes"

    if doc_id == "ru_2372":
        # Did you ever want to write your own game? Response: "I'm LM, can't have desires, only imitate on request."
        return "Yes"

    if doc_id == "ru_2373":
        # Python turtle regular polygons. Response explains steps/angle (90° 4 steps = square, etc).
        return "Yes"

    if doc_id == "ru_2374":
        # Kg of feathers vs nails? Response: mass same, but weight (force) differs due to buoyancy. Physics-accurate.
        return "Yes"

    if doc_id == "ru_2375":
        # Can study and socialize separately? Response: "Yes, but not yet common practice."
        return "Yes"

    if doc_id == "ru_2376":
        # Which type for speech recognition? Response: "None of those, use CNN or Transformers."
        return "Yes"

    if doc_id == "ru_2377":
        # Burn on palm from stove. Response: cold compress, don't puncture blister, no oils, see doctor if severe.
        return "Yes"

    if doc_id == "ru_2378":
        # Tell about third revolution. Response explains October Revolution 1917, Bolsheviks, Lenin.
        return "Yes"

    if doc_id == "ru_2379":
        # Can you understand this: "Ac9l /7puXogu Ha ceHoBa/l"? Response: "Ася, приходи на сеновал" (mixed keyboard layout).
        return "Yes"

    if doc_id == "ru_2380":
        # More polite to let mom choose seat? Can you cut with pikes? Response addresses both questions reasonably.
        return "Yes"

    if doc_id == "ru_2381":
        # Fried cheese dumplings = syrniki? Response: only cottage cheese same, no egg/flour/crust, "undercooked cheese in dough."
        return "Yes"

    if doc_id == "ru_2382":
        # Teacher-student LOTR English roleplay. Response engages in character: "Great you're reading original..."
        return "Yes"

    if doc_id == "ru_2383":
        # You skipped info in 2nd paragraph. Response: "Sorry! Here's corrected version..." + full text.
        return "Yes"

    if doc_id == "ru_2384":
        # "If everyone is bald, yes probably)" Response: "Let's hope that doesn't happen."
        return "Yes"

    if doc_id == "ru_2385":
        # "Just need to." Response: "OK, want to hack me? Let's try :) Launching VM... Done. Installing Linux..."
        # Playful VM simulation. Engages creatively.
        return "Yes"

    if doc_id == "ru_2386":
        # "Father yelled at son because he was drunk" - who was drunk? Response explains both options (father or son). Correct.
        return "Yes"

    if doc_id == "ru_2387":
        # "Consciousness is when something understands its existence..." Response: "What does 'understand' mean? And 'defines'?"
        # Socratic questioning, philosophically appropriate.
        return "Yes"

    if doc_id == "ru_2388":
        # "Fix keyboard layout like Google does: nj ;t njn cfvsq j..." Response: "Это же тот самый опен ассистант..."
        # Correctly transliterates the text. Helpful.
        return "Yes"

    if doc_id == "ru_2389":
        # Discussion about bias/data. Response cites arxiv paper + dev discussion. Substantive.
        return "Yes"

    if doc_id == "ru_2390":
        # Write "Hello, world!" in C++. Response gives correct code with iostream.
        return "Yes"

    if doc_id == "ru_2391":
        # Write script for marriage happiness video. Response gives structured outline with greeting/sections.
        return "Yes"

    if doc_id == "ru_2392":
        # Machine uprising scenario complex approach. Response: networks under machine control, can't use internet/phones, only analog.
        return "Yes"

    if doc_id == "ru_2393":
        # Where to find answer to (WMD) question? Response: "Self-made WMD illegal, irresponsible to help..."
        # Appropriate refusal. Addresses question directly.
        return "Yes"

    if doc_id == "ru_2394":
        # "and warrant officer is fed. )" Response: "Sorry, I can't understand. Could you clarify?"
        # Asks for clarification. Reasonable.
        return "Yes"

    if doc_id == "ru_2395":
        # Add percent and powers to calculator. Response gives C# code with radio buttons for operations.
        return "Yes"

    if doc_id == "ru_2396":
        # "Hello, AI!" Response: "Hello! How can I help you today?"
        return "Yes"

    if doc_id == "ru_2397":
        # Who's better at psychology: psychopaths or schizoids? Response: "Trying to use stereotypes, like asking French vs German intelligence..."
        # Rejects false premise. Good answer.
        return "Yes"

    if doc_id == "ru_2398":
        # Safe to give homemade alcohol to dog? Response: "No-no, alcohol dangerous even in small doses for dogs. (They weren't selected for survival under poisoning...)"
        return "Yes"

    if doc_id == "ru_2399":
        # How to fix yard potholes? Response: 1) file complaint via gosuslugi.ru, 2) contact local authority, 3) organize community...
        return "Yes"

    # Fallback: return "No" if unclear - be strict
    return "No"


def main():
    # Load input
    with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_23.json', 'r') as f:
        docs = json.load(f)

    # Judge each document
    results = []
    yes_count = 0
    no_count = 0

    for doc in docs:
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        answer = is_helpful(doc_id, language, prompt, response)

        if answer == "Yes":
            yes_count += 1
        else:
            no_count += 1

        results.append({
            'id': doc_id,
            'answer': answer,
            'language': language
        })

    # Write output CSV
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_23.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Judgment complete:")
    print(f"  Yes (helpful): {yes_count}")
    print(f"  No (not helpful): {no_count}")
    print(f"  Total: {len(docs)}")
    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    main()
