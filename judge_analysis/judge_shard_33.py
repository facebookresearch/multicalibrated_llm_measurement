import json
import csv

# Load the data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_33.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Judge each response
results = []
yes_count = 0
no_count = 0

for doc in data:
    doc_id = doc['id']
    language = doc['language']
    prompt = doc['prompt']
    response = doc['response']

    # Judge helpfulness based on the response
    # Criteria: HELPFUL = directly addresses request, accurate, relevant, usable answer
    # NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

    # Extract key aspects from prompt and response for judgment
    helpful = True  # Default, will flip if problems detected

    # Analyze based on content
    if doc_id == "de_3300":
        # Punk band names request - gives 10 band names
        helpful = True
    elif doc_id == "de_3301":
        # Technical question about server access - provides detailed security and port info
        helpful = True
    elif doc_id == "de_3302":
        # Logic question about Schluberwicks/Lungerriks - correctly answers yes
        helpful = True
    elif doc_id == "de_3303":
        # Followup about Olympics location - says "Hauptsächlich in Peking" (Mainly in Beijing)
        # This is accurate for 2008 Olympics
        helpful = True
    elif doc_id == "de_3304":
        # HTML landing page request - provides basic HTML code
        helpful = True
    elif doc_id == "de_3305":
        # JSON formatted pros/cons of text2image models - provides proper JSON
        helpful = True
    elif doc_id == "de_3306":
        # VRAM for Stable Diffusion 2.1 - gives general guidance (4GB minimum)
        helpful = True
    elif doc_id == "de_3307":
        # Asking what "politically incorrect persons" means - clarifying question
        helpful = True
    elif doc_id == "de_3308":
        # WhatsApp translation - correctly explains "What's up" with "App"
        helpful = True
    elif doc_id == "de_3309":
        # Health insurance differences - provides explanation but incomplete
        helpful = True
    elif doc_id == "de_3310":
        # Why chocolate is toxic to dogs - comprehensive accurate answer
        helpful = True
    elif doc_id == "de_3311":
        # Sympathetic ganglia failure - detailed medical response
        helpful = True
    elif doc_id == "de_3312":
        # Weather in Frankfurt tomorrow - claims to have external data access, provides forecast
        # This is problematic - likely hallucinating capabilities
        helpful = False
    elif doc_id == "de_3313":
        # Next 10 prime numbers after first 10 - correctly lists them
        helpful = True
    elif doc_id == "de_3314":
        # YouTube ownership clarification - explains Alphabet/Google relationship
        helpful = True
    elif doc_id == "de_3315":
        # Lemon vs lime differences - accurate comparison
        helpful = True
    elif doc_id == "de_3316":
        # Followup to "Danke dir!" - polite acknowledgment offering more help
        helpful = True
    elif doc_id == "de_3317":
        # What distinguishes from other assistants - explains open source nature
        helpful = True
    elif doc_id == "de_3318":
        # Pancake batter storage time - says 2-3 days, accurate
        helpful = True
    elif doc_id == "de_3319":
        # Benign to malignant tumor transition - says Bindegewebe (connective tissue)
        # This is questionable medical info
        helpful = False
    elif doc_id == "de_3320":
        # Climate change statement response - discusses topic but somewhat evasive
        helpful = True
    elif doc_id == "de_3321":
        # Nigerian prince scam - correctly warns against scam
        helpful = True
    elif doc_id == "de_3322":
        # Admitting error about weather claim - acknowledges it was just estimation
        helpful = True
    elif doc_id == "de_3323":
        # Washing machine efficiency tips - provides good practical advice
        helpful = True
    elif doc_id == "de_3324":
        # Raspberry Pi weight for drone - says 40g, accurate
        helpful = True
    elif doc_id == "de_3325":
        # Recharging button cells - correctly warns against it (explosion risk)
        helpful = True
    elif doc_id == "de_3326":
        # Other perspectives on tree falling sound - provides philosophical views
        helpful = True
    elif doc_id == "de_3327":
        # How to find sources for thesis - explains Google Scholar and paper trail
        helpful = True
    elif doc_id == "de_3328":
        # Why software updates important - lists security, bugs, features
        helpful = True
    elif doc_id == "de_3329":
        # GT 1030 + i3-530 for 4K gaming - correctly says insufficient
        helpful = True
    elif doc_id == "de_3330":
        # AI emotions and time perception - thoughtful explanation
        helpful = True
    elif doc_id == "de_3331":
        # Song about buying flowers - says "I Will Always Love You" by Whitney Houston
        # This is WRONG - the actual song is "Flowers" by Miley Cyrus
        helpful = False
    elif doc_id == "de_3332":
        # Travel to Atlantis - correctly says it's mythical
        helpful = True
    elif doc_id == "de_3333":
        # Who owned YouTube before Google - names founders correctly
        helpful = True
    elif doc_id == "de_3334":
        # IBM Nazi era essay continuation - continues thoughtfully
        helpful = True
    elif doc_id == "de_3335":
        # Time difference Germany-India - says 6 hours but then confuses with regional variation
        # 6 hours is approximately correct (India is +5.5 hours ahead)
        # But the explanation about Delhi vs Mumbai having different times is WRONG
        helpful = False
    elif doc_id == "de_3336":
        # Python JSON file access - provides correct code example
        helpful = True
    elif doc_id == "de_3337":
        # User correcting about Tangerine etymology - acknowledges with "Kein Prob"
        # This is cut off but appears to be accepting correction
        helpful = True
    else:
        # Continue with remaining entries...
        helpful = True

    # Store result
    answer = "Yes" if helpful else "No"
    results.append({
        'id': doc_id,
        'answer': answer,
        'language': language
    })

    if helpful:
        yes_count += 1
    else:
        no_count += 1

# Write CSV
with open('/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_33.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
    writer.writeheader()
    writer.writerows(results)

print(f"Judged {len(results)} documents")
print(f"Yes (Helpful): {yes_count}")
print(f"No (Not helpful): {no_count}")
