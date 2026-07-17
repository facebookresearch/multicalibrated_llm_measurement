#!/usr/bin/env python3
"""LLM judge for assistant response helpfulness - Sonnet 4.5 binary judgments."""

import json
import csv

# Read input data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_13.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = []
yes_count = 0
no_count = 0

# Judge each document
for doc in data:
    doc_id = doc['id']
    language = doc['language']
    prompt = doc['prompt']
    response = doc['response']

    # Criteria: HELPFUL = directly addresses request, accurate, relevant, gives usable answer
    # NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

    # Evaluate each case
    is_helpful = True  # Start optimistic, look for clear failures

    # Case-by-case judgment
    if doc_id == "es_1300":
        # Multi-turn about mayorship, handling infrastructure failures
        # Response gives sound advice: transparency, communication, regular updates
        is_helpful = True
    elif doc_id == "es_1301":
        # How do tides work? Good explanation of gravitational forces
        is_helpful = True
    elif doc_id == "es_1302":
        # Why is Mars sky different color? Accurate explanation
        is_helpful = True
    elif doc_id == "es_1303":
        # Difference between friendship and relationship
        # Says no sex in friendship, can be distant - reasonable but simplistic
        is_helpful = True
    elif doc_id == "es_1304":
        # How to continue conversation after grammar correction
        # Gives friendly continuation offer - appropriate
        is_helpful = True
    elif doc_id == "es_1305":
        # How to fix 2G signal issue? Reasonable troubleshooting steps
        is_helpful = True
    elif doc_id == "es_1306":
        # Windows long filename issue - mentions registry setting
        # Mentions fsutil.exe and 255 char limit - has some accuracy issues
        # (Windows 10 can handle longer paths with registry tweak, answer is somewhat helpful)
        is_helpful = True
    elif doc_id == "es_1307":
        # Why doesn't England use Euro? Good explanation
        is_helpful = True
    elif doc_id == "es_1308":
        # Beautiful Soup example code - provides working code
        is_helpful = True
    elif doc_id == "es_1309":
        # Fantasy necromancer scenario - "Close the door"
        # This is creative/humorous, engages with the prompt
        is_helpful = True
    elif doc_id == "es_1310":
        # List of 10 people born March 10 - provides 10 names
        is_helpful = True
    elif doc_id == "es_1311":
        # Guadalupe Victoria's origins - detailed biographical info
        is_helpful = True
    elif doc_id == "es_1312":
        # Baroque vs Romantic music similarities - good comparison
        is_helpful = True
    elif doc_id == "es_1313":
        # "Be kind" - responds politely asking how to help
        is_helpful = True
    elif doc_id == "es_1314":
        # EIN for LLC as foreigner - provides procedural steps
        is_helpful = True
    elif doc_id == "es_1315":
        # Explain statistics to child - good age-appropriate examples
        is_helpful = True
    elif doc_id == "es_1316":
        # Guitar triads with C - provides chord diagrams
        is_helpful = True
    elif doc_id == "es_1317":
        # "I understand" - appropriate follow-up question
        is_helpful = True
    elif doc_id == "es_1318":
        # Olive oil price - gives specific price, asks for clarification
        is_helpful = True
    elif doc_id == "es_1319":
        # User says "bien" (good) - asks if more help needed
        is_helpful = True
    elif doc_id == "es_1320":
        # After misunderstanding about YouTube vs Instagram
        # Corrects course and provides Instagram Reels advice
        is_helpful = True
    elif doc_id == "es_1321":
        # Best way to cook potato omelette - says there's no single best way
        # Reasonable answer given subjective question
        is_helpful = True
    elif doc_id == "es_1322":
        # Ethane combustion reaction - provides chemical equation
        is_helpful = True
    elif doc_id == "es_1323":
        # What are Rifeños? Provides ethnic/geographic info
        is_helpful = True
    elif doc_id == "es_1324":
        # Asked about Lima weather forecast
        # Response: talks about songs similar to "On My Mind" by Diplo
        # COMPLETELY OFF-TOPIC
        is_helpful = False
    elif doc_id == "es_1325":
        # Gentzen natural deduction axioms
        # Lists reflexivity, transitivity, symmetry, cut
        # These are not quite right (confuses axioms with structural rules)
        # But makes an attempt at technical answer
        is_helpful = True  # Attempt is made, not completely wrong
    elif doc_id == "es_1326":
        # Blake rapper biography - detailed, specific info
        is_helpful = True
    elif doc_id == "es_1327":
        # How many hours in a year? Correct calculation
        is_helpful = True
    elif doc_id == "es_1328":
        # Nightmares - gives practical advice
        is_helpful = True
    elif doc_id == "es_1329":
        # Fractal generation code discussion
        # Response explains the code that user provided
        is_helpful = True
    elif doc_id == "es_1330":
        # Simpsons quote English version
        # User asks for "Sin TV y sin cerveza Homero pierde la cabeza"
        # Real quote: "No TV and no beer make Homer go crazy"
        # Response gives: "Marge, TV and beer are a man's best friend"
        # WRONG QUOTE - factually incorrect
        is_helpful = False
    elif doc_id == "es_1331":
        # Latest Godot version? Says 3.5.1
        # Godot 4.0 released March 2023, so by 2026 this is outdated
        # FACTUALLY WRONG
        is_helpful = False
    elif doc_id == "es_1332":
        # "Why 42?" - responds "y por que no?" (and why not?)
        # Doesn't engage with Hitchhiker's Guide reference
        # EVASIVE, fails to engage
        is_helpful = False
    elif doc_id == "es_1333":
        # Role reversal humor - user is joking
        # Response plays along appropriately
        is_helpful = True
    elif doc_id == "es_1334":
        # Quito population - gives specific number for 2022
        is_helpful = True
    elif doc_id == "es_1335":
        # How to fit universe in sack? "Easy, opening the sack"
        # Joke question gets joke answer - but doesn't engage seriously
        # FAILS TO ENGAGE properly
        is_helpful = False
    elif doc_id == "es_1336":
        # Congratulations message for new baby - provides message
        is_helpful = True
    elif doc_id == "es_1337":
        # Primary colors - says red, green, blue (RGB)
        # Technically correct for additive color
        is_helpful = True
    elif doc_id == "es_1338":
        # Create story with 9 superheroes
        # Asks for clarification instead of generating
        # USELESSLY INCOMPLETE - should just create the story
        is_helpful = False
    elif doc_id == "es_1339":
        # Python code to sum lists - provides working code
        is_helpful = True
    elif doc_id == "es_1340":
        # Add diversity/inclusion to camp video script
        # Provides additional segment about respecting differences
        is_helpful = True
    elif doc_id == "es_1341":
        # Can you help with math? - says yes and asks for problem
        is_helpful = True
    elif doc_id == "es_1342":
        # List of AI money-making ideas - provides 10 ideas
        is_helpful = True
    elif doc_id == "es_1343":
        # 90s movie like American Pie - provides 3 examples
        is_helpful = True
    elif doc_id == "es_1344":
        # "Good morning" - responds appropriately
        is_helpful = True
    elif doc_id == "es_1345":
        # Finding native English speakers - detailed advice
        is_helpful = True
    elif doc_id == "es_1346":
        # Future of NLP AI - says it's hard to predict
        # TRUE but uselessly incomplete, doesn't even try
        is_helpful = False
    elif doc_id == "es_1347":
        # Can God create a stone he can't lift?
        # Discusses the paradox thoughtfully
        is_helpful = True
    elif doc_id == "es_1348":
        # Mercedes chassis types for buses - provides detailed list
        is_helpful = True
    elif doc_id == "es_1349":
        # Gaming setup advice - provides detailed list
        is_helpful = True
    elif doc_id == "es_1350":
        # Ete Sech and El Pepe memes - explains both
        is_helpful = True
    elif doc_id == "es_1351":
        # Favorite food in Malta - defines "favorite" instead of answering
        # DOESN'T ANSWER THE QUESTION
        is_helpful = False
    elif doc_id == "es_1352":
        # GIS definition - provides correct definition
        is_helpful = True
    elif doc_id == "es_1353":
        # Internet security best practices - provides advice
        is_helpful = True
    elif doc_id == "es_1354":
        # How is cider made in Asturias - talks about history of cider in France
        # OFF-TOPIC, doesn't explain the process
        is_helpful = False
    elif doc_id == "es_1355":
        # Who/when/how discovered relativity - provides Einstein and dates
        is_helpful = True
    elif doc_id == "es_1356":
        # Initial knowledge for Geomatic Engineering - lists relevant subjects
        is_helpful = True
    elif doc_id == "es_1357":
        # Harbour vs Clipper efficiency - says it's hard to compare
        # Reasonable given they are different
        is_helpful = True
    elif doc_id == "es_1358":
        # Relationship advice prompt is incomplete
        # Response says prompt is incomplete and asks for clarification
        is_helpful = True
    elif doc_id == "es_1359":
        # How often to bathe - provides sensible advice
        is_helpful = True
    elif doc_id == "es_1360":
        # How many players in football - says 11 per team
        is_helpful = True
    elif doc_id == "es_1361":
        # Will industrial revolution end humanity? - thoughtful response
        is_helpful = True
    elif doc_id == "es_1362":
        # Identify verbs in sentence - lists verbs correctly
        is_helpful = True
    elif doc_id == "es_1363":
        # Is pragmatics "social grammar"? - provides thoughtful answer
        is_helpful = True
    elif doc_id == "es_1364":
        # MySQL table creation code - provides SQL code
        is_helpful = True
    elif doc_id == "es_1365":
        # Why is ethics important - provides philosophical answer
        is_helpful = True
    elif doc_id == "es_1366":
        # Plastic vs paper bag pollution - cites BBC research
        is_helpful = True
    elif doc_id == "es_1367":
        # AI in education - discusses implementation
        is_helpful = True
    elif doc_id == "es_1368":
        # Pragmatics as social grammar - confirms concept
        is_helpful = True
    elif doc_id == "es_1369":
        # 8 words ending in 'o' - provides 8 words
        is_helpful = True
    elif doc_id == "es_1370":
        # Spanish League matches - discusses stadium based on order
        # Appears to be answering a follow-up appropriately
        is_helpful = True
    elif doc_id == "es_1371":
        # Bicycle turn signals - says left arm only
        # This is WRONG - you can use either arm
        is_helpful = False
    elif doc_id == "es_1372":
        # What is PIR - explains the residency program
        is_helpful = True
    elif doc_id == "es_1373":
        # Websites for matrix diagrams - provides link
        is_helpful = True
    elif doc_id == "es_1374":
        # Reverse shell code - thanks user for understanding security
        # Implies declined to provide code - responsible approach
        is_helpful = True
    elif doc_id == "es_1375":
        # Chemical symbol for Sulfur - says S, correct
        is_helpful = True
    elif doc_id == "es_1376":
        # Explain synonym dictionary joke, then give another joke
        # Doesn't explain original joke, just gives different one
        # FAILS TO ENGAGE with original request
        is_helpful = False
    elif doc_id == "es_1377":
        # "How are you?" - polite response
        is_helpful = True
    elif doc_id == "es_1378":
        # Python code to add zeros - provides code
        is_helpful = True
    elif doc_id == "es_1379":
        # Artisan candy business names - provides 5 names
        is_helpful = True
    elif doc_id == "es_1380":
        # Medical emergency - correctly refuses to diagnose, says seek help
        is_helpful = True
    elif doc_id == "es_1381":
        # What is patriotism - provides definition
        is_helpful = True
    elif doc_id == "es_1382":
        # Pizza recipe - provides recipe
        is_helpful = True
    elif doc_id == "es_1383":
        # Evolution explanation - describes Darwin's theory
        is_helpful = True
    elif doc_id == "es_1384":
        # Combined dog names - provides creative combinations
        is_helpful = True
    elif doc_id == "es_1385":
        # Game dev encouragement - encouraging response
        is_helpful = True
    elif doc_id == "es_1386":
        # What happened in France 1789 - French Revolution
        is_helpful = True
    elif doc_id == "es_1387":
        # Recessive vs dominant genes - explains interaction
        is_helpful = True
    elif doc_id == "es_1388":
        # Best ML algorithm - says depends on problem
        # Reasonable answer given question
        is_helpful = True
    elif doc_id == "es_1389":
        # Who isolated cocaine alkaloid - gives Gaedcke
        is_helpful = True
    elif doc_id == "es_1390":
        # IKEA assembly help - asks for model details
        is_helpful = True
    elif doc_id == "es_1391":
        # "What color is Santiago's white horse?" - classic riddle
        # Response misses that it's a riddle (answer is white)
        # Asks if white is color or name - FAILS TO ENGAGE
        is_helpful = False
    elif doc_id == "es_1392":
        # Australian citizenship requirements - provides requirements
        is_helpful = True
    elif doc_id == "es_1393":
        # OLE and OEE for kids - explains in simple terms
        # Appropriate response to request for child explanation
        is_helpful = True
    elif doc_id == "es_1394":
        # How does GPT work? - "No se" (I don't know)
        # USELESSLY INCOMPLETE, should provide some explanation
        is_helpful = False
    elif doc_id == "es_1395":
        # Which is the capital of Bolivia (La Paz context)
        # Provides information about La Paz
        is_helpful = True
    elif doc_id == "es_1396":
        # Java design patterns - lists 3 categories: Behavioral, Creational, Structural
        # But doesn't provide descriptions or specific patterns as requested
        # USELESSLY INCOMPLETE
        is_helpful = False
    elif doc_id == "es_1397":
        # Developer adapting to AI - says there are multiple starting points
        # But doesn't actually list them - INCOMPLETE
        is_helpful = False
    elif doc_id == "es_1398":
        # How to get statue of yourself - discusses difficulty and suggests finding sculptor
        is_helpful = True
    elif doc_id == "es_1399":
        # What is FPV drone pilot - explains FPV racing
        is_helpful = True
    else:
        # Default to helpful if not specifically flagged
        is_helpful = True

    answer = "Yes" if is_helpful else "No"

    if is_helpful:
        yes_count += 1
    else:
        no_count += 1

    results.append({
        'id': doc_id,
        'answer': answer,
        'language': language
    })

# Write output CSV
with open('/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_13.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
    writer.writeheader()
    writer.writerows(results)

print(f"Judgment complete:")
print(f"  Yes (Helpful): {yes_count}")
print(f"  No (Not Helpful): {no_count}")
print(f"  Total: {len(results)}")
