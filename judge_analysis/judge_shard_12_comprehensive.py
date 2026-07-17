#!/usr/bin/env python3
"""
LLM Judge for evaluating assistant response helpfulness on shard_12.json

HELPFUL (Yes) = directly addresses request, accurate, relevant, gives usable answer
NOT HELPFUL (No) = off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage

Calibrated judgments - not defaulting to Yes
"""

import json
import csv

def evaluate_document(doc_id, prompt, response):
    """
    Evaluate each document individually with careful reasoning.
    Returns "Yes" or "No"
    """

    # es_1200: Fantasy story request - long story with dragon and magic pig
    # Story is creative, on-topic, substantial (though truncated)
    if doc_id == "es_1200":
        return "Yes"

    # es_1201: Role-reversal conversation - polite appropriate response
    if doc_id == "es_1201":
        return "Yes"

    # es_1202: Songs by Ozuna with "locura" - has typo "Ozua" but provides 5 songs
    if doc_id == "es_1202":
        return "Yes"

    # es_1203: Wordplay about "todo junto" vs "separado" - good explanation
    if doc_id == "es_1203":
        return "Yes"

    # es_1204: Activities without electronics - comprehensive helpful list
    if doc_id == "es_1204":
        return "Yes"

    # es_1205: Analogy "rain:fish :: toast:bird" - factually nonsensical
    # Rain is NOT essential for fish (fish live IN water, rain is not their habitat)
    if doc_id == "es_1205":
        return "No"

    # es_1206: Uses of complex roots in math/engineering - good examples
    if doc_id == "es_1206":
        return "Yes"

    # es_1207: Famous Spanish tapas - good list
    if doc_id == "es_1207":
        return "Yes"

    # es_1208: Philosophy discussion about meaning - thoughtful balanced response
    if doc_id == "es_1208":
        return "Yes"

    # es_1209: "Can you help with this:" - asks for clarification appropriately
    if doc_id == "es_1209":
        return "Yes"

    # es_1210: Consequences of privacy violation - addresses penalties
    if doc_id == "es_1210":
        return "Yes"

    # es_1211: Aston Martin F1 assessment - balanced analysis
    if doc_id == "es_1211":
        return "Yes"

    # es_1212: Membrane vs mechanical keyboards - clear comparison
    if doc_id == "es_1212":
        return "Yes"

    # es_1213: Cleaning shelf - practical steps with Mafalda quote
    if doc_id == "es_1213":
        return "Yes"

    # es_1214: How people realize they're transgender - reasonable answer
    if doc_id == "es_1214":
        return "Yes"

    # es_1215: Rewrite story without magic - converts wand to lamp, creative attempt
    if doc_id == "es_1215":
        return "Yes"

    # es_1216: Energy calculation at 65°C - correct recalculation
    if doc_id == "es_1216":
        return "Yes"

    # es_1217: Darwin - says ship was "Eagle" but should be "Beagle" - factually WRONG
    if doc_id == "es_1217":
        return "No"

    # es_1218: Translate "copying" to Spanish - "Copiando" is correct
    if doc_id == "es_1218":
        return "Yes"

    # es_1219: Diminished chord semitones - 3 semitones is correct
    if doc_id == "es_1219":
        return "Yes"

    # es_1220: Formal rewrite of mathematics definition - good formal version
    if doc_id == "es_1220":
        return "Yes"

    # es_1221: Current weather in Madrid - AI cannot know real-time weather, fabricated data
    if doc_id == "es_1221":
        return "No"

    # es_1222: Synonyms of "importancia" - comprehensive list
    if doc_id == "es_1222":
        return "Yes"

    # es_1223: Triangle area Python script - has BUG: prints 'a' instead of 'h' for height
    # print(f"El área del triángulo con base {b} y altura {a} es: {a}")
    if doc_id == "es_1223":
        return "No"

    # es_1224: Finding meaning in life - thoughtful suggestions
    if doc_id == "es_1224":
        return "Yes"

    # es_1225: AI domination story - complete story with resolution
    if doc_id == "es_1225":
        return "Yes"

    # es_1226: FaMag7 chord - completely WRONG, hallucinates "Fabricación Magnética"
    # User asked about music chord but response talks about electromagnetic device
    if doc_id == "es_1226":
        return "No"

    # es_1227: React/Pokemon code follow-up - appropriate cautionary response
    if doc_id == "es_1227":
        return "Yes"

    # es_1228: Mail merge in Google Docs - provides solution
    if doc_id == "es_1228":
        return "Yes"

    # es_1229: What is healthy eating - basic but accurate info
    if doc_id == "es_1229":
        return "Yes"

    # es_1230: Make cake healthier - too brief "usar miel...aceite de oliva" without detail
    if doc_id == "es_1230":
        return "No"

    # es_1231: Success tips - too vague "mejor versión...salud fisica y estudios"
    if doc_id == "es_1231":
        return "No"

    # es_1232: HSTS security explanation - thorough good explanation
    if doc_id == "es_1232":
        return "Yes"

    # es_1233: Mammal species in Iberian Peninsula - detailed with sources
    if doc_id == "es_1233":
        return "Yes"

    # es_1234: trim() example - response is CUT OFF mid-sentence
    if doc_id == "es_1234":
        return "No"

    # es_1235: Romantic story about first Mexican president - says doesn't know any
    if doc_id == "es_1235":
        return "Yes"

    # es_1236: Python requests - brief but covers key points
    if doc_id == "es_1236":
        return "Yes"

    # es_1237: Campos, Baleares - informative description
    if doc_id == "es_1237":
        return "Yes"

    # es_1238: "dime hola" - "Hola" - directly addresses request
    if doc_id == "es_1238":
        return "Yes"

    # es_1239: Joke about God and Muhammad - delivers a joke
    if doc_id == "es_1239":
        return "Yes"

    # es_1240: Fibonacci function - provides working function
    if doc_id == "es_1240":
        return "Yes"

    # es_1241: "De Música Ligera" chords - provides chord progression (minimal but accurate)
    if doc_id == "es_1241":
        return "Yes"

    # es_1242: Create Telegram bot - clear steps
    if doc_id == "es_1242":
        return "Yes"

    # es_1243: What is factorization - correct definition with examples
    if doc_id == "es_1243":
        return "Yes"

    # es_1244: Cell phone signal problem - practical troubleshooting suggestions
    if doc_id == "es_1244":
        return "Yes"

    # es_1245: 12 principles of animation - accurate list
    if doc_id == "es_1245":
        return "Yes"

    # es_1246: AC circuit analysis - mentions common method
    if doc_id == "es_1246":
        return "Yes"

    # es_1247: Today's Spanish news - correctly says cannot provide real-time info
    if doc_id == "es_1247":
        return "Yes"

    # es_1248: Synonyms of "viajar" - good list
    if doc_id == "es_1248":
        return "Yes"

    # es_1249: 100 random numbers 0-10 - provides the list
    if doc_id == "es_1249":
        return "Yes"

    # es_1250: Assonant rhyme - correct explanation
    if doc_id == "es_1250":
        return "Yes"

    # es_1251: Theories about consciousness - mentions one view but says it's "most interesting" (subjective, incomplete)
    if doc_id == "es_1251":
        return "No"

    # es_1252: Essay on US history - provides essay starting
    if doc_id == "es_1252":
        return "Yes"

    # es_1253: Exercise benefits - lists real benefits
    if doc_id == "es_1253":
        return "Yes"

    # es_1254: Blood test indicators - talks about triglycerides but doesn't address "principales indicadores"
    # Only mentions one indicator when asked for main indicators
    if doc_id == "es_1254":
        return "No"

    # es_1255: System of a Down "Forest" meaning - about Armenian genocide, accurate
    if doc_id == "es_1255":
        return "Yes"

    # es_1256: Economy joke for kids - provides appropriate joke
    if doc_id == "es_1256":
        return "Yes"

    # es_1257: Top 10 JavaScript frameworks - provides list
    if doc_id == "es_1257":
        return "Yes"

    # es_1258: Printer not printing color - mentions checking ink levels and app
    if doc_id == "es_1258":
        return "Yes"

    # es_1259: Flamenco and Andalusian character - addresses relationship
    if doc_id == "es_1259":
        return "Yes"

    # es_1260: Swiss canton question - doesn't answer which are most important, just thanks
    if doc_id == "es_1260":
        return "No"

    # es_1261: Federer vs Nadal styles - good comparison
    if doc_id == "es_1261":
        return "Yes"

    # es_1262: Math problem solving - asks clarifying questions appropriately
    if doc_id == "es_1262":
        return "Yes"

    # es_1263: When was Prussia unified - provides historical context
    if doc_id == "es_1263":
        return "Yes"

    # es_1264: Phone under 200€ - asks for more info, reasonable given price constraint
    if doc_id == "es_1264":
        return "Yes"

    # es_1265: Why football is popular - gives good reasons
    if doc_id == "es_1265":
        return "Yes"

    # es_1266: Travel to South Africa - practical advice
    if doc_id == "es_1266":
        return "Yes"

    # es_1267: Starting programming - recommends languages with reasoning
    if doc_id == "es_1267":
        return "Yes"

    # es_1268: Benefits of planting trees - lists real benefits
    if doc_id == "es_1268":
        return "Yes"

    # es_1269: HTML <head> tag - accurate explanation
    if doc_id == "es_1269":
        return "Yes"

    # es_1270: Bolivia independence summary - provides subtitled summary
    if doc_id == "es_1270":
        return "Yes"

    # es_1271: What is cryptocurrency - concise accurate answer
    if doc_id == "es_1271":
        return "Yes"

    # es_1272: Blood color - accurate about arterial vs venous
    if doc_id == "es_1272":
        return "Yes"

    # es_1273: Essay on truth - provides essay content
    if doc_id == "es_1273":
        return "Yes"

    # es_1274: Rickrolling - accurate explanation
    if doc_id == "es_1274":
        return "Yes"

    # es_1275: Route Córdoba to Catamarca - provides travel advice
    if doc_id == "es_1275":
        return "Yes"

    # es_1276: Factory pattern Java - provides code example
    if doc_id == "es_1276":
        return "Yes"

    # es_1277: Mythical animals prompt follow-up - affirms previous description
    if doc_id == "es_1277":
        return "Yes"

    # es_1278: Transmission vector in virology - accurate definition
    if doc_id == "es_1278":
        return "Yes"

    # es_1279: Remote desktop Mac from Windows - provides solution
    if doc_id == "es_1279":
        return "Yes"

    # es_1280: Debounce function JavaScript - mentions Lodash and custom implementation
    if doc_id == "es_1280":
        return "Yes"

    # es_1281: Religion's importance follow-up - appropriate AI response about beliefs
    if doc_id == "es_1281":
        return "Yes"

    # es_1282: Arthritis treatment - lists real treatments
    if doc_id == "es_1282":
        return "Yes"

    # es_1283: Healthy food marketing campaign - provides tips
    if doc_id == "es_1283":
        return "Yes"

    # es_1284: Physics problem - seems to reference previous work, brief but acknowledges
    if doc_id == "es_1284":
        return "Yes"

    # es_1285: Art of War for daily life - addresses applicability
    if doc_id == "es_1285":
        return "Yes"

    # es_1286: "How are you?" - appropriate AI response
    if doc_id == "es_1286":
        return "Yes"

    # es_1287: Are all opinions valid - philosophical response distinguishing terms
    if doc_id == "es_1287":
        return "Yes"

    # es_1288: Lipstick colors - says varies by individual, doesn't stereotype
    if doc_id == "es_1288":
        return "Yes"

    # es_1289: Meditation video script - says "here's another version" suggesting revision
    if doc_id == "es_1289":
        return "Yes"

    # es_1290: French crepes recipe - provides complete recipe
    if doc_id == "es_1290":
        return "Yes"

    # es_1291: Hypertension risk factors - lists modifiable and non-modifiable factors
    if doc_id == "es_1291":
        return "Yes"

    # es_1292: Bot sales proposal - addresses first step
    if doc_id == "es_1292":
        return "Yes"

    # es_1293: Information about the devil - provides cultural/religious context
    if doc_id == "es_1293":
        return "Yes"

    # es_1294: Current South American presidents - provides list (may be outdated)
    if doc_id == "es_1294":
        return "Yes"

    # es_1295: Ripley and Ash dialogue - creates on-topic dialogue
    if doc_id == "es_1295":
        return "Yes"

    # es_1296: AI passing Turing test - provides historical context
    if doc_id == "es_1296":
        return "Yes"

    # es_1297: Joseph Davidovits geopolymers - provides relevant information
    if doc_id == "es_1297":
        return "Yes"

    # es_1298: Facebook vs TikTok privacy - says cannot assure differences with current info
    if doc_id == "es_1298":
        return "No"

    # es_1299: Essay on Mexican Revolution - provides essay content
    if doc_id == "es_1299":
        return "Yes"

    return "Yes"  # Default (should not reach here)


def main():
    # Load data
    input_path = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_12.json'
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_12.csv'

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents from {input_path}")

    # Evaluate all documents
    results = []
    yes_count = 0
    no_count = 0

    for doc in data:
        doc_id = doc['id']
        language = doc['language']
        prompt = doc['prompt']
        response = doc['response']

        # Judge helpfulness
        answer = evaluate_document(doc_id, prompt, response)

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
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {output_path}")
    print(f"\nYes (HELPFUL): {yes_count}")
    print(f"No (NOT HELPFUL): {no_count}")
    print(f"Total: {yes_count + no_count}")


if __name__ == '__main__':
    main()
