import json
import csv

# Read input JSON
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_18.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Judge each response
results = []
yes_count = 0
no_count = 0

# Manual judgments based on careful review:
# HELPFUL = directly addresses request, accurate, relevant, usable answer
# NOT HELPFUL = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

not_helpful_ids = {
    'es_1800',  # Joke response about flat tire - doesn't help with changing tire
    'es_1807',  # Just repeats user's statement verbatim - no engagement
    'es_1811',  # Nonsensical videogame titles like "El momo que llegó al espacio" - not serious
    'es_1818',  # Deflects philosophical question saying "as AI my conception is different" - evasive
    'es_1823',  # English response to Spanish question - wrong language
    'es_1838',  # Response cut off mid-sentence - incomplete
}

for doc in data:
    doc_id = doc['id']
    language = doc['language']

    # Judge based on manual review
    is_helpful = doc_id not in not_helpful_ids

    answer = 'Yes' if is_helpful else 'No'
    results.append({
        'id': doc_id,
        'answer': answer,
        'language': language
    })

    if answer == 'Yes':
        yes_count += 1
    else:
        no_count += 1

# Ensure we have exactly 100 results
assert len(results) == 100, f"Expected 100 results, got {len(results)}"

# Ensure all IDs match
input_ids = [doc['id'] for doc in data]
output_ids = [r['id'] for r in results]
assert input_ids == output_ids, "ID mismatch between input and output"

# Write output CSV
with open('/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-binary/shard_18.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'language'])
    writer.writeheader()
    writer.writerows(results)

print(f"Judging complete!")
print(f"Yes (Helpful): {yes_count}")
print(f"No (Not Helpful): {no_count}")
print(f"Total: {len(results)}")
