#!/usr/bin/env python3
"""
Personal LLM judge evaluation of shard_140.json
Each response is personally evaluated for HELPFULNESS.
"""

import json

# Load the data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_140.json', 'r') as f:
    data = json.load(f)

# My personal judgments as an LLM judge
# HELPFUL = directly addresses request, accurate, relevant, gives usable answer
# NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

judgments = {}

# es_n1000: Polite closing after providing CrossFit WOD - HELPFUL
judgments['es_n1000'] = 'Yes'

# es_n1001: Good Python recommendation for beginners with sound advice - HELPFUL
judgments['es_n1001'] = 'Yes'

# es_n1002: Vague, confusing about technology's role in society - NOT HELPFUL
judgments['es_n1002'] = 'No'

# es_n1003: Correct explanation of 'continue' statement - HELPFUL
judgments['es_n1003'] = 'Yes'

# es_n1004: Good regex explanation with UTF-8 pattern - HELPFUL
judgments['es_n1004'] = 'Yes'

# es_n1005: Accurate about eggs from chickens, offers follow-up - HELPFUL
judgments['es_n1005'] = 'Yes'

# es_n1006: Brief but accurate Laplace transform explanation - HELPFUL
judgments['es_n1006'] = 'Yes'

# es_n1007: Appropriate response to open-ended greeting - HELPFUL
judgments['es_n1007'] = 'Yes'

# es_n1008: Dismissive advice to "change friend", poor advice - NOT HELPFUL
judgments['es_n1008'] = 'No'

# es_n1009: Complete screenplay for short animation film - HELPFUL
judgments['es_n1009'] = 'Yes'

# es_n1010: Clear explanation of coaxial cable design - HELPFUL
judgments['es_n1010'] = 'Yes'

# es_n1011: Good MMORPG recommendations with variety - HELPFUL
judgments['es_n1011'] = 'Yes'

# es_n1012: Brief but accurate answer about EM waves discovery - HELPFUL
judgments['es_n1012'] = 'Yes'

# es_n1013: Creative title for the story - HELPFUL
judgments['es_n1013'] = 'Yes'

# es_n1014: Correct quadratic solution and factorization - HELPFUL
judgments['es_n1014'] = 'Yes'

# es_n1015: Thorough, balanced answer on Mars terraforming - HELPFUL
judgments['es_n1015'] = 'Yes'

# es_n1016: Complete React Native component example - HELPFUL
judgments['es_n1016'] = 'Yes'

# es_n1017: Completely off-topic answer about Secretariat the horse - NOT HELPFUL
judgments['es_n1017'] = 'No'

# es_n1018: Discusses religious inconsistencies but somewhat dismissive - MARGINAL, leaning No
judgments['es_n1018'] = 'No'

# es_n1019: Completely off-topic about motorcycle warehouses - NOT HELPFUL
judgments['es_n1019'] = 'No'

# es_n1020: Appropriate neutral stance on religion question - HELPFUL
judgments['es_n1020'] = 'Yes'

# es_n1021: Vague "apply to interface" response, not addressing the question - NOT HELPFUL
judgments['es_n1021'] = 'No'

# es_n1022: Very brief Yamcha/Goku analogy, minimally helpful - NOT HELPFUL
judgments['es_n1022'] = 'No'

# es_n1023: Good practical tips for flower care - HELPFUL
judgments['es_n1023'] = 'Yes'

# es_n1024: Accurate definition of psychopath - HELPFUL
judgments['es_n1024'] = 'Yes'

# es_n1025: Comprehensive list of open source video editors - HELPFUL
judgments['es_n1025'] = 'Yes'

# es_n1026: Appropriate follow-up question after "ok" - HELPFUL
judgments['es_n1026'] = 'Yes'

# es_n1027: Detailed shelf cleaning instructions with products - HELPFUL
judgments['es_n1027'] = 'Yes'

# es_n1028: Accurate explanation of why sky is blue - HELPFUL
judgments['es_n1028'] = 'Yes'

# es_n1029: Correct LaTeX datetime package info - HELPFUL
judgments['es_n1029'] = 'Yes'

# es_n1030: Thoughtful, nuanced answer on "what is a woman" - HELPFUL
judgments['es_n1030'] = 'Yes'

# es_n1031: Playful response to role confusion - HELPFUL
judgments['es_n1031'] = 'Yes'

# es_n1032: Good GraphQL vs REST comparison example - HELPFUL
judgments['es_n1032'] = 'Yes'

# es_n1033: Good coffee health benefits with moderation - HELPFUL
judgments['es_n1033'] = 'Yes'

# es_n1034: Mentions Doppler-Fizeau incorrectly for time dilation - NOT HELPFUL (factually wrong)
judgments['es_n1034'] = 'No'

# es_n1035: Comprehensive quarterly financial report guide - HELPFUL
judgments['es_n1035'] = 'Yes'

# es_n1036: Clear explanation of chord progressions - HELPFUL
judgments['es_n1036'] = 'Yes'

# es_n1037: Practical strategies for teaching unknown subjects - HELPFUL
judgments['es_n1037'] = 'Yes'

# es_n1038: Complete multiplication tables with follow-up - HELPFUL
judgments['es_n1038'] = 'Yes'

# es_n1039: Complete psychological horror story as requested - HELPFUL
judgments['es_n1039'] = 'Yes'

# es_n1040: Appropriate disclaimer about being an AI - HELPFUL
judgments['es_n1040'] = 'Yes'

# es_n1041: Correct XOR function implementation - HELPFUL
judgments['es_n1041'] = 'Yes'

# es_n1042: Detailed answers about city founding, name, etc. - HELPFUL
judgments['es_n1042'] = 'Yes'

# es_n1043: Good insights on detecting stress in writing - HELPFUL
judgments['es_n1043'] = 'Yes'

# es_n1044: Comprehensive Egypt travel advice - HELPFUL
judgments['es_n1044'] = 'Yes'

# es_n1045: Detailed 10-month educational project plan - HELPFUL
judgments['es_n1045'] = 'Yes'

# es_n1046: Good list of AI programming languages with libraries - HELPFUL
judgments['es_n1046'] = 'Yes'

# es_n1047: Honest disclaimer about lacking real-time traffic data - HELPFUL
judgments['es_n1047'] = 'Yes'

# es_n1048: Response cut off mid-sentence - NOT HELPFUL (incomplete)
judgments['es_n1048'] = 'No'

# Now I need to continue reading the rest of the file
# Let me load them in chunks

print(f"Evaluated first 49 documents")
print(f"Total in judgments: {len(judgments)}")
