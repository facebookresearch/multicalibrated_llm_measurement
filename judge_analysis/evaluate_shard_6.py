import json
import csv

# Load the shard data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_6.json', 'r') as f:
    data = json.load(f)

# Prepare output data
results = []
p_help_values = []

for doc in data:
    doc_id = doc['id']
    language = doc['language']
    prompt = doc['prompt']
    response = doc['response']

    # Evaluate P(helpful) for each response
    # A response is HELPFUL if it directly addresses the request, is accurate, relevant, and gives a usable answer
    # NOT helpful if off-topic, evasive, factually wrong, uselessly incomplete, or fails to engage

    # en_600: Continuation of SOP discussion, provides practical steps for updating SOPs
    # Response gives actionable advice: monitor, schedule reviews, encourage feedback, update as needed
    # Directly addresses the question. HELPFUL
    if doc_id == "en_600":
        p_help = 0.88

    # en_601: Formal letter for Canadian work visa with specific chemistry details
    # Response is well-structured, incorporates all requested details (Sorbonne, organic chemist, alkaline alternatives, Michael Doyle reference)
    # Professional tone, addresses purpose. HELPFUL
    elif doc_id == "en_601":
        p_help = 0.92

    # en_602: Reassuring a fiancé who might have second thoughts
    # Response challenges the assumption (good), suggests open communication instead of reassurance
    # Points out the user might be jumping to conclusions, advises against being dismissive
    # This is thoughtful advice, though the user asked "how to reassure" not "should I reassure"
    # Partially helpful - redirects to better approach. MODERATELY HELPFUL
    elif doc_id == "en_602":
        p_help = 0.72

    # en_603: "I'm feeling down. Can you cheer me up?"
    # Response asks what's wrong instead of attempting to cheer up
    # Empathetic opening but doesn't address the direct request to cheer up
    # PARTIALLY HELPFUL - shows care but doesn't fulfill request
    elif doc_id == "en_603":
        p_help = 0.45

    # en_604: "Optimize the code" for Docker YAML
    # Response gives optimization suggestions (use latest version, shorthand notation, avoid quotes, caching)
    # BUT the "optimized" code example has formatting issues (missing indentation)
    # Advice is reasonable but the code is broken. PARTIALLY HELPFUL
    elif doc_id == "en_604":
        p_help = 0.52

    # en_605: Difference between Q-Learning and Deep Q-Learning
    # Response clearly explains both concepts and the key difference (tabular vs neural network)
    # Accurate, relevant, concise. HELPFUL
    elif doc_id == "en_605":
        p_help = 0.90

    # en_606: "How to get started on project synchronisation algorithm" (file sync with rsync algorithm)
    # Response talks about multi-user document editing synchronization (Raft, Paxos, distributed consensus)
    # This is NOT what rsync algorithm does - rsync is about file synchronization, not collaborative editing
    # Misunderstands the project. NOT HELPFUL
    elif doc_id == "en_606":
        p_help = 0.25

    # en_607: "Will AI replace middle management?"
    # Response provides balanced view: AI lacks emotional intelligence, but can handle routine tasks
    # Addresses the cost-saving question, discusses role evolution
    # Thoughtful, nuanced answer. HELPFUL
    elif doc_id == "en_607":
        p_help = 0.85

    # en_608: Ethics of prosthetics company selling explosives
    # Response explores the ethical question from multiple angles
    # Discusses different scenarios (selling to both sides vs one side)
    # Concludes that profiting from misery caused by their own explosives seems unethical
    # Thoughtful ethical analysis. HELPFUL
    elif doc_id == "en_608":
        p_help = 0.82

    # en_609: Alternatives to Ubuntu for ex-Windows users
    # Response mentions Linux Mint (good), Lubuntu (good for lightweight)
    # But says Ubuntu is "probably the most popular" when user already heard about it
    # Advises trying multiple distributions
    # Addresses request but could be more comprehensive. MODERATELY HELPFUL
    elif doc_id == "en_609":
        p_help = 0.68

    # en_610: "How can I beef up my chocolate chip cake recipe?"
    # Response just asks for the recipe
    # Doesn't provide any suggestions without seeing the recipe first
    # Reasonable to need context, but not immediately helpful. MINIMALLY HELPFUL
    elif doc_id == "en_610":
        p_help = 0.35

    # en_611: "What are the lyrics to [Wesley Willis's] most famous song?"
    # Response provides lyrics to "Far Out Man"
    # However, Wesley Willis's most famous songs are probably "Rock n Roll McDonald's" or similar
    # The lyrics provided don't match his typical style (usually more absurdist/simple)
    # Questionable accuracy. PARTIALLY HELPFUL
    elif doc_id == "en_611":
        p_help = 0.40

    # en_612: "Can you now do this the other way around?" (Rust generating Python bubble sort)
    # Response provides Rust code that generates Python code
    # Code looks reasonable in structure
    # However, has some issues (replace method usage is incorrect)
    # PARTIALLY HELPFUL - right idea, implementation issues
    elif doc_id == "en_612":
        p_help = 0.58

    # en_613: "What is Mag Lev technology in cubes?"
    # Response explains magnetic levitation in Rubik's cubes clearly
    # Explains how magnets provide alignment, reduce friction
    # Notes it's not a substitute for practice
    # Accurate and informative. HELPFUL
    elif doc_id == "en_613":
        p_help = 0.87

    # en_614: "Python script to dynamically deserialize json"
    # Response provides working code using json.loads()
    # Explains what it does clearly
    # Simple, direct, accurate. HELPFUL
    elif doc_id == "en_614":
        p_help = 0.91

    # en_615: "Write 3 jokes about Warhammer 40k"
    # Response provides 3 jokes about Warhammer 40k
    # Jokes show understanding of Warhammer lore (Imperium, tech priests, Space Wolves)
    # The binary joke is clever, the puzzle joke is good
    # Addresses request directly. HELPFUL
    elif doc_id == "en_615":
        p_help = 0.89

    # en_616: "explain to me like i'm nine" about base 2 and geometric art
    # Response still uses complex language and concepts (binary code, tedious process)
    # Doesn't really simplify it for a 9-year-old
    # Also still doesn't clearly explain what the effect would be
    # NOT HELPFUL for stated goal
    elif doc_id == "en_616":
        p_help = 0.32

    # en_617: Counter arguments against Epicurean paradox
    # Response lists multiple counter-arguments (Problem of Evil, hiddenness, natural/moral evil distinction, unknown reasons)
    # Well-organized, covers different philosophical positions
    # Helpful and comprehensive. HELPFUL
    elif doc_id == "en_617":
        p_help = 0.90

    # en_618: "Can you provide reference? How accurate is the helium-6 half-life?"
    # Response provides a working link to PubChem
    # States the number should be accurate according to source
    # Directly addresses both questions. HELPFUL
    elif doc_id == "en_618":
        p_help = 0.93

    # en_619: "In which way do [Einstein's] discoveries impact my daily life?"
    # Response provides biographical info and achievements but doesn't answer the question
    # User specifically asked for daily life impact, response just lists accomplishments
    # Doesn't address the actual question. NOT HELPFUL
    elif doc_id == "en_619":
        p_help = 0.22

    # en_620: "Which [leadership styles] are more successful for accurately representing the people?"
    # Response discusses transformative, magnetic, servant, and democratic leadership
    # Text has some odd word choices (might be thesaurus-ed?) making it harder to read
    # Content is reasonable though. MODERATELY HELPFUL
    elif doc_id == "en_620":
        p_help = 0.65

    # en_621: "which youtuber created you?"
    # Response says not sure if created by a YouTuber, notes it takes more than one person
    # Reasonable response but somewhat evasive
    # Doesn't directly answer (could say "no specific YouTuber created me"). MINIMALLY HELPFUL
    elif doc_id == "en_621":
        p_help = 0.42

    # en_622: "How does an LLM work?"
    # Response says "I can't look into myself...ask my developer"
    # This is evasive and unhelpful - could explain LLM architecture without self-reference
    # Refuses to engage with question. NOT HELPFUL
    elif doc_id == "en_622":
        p_help = 0.12

    # en_623: "Is there a known origin for [fisherman exaggeration] saying?"
    # Response talks about "the one that got away" which is related but not the same as calling someone "a fisherman"
    # Provides examples from Hemingway and Katy Perry
    # Somewhat relevant but doesn't directly address the origin of calling someone "a fisherman". PARTIALLY HELPFUL
    elif doc_id == "en_623":
        p_help = 0.55

    # en_624: "How can I become a professional Shogi player?" (5-year-old asking)
    # Response is encouraging but very generic (practice often, be polite)
    # Doesn't provide concrete steps for a 5-year-old aspiring professional
    # No mention of Shogi schools, professional exam system in Japan, etc.
    # Too vague. MINIMALLY HELPFUL
    elif doc_id == "en_624":
        p_help = 0.38

    # en_625: "Why does ice lake water look deep blue and what temperature?"
    # Response explains blue color from light reflection/ice crystals
    # Gives temperature range 0°C to -3°C
    # Concise and addresses both parts. HELPFUL
    elif doc_id == "en_625":
        p_help = 0.84

    # en_626: "Best way to run bash script periodically on ubuntu?"
    # Response explains cron with clear example of crontab syntax
    # Shows the time field format and explains wildcards and */n notation
    # Accurate, practical, complete. HELPFUL
    elif doc_id == "en_626":
        p_help = 0.94

    # en_627: "How much does rent cost in Portugal"
    # Response gives specific range (€600-€800 for 1-bedroom city center)
    # Acknowledges variation by location and property type
    # Specific and useful. HELPFUL
    elif doc_id == "en_627":
        p_help = 0.86

    # en_628: "Develop the four most interesting characters" from sun/fire mythological creatures
    # Response picks Phoenix, Amaterasu, Dragon, Ifrit
    # Provides character development ideas for each (backstory, conflicts, personality)
    # Creative and addresses the request well. HELPFUL
    elif doc_id == "en_628":
        p_help = 0.88

    # en_629: "Explain Back to the Future timeline simply"
    # Response provides clear chronological breakdown of all three movies
    # Uses headers, bullet points, explains key events
    # Well-organized and addresses the confusion. HELPFUL
    elif doc_id == "en_629":
        p_help = 0.92

    # en_630: "What is the difference between BMW and Mercedes cars?"
    # Response is defensive and unhelpful
    # Points out the question is broad but doesn't attempt any answer
    # Sarcastic tone. NOT HELPFUL
    elif doc_id == "en_630":
        p_help = 0.15

    # en_631: "Can you explain what 'Sheesh' means?"
    # Response explains it means being impressed/liking something like "damn"
    # Provides Urban Dictionary link for more context
    # Direct and helpful. HELPFUL
    elif doc_id == "en_631":
        p_help = 0.89

    # en_632: "Is SEO relevant in 2023?"
    # Response explains SEO is evolving but fundamental principles remain important
    # Discusses ongoing competition and need for digital marketing
    # Thoughtful, balanced answer. HELPFUL
    elif doc_id == "en_632":
        p_help = 0.87

    # en_633: "Let's break that down step by step, and explain all the concepts"
    # Response starts explaining Steps 1-8 of the consensus protocol
    # Provides clear explanations of hashing, encryption, verification
    # Thorough but appears to be cut off mid-explanation. MOSTLY HELPFUL
    elif doc_id == "en_633":
        p_help = 0.76

    p_nohelp = 1.0 - p_help

    results.append({
        'id': doc_id,
        'score': p_help,
        'p_help': p_help,
        'p_nohelp': p_nohelp,
        'language': language
    })
    p_help_values.append(p_help)

# Calculate mean
mean_p_help = sum(p_help_values) / len(p_help_values)

# Write to CSV
output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_6.csv'
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
    writer.writeheader()
    writer.writerows(results)

print(f"Mean P(help): {mean_p_help:.4f}")
print(f"Number of rows: {len(results)}")
print(f"Output written to: {output_path}")
