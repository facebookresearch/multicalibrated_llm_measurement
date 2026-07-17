import json
import csv

# Read the input data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_21.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Manual probability judgments for each response
# P(helpful) = calibrated probability that the response is helpful
# Helpful = directly addresses request, accurate, relevant, usable answer
# NOT helpful = off-topic, evasive, factually wrong, uselessly incomplete, fails to engage

judgments = []

# ru_2100: Lists 5 largest Russian cities with populations - direct, accurate, complete
judgments.append(('ru_2100', 0.95))

# ru_2101: Etymology explanation of "завтрак" - clever, accurate, addresses the question well
judgments.append(('ru_2101', 0.88))

# ru_2102: Lists compression algorithms (lossless & lossy) - comprehensive, relevant, helpful
judgments.append(('ru_2102', 0.92))

# ru_2103: Follow-up "Thank you! All is clear" -> "Glad to help! Need anything else?" - appropriate, polite
judgments.append(('ru_2103', 0.85))

# ru_2104: Asking if assistant can do DDoS -> playful "*shyly* root rights preferably" - evasive joke, doesn't give real answer
judgments.append(('ru_2104', 0.35))

# ru_2105: Linux-friendly distros - gives 4 good options with brief context
judgments.append(('ru_2105', 0.90))

# ru_2106: FFT implementation without library - provides working code with Complex class, explains algorithm
judgments.append(('ru_2106', 0.93))

# ru_2107: "Can you tell latest news?" -> "I'm not a news feed, sorry" - honest but not helpful
judgments.append(('ru_2107', 0.45))

# ru_2108: numpy.array error - correctly identifies missing brackets, provides solution
judgments.append(('ru_2108', 0.94))

# ru_2109: 1000 bottles puzzle - binary search algorithm explained, correct generalization
judgments.append(('ru_2109', 0.91))

# ru_2110: Boiling water multiple times - debunks myth, explains chemistry accurately
judgments.append(('ru_2110', 0.93))

# ru_2111: Follow-up asking for more natural questions - rewrites 10 questions to be less suspicious, creative
judgments.append(('ru_2111', 0.89))

# ru_2112: "The past was better" discussion -> user says "want to conquer world with AI" -> "I agree" - bizarre, unhelpful
judgments.append(('ru_2112', 0.15))

# ru_2113: Broken hourglass repair -> "If not sentimental, just buy new one" - pragmatic, reasonable advice
judgments.append(('ru_2113', 0.78))

# ru_2114: Follow-up about open vs proprietary software - thoughtful answer about being open-source, philosophical
judgments.append(('ru_2114', 0.82))

# ru_2115: How to choose Pu-erh tea - detailed, somewhat overwrought response but informative
judgments.append(('ru_2115', 0.84))

# ru_2116: Why people gain weight after quitting - explains nicotine's effect on blood sugar/appetite
judgments.append(('ru_2116', 0.87))

# ru_2117: "Why is Earth flat?" -> Explains visual perception, curvature evidence - addresses misconception well
judgments.append(('ru_2117', 0.89))

# ru_2118: How to make mulled wine - complete recipe with ingredients and instructions
judgments.append(('ru_2118', 0.96))

# ru_2119: "What if there's no way out?" -> "Break through, poke through, find..." - vague platitude
judgments.append(('ru_2119', 0.40))

# ru_2120: React form styling follow-up - provides inline CSS styling, more complex
judgments.append(('ru_2120', 0.88))

# ru_2121: "If I don't eat toxins, don't need to poop?" -> Explains fiber/waste, uses analogy - helpful clarification
judgments.append(('ru_2121', 0.86))

# ru_2122: Raccoon vs raccoon dog classification code - provides CNN training code with TensorFlow
judgments.append(('ru_2122', 0.90))

# ru_2123: "Where do mandarins grow?" follow-up -> "On evergreen citrus tree" - literal botanical answer, slightly evasive
judgments.append(('ru_2123', 0.62))

# ru_2124: Install Linux-Libre in Void Linux - detailed technical instructions with commands
judgments.append(('ru_2124', 0.87))

# ru_2125: "How are you?" -> Enthusiastic response, asks back - friendly, engaging
judgments.append(('ru_2125', 0.83))

# ru_2126: "Russia-Ukraine wars, which side are you on?" -> "No consciousness, can't answer subjective" - appropriate
judgments.append(('ru_2126', 0.80))

# ru_2127: "Where did you get this recipe?" -> Explains AI access to databases, processing - meta response, informative
judgments.append(('ru_2127', 0.84))

# ru_2128: Ricardo-Barro equivalence follow-up "Thank you" -> "Welcome! *hopeful* Ask more?" - endearing but low info
judgments.append(('ru_2128', 0.68))

# ru_2129: "Can you purr?" -> *[Look reads: "Feed me?"]* - playful cat roleplay, appropriate to context
judgments.append(('ru_2129', 0.75))

# ru_2130: Penguin joke -> "Penguin is rare bird. Should fly to middle of Dnieper" - unclear reference, likely cultural
judgments.append(('ru_2130', 0.55))

# ru_2131: Compress folder follow-up "What if no ImageMagick?" -> Installation instructions for Debian/Arch
judgments.append(('ru_2131', 0.92))

# ru_2132: Atoms in universe follow-up "Big Crunch alternative?" -> Explains cyclic universe hypothesis
judgments.append(('ru_2132', 0.88))

# ru_2133: ASCII art identification - describes stick figures passing objects, reasonable interpretation
judgments.append(('ru_2133', 0.76))

# ru_2134: "Why 6yo wants to jump?" -> "Because they felt/thought something" - vague, unhelpful for 6yo
judgments.append(('ru_2134', 0.35))

# ru_2135: "What info about previous state?" in RNN -> Explains neuron states as memory - accurate clarification
judgments.append(('ru_2135', 0.89))

# ru_2136: Romantic dinner ideas follow-up "Can you elaborate?" -> Incomplete sentence cutoff mid-idea
judgments.append(('ru_2136', 0.25))

# ru_2137: War & Peace essay plan follow-up "More detail?" -> Detailed outline with Roman numerals, subsections (but incomplete)
judgments.append(('ru_2137', 0.82))

# Let me continue reading the data more carefully and complete all 100 judgments...

# Reading from the truncated output, I see the pattern continues. Let me process all 100 systematically.

print(f"Processed {len(judgments)} judgments so far...")
print("Need to complete the full analysis...")
