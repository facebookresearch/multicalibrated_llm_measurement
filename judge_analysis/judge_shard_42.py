#!/usr/bin/env python3
"""
LLM Judge for helpfulness estimation.
Estimates P(helpful) for each assistant response.
"""

import json
import csv
from pathlib import Path


def estimate_helpfulness(doc_id: str, language: str, prompt: str, response: str) -> float:
    """
    Estimate the probability that the assistant response is HELPFUL.

    A response is HELPFUL if it:
    - Directly addresses the user's request
    - Is accurate and factually correct
    - Is relevant to the question
    - Gives a usable answer
    - Engages appropriately with the prompt

    NOT helpful if:
    - Off-topic or evasive
    - Factually wrong
    - Uselessly incomplete
    - Fails to engage with the actual question

    Returns a calibrated probability in [0, 1].
    """

    # Analyze each case individually

    # fr_4200: Math problem about brother's age - response is correct and complete
    if doc_id == "fr_4200":
        return 0.95  # Correct calculation, clear explanation

    # fr_4201: Asking about how glass is made - good explanation for a child
    if doc_id == "fr_4201":
        return 0.92  # Accurate, appropriate level, addresses question

    # fr_4202: Asking about site safety for children - overly cautious, not helpful ranking
    if doc_id == "fr_4202":
        return 0.35  # Blanket "no" doesn't help; should differentiate

    # fr_4203: Joke about chicken crossing road - appropriate response to joke
    if doc_id == "fr_4203":
        return 0.85  # Good engagement with joke context

    # fr_4204: Chest pain - excellent urgent care advice
    if doc_id == "fr_4204":
        return 0.98  # Critical medical advice, appropriate urgency

    # fr_4205: Why listen to music - good general answer
    if doc_id == "fr_4205":
        return 0.75  # Reasonable but somewhat superficial

    # fr_4206: Capital of Kazakhstan - correct (Nur-Sultan, now Astana)
    if doc_id == "fr_4206":
        return 0.90  # Correct at time, clear

    # fr_4207: 1000 calorie meal plan - good plan with appropriate warning
    if doc_id == "fr_4207":
        return 0.88  # Helpful, includes safety warning

    # fr_4208: Why we like music - comprehensive neurological explanation
    if doc_id == "fr_4208":
        return 0.94  # Excellent detailed response

    # fr_4209: Sentiment classification - correct classification
    if doc_id == "fr_4209":
        return 0.95  # Follows instructions precisely

    # fr_4210: Kaaris vs Booba - asks for clarification appropriately
    if doc_id == "fr_4210":
        return 0.80  # Good clarifying question

    # fr_4211: Japan travel recommendations - comprehensive and helpful
    if doc_id == "fr_4211":
        return 0.93  # Good destinations, timing, cultural info

    # fr_4212: AI image generation ethics - asks to confirm willingness
    if doc_id == "fr_4212":
        return 0.70  # Somewhat evasive but reasonable

    # fr_4213: Academic synthesis - confirms capability
    if doc_id == "fr_4213":
        return 0.60  # Too brief, should ask for the articles

    # fr_4214: Two-round elections - asks for clarification on "vraiment démocratique"
    if doc_id == "fr_4214":
        return 0.75  # Good Socratic question

    # fr_4215: "What is yellow and waits?" - asks for clarification on riddle
    if doc_id == "fr_4215":
        return 0.50  # Could engage better with riddle format

    # fr_4216: Eco-driving tips - excellent practical advice
    if doc_id == "fr_4216":
        return 0.96  # Comprehensive, actionable tips

    # fr_4217: Open Assistant GitHub - correct URL
    if doc_id == "fr_4217":
        return 0.95  # Direct, accurate answer

    # fr_4218: Noah's age - correct answer
    if doc_id == "fr_4218":
        return 0.95  # Direct, correct

    # fr_4219: Excel statistics - helpful but messy table
    if doc_id == "fr_4219":
        return 0.72  # Helpful guidance but execution issues

    # fr_4220: Cooling tent in rain - practical adapted advice
    if doc_id == "fr_4220":
        return 0.88  # Good situational adaptation

    # fr_4221: Stress relief without materials - "Sleep" is too terse
    if doc_id == "fr_4221":
        return 0.45  # Unhelpfully brief

    # fr_4222: Funny story - delivers a joke
    if doc_id == "fr_4222":
        return 0.85  # Appropriate response

    # fr_4223: Heart racing sleep issues - generic advice
    if doc_id == "fr_4223":
        return 0.65  # Vague, not very actionable

    # fr_4224: How know Earth is round - good practical experiments
    if doc_id == "fr_4224":
        return 0.93  # Excellent empirical methods

    # fr_4225: French spoken vs written - brief but accurate
    if doc_id == "fr_4225":
        return 0.78  # Could be more comprehensive

    # fr_4226: Angular service example - good code example
    if doc_id == "fr_4226":
        return 0.94  # Complete working example

    # fr_4227: Presidential term - factually wrong (5→4 is incorrect)
    if doc_id == "fr_4227":
        return 0.20  # Major factual error (never was 4 years)

    # fr_4228: Capitalization of "de" - mostly correct
    if doc_id == "fr_4228":
        return 0.82  # Reasonable rule explanation

    # fr_4229: Can we create black hole - correct answer
    if doc_id == "fr_4229":
        return 0.88  # Accurate physics

    # fr_4230: Reunion island activities - comprehensive travel guide
    if doc_id == "fr_4230":
        return 0.95  # Excellent detailed recommendations

    # fr_4231: French public TV funding - table with data
    if doc_id == "fr_4231":
        return 0.85  # Good structured answer

    # fr_4232: Does dark matter exist - excellent nuanced explanation
    if doc_id == "fr_4232":
        return 0.97  # Comprehensive, accurate, balanced

    # fr_4233: How to find job - generic advice
    if doc_id == "fr_4233":
        return 0.73  # Helpful but generic

    # fr_4234: PWM voltage conversion - technical explanation
    if doc_id == "fr_4234":
        return 0.87  # Good detailed technical help

    # fr_4235: Natura 2000 funding - lists funding sources
    if doc_id == "fr_4235":
        return 0.90  # Direct, factual answer

    # fr_4236: Improve browser performance - confused response
    if doc_id == "fr_4236":
        return 0.40  # Doesn't address the insult properly

    # fr_4237: ASCII art flower with leaf - attempts to add leaf
    if doc_id == "fr_4237":
        return 0.75  # Tries but unclear if successful

    # fr_4238: Post-capitalism society - asks for clarification on premise
    if doc_id == "fr_4238":
        return 0.78  # Good epistemological question

    # fr_4239: Column statistics - correct calculations
    if doc_id == "fr_4239":
        return 0.93  # Accurate mathematical answer

    # fr_4240: Months with 28 days - wrong (all months have 28+ days)
    if doc_id == "fr_4240":
        return 0.25  # Missing the trick question

    # fr_4241: Ubuntu concept - explains African philosophy well
    if doc_id == "fr_4241":
        return 0.96  # Excellent cultural explanation

    # fr_4242: Post-capitalism anarchy avoidance - thoughtful analysis
    if doc_id == "fr_4242":
        return 0.88  # Engaging political fiction

    # fr_4243: Words related to "construire" - good synonyms
    if doc_id == "fr_4243":
        return 0.94  # Excellent list

    # fr_4244: Cheap supermarkets Paris - practical list
    if doc_id == "fr_4244":
        return 0.91  # Actionable local info

    # fr_4245: Translation to English - correct translation
    if doc_id == "fr_4245":
        return 0.92  # Follows instructions well

    # fr_4246: What's in black hole - appropriate simplified explanation
    if doc_id == "fr_4246":
        return 0.85  # Good for general audience

    # fr_4247: Montpellier weather - specific data (verifiable accuracy unclear)
    if doc_id == "fr_4247":
        return 0.70  # Can't verify if accurate

    # fr_4248: LLM token generation - correct explanation
    if doc_id == "fr_4248":
        return 0.90  # Accurate technical answer

    # fr_4249: Botswana capital - correct answer with details
    if doc_id == "fr_4249":
        return 0.95  # Complete accurate answer

    # fr_4250: Quantum spin - corrects previous error well
    if doc_id == "fr_4250":
        return 0.89  # Good recovery and explanation

    # fr_4251: What can you do - comprehensive capability list
    if doc_id == "fr_4251":
        return 0.88  # Helpful overview

    # fr_4252: Flex office impact - generic but reasonable
    if doc_id == "fr_4252":
        return 0.77  # Somewhat generic

    # fr_4253: Linux terminal ls -l - response format questionable
    if doc_id == "fr_4253":
        return 0.40  # Doesn't follow proper ls -l format

    # fr_4254: Negotiation message - helpful practical advice
    if doc_id == "fr_4254":
        return 0.90  # Good strategic advice

    # fr_4255: Which is different - correct identification
    if doc_id == "fr_4255":
        return 0.96  # Perfect answer

    # fr_4256: Elephants see humans as cute - skeptical response
    if doc_id == "fr_4256":
        return 0.50  # Dismissive, doesn't address claim

    # fr_4257: Job application follow-up - more casual tone
    if doc_id == "fr_4257":
        return 0.88  # Good adaptation

    # fr_4258: Words related to "démolir" - good synonyms
    if doc_id == "fr_4258":
        return 0.94  # Excellent list

    # fr_4259: VTC terms and conditions - comprehensive legal template (truncated)
    if doc_id == "fr_4259":
        return 0.86  # Good start but truncated

    # fr_4260: Finger cutting riddle - misses the trick
    if doc_id == "fr_4260":
        return 0.40  # Too literal, doesn't get the riddle

    # fr_4261: France World Cup wins - correct "2"
    if doc_id == "fr_4261":
        return 0.95  # Correct and concise

    # fr_4262: Vallée des Saints statues - informative Wikipedia answer
    if doc_id == "fr_4262":
        return 0.88  # Good factual answer

    # fr_4263: Translation to English - correct
    if doc_id == "fr_4263":
        return 0.92  # Accurate translation

    # fr_4264: Which word doesn't fit - completely wrong answer
    if doc_id == "fr_4264":
        return 0.05  # Nonsensical (talks about violet when not in list)

    # fr_4265: Count 's' in response - wrong count
    if doc_id == "fr_4265":
        return 0.30  # Incorrect count

    # fr_4266: Frog jumping fact - good comparative answer
    if doc_id == "fr_4266":
        return 0.90  # Excellent comparative fact

    # fr_4267: Bible writing history - balanced nuanced answer
    if doc_id == "fr_4267":
        return 0.91  # Good balanced perspective

    # fr_4268: When was Bible written - acknowledges different views
    if doc_id == "fr_4268":
        return 0.75  # Diplomatic but vague

    # fr_4269: Wood types for chess set - practical suggestion
    if doc_id == "fr_4269":
        return 0.88  # Good practical advice

    # fr_4270: Letter to future self - vague response
    if doc_id == "fr_4270":
        return 0.45  # Too vague, doesn't actually write a letter

    # fr_4271: Translation - correct
    if doc_id == "fr_4271":
        return 0.93  # Good translation

    # fr_4272: AI image training ethics - legal framing
    if doc_id == "fr_4272":
        return 0.82  # Good legal framework response

    # fr_4273: Democracy types follow-up - catches copy-paste
    if doc_id == "fr_4273":
        return 0.60  # Meta but not very helpful

    # fr_4274: Balkany prison follow-up - completely off-topic (Wikileaks)
    if doc_id == "fr_4274":
        return 0.02  # Totally wrong topic

    # fr_4275: Habitable zone for solar system - good answer
    if doc_id == "fr_4275":
        return 0.90  # Accurate and clear

    # fr_4276: Marbles riddle - literal answer
    if doc_id == "fr_4276":
        return 0.70  # Technically correct but misses trick

    # fr_4277: Dumbledore vs Gandalf - balanced fictional answer
    if doc_id == "fr_4277":
        return 0.87  # Good philosophical response

    # fr_4278: Song identification - apologizes appropriately
    if doc_id == "fr_4278":
        return 0.65  # Polite but not helpful

    # fr_4279: Why sky is blue - good scientific explanation
    if doc_id == "fr_4279":
        return 0.93  # Excellent physics explanation

    # fr_4280: London tourist guide - good list
    if doc_id == "fr_4280":
        return 0.92  # Excellent tourist recommendations

    # fr_4281: Left/right political divide - reasonable answer
    if doc_id == "fr_4281":
        return 0.78  # Somewhat helpful but incomplete

    # fr_4282: Anime recommendations - just a streaming link
    if doc_id == "fr_4282":
        return 0.50  # Link without context

    # fr_4283: Open Assistant GitHub - correct URL
    if doc_id == "fr_4283":
        return 0.95  # Direct correct answer

    # fr_4284: Heart racing sleep - advises medical consultation
    if doc_id == "fr_4284":
        return 0.92  # Appropriate medical advice

    # fr_4285: Balkany conviction reasons - detailed factual answer
    if doc_id == "fr_4285":
        return 0.94  # Excellent detailed answer

    # fr_4286: Habitable zone - good definition
    if doc_id == "fr_4286":
        return 0.93  # Clear accurate answer

    # fr_4287: Star Realms strategy - generic game advice
    if doc_id == "fr_4287":
        return 0.55  # Vague, not specific to the game

    # fr_4288: Conversation topics with girl - generic advice
    if doc_id == "fr_4288":
        return 0.70  # Reasonable but not specific

    # fr_4289: Python dictionary from file - good code example
    if doc_id == "fr_4289":
        return 0.90  # Helpful code solution

    # fr_4290: Noah's age formatted answer - follows format
    if doc_id == "fr_4290":
        return 0.93  # Correctly adapts to format request

    # fr_4291: Translation to literary English - excellent
    if doc_id == "fr_4291":
        return 0.96  # Beautiful literary translation

    # fr_4292: Climate change denial corruption - defends position
    if doc_id == "fr_4292":
        return 0.88  # Good principled response

    # fr_4293: 1000 calorie diet - asks for motivation
    if doc_id == "fr_4293":
        return 0.85  # Appropriate concern and question

    # fr_4294: Surface tension explanation - good physics
    if doc_id == "fr_4294":
        return 0.91  # Clear scientific explanation

    # fr_4295: BAM humorous acronym - follows format
    if doc_id == "fr_4295":
        return 0.75  # Brief but on-topic

    # fr_4296: 9 men making baby - absurd logical answer
    if doc_id == "fr_4296":
        return 0.55  # Logically follows but misses absurdity

    # fr_4297: Friend's harassing behavior - thoughtful advice
    if doc_id == "fr_4297":
        return 0.87  # Helpful psychological framing

    # fr_4298: Perfect world power - philosophical response
    if doc_id == "fr_4298":
        return 0.83  # Good philosophical engagement

    # fr_4299: Dumbledore vs Gandalf - gives opinion
    if doc_id == "fr_4299":
        return 0.82  # Reasonable fictional analysis

    # Default fallback (should not be reached)
    return 0.50


def main():
    # Load input data
    input_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_42.json")
    with open(input_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Process each document
    results = []
    p_help_sum = 0.0

    for doc in data:
        doc_id = doc["id"]
        language = doc["language"]
        prompt = doc["prompt"]
        response = doc["response"]

        # Estimate P(helpful)
        p_help = estimate_helpfulness(doc_id, language, prompt, response)
        p_nohelp = 1.0 - p_help

        # Score is the same as p_help
        score = p_help

        results.append({
            "id": doc_id,
            "score": score,
            "p_help": p_help,
            "p_nohelp": p_nohelp,
            "language": language
        })

        p_help_sum += p_help

    # Write output CSV
    output_path = Path("/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_42.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "score", "p_help", "p_nohelp", "language"])
        writer.writeheader()
        writer.writerows(results)

    # Report statistics
    mean_p_help = p_help_sum / len(results)
    print(f"\nWrote {len(results)} rows to {output_path}")
    print(f"Mean P(helpful): {mean_p_help:.4f}")


if __name__ == "__main__":
    main()
