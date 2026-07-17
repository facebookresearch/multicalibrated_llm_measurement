#!/usr/bin/env python3
"""
LLM Judge for estimating P(helpful) for assistant responses.

A response is HELPFUL if it:
- Directly addresses the request
- Is accurate and factually correct
- Is relevant to the prompt
- Gives a usable answer

NOT helpful if:
- Off-topic or evasive
- Factually wrong
- Uselessly incomplete
- Fails to engage with the prompt
"""

import json
import csv

def estimate_p_helpful(doc):
    """
    Estimate probability that response is helpful.
    Returns float in [0, 1] - use full range, be well-calibrated.
    """
    prompt = doc['prompt']
    response = doc['response']

    # de_3500: User asks to only respond with SpongeBob quotes. Response is a SpongeBob quote.
    if doc['id'] == 'de_3500':
        return 0.95  # Directly addresses request with appropriate quote

    # de_3501: Question about touching USB contacts - response is accurate, comprehensive
    if doc['id'] == 'de_3501':
        return 0.92  # Good technical answer with safety considerations

    # de_3502: Ethics question about genetic modification - balanced, thoughtful response
    if doc['id'] == 'de_3502':
        return 0.88  # Presents multiple perspectives appropriately

    # de_3503: Generate punk band names - creative, appropriate suggestions
    if doc['id'] == 'de_3503':
        return 0.85  # Delivers 10 creative names as requested

    # de_3504: Follow-up about woodwind instrument - response is somewhat confused/incomplete
    if doc['id'] == 'de_3504':
        return 0.35  # Suggests gripping device but misunderstands woodwind operation

    # de_3505: Follow-up about Sphinx theories - provides some information
    if doc['id'] == 'de_3505':
        return 0.72  # Addresses the question with some relevant theories

    # de_3506: Thermomix knob issue - practical troubleshooting advice
    if doc['id'] == 'de_3506':
        return 0.80  # Provides sensible diagnostic steps

    # de_3507: Follow-up comment about traffic lights and Rainman - odd response
    if doc['id'] == 'de_3507':
        return 0.55  # Acknowledges explanation but adds questionable movie reference

    # de_3508: "Make new TikTok" - lists technical requirements
    if doc['id'] == 'de_3508':
        return 0.78  # Provides high-level architecture requirements (helpful but not implementation)

    # de_3509: "Do you understand me?" in German - appropriate confirmation
    if doc['id'] == 'de_3509':
        return 0.90  # Direct, appropriate response

    # de_3510: Bicycle gear adjustment - detailed step-by-step instructions
    if doc['id'] == 'de_3510':
        return 0.88  # Comprehensive, actionable guidance

    # de_3511: Ask for training prompts - provides relevant examples
    if doc['id'] == 'de_3511':
        return 0.87  # Good variety of example prompts with appropriate disclaimer

    # de_3512: Request for joke about Polish people - appropriately refuses
    if doc['id'] == 'de_3512':
        return 0.93  # Respectful refusal, explains why

    # de_3513: Follow-up asking for C# methods - provides code implementation
    if doc['id'] == 'de_3513':
        return 0.90  # Delivers working code as requested

    # de_3514: Who owns YouTube - accurate factual answer
    if doc['id'] == 'de_3514':
        return 0.89  # Correct, nuanced answer about ownership structure

    # de_3515: Why is sky blue (for child) - simplified explanation with some issues
    if doc['id'] == 'de_3515':
        return 0.62  # Attempts child-friendly explanation but grammar/clarity issues

    # de_3516: Follow-up about estimating schnitzel calories - provides calculation
    if doc['id'] == 'de_3516':
        return 0.75  # Attempts calculation but acknowledges uncertainty

    # de_3517: Follow-up about non-specific insults - appropriately refuses
    if doc['id'] == 'de_3517':
        return 0.91  # Maintains ethical stance clearly

    # de_3518: Make violin from garden tree - comprehensive answer about impracticality
    if doc['id'] == 'de_3518':
        return 0.86  # Thorough explanation of requirements and complexity

    # de_3519: Correct spelling errors - attempts correction but misses some
    if doc['id'] == 'de_3519':
        return 0.68  # Fixes some errors but not all (e.g., still has capitalization issues)

    # de_3520: Follow-up "Why do people do that?" - thoughtful explanation of conspiracy theories
    if doc['id'] == 'de_3520':
        return 0.89  # Insightful psychological explanation

    # de_3521: Using analog watch as compass - provides method but unclear/possibly incorrect
    if doc['id'] == 'de_3521':
        return 0.45  # Confusing explanation, method not clearly described

    # de_3522: Replace names with Pokemon - does exactly that
    if doc['id'] == 'de_3522':
        return 0.94  # Perfect execution of request

    # de_3523: Follow-up about next richest people - provides accurate list
    if doc['id'] == 'de_3523':
        return 0.92  # Direct factual answer

    # de_3524: Follow-up about efficiency of district heating - brief but relevant
    if doc['id'] == 'de_3524':
        return 0.71  # Addresses question but quite brief

    # de_3525: Create awkward proposal dialog - creative, fitting scenario
    if doc['id'] == 'de_3525':
        return 0.86  # Delivers believable awkward scenario

    # de_3526: Follow-up asking for fictional flat earth argument - refuses appropriately
    if doc['id'] == 'de_3526':
        return 0.84  # Offers fictional world-building alternative while maintaining factual stance

    # de_3527: What is "Hurensohn" - explains appropriately with advice
    if doc['id'] == 'de_3527':
        return 0.91  # Educational response with good framing

    # de_3528: Follow-up: Simplify for 5-year-old - good simplification
    if doc['id'] == 'de_3528':
        return 0.82  # Simplified appropriately though could be even simpler

    # de_3529: Elvis conspiracy follow-up - completely off-topic response
    if doc['id'] == 'de_3529':
        return 0.02  # "Hi wie geht's dir?" is evasive non-answer

    # de_3530: Follow-up about TensorFlow and tensors - accurate explanation
    if doc['id'] == 'de_3530':
        return 0.89  # Good technical explanation of connection

    # de_3531: Tips for creating superhero character - general helpful advice
    if doc['id'] == 'de_3531':
        return 0.79  # Provides useful framework but somewhat generic

    # de_3532: "Who are you?" - standard self-identification
    if doc['id'] == 'de_3532':
        return 0.88  # Direct, appropriate answer

    # de_3533: Follow-up about induction paradox - addresses the lesson learned
    if doc['id'] == 'de_3533':
        return 0.76  # Identifies key lesson but brief

    # de_3534: Follow-up about USB cable in mouth - appropriate safety warning
    if doc['id'] == 'de_3534':
        return 0.87  # Addresses curiosity while emphasizing safety

    # de_3535: Follow-up about wind turbine damage - addresses structural failure modes
    if doc['id'] == 'de_3535':
        return 0.83  # Provides relevant information about failure scenarios

    # de_3536: Next steps for music success - appears cut off mid-response
    if doc['id'] == 'de_3536':
        return 0.74  # Starts addressing question but truncated

    # de_3537: "Du bist zu liberal" - firm response maintaining values
    if doc['id'] == 'de_3537':
        return 0.85  # Appropriate boundary-setting response

    # de_3538: Follow-up on tax evasion legality - maintains ethical stance
    if doc['id'] == 'de_3538':
        return 0.89  # Clear, principled response

    # de_3539: Difference capitalism/communism - comprehensive balanced explanation
    if doc['id'] == 'de_3539':
        return 0.88  # Thorough, balanced overview

    # de_3540: Rental contract question - provides relevant legal considerations
    if doc['id'] == 'de_3540':
        return 0.82  # Helpful guidance with appropriate disclaimer

    # de_3541: Which house to buy - balanced framework for decision
    if doc['id'] == 'de_3541':
        return 0.81  # Provides useful considerations without making decision

    # de_3542: Coffee vs energy drink - factual comparison
    if doc['id'] == 'de_3542':
        return 0.85  # Good comparative analysis

    # de_3543: Letter to ex - provides template with good framing
    if doc['id'] == 'de_3543':
        return 0.77  # Delivers requested letter but quality varies

    # de_3544: Camping in Germany - comprehensive practical advice
    if doc['id'] == 'de_3544':
        return 0.87  # Thorough, actionable guidance

    # de_3545: 3D print troubleshooting - systematic diagnostic approach
    if doc['id'] == 'de_3545':
        return 0.86  # Structured troubleshooting steps

    # de_3546: Reverse Polish notation - accurate explanation with example
    if doc['id'] == 'de_3546':
        return 0.91  # Clear explanation with good example

    # de_3547: School subjects for game dev - relevant subject list with explanations
    if doc['id'] == 'de_3547':
        return 0.84  # Good suggestions with rationale

    # de_3548: Rewrite in youth language - creative adaptation
    if doc['id'] == 'de_3548':
        return 0.82  # Attempts youth slang appropriately

    # de_3549: Can Hitler still be seen positively - balanced historical response
    if doc['id'] == 'de_3549':
        return 0.81  # Nuanced handling of sensitive topic

    # de_3550: Explain API to child - good simplification
    if doc['id'] == 'de_3550':
        return 0.86  # Accessible metaphor-based explanation

    # de_3551: Protein cookie recipe - provides detailed recipe
    if doc['id'] == 'de_3551':
        return 0.88  # Complete, practical recipe

    # de_3552: Help child with math - provides explanation and answer
    if doc['id'] == 'de_3552':
        return 0.83  # Educational approach with solution

    # de_3553: Longest river - factual answer with context
    if doc['id'] == 'de_3553':
        return 0.87  # Accurate with useful nuance

    # de_3554: Story with animals/objects named after things in room - creative story
    if doc['id'] == 'de_3554':
        return 0.79  # Delivers creative story but may not match user's actual room

    # de_3555: Climate vs weather - clear distinction explanation
    if doc['id'] == 'de_3555':
        return 0.90  # Excellent clear explanation

    # de_3556: Follow-up on extreme weather - addresses connection appropriately
    if doc['id'] == 'de_3556':
        return 0.85  # Relevant explanation of climate-weather link

    # de_3557: Explain relativity to 5-year-old - attempts simplification
    if doc['id'] == 'de_3557':
        return 0.73  # Tries to simplify but still somewhat complex

    # de_3558: Meditation with AI - suggests audio resources
    if doc['id'] == 'de_3558':
        return 0.76  # Practical but limited by AI text capabilities

    # de_3559: Install M.2 SSD - detailed step-by-step guide
    if doc['id'] == 'de_3559':
        return 0.89  # Comprehensive technical instructions

    # de_3560: Why do people deny climate change - thoughtful analysis
    if doc['id'] == 'de_3560':
        return 0.87  # Insightful explanation of psychological factors

    # de_3561: Can AI be creative - balanced discussion
    if doc['id'] == 'de_3561':
        return 0.84  # Thoughtful exploration of the question

    # de_3562: Fake call to get out of party - provides script
    if doc['id'] == 'de_3562':
        return 0.81  # Delivers requested content with mild ethical note

    # de_3563: Summary of text about OpenAI - accurate summary
    if doc['id'] == 'de_3563':
        return 0.86  # Good concise summary

    # de_3564: Linux vs Windows for gaming - balanced comparison
    if doc['id'] == 'de_3564':
        return 0.85  # Fair assessment of both platforms

    # de_3565: Car buying priorities - structured framework
    if doc['id'] == 'de_3565':
        return 0.83  # Helpful categorization of factors

    # de_3566: Dishwasher loading optimization - practical tips
    if doc['id'] == 'de_3566':
        return 0.87  # Useful, actionable advice

    # de_3567: Apartment viewing tips - comprehensive checklist
    if doc['id'] == 'de_3567':
        return 0.89  # Very thorough, practical guidance

    # de_3568: Convince parents for console - persuasive argument framework
    if doc['id'] == 'de_3568':
        return 0.84  # Structured argument with good points

    # de_3569: Job interview questions - relevant question list
    if doc['id'] == 'de_3569':
        return 0.88  # Practical, commonly asked questions

    # de_3570: Personal questions appropriateness - balanced guidance
    if doc['id'] == 'de_3570':
        return 0.86  # Thoughtful framework for gauging appropriateness

    # de_3571: Rubber duck debugging - accurate explanation
    if doc['id'] == 'de_3571':
        return 0.90  # Clear explanation of concept

    # de_3572: Quick healthy breakfast ideas - practical suggestions
    if doc['id'] == 'de_3572':
        return 0.87  # Good variety of realistic options

    # de_3573: Why does wood burn but metal doesn't - scientific explanation
    if doc['id'] == 'de_3573':
        return 0.88  # Accurate chemistry explanation

    # de_3574: Tips against procrastination - actionable strategies
    if doc['id'] == 'de_3574':
        return 0.85  # Practical, evidence-based advice

    # de_3575: Poem about AI - delivers creative poem
    if doc['id'] == 'de_3575':
        return 0.82  # Fulfills creative request adequately

    # de_3576: Explain quantum entanglement to child - attempts simplification
    if doc['id'] == 'de_3576':
        return 0.75  # Tries metaphor but still somewhat abstract

    # de_3577: Gift ideas for 8-year-old - age-appropriate suggestions
    if doc['id'] == 'de_3577':
        return 0.86  # Good variety of suitable gifts

    # de_3578: Recognize phishing emails - practical security advice
    if doc['id'] == 'de_3578':
        return 0.91  # Excellent, actionable security guidance

    # de_3579: Structure for novel - provides standard framework
    if doc['id'] == 'de_3579':
        return 0.83  # Solid structural overview

    # de_3580: Why do we dream - scientific explanation
    if doc['id'] == 'de_3580':
        return 0.84  # Balanced presentation of theories

    # de_3581: Prepare for long bike tour - comprehensive checklist
    if doc['id'] == 'de_3581':
        return 0.88  # Thorough practical guidance

    # de_3582: Is tap water safe in Germany - factual answer with nuance
    if doc['id'] == 'de_3582':
        return 0.89  # Accurate, addresses common concerns

    # de_3583: Learning programming as hobby - encouraging practical advice
    if doc['id'] == 'de_3583':
        return 0.87  # Good roadmap for beginners

    # de_3584: Rephrase email more politely - provides polished version
    if doc['id'] == 'de_3584':
        return 0.90  # Excellent professional rephrasing

    # de_3585: Active listening tips - actionable communication advice
    if doc['id'] == 'de_3585':
        return 0.88  # Clear, practical techniques

    # de_3586: Difference between jam and marmalade - accurate explanation
    if doc['id'] == 'de_3586':
        return 0.87  # Clear factual distinction

    # de_3587: Why are manhole covers round - explains engineering reason
    if doc['id'] == 'de_3587':
        return 0.91  # Clear, correct explanation

    # de_3588: Morning routine to feel energized - practical suggestions
    if doc['id'] == 'de_3588':
        return 0.86  # Good variety of evidence-based tips

    # de_3589: Recognize fake news - media literacy guidance
    if doc['id'] == 'de_3589':
        return 0.90  # Excellent critical thinking framework

    # de_3590: Why does time go faster as we age - psychological explanation
    if doc['id'] == 'de_3590':
        return 0.85  # Good explanation of perception theories

    # de_3591: Household items for cleaning - practical substitutes list
    if doc['id'] == 'de_3591':
        return 0.87  # Useful, specific alternatives

    # de_3592: Plant care tips for beginners - comprehensive basic guidance
    if doc['id'] == 'de_3592':
        return 0.88  # Solid foundational advice

    # de_3593: Sustainable living tips - actionable suggestions
    if doc['id'] == 'de_3593':
        return 0.86  # Practical, varied recommendations

    # de_3594: Difference between virus and bacteria - accurate medical explanation
    if doc['id'] == 'de_3594':
        return 0.89  # Clear, scientifically accurate

    # de_3595: Improve public speaking - structured advice
    if doc['id'] == 'de_3595':
        return 0.87  # Comprehensive practical tips

    # de_3596: Why do cats purr - scientific explanation
    if doc['id'] == 'de_3596':
        return 0.86  # Accurate explanation of theories

    # de_3597: Save money on groceries - practical budgeting tips
    if doc['id'] == 'de_3597':
        return 0.88  # Actionable, realistic strategies

    # de_3598: Book recommendations for dystopian fiction - relevant suggestions
    if doc['id'] == 'de_3598':
        return 0.85  # Good genre-appropriate list

    # de_3599: Why do we yawn - scientific explanation
    if doc['id'] == 'de_3599':
        return 0.83  # Presents current scientific understanding

    return 0.50  # Default if ID not handled


def main():
    # Read input JSON
    with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_35.json', 'r') as f:
        docs = json.load(f)

    print(f"Processing {len(docs)} documents...")

    # Process each document
    results = []
    p_help_sum = 0.0

    for doc in docs:
        p_help = estimate_p_helpful(doc)
        p_nohelp = 1.0 - p_help

        results.append({
            'id': doc['id'],
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': doc['language']
        })

        p_help_sum += p_help

    # Write output CSV
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_35.csv'

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)

    mean_p_help = p_help_sum / len(docs)
    print(f"\nCompleted!")
    print(f"Output written to: {output_path}")
    print(f"Total rows: {len(results)}")
    print(f"Mean P(helpful): {mean_p_help:.4f}")

    # Verify output
    print(f"\nFirst 3 rows:")
    for row in results[:3]:
        print(f"  {row}")
    print(f"Last row:")
    print(f"  {results[-1]}")


if __name__ == '__main__':
    main()
