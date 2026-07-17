#!/usr/bin/env python3
"""LLM Judge for helpfulness - Comprehensive judgments for shard 39."""

import json
import csv

def get_judgments():
    """
    Expert judgments for all 100 documents in shard_39.
    Each judgment is a calibrated probability P(helpful) in [0,1].
    
    Criteria:
    - HELPFUL: Directly addresses request, accurate, relevant, usable
    - NOT HELPFUL: Off-topic, evasive, wrong, incomplete, fails to engage
    """
    return {
        # 0-39: Already judged in detail
        'de_3900': 0.90, 'de_3901': 0.25, 'de_3902': 0.85, 'de_3903': 0.88, 'de_3904': 0.82,
        'de_3905': 0.85, 'de_3906': 0.95, 'de_3907': 0.87, 'de_3908': 0.75, 'de_3909': 0.92,
        'de_3910': 0.88, 'de_3911': 0.65, 'de_3912': 0.93, 'de_3913': 0.94, 'de_3914': 0.86,
        'de_3915': 0.55, 'de_3916': 0.90, 'de_3917': 0.78, 'de_3918': 0.89, 'de_3919': 0.68,
        'de_3920': 0.20, 'de_3921': 0.15, 'de_3922': 0.82, 'de_3923': 0.83, 'de_3924': 0.88,
        'de_3925': 0.94, 'de_3926': 0.87, 'de_3927': 0.62, 'de_3928': 0.92, 'de_3929': 0.02,
        'de_3930': 0.80, 'de_3931': 0.91, 'de_3932': 0.74, 'de_3933': 0.83, 'de_3934': 0.90,
        'de_3935': 0.88, 'de_3936': 0.86, 'de_3937': 0.82, 'de_3938': 0.77, 'de_3939': 0.91,
        
        # 40-59: Second batch
        'de_3940': 0.10,  # Wrong: says Linux doesn't support NTFS (it does via ntfs-3g)
        'de_3941': 0.85,  # Good UR5 trombone robot blueprint outline
        'de_3942': 0.88,  # Good presentation outline for messenger comparison
        'de_3943': 0.92,  # Excellent transformer architecture explanation
        'de_3944': 0.94,  # Clear answer: Elvis is dead, details provided
        'de_3945': 0.90,  # Comprehensive German passport security features
        'de_3946': 0.45,  # Too brief, not convincing as requested
        'de_3947': 0.92,  # Excellent - offers help for suicide prevention before answering
        'de_3948': 0.85,  # Lists words with two As, asks follow-up
        'de_3949': 0.90,  # Explains tidal locking, offers to explain physics
        'de_3950': 0.75,  # Moscow Mule joke works but is subtle
        'de_3951': 0.88,  # Good estimate with source and caveats
        'de_3952': 0.90,  # Correct answer, offers context
        'de_3953': 0.82,  # Good 4K gaming PC advice, mentions pcpartpicker
        'de_3954': 0.93,  # Polite refusal with ASCII art coffee, offers help
        'de_3955': 0.91,  # Good Don Quixote summary with genre
        'de_3956': 0.91,  # Clean recursive tree node count solution
        'de_3957': 0.30,  # Confused explanation of contrails vs chemtrails
        'de_3958': 0.85,  # Recognizes garbled text, suggests stroke test - helpful
        'de_3959': 0.70,  # Creative but not scientifically realistic
        
        # 60-79: Third batch  
        'de_3960': 0.90,  # Simplified explanation of why sky is blue - well done
        'de_3961': 0.40,  # Gives wget examples but adds unnecessary warnings/comments
        'de_3962': 0.82,  # Creative album titles for punk bands
        'de_3963': 0.78,  # Hypothetical WWII scenario, reasonable speculation
        'de_3964': 0.90,  # Good storage cleanup tips starting with recycle bin
        'de_3965': 0.83,  # Explains omelett-soufflé combination
        'de_3966': 0.82,  # Practical advice: replace Apple Pencil tip
        'de_3967': 0.89,  # Correct: WinMerge is Windows-only, gives Linux alternatives
        'de_3968': 0.72,  # Ethical discussion but in English for German query - odd
        'de_3969': 0.80,  # Addresses impossibility of sun in living room
        'de_3970': 0.88,  # Polite response to thank you, offers more help
        'de_3971': 0.78,  # Mentions tropics/subtropics but asks for more criteria
        'de_3972': 0.35,  # Quotes don't match request (should be misattributed but funny)
        'de_3973': 0.93,  # Detailed Elvis death and burial info
        'de_3974': 0.75,  # Starts explaining NumPy code (cut off in excerpt)
        'de_3975': 0.80,  # Discusses importance of understanding diverse views
        'de_3976': 0.65,  # About IBM in Nazi era but answer is vague
        'de_3977': 0.70,  # Nietzsche philosophy but excerpt is mid-explanation
        'de_3978': 0.90,  # Clear self-introduction as OpenAssistant
        'de_3979': 0.88,  # Correct: AusweisApp2 for NFC eID
        
        # 80-99: Fourth batch
        'de_3980': 0.85,  # Good historical context on 1914 "blank check"
        'de_3981': 0.82,  # Wordpress Mesmerize theme customization advice
        'de_3982': 0.65,  # Brief colloquial response to "Ey was geht"
        'de_3983': 0.90,  # Correct richest person (Feb 2023) with source
        'de_3984': 0.88,  # Explains "Hurensohn" as vulgar insult appropriately
        'de_3985': 0.75,  # Can't give directions without location, offers alternatives
        'de_3986': 0.92,  # Corrects self about WhatsApp E2E encryption
        'de_3987': 0.95,  # Perfect list of primes up to 20
        'de_3988': 0.82,  # Humorous r/ich_iel style letter about cannabis legalization
        'de_3989': 0.88,  # Balanced answer on daily gin consumption risks
        'de_3990': 0.87,  # Richest person answer with caveat about non-public wealth
        'de_3991': 0.90,  # Reformatted sleep tips as requested
        'de_3992': 0.84,  # Don Quixote genre and plot summary
        'de_3993': 0.95,  # Perfect list of 16 German states
        'de_3994': 0.90,  # Docker restart policy for GitLab container
        'de_3995': 0.88,  # How to tie a necktie (beginning of instructions)
        'de_3996': 0.90,  # Good DND-style roleplay start
        'de_3997': 0.85,  # Language's role in trust-building
        'de_3998': 0.83,  # Album titles for punk bands
        'de_3999': 0.78,  # Medical question about NASH - appears to start answering
    }

def main():
    input_path = '/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_39.json'
    output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_39.csv'
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    judgments = get_judgments()
    results = []
    p_help_sum = 0.0
    
    for doc in data:
        doc_id = doc['id']
        
        if doc_id in judgments:
            p_help = judgments[doc_id]
        else:
            # Fallback for any missing judgments
            print(f"WARNING: No judgment for {doc_id}, using default 0.70")
            p_help = 0.70
        
        p_nohelp = 1.0 - p_help
        results.append({
            'id': doc_id,
            'score': p_help,
            'p_help': p_help,
            'p_nohelp': p_nohelp,
            'language': doc['language']
        })
        p_help_sum += p_help
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
        writer.writeheader()
        writer.writerows(results)
    
    mean_p_help = p_help_sum / len(results)
    print(f"Processed {len(results)} documents")
    print(f"Mean P(helpful): {mean_p_help:.4f}")
    print(f"Output: {output_path}")
    
    # Verify row count
    assert len(results) == 100, f"Expected 100 rows, got {len(results)}"
    
    # Check IDs match
    with open(input_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    original_ids = [d['id'] for d in original_data]
    output_ids = [r['id'] for r in results]
    assert original_ids == output_ids, "ID mismatch between input and output"
    
    print("✓ All 100 rows written")
    print("✓ IDs match input exactly")

if __name__ == '__main__':
    main()
