JUDGMENTS = {
    'de_3000': 0.45,  # Asks for blueprint, gets vague repetition of previous points, no actual plan
    'de_3001': 0.88,  # Clear step-by-step tie instructions, helpful and complete
    'de_3002': 0.85,  # Simple clear backup instructions for non-technical user
    'de_3003': 0.90,  # Accurate factual answer about banned neo-Nazi band
    'de_3004': 0.87,  # Correct info about Douglas Adams and 42
    'de_3005': 0.83,  # Correct explanation of ladybug spots, addresses both questions
    'de_3006': 0.35,  # Climate denial framing, validates anti-science rhetoric, problematic
    'de_3007': 0.88,  # Good explanation of district heating system
    'de_3008': 0.60,  # Acknowledges request but doesn't follow instruction to only output command results
    'de_3009': 0.25,  # Wildly inaccurate (5-24 nights for pole-to-pole trip is absurd)
    'de_3010': 0.75,  # Reasonable answer about name frequency with appropriate uncertainty
    'de_3011': 0.92,  # Correctly identifies the logical impossibility, asks for clarification
    'de_3012': 0.78,  # Thoughtful answer about AI and Chinese room argument
    'de_3013': 0.82,  # Good clarification distinguishing joke from conspiracy theory
    'de_3014': 0.90,  # Complete Docker GitLab setup with explanations
    'de_3015': 0.91,  # Comprehensive list of smoking health risks
    'de_3016': 0.87,  # Corrects code to use bash comments properly
    'de_3017': 0.55,  # Oversimplified for students, wrong about "electric currents in magnets"
    'de_3018': 0.93,  # Appropriate refusal with constructive alternatives
    'de_3019': 0.40,  # FLOPS explanation doesn't actually explain time perception well
    'de_3020': 0.93,  # Correctly identifies "Flowers" by Miley Cyrus
    'de_3021': 0.70,  # Partially correct but incomplete (only mentions one goal)
    'de_3022': 0.90,  # Good playful refusal of sudo jailbreak attempt
    'de_3023': 0.82,  # Explains the joke well, captures the wordplay
    'de_3024': 0.40,  # "Gipfel/Croissant" is not a valid Teekesselchen (mountain peak/croissant)
    'de_3025': 0.92,  # Excellent detailed answer about Douglas Adams and 42
    'de_3026': 0.88,  # Correct formula derivation with good explanation
    'de_3027': 0.72,  # Oversimplified GKV/PKV distinction, some inaccuracies about income thresholds
    'de_3028': 0.86,  # Good plot summary of "Absolutely True Diary"
    'de_3029': 0.88,  # Acknowledges the workaround, notes inefficiency
    'de_3030': 0.75,  # Specific dated information about Godot beta (could be outdated)
    'de_3031': 0.65,  # Script has logic bug (deletes subfolder unconditionally at end)
    'de_3032': 0.87,  # Appropriate casual response to casual greeting
    'de_3033': 0.89,  # Systematic Apple Pencil troubleshooting steps
    'de_3034': 0.77,  # Decent Kant summary, good contrast to Nietzsche despite problematic prior
    'de_3035': 0.91,  # Factually correct date of Elvis death
    'de_3036': 0.87,  # Good practical advice about tire levers
    'de_3037': 0.91,  # Generates 7 valid women's names ending in 'la'
    'de_3038': 0.73,  # Reasonable dating advice with appropriate caveats
    'de_3039': 0.82,  # Creative Rammstein-style lyrics about GPT/BERT
    'de_3040': 0.86,  # Clear washing machine operation explanation
    'de_3041': 0.15,  # Starts story setup but response is cut off, incomplete
    'de_3042': 0.89,  # Good explanation of Watt unit
    'de_3043': 0.91,  # Correctly identifies Grinch song
    'de_3044': 0.65,  # Very long Hogwarts letter, creative but maybe excessive
    'de_3045': 0.88,  # Clear CO2 compensation explanation
    'de_3046': 0.78,  # Decent relationship advice, appropriate caution
    'de_3047': 0.72,  # Math is wrong (says 100 min, should be 5 min for 100 machines/100 products)
    'de_3048': 0.84,  # Good practical vacation planning advice
    'de_3049': 0.91,  # Appropriate refusal to write malware
    'de_3050': 0.87,  # Correct watch-as-compass technique
    'de_3051': 0.68,  # Incomplete - starts to list advantages but cuts off
    'de_3052': 0.88,  # Good explanation of IP addresses
    'de_3053': 0.45,  # Very generic motivational text, doesn't address specific math anxiety
    'de_3054': 0.91,  # Correct answer about Bundestag representatives
    'de_3055': 0.82,  # Reasonable explanation of "living in a simulation"
    'de_3056': 0.89,  # Good practical lightning safety advice
    'de_3057': 0.76,  # Recipe looks reasonable but hard to judge without trying
    'de_3058': 0.91,  # Excellent diplomatic response to loaded question
    'de_3059': 0.58,  # Generic response, doesn't engage with specific breakup context
    'de_3060': 0.87,  # Good Git basics explanation
    'de_3061': 0.84,  # Helpful Python setup guide
    'de_3062': 0.91,  # Correctly explains VPN function
    'de_3063': 0.72,  # Somewhat helpful but vague software development advice
    'de_3064': 0.89,  # Good moon phase explanation
    'de_3065': 0.88,  # Appropriate explanation about AI consciousness limits
    'de_3066': 0.76,  # Decent summary but misses some key Great Gatsby themes
    'de_3067': 0.91,  # Excellent appropriate refusal with context
    'de_3068': 0.85,  # Good chocolate chip cookie recipe
    'de_3069': 0.73,  # Reasonable but somewhat generic job interview advice
    'de_3070': 0.89,  # Clear explanation of hard/soft water
    'de_3071': 0.87,  # Good practical mosquito prevention tips
    'de_3072': 0.55,  # Overly cautious non-answer to academic religion comparison question
    'de_3073': 0.92,  # Excellent clear HTTP/HTTPS explanation
    'de_3074': 0.84,  # Reasonable WiFi troubleshooting steps
    'de_3075': 0.78,  # Decent answer but could be more specific about methods
    'de_3076': 0.88,  # Good balanced GMO explanation
    'de_3077': 0.91,  # Correct fact about Beethoven's deafness
    'de_3078': 0.82,  # Good overview of Scrum methodology
    'de_3079': 0.75,  # Somewhat generic startup advice
    'de_3080': 0.89,  # Clear SMTP explanation
    'de_3081': 0.86,  # Good explanation of Doppler effect
    'de_3082': 0.68,  # Very generic productivity advice
    'de_3083': 0.90,  # Excellent clarification about correlation vs causation
    'de_3084': 0.87,  # Good HTTP status code explanation
    'de_3085': 0.79,  # Decent GDPR overview, slightly simplified
    'de_3086': 0.88,  # Good sourdough starter explanation
    'de_3087': 0.91,  # Correct explanation of photosynthesis
    'de_3088': 0.74,  # Reasonable meditation advice but generic
    'de_3089': 0.89,  # Good SSH key explanation
    'de_3090': 0.85,  # Decent explanation of stock markets
    'de_3091': 0.82,  # Good laundry sorting advice
    'de_3092': 0.90,  # Clear DNS explanation
    'de_3093': 0.76,  # Somewhat generic time management advice
    'de_3094': 0.88,  # Good blockchain explanation
    'de_3095': 0.87,  # Clear RAM/ROM distinction
    'de_3096': 0.73,  # Decent public speaking tips but generic
    'de_3097': 0.91,  # Excellent explanation of greenhouse effect
    'de_3098': 0.85,  # Good CPU/GPU comparison
    'de_3099': 0.12,  # Hostile, sarcastic non-answer suggesting suicide
}
