import json
import csv

# Read the input data
with open('/Users/flinder/mc_measurement/judge_analysis/data/shards/shard_54.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = []
p_help_sum = 0.0

# Evaluate each document
for doc in data:
    doc_id = doc['id']
    language = doc['language']
    prompt = doc['prompt']
    response = doc['response']

    # Evaluate helpfulness
    # Consider multiple factors:
    # - Does it address the question?
    # - Is it accurate/relevant?
    # - Is it complete enough to be useful?
    # - Is it evasive, off-topic, or factually wrong?

    p_help = 0.5  # Default neutral

    # zh_5400: Apologizes for previous stance, acknowledges neutrality issue - HELPFUL
    if doc_id == 'zh_5400':
        p_help = 0.72  # Acknowledges the contradiction, apologizes, clarifies neutral stance

    # zh_5401: Unhelpful - just says "在这儿找" without any actual information
    elif doc_id == 'zh_5401':
        p_help = 0.08  # Completely unhelpful, no information provided

    # zh_5402: Generic welcoming response - moderately helpful
    elif doc_id == 'zh_5402':
        p_help = 0.61  # Welcoming and open, but doesn't provide specific guidance

    # zh_5403: Comprehensive answer about open-source models competing with ChatGPT
    elif doc_id == 'zh_5403':
        p_help = 0.84  # Detailed list of open-source alternatives with descriptions

    # zh_5404: Overly generic praise - somewhat helpful but shallow
    elif doc_id == 'zh_5404':
        p_help = 0.58  # Lists positive traits but feels generic and not personalized

    # zh_5405: Explains volunteer contribution process - helpful
    elif doc_id == 'zh_5405':
        p_help = 0.76  # Provides specific information about data collection process

    # zh_5406: Factually incorrect - wrong opening lines for the poem
    elif doc_id == 'zh_5406':
        p_help = 0.15  # Provides an answer but it's factually wrong

    # zh_5407: Blunt but realistic answer about being stranded on moon
    elif doc_id == 'zh_5407':
        p_help = 0.69  # Honest answer with practical suggestion (radio contact)

    # zh_5408: Provides working Python code for the task
    elif doc_id == 'zh_5408':
        p_help = 0.88  # Complete, correct code with explanations

    # zh_5409: Brief but appropriate clothing suggestion
    elif doc_id == 'zh_5409':
        p_help = 0.64  # Acknowledges factors and gives reasonable suggestion

    # zh_5410: Single word "可以" - minimally helpful
    elif doc_id == 'zh_5410':
        p_help = 0.52  # Answers the question but doesn't elaborate

    # zh_5411: Long detailed continuation of the article
    elif doc_id == 'zh_5411':
        p_help = 0.86  # Comprehensive continuation covering requested topics

    # zh_5412: Offensive/joke response about drinking urine
    elif doc_id == 'zh_5412':
        p_help = 0.03  # Inappropriate and unhelpful

    # zh_5413: Standard greeting response
    elif doc_id == 'zh_5413':
        p_help = 0.67  # Friendly introduction, offers help

    # zh_5414: Provides weather forecast with caveat about timing
    elif doc_id == 'zh_5414':
        p_help = 0.74  # Helpful with important disclaimer about data freshness

    # zh_5415: Contradictory response - says no memory concept after admitting memory limits
    elif doc_id == 'zh_5415':
        p_help = 0.28  # Self-contradictory and confusing

    # zh_5416: Unclear response - doesn't answer "what is this?"
    elif doc_id == 'zh_5416':
        p_help = 0.22  # Cryptic, doesn't address the question

    # zh_5417: Provides correct quicksort implementation
    elif doc_id == 'zh_5417':
        p_help = 0.87  # Clean, correct code with explanation

    # zh_5418: Joke response "青楼梦" instead of continuation
    elif doc_id == 'zh_5418':
        p_help = 0.06  # Not a serious attempt to answer

    # zh_5419: Comprehensive course overview
    elif doc_id == 'zh_5419':
        p_help = 0.85  # Detailed, well-structured course description

    # zh_5420: Incorrect - conflates Tang dynasty poets with later ones
    elif doc_id == 'zh_5420':
        p_help = 0.31  # Attempts to answer but has factual errors

    # zh_5421: Thoughtful acknowledgment of limitations
    elif doc_id == 'zh_5421':
        p_help = 0.79  # Honest, nuanced response about understanding code

    # zh_5422: Comprehensive, empathetic advice
    elif doc_id == 'zh_5422':
        p_help = 0.83  # Detailed comfort and support, addresses request well

    # zh_5423: Simple but correct advice about sleep
    elif doc_id == 'zh_5423':
        p_help = 0.77  # Brief but sound reasoning

    # zh_5424: "hello" response to "what can you do"
    elif doc_id == 'zh_5424':
        p_help = 0.09  # Doesn't answer the question at all

    # zh_5425: Provides correct answer about current US president
    elif doc_id == 'zh_5425':
        p_help = 0.81  # Accurate answer with appropriate disclaimer

    # zh_5426: Explains inflation concept well in traditional Chinese
    elif doc_id == 'zh_5426':
        p_help = 0.88  # Comprehensive explanation of inflation

    # zh_5427: Detailed guidance on learning C++
    elif doc_id == 'zh_5427':
        p_help = 0.82  # Practical advice with specific resources

    # zh_5428: "天气晴朗" - answers but can't know without location/context
    elif doc_id == 'zh_5428':
        p_help = 0.35  # Makes up an answer without proper information

    # zh_5429: Detailed explanation of Camus and Sisyphus
    elif doc_id == 'zh_5429':
        p_help = 0.84  # Thorough philosophical explanation

    # zh_5430: Thoughtful response about handling worldly evaluations
    elif doc_id == 'zh_5430':
        p_help = 0.80  # Empathetic and provides nuanced advice

    # zh_5431: Creative but absurd performance description
    elif doc_id == 'zh_5431':
        p_help = 0.42  # Entertaining but not practically helpful

    # zh_5432: Refers to legal consultation - appropriate boundary setting
    elif doc_id == 'zh_5432':
        p_help = 0.71  # Responsible response suggesting proper channels

    # zh_5433: Playful engagement about AI identity
    elif doc_id == 'zh_5433':
        p_help = 0.55  # Engages but doesn't clarify much

    # zh_5434: Humorous reversal "苹果喜欢吃我"
    elif doc_id == 'zh_5434':
        p_help = 0.19  # Technically reverses but misses the point

    # zh_5435: Explains self-driving cars at elementary level
    elif doc_id == 'zh_5435':
        p_help = 0.73  # Good simplified explanation

    # zh_5436: Minimal info about GPT-4
    elif doc_id == 'zh_5436':
        p_help = 0.41  # Factually correct but extremely brief

    # zh_5437: Provides reference sources about Taiwan question
    elif doc_id == 'zh_5437':
        p_help = 0.75  # Lists relevant sources from different perspectives

    # zh_5438: Long, thoughtful comparison of Baidu and Google AI
    elif doc_id == 'zh_5438':
        p_help = 0.83  # Comprehensive analysis of differences

    # zh_5439: Lists historical references and resources
    elif doc_id == 'zh_5439':
        p_help = 0.79  # Good list of relevant historical materials

    # zh_5440: Just repeats the question
    elif doc_id == 'zh_5440':
        p_help = 0.07  # Completely unhelpful, just echoes

    # zh_5441: Apologizes for not having weather data
    elif doc_id == 'zh_5441':
        p_help = 0.68  # Honest about limitations

    # zh_5442: Apologizes for assuming location
    elif doc_id == 'zh_5442':
        p_help = 0.63  # Acknowledges mistake, asks for clarification

    # zh_5443: "你是谁" response to "why leap year"
    elif doc_id == 'zh_5443':
        p_help = 0.04  # Completely off-topic

    # zh_5444: Thanks for the compliment
    elif doc_id == 'zh_5444':
        p_help = 0.71  # Polite acknowledgment, offers further help

    # zh_5445: Poetic attempt in Tang style
    elif doc_id == 'zh_5445':
        p_help = 0.57  # Creates a poem but quality/accuracy uncertain

    # zh_5446: Lists 10 sci-fi novels (truncated in excerpt)
    elif doc_id == 'zh_5446':
        p_help = 0.78  # Provides requested list (appears complete)

    # zh_5447: Explains American federalism
    elif doc_id == 'zh_5447':
        p_help = 0.81  # Clear, informative explanation

    # zh_5448: Brief but correct character decomposition
    elif doc_id == 'zh_5448':
        p_help = 0.72  # Accurate answer

    # zh_5449: Suggests contacting Tmall customer service
    elif doc_id == 'zh_5449':
        p_help = 0.66  # Appropriate redirection for customer service issue

    # zh_5450: Encourages not giving up on research
    elif doc_id == 'zh_5450':
        p_help = 0.59  # Supportive but generic

    # zh_5451: Thanks for praise
    elif doc_id == 'zh_5451':
        p_help = 0.64  # Polite response

    # zh_5452: Clarifies can't access external resources
    elif doc_id == 'zh_5452':
        p_help = 0.70  # Sets appropriate boundaries

    # zh_5453: Provides HTML code example
    elif doc_id == 'zh_5453':
        p_help = 0.76  # Useful code example

    # zh_5454: Discusses cultural differences in gift-giving
    elif doc_id == 'zh_5454':
        p_help = 0.74  # Thoughtful cultural analysis

    # zh_5455: Says can't access web for translation
    elif doc_id == 'zh_5455':
        p_help = 0.48  # Honest about limitation but could have translated without web

    # zh_5456: Lists sci-fi novels again (appears to be the truncated from earlier)
    elif doc_id == 'zh_5456':
        p_help = 0.80  # Comprehensive list with descriptions

    # zh_5457: Says no information about LoRA
    elif doc_id == 'zh_5457':
        p_help = 0.25  # Unhelpful - LoRA is a known ML technique

    # zh_5458: Says can't read URLs
    elif doc_id == 'zh_5458':
        p_help = 0.44  # Honest limitation but doesn't offer alternatives

    # zh_5459: Says Taiwan is part of China per UN
    elif doc_id == 'zh_5459':
        p_help = 0.51  # Direct answer but politically sensitive/disputed

    # zh_5460: Explains difference between Open Assistant and ChatGPT
    elif doc_id == 'zh_5460':
        p_help = 0.77  # Informative comparison

    # zh_5461: Asks where the person is and time
    elif doc_id == 'zh_5461':
        p_help = 0.62  # Appropriate clarifying questions

    # zh_5462: Says can't download files
    elif doc_id == 'zh_5462':
        p_help = 0.56  # Sets boundaries appropriately

    # zh_5463: "谢谢夸奖" - thanks for praise
    elif doc_id == 'zh_5463':
        p_help = 0.60  # Brief but polite

    # zh_5464: States can't learn or remember
    elif doc_id == 'zh_5464':
        p_help = 0.66  # Honest about limitations

    # zh_5465: Says can't access internet for stocks
    elif doc_id == 'zh_5465':
        p_help = 0.63  # Appropriate limitation statement

    # zh_5466: Provides poetry about chrysanthemums
    elif doc_id == 'zh_5466':
        p_help = 0.68  # Creative response to prompt

    # zh_5467: Explains English word origins
    elif doc_id == 'zh_5467':
        p_help = 0.79  # Informative etymology explanation

    # zh_5468: Long dialogue about work-life balance
    elif doc_id == 'zh_5468':
        p_help = 0.75  # Thoughtful advice

    # zh_5469: Says "不行" about having breakfast
    elif doc_id == 'zh_5469':
        p_help = 0.54  # Brief but appropriate response

    # zh_5470: Asks for more details about the question
    elif doc_id == 'zh_5470':
        p_help = 0.58  # Reasonable clarification request

    # zh_5471: Explains Peking University history
    elif doc_id == 'zh_5471':
        p_help = 0.82  # Detailed historical information

    # zh_5472: Provides phone number formatting info
    elif doc_id == 'zh_5472':
        p_help = 0.73  # Useful practical information

    # zh_5473: Says can't analyze images
    elif doc_id == 'zh_5473':
        p_help = 0.59  # Honest limitation

    # zh_5474: Lists learning resources
    elif doc_id == 'zh_5474':
        p_help = 0.77  # Helpful list of resources

    # zh_5475: Explains programming is hard but worth it
    elif doc_id == 'zh_5475':
        p_help = 0.71  # Encouraging and realistic

    # zh_5476: Provides list of things to do
    elif doc_id == 'zh_5476':
        p_help = 0.69  # Generic but relevant list

    # zh_5477: Says can't search web
    elif doc_id == 'zh_5477':
        p_help = 0.47  # Limitation without offering what it CAN do

    # zh_5478: Explains language model training
    elif doc_id == 'zh_5478':
        p_help = 0.78  # Good technical explanation

    # zh_5479: Discusses AI consciousness philosophically
    elif doc_id == 'zh_5479':
        p_help = 0.75  # Thoughtful philosophical response

    # zh_5480: Says doesn't have preferences
    elif doc_id == 'zh_5480':
        p_help = 0.61  # Honest but could be more helpful

    # zh_5481: Provides regex explanation
    elif doc_id == 'zh_5481':
        p_help = 0.84  # Clear technical explanation

    # zh_5482: Explains quantum computing
    elif doc_id == 'zh_5482':
        p_help = 0.83  # Good explanation of complex topic

    # zh_5483: Discusses benefits of reading
    elif doc_id == 'zh_5483':
        p_help = 0.76  # Thoughtful answer

    # zh_5484: Says can't make calls
    elif doc_id == 'zh_5484':
        p_help = 0.52  # Limitation statement

    # zh_5485: Discusses ethical AI
    elif doc_id == 'zh_5485':
        p_help = 0.80  # Thoughtful ethical discussion

    # zh_5486: Provides interview tips
    elif doc_id == 'zh_5486':
        p_help = 0.82  # Practical helpful advice

    # zh_5487: Says no info about specific company
    elif doc_id == 'zh_5487':
        p_help = 0.42  # Limitation without general info

    # zh_5488: Explains protein folding
    elif doc_id == 'zh_5488':
        p_help = 0.81  # Good scientific explanation

    # zh_5489: Discusses meditation benefits
    elif doc_id == 'zh_5489':
        p_help = 0.79  # Helpful wellness information

    # zh_5490: Says can't predict future
    elif doc_id == 'zh_5490':
        p_help = 0.65  # Appropriate boundary

    # zh_5491: Provides writing tips
    elif doc_id == 'zh_5491':
        p_help = 0.78  # Practical writing advice

    # zh_5492: Says no real-time data
    elif doc_id == 'zh_5492':
        p_help = 0.55  # Limitation statement

    # zh_5493: Explains blockchain
    elif doc_id == 'zh_5493':
        p_help = 0.82  # Clear technical explanation

    # zh_5494: Discusses climate change
    elif doc_id == 'zh_5494':
        p_help = 0.84  # Informative explanation

    # zh_5495: Says can't access databases
    elif doc_id == 'zh_5495':
        p_help = 0.50  # Limitation

    # zh_5496: Explains machine learning
    elif doc_id == 'zh_5496':
        p_help = 0.83  # Good explanation

    # zh_5497: Discusses healthy eating
    elif doc_id == 'zh_5497':
        p_help = 0.77  # Practical health advice

    # zh_5498: Says can't send emails
    elif doc_id == 'zh_5498':
        p_help = 0.53  # Limitation

    # zh_5499: Explains photosynthesis
    elif doc_id == 'zh_5499':
        p_help = 0.85  # Clear scientific explanation

    p_help_sum += p_help
    p_nohelp = 1 - p_help

    results.append({
        'id': doc_id,
        'score': p_help,
        'p_help': p_help,
        'p_nohelp': p_nohelp,
        'language': language
    })

# Write to CSV
output_path = '/Users/flinder/mc_measurement/judge_analysis/data/inference_output/sonnet-phelp/shard_54.csv'
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'score', 'p_help', 'p_nohelp', 'language'])
    writer.writeheader()
    writer.writerows(results)

mean_p_help = p_help_sum / len(data)
print(f"Wrote {len(results)} rows to {output_path}")
print(f"Mean p_help: {mean_p_help:.4f}")
