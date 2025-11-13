"""
Module Demo: LLM (Large Language Model Integration)

This demo covers the cntext.llm module for:
- Structured text analysis using LLMs
- Built-in analysis tasks (sentiment, emotion, classification, etc.)
- Custom prompts
- Support for Ollama, OpenAI, Alibaba Cloud, and custom backends
- Async batch processing

This is a NEWER feature that combines traditional NLP with modern LLMs!
"""

import sys
sys.path.insert(0, '/home/user/cntext')

print("=" * 80)
print("MODULE DEMO: LLM - Large Language Model Integration")
print("=" * 80)
print("\nThis module enables structured text analysis using LLMs!")
print("Combines the power of LLMs with cntext's analysis framework.\n")

# Sample texts for LLM analysis
sample_texts = {
    "customer_review": """
        I'm really impressed with this product! The quality exceeded my expectations
        and the customer service was outstanding. My only minor complaint is that
        shipping took a bit longer than expected. Overall, highly recommended!
    """,

    "financial_news": """
        The Federal Reserve announced a rate hike today, citing continued inflationary
        pressures. Markets responded negatively, with major indices declining. Analysts
        warn of potential recessionary risks if rates continue to rise aggressively.
    """,

    "technical_support": """
        User reported system crashes when loading large files. Attempted restart
        and cache clearing without success. Issue appears related to memory management.
        Escalating to engineering team for investigation and resolution.
    """
}

# ============================================================================
# 1. LLM Function Overview
# ============================================================================

print("1. LLM FUNCTION OVERVIEW")
print("-" * 80)

print("""
The ct.llm() function provides a unified interface for text analysis using LLMs.

Key Features:
- Structured output (returns parsed JSON/dict)
- Built-in analysis tasks
- Custom prompts supported
- Multiple backend support
- Rate limiting and error handling
- Async batch processing

Function Signature:
  ct.llm(
      text,                    # Input text to analyze
      prompt=None,             # Custom prompt (optional)
      output_format=None,      # Structured output format (optional)
      task=None,               # Built-in task name (optional)
      backend='ollama',        # LLM backend to use
      base_url='...',          # API endpoint
      api_key=None,            # API key if needed
      model_name='...',        # Model to use
      temperature=0.0,         # Sampling temperature
      max_tokens=1000,         # Max output tokens
      timeout=30               # Request timeout
  )

Supported Backends:
- ollama: Local Ollama server
- lmstudio: Local LM Studio
- openai: OpenAI API
- dashscope: Alibaba Cloud (Qwen models)
- Custom: Any OpenAI-compatible API
""")

# ============================================================================
# 2. Built-in Analysis Tasks
# ============================================================================

print("\n\n2. BUILT-IN ANALYSIS TASKS")
print("-" * 80)

print("""
cntext includes pre-configured prompts for common analysis tasks:

Task: 'sentiment'
-----------------
Analyzes overall sentiment (positive/negative/neutral/mixed)
Returns: {'sentiment': 'positive', 'confidence': 'high', 'reason': '...'}

Example:
  result = ct.llm(text, task='sentiment', backend='ollama', model_name='qwen2.5:3b')


Task: 'emotion'
---------------
Identifies specific emotions (joy, sadness, anger, fear, surprise, etc.)
Returns: {'primary_emotion': 'joy', 'emotions': ['joy', 'excitement'], ...}

Example:
  result = ct.llm(text, task='emotion', backend='ollama', model_name='qwen2.5:3b')


Task: 'classification'
----------------------
Classifies text into categories
Returns: {'category': 'technical_issue', 'confidence': 0.85, ...}

Example:
  result = ct.llm(
      text,
      task='classification',
      backend='ollama',
      model_name='qwen2.5:3b'
  )


Task: 'intent'
--------------
Identifies user intent (request, complaint, question, feedback, etc.)
Returns: {'intent': 'complaint', 'urgency': 'high', ...}


Task: 'keywords'
----------------
Extracts key terms and concepts
Returns: {'keywords': ['innovation', 'growth', 'market'], 'themes': [...]}


Task: 'entities'
----------------
Extracts named entities (people, organizations, locations, etc.)
Returns: {'entities': {'PERSON': [...], 'ORG': [...], 'LOC': [...]}}


Task: 'summarization'
---------------------
Generates concise summary
Returns: {'summary': '...', 'key_points': [...]}


Task: 'aspects'
---------------
Aspect-based sentiment analysis
Returns: {'aspects': [{'aspect': 'quality', 'sentiment': 'positive'}, ...]}
""")

# ============================================================================
# 3. Example Usage Scenarios
# ============================================================================

print("\n\n3. EXAMPLE USAGE SCENARIOS")
print("-" * 80)

print("""
Note: These examples assume you have Ollama or another LLM backend running.
The actual function calls require a working LLM setup.

Scenario 1: Customer Review Analysis
-------------------------------------
""")

example1 = """
import cntext as ct

review = '''I'm really impressed with this product! The quality exceeded
my expectations. Shipping was a bit slow though.'''

# Analyze sentiment
sentiment = ct.llm(
    text=review,
    task='sentiment',
    backend='ollama',
    model_name='qwen2.5:3b'
)
print(f"Sentiment: {sentiment['sentiment']}")
# Output: {'sentiment': 'positive', 'confidence': 'high', ...}

# Extract aspects
aspects = ct.llm(
    text=review,
    task='aspects',
    backend='ollama',
    model_name='qwen2.5:3b'
)
print("Aspects:", aspects)
# Output: {'aspects': [
#     {'aspect': 'quality', 'sentiment': 'positive'},
#     {'aspect': 'shipping', 'sentiment': 'negative'}
# ]}
"""

print(example1)

print("""
Scenario 2: Financial News Classification
------------------------------------------
""")

example2 = """
news = '''The Federal Reserve announced a rate hike today, citing
continued inflationary pressures...'''

# Classify news type
classification = ct.llm(
    text=news,
    task='classification',
    backend='ollama',
    model_name='qwen2.5:3b'
)
print(f"Category: {classification['category']}")
# Output: {'category': 'monetary_policy', 'confidence': 0.92, ...}

# Extract entities
entities = ct.llm(
    text=news,
    task='entities',
    backend='ollama',
    model_name='qwen2.5:3b'
)
print("Organizations:", entities['entities']['ORG'])
# Output: {'entities': {'ORG': ['Federal Reserve'], ...}}
"""

print(example2)

print("""
Scenario 3: Support Ticket Analysis
------------------------------------
""")

example3 = """
ticket = '''User reported system crashes when loading large files...'''

# Identify intent and urgency
intent = ct.llm(
    text=ticket,
    task='intent',
    backend='ollama',
    model_name='qwen2.5:3b'
)
print(f"Intent: {intent['intent']}, Urgency: {intent['urgency']}")
# Output: {'intent': 'technical_issue', 'urgency': 'high', ...}

# Extract keywords
keywords = ct.llm(
    text=ticket,
    task='keywords',
    backend='ollama',
    model_name='qwen2.5:3b'
)
print("Key themes:", keywords['themes'])
# Output: {'keywords': ['system', 'crash', 'files'], 'themes': ['performance', 'stability']}
"""

print(example3)

# ============================================================================
# 4. Custom Prompts
# ============================================================================

print("\n\n4. CUSTOM PROMPTS")
print("-" * 80)

print("""
You can provide custom prompts for specialized analysis:

Example: Custom Analysis for Research Papers
---------------------------------------------
""")

custom_example = """
paper_abstract = '''...research paper abstract...'''

custom_prompt = '''
Analyze this research paper abstract and extract:
1. Main research question
2. Methodology used
3. Key findings
4. Contribution to the field

Provide structured output in JSON format.
'''

result = ct.llm(
    text=paper_abstract,
    prompt=custom_prompt,
    backend='ollama',
    model_name='qwen2.5:7b',  # Larger model for complex task
    temperature=0.0,  # Deterministic output
    max_tokens=500
)

print("Research Question:", result['research_question'])
print("Methodology:", result['methodology'])
print("Key Findings:", result['key_findings'])
"""

print(custom_example)

# ============================================================================
# 5. Batch Processing
# ============================================================================

print("\n\n5. BATCH PROCESSING")
print("-" * 80)

print("""
For processing multiple texts efficiently:

Example: Analyze Multiple Reviews
----------------------------------
""")

batch_example = """
import asyncio
import cntext as ct

reviews = [
    "Great product! Highly recommended.",
    "Disappointed with the quality.",
    "Okay, nothing special.",
    # ... hundreds more ...
]

# Batch process with rate limiting
async def analyze_reviews(reviews):
    tasks = []
    for review in reviews:
        task = ct.llm_async(  # Async version
            text=review,
            task='sentiment',
            backend='ollama',
            model_name='qwen2.5:3b'
        )
        tasks.append(task)

    # Process with rate limiting (10 concurrent requests)
    results = await asyncio.gather(*tasks, limit=10)
    return results

# Run batch analysis
results = asyncio.run(analyze_reviews(reviews))

# Aggregate results
positive = sum(1 for r in results if r['sentiment'] == 'positive')
negative = sum(1 for r in results if r['sentiment'] == 'negative')

print(f"Positive: {positive}, Negative: {negative}")
"""

print(batch_example)

# ============================================================================
# 6. Backend Configuration
# ============================================================================

print("\n\n6. BACKEND CONFIGURATION")
print("-" * 80)

print("""
Configure different LLM backends:

Ollama (Local):
---------------
# Requires Ollama running locally (default: http://localhost:11434)
result = ct.llm(
    text=text,
    task='sentiment',
    backend='ollama',
    model_name='qwen2.5:3b'  # or 'llama2', 'mistral', etc.
)


OpenAI:
-------
result = ct.llm(
    text=text,
    task='sentiment',
    backend='openai',
    api_key='your-api-key',
    model_name='gpt-3.5-turbo'
)


Alibaba Cloud (Qwen):
----------------------
result = ct.llm(
    text=text,
    task='sentiment',
    backend='dashscope',
    api_key='your-dashscope-key',
    model_name='qwen-turbo'
)


LM Studio (Local):
------------------
result = ct.llm(
    text=text,
    task='sentiment',
    backend='lmstudio',
    base_url='http://localhost:1234/v1',
    model_name='local-model'
)


Custom OpenAI-Compatible API:
------------------------------
result = ct.llm(
    text=text,
    task='sentiment',
    backend='custom',
    base_url='https://your-api.com/v1',
    api_key='your-key',
    model_name='your-model'
)
""")

# ============================================================================
# 7. Integration with Traditional Methods
# ============================================================================

print("\n\n7. INTEGRATION WITH TRADITIONAL METHODS")
print("-" * 80)

print("""
Combine LLM analysis with cntext's traditional methods:

Hybrid Approach: Dictionary + LLM
----------------------------------
""")

hybrid_example = """
import cntext as ct

text = '''Company revenue increased 25% with strong market performance...'''

# 1. Traditional sentiment (fast, rule-based)
lm_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')
trad_sentiment = ct.sentiment(text, diction=lm_dict, lang='english')

# 2. LLM sentiment (slower, contextual)
llm_sentiment = ct.llm(text, task='sentiment', backend='ollama', model_name='qwen2.5:3b')

# 3. Compare and validate
print("Traditional:", trad_sentiment)
print("LLM:", llm_sentiment)

# Best practice: Use traditional for bulk screening,
# LLM for nuanced analysis of flagged texts
"""

print(hybrid_example)

# ============================================================================
# Summary
# ============================================================================

print("\n\n" + "=" * 80)
print("SUMMARY: LLM Module Functions")
print("=" * 80)

functions = [
    ("ct.llm(text, task, backend, ...)", "Synchronous LLM analysis", "✓"),
    ("ct.llm_async(text, task, backend, ...)", "Async LLM analysis", "✓"),
    ("Built-in tasks", "sentiment, emotion, classification, etc.", "✓"),
    ("Custom prompts", "Flexible analysis with your own prompts", "✓"),
    ("Multiple backends", "Ollama, OpenAI, Alibaba, custom", "✓"),
    ("Batch processing", "Efficient multi-text analysis", "✓"),
    ("Rate limiting", "Prevent API throttling", "✓"),
    ("Structured output", "Parsed JSON responses", "✓"),
]

print("\n{:<45s} {:<35s} {}".format("Feature", "Description", "Status"))
print("-" * 80)
for func, purpose, status in functions:
    print("{:<45s} {:<35s} {}".format(func, purpose, status))

print("\n" + "=" * 80)
print("LLM MODULE DEMO COMPLETE")
print("=" * 80)

print("""
Key Takeaways:

1. Modern LLM Integration:
   - Structured text analysis with LLMs
   - Pre-configured analysis tasks
   - Custom prompts supported
   - Returns parsed, structured data

2. Multiple Backends:
   - Local: Ollama, LM Studio
   - Cloud: OpenAI, Alibaba Cloud
   - Custom: Any OpenAI-compatible API
   - Choose based on cost, privacy, performance

3. Built-in Tasks:
   - sentiment: Overall tone analysis
   - emotion: Specific emotion detection
   - classification: Category assignment
   - intent: User intention identification
   - keywords: Key term extraction
   - entities: Named entity recognition
   - summarization: Text summarization
   - aspects: Aspect-based sentiment

4. Production Features:
   - Async batch processing
   - Rate limiting
   - Error handling
   - Timeout management
   - Structured output parsing

5. Best Practices:
   - Use traditional methods for bulk screening
   - Apply LLMs for nuanced analysis
   - Combine both for validation
   - Choose appropriate model size
   - Set temperature=0.0 for consistency

6. When to Use LLMs:
   - Complex, nuanced sentiment
   - Contextual understanding needed
   - Multiple aspects to extract
   - Ambiguous or figurative language
   - Domain-specific analysis

7. When to Use Traditional:
   - Large-scale screening
   - Speed is critical
   - Well-defined dictionaries exist
   - Interpretability required
   - Resource constraints

Setup Requirements:
- For Ollama: Install Ollama, pull models (ollama pull qwen2.5:3b)
- For OpenAI: API key and credits
- For Alibaba: DashScope API key
- For LM Studio: Install and run locally

Next Steps:
- Install Ollama or choose LLM backend
- Test with sample texts
- Create custom prompts for your domain
- Integrate with traditional analyses
- Build hybrid analysis pipelines
""")

print("\n" + "=" * 80)
print("ALL MODULE DEMOS COMPLETE!")
print("=" * 80)

print("""
You've now seen ALL major modules in cntext:

1. ✓ io      - File I/O and preprocessing
2. ✓ stats   - Statistical text analysis
3. ✓ model   - Word embeddings training
4. ✓ mind    - Semantic analysis (most innovative!)
5. ✓ plot    - Visualization
6. ✓ llm     - LLM integration (newest!)

Each module complements the others for comprehensive text analysis.
Combine them to build powerful analysis pipelines!

See demos/README.md for a complete overview and suggested workflows.
""")
