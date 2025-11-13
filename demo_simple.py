"""
Simplified English Text Analysis Workflow Demonstration

This demonstrates the key improvements made to cntext for English text analysis
without requiring full package installation.
"""

print("=" * 80)
print("ENGLISH TEXT ANALYSIS - WORKFLOW DEMONSTRATION")
print("=" * 80)
print("\nThis demo shows the enhancements made to cntext for English text analysis.")
print()

# ============================================================================
# SAMPLE DATA
# ============================================================================

print("\n" + "=" * 80)
print("SAMPLE DATA: Three Company Annual Reports")
print("=" * 80)

companies = {
    "TechInnovate Corp (Positive)": """
        TechInnovate Corporation demonstrated exceptional performance this quarter,
        achieving record revenues of $250 million, representing a 45% increase.
        Our innovative AI-driven products continue to disrupt the market, with
        customer satisfaction reaching all-time highs. The company successfully
        launched three groundbreaking products. Strong partnerships with major
        enterprises validate our innovative approach. We anticipate continued
        growth driven by our commitment to innovation and excellence.
    """,

    "TraditionalManufacture Inc (Negative)": """
        TraditionalManufacture Inc. reported stable revenues despite challenging
        market conditions and increasing competition. The company faced significant
        uncertainty regarding regulatory changes and potential litigation risks.
        Operations remained steady, though concerns persist about supply chain
        disruptions and rising costs. Management acknowledges difficulties in
        adapting. Conservative strategies helped mitigate risks, but growth remains
        constrained. The company experienced setbacks and encountered obstacles.
        Profitability declined due to inefficiencies. Uncertainty clouds the outlook.
    """,

    "BalancedSolutions LLC (Neutral)": """
        BalancedSolutions LLC achieved solid results with revenues of $200 million,
        reflecting modest growth of 8%. The company maintained its market position
        through reliable service delivery. While facing some challenges, including
        competitive pressures and uncertainties, the organization demonstrated
        resilience. Strategic investments in technology yielded positive returns,
        though some initiatives faced delays. The outlook remains cautiously optimistic.
    """
}

for company, text in companies.items():
    print(f"\n{company}:")
    word_count = len(text.split())
    print(f"  Word count: {word_count}")

# ============================================================================
# DEMONSTRATION 1: Enhanced English Tokenization
# ============================================================================

print("\n" + "=" * 80)
print("DEMO 1: ENHANCED ENGLISH TOKENIZATION")
print("=" * 80)

test_text = "The companies are running innovative programs. Revenue increased 25% in 2024!"

print(f"\nTest text: {test_text}")

print("\n--- OLD APPROACH (Simple split) ---")
old_tokens = test_text.lower().split()
print(f"Tokens: {old_tokens}")
print(f"Issues: Includes punctuation, no lemmatization, treats 'companies' and 'company' as different")

print("\n--- NEW APPROACH (Enhanced tokenization) ---")
print("Features:")
print("  ✓ Proper word boundaries")
print("  ✓ Punctuation removal")
print("  ✓ Optional lemmatization: 'companies' → 'company', 'running' → 'run'")
print("  ✓ Number normalization: '25%' → '_num_'")
print("  ✓ Stopword removal")

new_tokens_basic = [w.strip('.,!?%') for w in test_text.lower().split()
                    if w.strip('.,!?%') and not w.strip('.,!?%') in ['the', 'in', 'are']]
print(f"\nBasic enhanced: {new_tokens_basic}")
print(f"With lemmatization: ['company', 'run', 'innovative', 'program', 'revenue', 'increase', '_num_', '2024']")

# ============================================================================
# DEMONSTRATION 2: Sentiment Analysis Workflow
# ============================================================================

print("\n" + "=" * 80)
print("DEMO 2: SENTIMENT ANALYSIS (Loughran-McDonald Financial Dictionary)")
print("=" * 80)

print("\nFinancial Sentiment Dictionary:")
print("  - Positive: growth, success, innovative, exceptional, strong, excellent")
print("  - Negative: challenge, uncertainty, difficulty, risk, concern, decline")
print("  - Uncertainty: may, might, could, uncertain, depends, potential")

print("\n--- Manual Sentiment Analysis of Our Sample Texts ---")

# Simple word-based sentiment scoring
positive_words = ['exceptional', 'innovative', 'success', 'growth', 'strong',
                  'excellent', 'groundbreaking', 'achievement', 'increase']
negative_words = ['challenge', 'uncertainty', 'difficulty', 'risk', 'concern',
                  'decline', 'setback', 'obstacle', 'constrained', 'disruption']

for company, text in companies.items():
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    if pos_count + neg_count > 0:
        sentiment_ratio = (pos_count - neg_count) / (pos_count + neg_count)
    else:
        sentiment_ratio = 0

    tone = "POSITIVE" if sentiment_ratio > 0.2 else "NEGATIVE" if sentiment_ratio < -0.2 else "NEUTRAL"

    print(f"\n{company}:")
    print(f"  Positive words found: {pos_count}")
    print(f"  Negative words found: {neg_count}")
    print(f"  Net sentiment: {sentiment_ratio:.2f} ({tone})")

# ============================================================================
# DEMONSTRATION 3: Word Frequency (Before/After Lemmatization)
# ============================================================================

print("\n" + "=" * 80)
print("DEMO 3: WORD FREQUENCY - Impact of Lemmatization")
print("=" * 80)

sample = """
The companies are running innovative programs. Innovation drives progress.
The company runs multiple innovations. Running companies need innovation.
"""

print(f"Sample text: {sample.strip()}")

print("\n--- WITHOUT Lemmatization ---")
from collections import Counter
words_no_lemma = [w.strip('.,!?').lower() for w in sample.split() if len(w.strip('.,!?')) > 3]
freq_no_lemma = Counter(words_no_lemma)
print("Top words:", dict(freq_no_lemma.most_common(5)))
print("Notice: 'companies', 'company', 'running', 'runs' all counted separately")

print("\n--- WITH Lemmatization (Simulated) ---")
# Simulate lemmatization
lemma_map = {
    'companies': 'company', 'running': 'run', 'runs': 'run',
    'innovations': 'innovation', 'drives': 'drive'
}
words_lemma = []
for w in words_no_lemma:
    words_lemma.append(lemma_map.get(w, w))
freq_lemma = Counter(words_lemma)
print("Top words:", dict(freq_lemma.most_common(5)))
print("Notice: Word forms merged → more accurate frequency counts")

# ============================================================================
# DEMONSTRATION 4: Text Similarity
# ============================================================================

print("\n" + "=" * 80)
print("DEMO 4: TEXT SIMILARITY ANALYSIS")
print("=" * 80)

print("\nComparing how similar the three company reports are to each other.")
print("Higher score = more similar content/vocabulary")

# Simple Jaccard similarity (word overlap)
def simple_similarity(text1, text2):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0

companies_list = list(companies.items())
print("\nSimilarity Matrix:")
print(f"{'':35s}", end='')
for i, (comp, _) in enumerate(companies_list):
    short_name = comp.split('(')[0].strip()[:15]
    print(f"{short_name:>15s}", end='')
print()

for i, (comp1, text1) in enumerate(companies_list):
    short_name1 = comp1.split('(')[0].strip()[:30]
    print(f"{short_name1:35s}", end='')
    for j, (comp2, text2) in enumerate(companies_list):
        sim = simple_similarity(text1, text2)
        print(f"{sim:>15.3f}", end='')
    print()

# ============================================================================
# DEMONSTRATION 5: Keyword in Context
# ============================================================================

print("\n" + "=" * 80)
print("DEMO 5: KEYWORD IN CONTEXT")
print("=" * 80)

keywords = ['revenue', 'growth', 'challenge']
print(f"\nSearching for keywords: {keywords}")

for company, text in companies.items():
    print(f"\n{company}:")
    words = text.split()
    found_any = False

    for keyword in keywords:
        for i, word in enumerate(words):
            if keyword in word.lower():
                # Get context (3 words before and after)
                start = max(0, i-3)
                end = min(len(words), i+4)
                context = ' '.join(words[start:end])
                print(f"  '{keyword}': ...{context}...")
                found_any = True
                break  # Show only first occurrence

    if not found_any:
        print("  (No keywords found)")

# ============================================================================
# DEMONSTRATION 6: Code Improvements Summary
# ============================================================================

print("\n" + "=" * 80)
print("DEMO 6: KEY CODE IMPROVEMENTS IN THIS FORK")
print("=" * 80)

improvements = [
    {
        "File": "cntext/english_nlp.py",
        "Status": "✓ NEW MODULE",
        "Description": "Enhanced English tokenization with spaCy/NLTK, lemmatization support"
    },
    {
        "File": "cntext/model/sopmi.py",
        "Status": "✓ BUG FIXED",
        "Description": "Now properly handles English text (was always using jieba for all languages)"
    },
    {
        "File": "cntext/stats/utils.py",
        "Status": "✓ ENHANCED",
        "Description": "word_count() now supports lemmatize=True parameter"
    },
    {
        "File": "cntext/model/utils.py",
        "Status": "✓ ENHANCED",
        "Description": "preprocess_line() improved with better tokenization and lemmatization"
    },
    {
        "File": "setup.py & pyproject.toml",
        "Status": "✓ UPDATED",
        "Description": "Added spaCy as optional dependency: pip install cntext[english]"
    }
]

print("\n")
for item in improvements:
    print(f"{item['Status']:15s} {item['File']}")
    print(f"{'':15s} → {item['Description']}")
    print()

# ============================================================================
# DEMONSTRATION 7: Comparison Table
# ============================================================================

print("\n" + "=" * 80)
print("DEMO 7: BEFORE vs. AFTER COMPARISON")
print("=" * 80)

comparison = [
    ("Tokenization", "text.split()", "spaCy/NLTK tokenization"),
    ("Lemmatization", "❌ Not available", "✓ Optional parameter"),
    ("Punctuation", "Manual removal", "✓ Automatic"),
    ("Number handling", "Basic", "✓ Decimals supported"),
    ("SoPmi English", "❌ Broken (used jieba)", "✓ Fixed"),
    ("Quality", "Basic", "✓ Research-grade"),
]

print(f"\n{'Feature':<20s} {'Before (Original)':<30s} {'After (This Fork)':<30s}")
print("-" * 80)
for feature, before, after in comparison:
    print(f"{feature:<20s} {before:<30s} {after:<30s}")

# ============================================================================
# DEMONSTRATION 8: Usage Examples
# ============================================================================

print("\n" + "=" * 80)
print("DEMO 8: HOW TO USE THE ENHANCED FEATURES")
print("=" * 80)

print("""
1. BASIC WORD COUNT (works without spaCy):

   import cntext as ct
   text = "The companies are running innovative programs."
   counts = ct.word_count(text, lang='english')
   # Result: {'companies': 1, 'running': 1, 'innovative': 1, ...}

2. WITH LEMMATIZATION (requires spaCy):

   counts = ct.word_count(text, lang='english', lemmatize=True)
   # Result: {'company': 1, 'run': 1, 'innovative': 1, ...}
   # Notice: 'companies' → 'company', 'running' → 'run'

3. SENTIMENT ANALYSIS:

   lm_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')
   sentiment = ct.sentiment(text, diction=lm_dict, lang='english')
   # Result: {'Positive': 2, 'Negative': 0, 'Uncertainty': 1, ...}

4. WORD EMBEDDINGS WITH BETTER PREPROCESSING:

   wv = ct.Word2Vec(
       corpus_file='corpus.txt',
       lang='english',
       lemmatize=True  # New parameter!
   )

5. SEMANTIC PROJECTION (Measure abstract concepts):

   # Create axis: "innovative vs. traditional"
   axis = ct.generate_concept_axis(
       wv,
       poswords=['innovation', 'creative', 'novel'],
       negwords=['traditional', 'conventional', 'conservative']
   )

   score = ct.project_text(wv, text, axis, lang='english')
   # Score > 0 = more innovative, Score < 0 = more traditional

6. INSTALL WITH ENHANCED SUPPORT:

   pip install cntext[english]
   python -m spacy download en_core_web_sm
""")

# ============================================================================
# CONCLUSION
# ============================================================================

print("\n" + "=" * 80)
print("DEMONSTRATION COMPLETE")
print("=" * 80)

print("""
What This Fork Provides:

✓ Better English tokenization (not just text.split())
✓ Optional lemmatization support for more accurate analysis
✓ Bug fixes in English language handling
✓ spaCy integration with graceful NLTK fallback
✓ All Chinese functionality preserved
✓ Backward compatible - all changes are opt-in

Key Use Cases:

1. Academic Research: Analyze English text in social science studies
2. Financial Analysis: Sentiment analysis of earnings calls, reports
3. Content Analysis: Study language patterns, semantic shifts over time
4. Market Research: Analyze customer feedback, reviews, surveys
5. Cultural Studies: Measure attitudes, biases, cultural concepts

Next Steps:

1. Test with your own English text data
2. Explore the 4 built-in English dictionaries
3. Train domain-specific word embeddings
4. Use semantic projection to measure abstract concepts
5. Contribute additional English dictionaries

Thank you for using this English-enhanced fork of cntext!

Repository: https://github.com/yourusername/cntext (update with your URL)
Original: https://github.com/hiDaDeng/cntext
""")

print("=" * 80)
