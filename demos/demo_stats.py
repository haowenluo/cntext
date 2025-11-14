"""
Module Demo: Stats (Statistical Text Analysis)

This demo covers all functions in the cntext.stats module for:
- Word frequency and counting
- Sentiment analysis
- Readability metrics
- Text similarity measures
- Economic indices (EPU, FEPU)
- Keyword extraction
"""

import sys
sys.path.insert(0, '/home/user/cntext')

print("=" * 80)
print("MODULE DEMO: STATS - Statistical Text Analysis")
print("=" * 80)
print()

# Sample texts for analysis
sample_texts = {
    "positive_tech": """
        TechInnovate achieved exceptional results this quarter, with revenue
        growing 45%. Our innovative products continue to disrupt markets and
        delight customers. Strong partnerships drive continued success. The
        future looks bright with expanding opportunities.
    """,

    "negative_concern": """
        The company faces significant challenges amid uncertain economic conditions.
        Revenue declined due to operational difficulties and rising costs. Management
        expresses concern about regulatory risks and potential litigation. Market
        obstacles constrain growth prospects.
    """,

    "neutral_report": """
        The company reported quarterly results in line with expectations. Revenue
        remained stable despite competitive pressures. Management continues to focus
        on operational efficiency while navigating evolving market dynamics. Some
        initiatives progressed while others faced delays.
    """
}

# ============================================================================
# 1. Word Frequency Analysis
# ============================================================================

print("1. WORD FREQUENCY ANALYSIS")
print("-" * 80)

try:
    import cntext as ct
    from collections import Counter

    text = sample_texts["positive_tech"]

    print("\n1.1 Basic word count (no lemmatization):")
    counts = ct.word_count(text, lang='english', lemmatize=False)
    print(f"Top 10 words:")
    for word, count in counts.most_common(10):
        print(f"  {word}: {count}")

    print("\n1.2 Word count WITH lemmatization:")
    print("(Note: Requires spaCy for best results, falls back to NLTK)")
    counts_lemma = ct.word_count(text, lang='english', lemmatize=True)
    print(f"Top 10 lemmatized words:")
    for word, count in counts_lemma.most_common(10):
        print(f"  {word}: {count}")

    print("\n1.3 Return as DataFrame:")
    df = ct.word_count(text, lang='english', return_df=True)
    print(df.head(10))

    print("\n✓ Word counting works!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 2. Sentiment Analysis
# ============================================================================

print("\n\n2. SENTIMENT ANALYSIS")
print("-" * 80)

print("\n2.1 Load sentiment dictionary:")
try:
    # Load Loughran-McDonald financial sentiment dictionary
    lm_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')
    print(f"Dictionary: {lm_dict['Name']}")
    print(f"Categories: {', '.join(lm_dict['Category'])}")

    print("\n2.2 Analyze sentiment for each text:")
    for name, text in sample_texts.items():
        sentiment = ct.sentiment(text, diction=lm_dict, lang='english')

        print(f"\n{name}:")
        print(f"  Positive:    {sentiment.get('Positive', 0):3d} words")
        print(f"  Negative:    {sentiment.get('Negative', 0):3d} words")
        print(f"  Uncertainty: {sentiment.get('Uncertainty', 0):3d} words")
        print(f"  Litigious:   {sentiment.get('Litigious', 0):3d} words")

        # Calculate net sentiment
        pos = sentiment.get('Positive', 0)
        neg = sentiment.get('Negative', 0)
        if pos + neg > 0:
            net = (pos - neg) / (pos + neg)
            tone = "POSITIVE" if net > 0.2 else "NEGATIVE" if net < -0.2 else "NEUTRAL"
            print(f"  Net Sentiment: {net:+.2f} ({tone})")

    print("\n✓ Sentiment analysis works!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n2.3 Try NRC emotion dictionary:")
try:
    nrc_dict = ct.read_yaml_dict('en_common_NRC.yaml')
    print(f"\nDictionary: {nrc_dict['Name']}")
    print(f"Emotion categories: {', '.join(nrc_dict['Category'][:8])}")

    text = sample_texts["positive_tech"]
    emotions = ct.sentiment(text, diction=nrc_dict, lang='english')

    print(f"\nEmotion analysis of positive_tech:")
    for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {emotion}: {count}")

    print("\n✓ NRC emotion analysis works!")

except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# 3. Readability Metrics
# ============================================================================

print("\n\n3. READABILITY METRICS")
print("-" * 80)

try:
    print("\nAnalyzing readability for each text:")
    print("(Higher scores = more complex/difficult to read)\n")

    for name, text in sample_texts.items():
        readability = ct.readability(text, lang='english')

        print(f"{name}:")
        print(f"  Flesch Reading Ease: {readability.get('Flesch', 0):.2f}")
        print(f"  Fog Index:           {readability.get('Fog', 0):.2f}")
        print(f"  SMOG:                {readability.get('SMOG', 0):.2f}")
        print(f"  Coleman-Liau:        {readability.get('Coleman_Liau', 0):.2f}")
        print(f"  ARI:                 {readability.get('ARI', 0):.2f}")
        print()

    print("✓ Readability metrics work!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 4. Text Similarity
# ============================================================================

print("\n\n4. TEXT SIMILARITY MEASURES")
print("-" * 80)

try:
    text1 = sample_texts["positive_tech"]
    text2 = sample_texts["negative_concern"]
    text3 = sample_texts["neutral_report"]

    print("\n4.1 Cosine Similarity:")
    sim_12 = ct.cosine_sim(text1, text2, lang='english')
    sim_13 = ct.cosine_sim(text1, text3, lang='english')
    sim_23 = ct.cosine_sim(text2, text3, lang='english')

    print(f"  Positive vs Negative: {sim_12:.3f}")
    print(f"  Positive vs Neutral:  {sim_13:.3f}")
    print(f"  Negative vs Neutral:  {sim_23:.3f}")

    print("\n4.2 Jaccard Similarity:")
    jac_12 = ct.jaccard_sim(text1, text2, lang='english')
    jac_13 = ct.jaccard_sim(text1, text3, lang='english')
    jac_23 = ct.jaccard_sim(text2, text3, lang='english')

    print(f"  Positive vs Negative: {jac_12:.3f}")
    print(f"  Positive vs Neutral:  {jac_13:.3f}")
    print(f"  Negative vs Neutral:  {jac_23:.3f}")

    print("\n4.3 Minimum Edit Distance:")
    # Note: This compares character-level differences
    # Works best for comparing similar texts
    short_text1 = "innovation drives growth"
    short_text2 = "innovation drives progress"

    distance = ct.minedit_sim(short_text1, short_text2, lang='english')
    print(f"  '{short_text1}' vs")
    print(f"  '{short_text2}'")
    print(f"  Distance: {distance:.3f} (lower = more similar)")

    print("\n✓ Similarity measures work!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 5. Keyword in Context
# ============================================================================

print("\n\n5. KEYWORD IN CONTEXT")
print("-" * 80)

try:
    text = sample_texts["positive_tech"]
    keywords = ['revenue', 'innovative', 'success']

    print(f"\nSearching for keywords: {keywords}")
    print(f"Window size: 3 words on each side\n")

    contexts = ct.word_in_context(text, keywords, window=3, lang='english')

    if len(contexts) > 0:
        print(f"Found {len(contexts)} occurrences:\n")
        for idx, row in contexts.iterrows():
            print(f"  Keyword: '{row['keyword']}'")
            print(f"  Context: ...{row['context']}...")
            print()
    else:
        print("No keywords found")

    print("✓ Keyword in context works!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 6. Text Concentration (HHI)
# ============================================================================

print("\n\n6. TEXT CONCENTRATION (Herfindahl-Hirschman Index)")
print("-" * 80)

try:
    print("\nCalculating word concentration for each text:")
    print("(Higher HHI = more concentrated/repetitive vocabulary)\n")

    for name, text in sample_texts.items():
        hhi = ct.word_hhi(text)
        print(f"  {name}: {hhi:.4f}")

    print("\n✓ HHI calculation works!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 7. Weighted Sentiment Analysis
# ============================================================================

print("\n\n7. WEIGHTED SENTIMENT ANALYSIS")
print("-" * 80)

try:
    print("\nUsing valence-based dictionary (with weights):")

    # Load concreteness dictionary (has numerical values)
    conc_dict = ct.read_yaml_dict('en_valence_Concreteness.yaml')
    print(f"Dictionary: {conc_dict['Name']}")

    text = "The innovative company achieved concrete results through strategic planning."

    # This gives weighted scores based on word values
    valence = ct.sentiment_by_valence(text, diction=conc_dict, lang='english')

    print(f"\nWeighted sentiment: {valence}")
    print("\n✓ Weighted sentiment analysis works!")

except Exception as e:
    print(f"Note: This requires a valence dictionary with numerical values")
    print(f"Error: {e}")

# ============================================================================
# 8. Economic Indices
# ============================================================================

print("\n\n8. ECONOMIC POLICY UNCERTAINTY INDICES")
print("-" * 80)

print("\n8.1 EPU (Economic Policy Uncertainty):")
print("Note: epu() is designed for Chinese news text analysis")
print("For English EPU, you would need English news corpus and EPU dictionary")
print("""
Example usage for Chinese:
    epu_df = ct.epu(
        df,
        text_column='text',
        date_column='date',
        ep_pattern='',  # Economic+Policy pattern
        u_pattern=''    # Uncertainty pattern
    )
""")

print("\n8.2 FEPU (Firm-level Economic Policy Uncertainty):")
print("Note: fepu() is designed for Chinese MD&A text from annual reports")
print("""
Example usage:
    mda_text = ct.extract_mda(annual_report_text)
    fepu_score = ct.fepu(
        mda_text,
        ep_pattern='',  # Economic+Policy pattern
        u_pattern=''    # Uncertainty pattern
    )
""")

# ============================================================================
# Summary
# ============================================================================

print("\n\n" + "=" * 80)
print("SUMMARY: Stats Module Functions")
print("=" * 80)

functions = [
    ("ct.word_count(text, lang, lemmatize)", "Word frequency analysis", "✓ Enhanced"),
    ("ct.sentiment(text, dict, lang)", "Dictionary-based sentiment", "✓"),
    ("ct.sentiment_by_valence(text, dict, lang)", "Weighted sentiment", "✓"),
    ("ct.readability(text, lang)", "Multiple readability formulas", "✓"),
    ("ct.cosine_sim(text1, text2, lang)", "Cosine similarity", "✓"),
    ("ct.jaccard_sim(text1, text2, lang)", "Jaccard similarity", "✓"),
    ("ct.minedit_sim(text1, text2, lang)", "Edit distance", "✓"),
    ("ct.word_in_context(text, keywords, window, lang)", "Extract keyword contexts", "✓"),
    ("ct.word_hhi(text)", "Text concentration index", "✓"),
    ("ct.semantic_brand_score(text, brands, lang)", "Brand importance", "Available"),
    ("ct.epu(df, ...)", "Economic Policy Uncertainty", "Chinese only"),
    ("ct.fepu(text, ...)", "Firm-level EPU", "Chinese only"),
]

print("\n{:<45s} {:<35s} {}".format("Function", "Purpose", "Status"))
print("-" * 80)
for func, purpose, status in functions:
    print("{:<45s} {:<35s} {}".format(func, purpose, status))

print("\n" + "=" * 80)
print("STATS MODULE DEMO COMPLETE")
print("=" * 80)

print("""
Key Takeaways:

1. Word Frequency:
   - word_count() enhanced with lemmatize parameter
   - Returns Counter or DataFrame
   - Better accuracy with lemmatization

2. Sentiment Analysis:
   - Multiple dictionaries available (LM, NRC, etc.)
   - Unweighted: sentiment()
   - Weighted: sentiment_by_valence()
   - Works with any YAML dictionary

3. Readability:
   - Multiple formulas: Flesch, Fog, SMOG, Coleman-Liau, ARI
   - All calibrated for English text
   - Higher scores = more difficult

4. Similarity Measures:
   - Cosine: Best for semantic similarity
   - Jaccard: Best for word overlap
   - MinEdit: Best for near-duplicate detection

5. Keyword Analysis:
   - word_in_context() extracts surrounding text
   - Configurable window size
   - Returns DataFrame for analysis

6. English Focus:
   - All core functions work with English
   - Some specialized functions (EPU, FEPU) are Chinese-specific
   - All enhanced with better tokenization

Next Steps:
- See demo_model.py for word embeddings
- See demo_mind.py for semantic analysis
- Explore other built-in English dictionaries
""")
