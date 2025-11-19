# MD&A Analysis Quick Reference
## Fast implementation guide for SEC filing analysis

---

## IMMEDIATE USE: Extract & Analyze MD&A

### 1. Load and Preprocess MD&A
```python
import cntext as ct
import pandas as pd

# Load from PDF 10-K
text = ct.read_pdf('apple_10k_2023.pdf')

# Clean up
text = ct.fix_contractions(text)  # "don't" → "do not"
text = ct.clean_text(text, lang='english')
```

### 2. One-Liner Sentiment Analysis
```python
# Load Loughran-McDonald dictionary (designed for financial texts)
lm = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')

# Analyze sentiment
result = ct.sentiment(text, diction=lm, lang='english')

print(f"Positive: {result['Positive_num']}")
print(f"Negative: {result['Negative_num']}")
print(f"Uncertainty: {result['Uncertainty_num']}")
print(f"Litigious: {result['Litigious_num']}")
```

### 3. Readability Analysis
```python
# Get all 6 readability metrics
readability = ct.readability(text, lang='english')

# Higher numbers = harder to read
# Typical 10-K: Fog Index 12-15, Flesch Grade 10-14
print(f"Flesch Kincaid Grade: {readability['flesch_kincaid_grade_level']}")
print(f"Fog Index: {readability['fog_index']}")
print(f"SMOG Index: {readability['smog_index']}")
```

---

## BATCH PROCESS: Large MD&A Dataset (600MB+)

### Memory-Efficient Approach
```python
import pandas as pd
import cntext as ct
from tqdm import tqdm
tqdm.pandas()  # Enable progress bar

# Load large CSV
df = pd.read_csv('mda_data.csv', low_memory=False)
print(f"Loaded {len(df)} MD&A sections")

# Preprocess in bulk
df['text'] = df['text'].apply(
    lambda x: ct.clean_text(ct.fix_contractions(str(x)), lang='english')
)

# Load dictionary once (expensive, do outside loop!)
lm = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')

# Apply sentiment in parallel batches
def get_sentiment(text):
    try:
        result = ct.sentiment(text, diction=lm, lang='english')
        return result
    except:
        return None

df_sentiment = df['text'].progress_apply(get_sentiment).apply(pd.Series)
df = pd.concat([df, df_sentiment], axis=1)

# Apply readability
df_readability = df['text'].progress_apply(
    lambda x: ct.readability(x, lang='english')
).apply(pd.Series)
df = pd.concat([df, df_readability], axis=1)

# Save results
df.to_csv('mda_analyzed.csv', index=False)
print(f"Analysis complete. Results saved.")
```

---

## SEMANTIC ANALYSIS: Measure Business Concepts

### 3-Step Process

**Step 1: Train Word Embeddings on Your MD&A Corpus**
```python
# First, create corpus file (one sentence per line)
# Or use existing corpus of MD&A texts

wv = ct.Word2Vec(
    corpus_file='all_mda_texts.txt',
    lang='english',
    vector_size=100,      # Word vector dimensions
    window_size=5,        # Context window
    min_count=5,          # Minimum frequency
    workers=4
)
```

**Step 2: Define Concept Axes**
```python
# Axis 1: Optimism vs. Pessimism
optimism_axis = ct.generate_concept_axis(
    wv,
    poswords=['growth', 'expansion', 'success', 'innovation', 'strong'],
    negwords=['decline', 'loss', 'risk', 'uncertainty', 'weak']
)

# Axis 2: Digital Transformation
digital_axis = ct.generate_concept_axis(
    wv,
    poswords=['digital', 'technology', 'AI', 'automation', 'innovation'],
    negwords=['traditional', 'manual', 'legacy', 'conventional']
)

# Axis 3: Risk Aversion
risk_axis = ct.generate_concept_axis(
    wv,
    poswords=['caution', 'conservative', 'careful', 'prudent'],
    negwords=['aggressive', 'bold', 'adventurous', 'risk-taking']
)
```

**Step 3: Score Each Document**
```python
# Project each MD&A onto the axes
df['optimism_score'] = df['text'].apply(
    lambda x: ct.project_text(wv, x, optimism_axis, lang='english')
)

df['digital_score'] = df['text'].apply(
    lambda x: ct.project_text(wv, x, digital_axis, lang='english')
)

df['risk_aversion'] = df['text'].apply(
    lambda x: ct.project_text(wv, x, risk_axis, lang='english')
)

# Positive = toward poswords, Negative = toward negwords
```

---

## ADVANCED: Track Trends Over Time

```python
# Group by year and analyze trends
trends = df.groupby('fiscal_year').agg({
    'Positive_num': 'mean',
    'Negative_num': 'mean',
    'Uncertainty_num': 'mean',
    'optimism_score': 'mean',
    'digital_score': 'mean',
    'flesch_kincaid_grade_level': 'mean'
}).round(2)

print("Trends over time:")
print(trends)

# Plot trends
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

trends[['Positive_num', 'Negative_num']].plot(ax=axes[0,0])
axes[0,0].set_title('Sentiment Trend')

trends['optimism_score'].plot(ax=axes[0,1])
axes[0,1].set_title('Optimism Score Trend')

trends['digital_score'].plot(ax=axes[1,0])
axes[1,0].set_title('Digital Transformation Trend')

trends['flesch_kincaid_grade_level'].plot(ax=axes[1,1])
axes[1,1].set_title('Readability Trend')

plt.tight_layout()
plt.savefig('mda_trends.png', dpi=100)
```

---

## MINIMAL EXAMPLE: 5-Line Analysis

```python
import cntext as ct

text = ct.read_pdf('10k.pdf')
text = ct.clean_text(text, lang='english')
lm = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')
sentiment = ct.sentiment(text, diction=lm, lang='english')
readability = ct.readability(text, lang='english')

print(f"Sentiment: Pos={sentiment['Positive_num']}, Neg={sentiment['Negative_num']}, Unc={sentiment['Uncertainty_num']}")
print(f"Readability: Flesch-Kincaid Grade {readability['flesch_kincaid_grade_level']}")
```

---

## DICTIONARIES FOR DIFFERENT ASPECTS

| Aspect | Dictionary | Key Metrics |
|--------|-----------|-------------|
| **Financial Tone** | Loughran-McDonald | Positive, Negative, Uncertainty, Litigious |
| **General Sentiment** | Lexicoder (LSD2015) | Positive, Negative |
| **Emotions** | NRC Emotion | Joy, Fear, Anger, Sadness, Anticipation, Trust, Disgust, Surprise |
| **Abstractness** | Concreteness | Word concreteness scores |
| **Complexity** | Built-in | Readability indices (Flesch, Fog, etc.) |

---

## COMMON MD&A METRICS TO TRACK

### Core Metrics
```python
# Sentiment strength
positive_rate = result['Positive_num'] / result['word_num'] * 100
negative_rate = result['Negative_num'] / result['word_num'] * 100
net_sentiment = (result['Positive_num'] - result['Negative_num']) / \
                (result['Positive_num'] + result['Negative_num'] + 1)

# Uncertainty perception
uncertainty_rate = result['Uncertainty_num'] / result['word_num'] * 100

# Litigation risk mentions
litigation_risk = result['Litigious_num']

# Readability/Disclosure
grade_level = readability['flesch_kincaid_grade_level']
fog_index = readability['fog_index']

# Semantic concepts (if embeddings trained)
optimism_score  # Higher = more optimistic
digital_score   # Higher = more digitalization-focused
```

### Aggregation Examples
```python
# By company
by_company = df.groupby('company_name')[
    ['Positive_num', 'Negative_num', 'optimism_score', 
     'flesch_kincaid_grade_level']
].mean()

# By industry
by_industry = df.groupby('industry')[
    ['Positive_num', 'Negative_num', 'digital_score']
].mean()

# By year
by_year = df.groupby('fiscal_year')[
    ['Positive_num', 'Negative_num', 'uncertainty_num', 'digital_score']
].mean()
```

---

## PERFORMANCE NOTES

### Processing Times (Typical)
- Sentiment analysis: ~50-100 documents/second
- Readability: ~100-200 documents/second
- Word embeddings: ~100KB corpus = few seconds
- Semantic projection: ~1000 documents/second (post-training)

### Memory Requirements
- Small dataset (< 1M words): <500 MB
- Medium dataset (100M words): ~2 GB
- Large dataset (> 1GB text): Use batch processing + iteration

### Optimization Tips
1. Load dictionaries ONCE, outside loops
2. Use list comprehension instead of apply() when possible
3. Process in batches (100-1000 documents at a time)
4. Use multiprocessing for independent documents
5. Cache embeddings after training

---

## COMMON ISSUES & FIXES

### Issue: "KeyError: word not in vocabulary"
**Fix**: Wrap in try-except or check if word exists first
```python
result = ct.project_text(wv, text, axis, lang='english')
if pd.isna(result):
    print("No valid words found in text")
```

### Issue: Memory error with large corpus
**Fix**: Use chunksize parameter in Word2Vec
```python
wv = ct.Word2Vec(corpus_file='huge.txt', chunksize=50000)
```

### Issue: Sentiment returns 0 across the board
**Fix**: Check if stopwords are being removed; they shouldn't contain sentiment words
```python
# Verify dictionary loaded correctly
print(ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')['Dictionary'].keys())
```

### Issue: Readability metrics give extreme values
**Fix**: Check for very short texts or unusual formatting
```python
# Filter out texts < 50 words
df = df[df['text'].str.split().str.len() >= 50]
```

---

## RECOMMENDED WORKFLOW FOR 600MB DATASET

```python
import pandas as pd
import cntext as ct
from pathlib import Path

# 1. LOAD
print("Loading...")
df = pd.read_csv('mda_600mb.csv', low_memory=False)

# 2. CLEAN
print("Cleaning...")
df['text_clean'] = df['mda_text'].apply(
    lambda x: ct.clean_text(ct.fix_contractions(str(x)), lang='english')
)

# 3. QUICK METRICS (fast)
print("Sentiment analysis...")
lm = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')
df_sentiment = df['text_clean'].apply(
    lambda x: ct.sentiment(x, diction=lm, lang='english')
).apply(pd.Series)
df = pd.concat([df, df_sentiment], axis=1)

print("Readability analysis...")
df_readability = df['text_clean'].apply(
    lambda x: ct.readability(x, lang='english')
).apply(pd.Series)
df = pd.concat([df, df_readability], axis=1)

# 4. SAVE INTERMEDIATE
print("Saving intermediate results...")
df.to_csv('mda_basic_metrics.csv', index=False)

# 5. SEMANTIC ANALYSIS (optional, slower)
print("Training embeddings (this may take a while)...")
# Export clean texts to file
with open('mda_corpus.txt', 'w') as f:
    f.write('\n'.join(df['text_clean']))

wv = ct.Word2Vec('mda_corpus.txt', lang='english', vector_size=100, min_count=5)

# 6. PROJECT ONTO CONCEPTS
print("Measuring semantic concepts...")
optimism = ct.generate_concept_axis(
    wv,
    poswords=['growth', 'opportunity', 'success'],
    negwords=['decline', 'loss', 'risk']
)

df['optimism_score'] = df['text_clean'].apply(
    lambda x: ct.project_text(wv, x, optimism, lang='english')
)

# 7. FINAL OUTPUT
print("Finalizing...")
df.to_csv('mda_complete_analysis.csv', index=False)
print("Done!")
```

---

**Ready to analyze MD&A files. Start with the Minimal Example above.**
