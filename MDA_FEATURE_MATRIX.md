# MD&A Feature Matrix
## Complete List of All Applicable Metrics for SEC Financial Documents

---

## SENTIMENT & TONE ANALYSIS

### Sentiment Analysis Functions

| Function | Library | Parameters | Output | MD&A Relevance | Financial Specific |
|----------|---------|-----------|--------|-----------------|-------------------|
| `sentiment()` | stats | text, dictionary, lang | Count by category | Positive, Negative, Uncertainty | YES |
| `sentiment_by_valence()` | stats | text, dictionary, lang | Numeric scores | Weighted sentiment scoring | YES |

### Available Dictionaries for MD&A

| Dictionary Name | Categories | Size | Best For | Financial Relevance |
|---|---|---|---|---|
| **Loughran-McDonald** | Positive, Negative, Uncertainty, Litigious, StrongModal, WeakModal, Constraining | ~2,000 words | SEC filings, 10-K/10-Q | EXCELLENT |
| **Lexicoder (LSD2015)** | Positive, Negative | ~4,500 words | General sentiment | GOOD |
| **NRC Emotion** | Anger, Fear, Joy, Sadness, Anticipation, Trust, Disgust, Surprise, Negative, Positive | ~14,000 words | Emotional tone | MODERATE |
| **SentiWS** | Valence scores (-1 to 1) | ~17,000 words | Fine-grained sentiment | MODERATE |
| **Concreteness** | Concreteness ratings (0-5) | ~40,000 words | Abstract vs. concrete language | MODERATE |
| **ANEW** | Pleasure, Arousal, Dominance | ~1,000 words | Dimensional emotions | LOW |

### Sentiment Metrics to Calculate

From Loughran-McDonald dictionary:

```
1. Positive Count (raw)
2. Positive Rate (% of document)
3. Negative Count (raw)
4. Negative Rate (% of document)
5. Uncertainty Count (raw)
6. Uncertainty Rate (% of document)
7. Litigious Count (raw)
8. StrongModal Count (modal words: must, will)
9. WeakModal Count (modal words: may, might)
10. Constraining Count (limiting words)

Derived Metrics:
11. Net Sentiment = (Pos - Neg) / (Pos + Neg + 1)
12. Sentiment Ratio = Pos / Neg (ratio, handle division by zero)
13. Risk Perception = (Uncertainty + Litigious) / Total Words
14. Modality Strength = StrongModal / (StrongModal + WeakModal)
```

---

## READABILITY & COMPLEXITY ANALYSIS

### Readability Metrics (Built-in)

| Metric | Function | Formula Basis | Interpretation | MD&A Use |
|--------|----------|---------------|-----------------|----------|
| **Flesch Reading Ease** | readability() | Syllables + word length | 0-100: 0=hard, 100=easy | Document accessibility |
| **Flesch-Kincaid Grade Level** | readability() | Syllables + sentence length | US grade level (0-18) | Required education level |
| **Gunning Fog Index** | readability() | Complex words + sentence length | Years of education needed | Disclosure clarity |
| **SMOG Index** | readability() | Complex words + sentence count | Grade level estimate | Readability assessment |
| **Coleman-Liau Index** | readability() | Characters + words/sentence | Grade level (US) | Alternative readability |
| **Automated Readability Index (ARI)** | readability() | Characters + words/sentence | Grade level (US) | Machine readability |
| **Rix Index** | readability() | Complex words/sentence | Simple readability measure | Complexity proxy |

### Expected Ranges for 10-K MD&A

```
Typical 10-K MD&A Readability Profile:
- Flesch Reading Ease: 20-40 (difficult to very difficult)
- Flesch-Kincaid Grade: 12-15 (college/post-college level)
- Gunning Fog: 12-15 (12-15 years of education)
- SMOG: 13-16 (college-level)
- Coleman-Liau: 10-14 (high school to college)
- ARI: 12-16 (college-level)
- RIX: 40-70 (readability score)
```

---

## WORD & VOCABULARY ANALYSIS

### Word Frequency & Vocabulary Metrics

| Function | Input | Output | MD&A Use | Calculations |
|----------|-------|--------|----------|---|
| `word_count()` | text | Counter dict | Identify key topics | Basic frequency |
| `word_count(..., lemmatize=True)` | text | Counter dict | Group related words | "companies"+"company" |
| `word_hhi()` | text | 0-1 float | Vocabulary concentration | Herfindahl-Hirschman Index |
| `word_in_context()` | text, keywords | DataFrame | Analyze keyword context | Word windows |

### Derived Vocabulary Metrics

```
From word_count():
1. Most frequent words (top 10, 20, 50)
2. Vocabulary size (unique words)
3. Type-token ratio = Unique words / Total words
4. Hapax rate = Words appearing once / Total unique
5. Repetition rate = (Total words - Unique) / Total words

From word_hhi():
6. HHI Index = Sum(word_freq^2) - measures concentration
   0.0 = maximum diversity (all words equal frequency)
   1.0 = minimum diversity (one word dominates)
```

---

## TEXT SIMILARITY & COMPARISON ANALYSIS

### Text Comparison Functions

| Function | Inputs | Output | Use Cases |
|----------|--------|--------|-----------|
| `cosine_sim()` | text1, text2 | 0-1 float | Compare MD&A across years or companies |
| `jaccard_sim()` | text1, text2 | 0-1 float | Set-based similarity |
| `minedit_sim()` | text1, text2 | float (edit distance) | Minimum edits needed |
| `simple_sim()` | text1, text2 | 0-1 float | Diff-based similarity |

### MD&A Comparison Examples

```
Use Cases:
1. Compare same company's MD&A year-over-year
   - Low similarity = strategic shift
   - High similarity = consistent strategy

2. Compare competitor MD&A sections
   - Identify similar risk disclosures
   - Find common business challenges

3. Industry benchmarking
   - Compare MD&A by industry sector
   - Identify unique disclosures

4. Regulatory change impact
   - Compare MD&A before/after regulation
```

---

## SEMANTIC & CONCEPT ANALYSIS

### Word Embedding Training

| Function | Output | Data Requirements | MD&A Use |
|----------|--------|-------------------|----------|
| `Word2Vec()` | KeyedVectors | Plain text corpus | Train custom embeddings on MD&A |
| `GloVe()` | KeyedVectors | Plain text corpus | Alternative to Word2Vec |
| `FastText()` | KeyedVectors | Plain text corpus | Handle rare words, typos |
| `expand_dictionary()` | Expanded dict | wv + seed dictionary | Expand sentiment dictionaries |

### Semantic Projection (Advanced)

| Function | Input | Output | MD&A Concept Examples |
|----------|-------|--------|----------------------|
| `generate_concept_axis()` | pos/neg words, embeddings | Vector axis | Define business concepts |
| `project_text()` | text, axis, embeddings | Float | Measure concept strength |
| `project_word()` | word, concept, embeddings | Float | Single word projection |
| `sematic_projection()` | words, pos/neg, embeddings | List of tuples | Project word list |
| `sematic_distance()` | words1, words2, embeddings | Float | Distance between concepts |
| `semantic_centroid()` | words, embeddings | Vector | Center of concept |
| `wepa()` | text, pos/neg, embeddings | Float | Optimized projection |

### Example Business Concept Axes for MD&A

```
Axis 1: Optimism vs. Pessimism
  POS: growth, expansion, success, innovation, strong, opportunity
  NEG: decline, loss, risk, uncertainty, weak, challenge

Axis 2: Digital Transformation
  POS: digital, technology, AI, automation, innovation, cloud
  NEG: traditional, manual, legacy, conventional, analog

Axis 3: Risk Aversion
  POS: caution, conservative, careful, prudent, safeguard
  NEG: aggressive, bold, adventurous, risk-taking, exposure

Axis 4: Regulatory Compliance
  POS: compliance, regulation, policy, governance, adherence
  NEG: violation, breach, non-compliance, penalty, litigation

Axis 5: Sustainability Focus
  POS: environment, sustainable, green, ESG, climate
  NEG: pollution, waste, carbon, environmental-risk

Axis 6: Internationalization
  POS: global, international, export, overseas, emerging
  NEG: domestic, local, regional, US-centric

Axis 7: Supply Chain Complexity
  POS: supply-chain, logistics, sourcing, procurement
  NEG: integration, vertical, proprietary, in-house

Axis 8: Customer-Centric
  POS: customer, experience, satisfaction, retention, loyalty
  NEG: churn, attrition, dissatisfaction, complaint

Axis 9: Innovation Culture
  POS: innovation, R&D, research, development, patent
  NEG: commodity, mature, stagnant, incremental
```

### Semantic Diversity Metrics

| Function | Input | Output | MD&A Use |
|----------|-------|--------|----------|
| `divergent_association_task()` | words | Float (0-100) | Creativity/diversity of concepts |
| `discursive_diversity_score()` | words | Float (0-1) | Language/cognitive diversity |

---

## SPECIALIZED FINANCIAL METRICS

### Brand/Concept Importance

| Function | Input | Output | MD&A Use |
|----------|-------|--------|----------|
| `semantic_brand_score()` | text, brands/topics, embeddings | DataFrame with SBS scores | Measure importance of topics |

Components:
- PREVALENCE: How often mentioned (vs. average)
- DIVERSITY: How diverse the connections
- CONNECTIVITY: How central in semantic network
- **SBS** = PREVALENCE + DIVERSITY + CONNECTIVITY

### Policy Uncertainty (Chinese-focused, but adaptable)

| Function | Input | Output | MD&A Adaptation |
|----------|-------|--------|-----------------|
| `fepu()` | text, patterns | Float | Firm-level perceived uncertainty |
| `epu()` | DataFrame with dates | Series | Time-series uncertainty index |

Adaptable for MD&A with custom patterns:
```python
# Regulatory uncertainty
reg_pattern = 'regulation|regulatory|compliance|legal'
unc_pattern = 'uncertain|risk|complexity|challenge'

# Supply chain uncertainty
supply_pattern = 'supply|chain|sourcing|procurement'
unc_pattern = 'disruption|risk|constraint|limitation'
```

---

## COMPLETE ANALYSIS CHECKLIST FOR MD&A

### Phase 1: Basic Metrics (Fast - ~5 minutes for 100 documents)

- [ ] Word count & length
- [ ] Sentiment (Loughran-McDonald)
  - [ ] Positive count & rate
  - [ ] Negative count & rate
  - [ ] Uncertainty count & rate
  - [ ] Litigious count
  - [ ] Net sentiment
  - [ ] Risk perception rate
- [ ] Readability (all 6 indices)
  - [ ] Flesch-Kincaid Grade
  - [ ] Fog Index
  - [ ] SMOG Index
  - [ ] Coleman-Liau Index
  - [ ] ARI Index
  - [ ] RIX Index
- [ ] Word frequency (top 20 words)

### Phase 2: Vocabulary Analysis (Medium - ~15 minutes)

- [ ] Vocabulary size & type-token ratio
- [ ] HHI vocabulary concentration
- [ ] Hapax rate (words appearing once)
- [ ] Repetition analysis
- [ ] Stopword analysis

### Phase 3: Advanced Semantics (Slow - hours to days, one-time)

- [ ] Train Word2Vec embeddings on corpus
- [ ] Define concept axes (8-10 business concepts)
- [ ] Project all documents onto axes
- [ ] Measure semantic distances
- [ ] Analyze divergent thinking (DAT) scores
- [ ] Calculate discursive diversity

### Phase 4: Comparison & Trends (Medium - ~30 minutes)

- [ ] Year-over-year similarity
- [ ] Competitor comparisons
- [ ] Concept score trends over time
- [ ] Industry benchmarking
- [ ] Regulatory impact analysis

---

## IMPLEMENTATION ROADMAP

### Week 1: Foundation
```
Day 1: Load & preprocess basic MD&A sample
Day 2: Calculate sentiment with Loughran-McDonald
Day 3: Calculate readability metrics
Day 4: Word frequency analysis
Day 5: Compare documents
```

### Week 2: Advanced
```
Day 1: Train Word2Vec on full corpus
Day 2: Define business concept axes
Day 3: Project documents onto axes
Day 4: Trend analysis over years
Day 5: Final validation & documentation
```

---

## OUTPUT METRICS SUMMARY

### For Each Document:

```python
{
    # Identifiers
    'company_id': str,
    'company_name': str,
    'filing_date': date,
    'fiscal_year': int,
    
    # Text Properties
    'word_count': int,
    'sentence_count': int,
    'unique_words': int,
    'type_token_ratio': float,
    
    # Sentiment Metrics
    'positive_count': int,
    'positive_rate': float,
    'negative_count': int,
    'negative_rate': float,
    'uncertainty_count': int,
    'uncertainty_rate': float,
    'litigious_count': int,
    'strong_modal_count': int,
    'weak_modal_count': int,
    'net_sentiment': float,
    'risk_perception': float,
    
    # Readability Metrics
    'flesch_reading_ease': float,
    'flesch_kincaid_grade': float,
    'gunning_fog_index': float,
    'smog_index': float,
    'coleman_liau_index': float,
    'ari_index': float,
    'rix_index': float,
    'readability_avg': float,
    
    # Vocabulary Metrics
    'vocabulary_size': int,
    'hhi_index': float,
    'repetition_rate': float,
    'hapax_rate': float,
    
    # Concept Scores (if embeddings trained)
    'optimism_score': float,
    'digital_score': float,
    'risk_aversion': float,
    'regulatory_focus': float,
    'sustainability_focus': float,
    'internationalization': float,
    
    # Similarity Scores
    'prev_year_similarity': float,
    'competitor_avg_similarity': float,
    
    # Diversity Scores
    'dat_score': float,
    'discursive_diversity': float,
}
```

---

## PERFORMANCE & SCALABILITY

### Time Complexity

```
Single Document:
- Sentiment analysis: O(n) where n = word count (~100ms for 5000 words)
- Readability: O(n) (~50ms)
- Word frequency: O(n) (~100ms)
- Word embeddings (training): O(corpus_size * window * embed_dim)
  ~1-5 hours for 1GB corpus
- Projection onto axis: O(n * embed_dim) (~1ms)

Batch Processing:
- 100 documents: ~30 seconds total (sentiment + readability)
- 1,000 documents: ~3 minutes
- 10,000 documents: ~30 minutes
- 100,000+ documents: use parallel processing
```

### Memory Requirements

```
Base library: ~500MB
Loaded embeddings (100M corpus, 100-dim): ~1GB
Dictionary in memory: <100MB
Processing 100 documents: ~200MB
Processing 1,000 documents: ~2GB (use batches)
Processing 10,000 documents: use chunked processing
```

---

## VALIDATION & QUALITY CHECKS

### Before Analysis:
```python
# Check data quality
- Null/missing text values
- Text length distribution
- Encoding issues (use fix_text())
- Duplicate documents
- Language detection (should be English)
```

### After Analysis:
```python
# Validate outputs
- Sentiment totals make sense
- Readability scores in expected ranges
- No NaN/Inf values
- Word counts match expectations
- Correlation between metrics is reasonable
```

### Example Ranges for Validation:

```python
# Expected ranges
word_count: 10,000 - 50,000 (typical MD&A)
positive_rate: 1-5% of words
negative_rate: 1-4% of words
uncertainty_rate: 1-3% of words
flesch_kincaid_grade: 10-16
fog_index: 10-15
readability_avg: similar to above ranges

# Red flags
positive_rate < 0.5% or > 10%
negative_rate < 0.5% or > 8%
fog_index < 5 or > 20 (unless extreme text)
word_count < 1000 (usually incomplete)
```

---

**This matrix provides the complete toolkit for MD&A analysis.**

