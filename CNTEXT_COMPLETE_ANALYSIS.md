# CNTEXT Repository Analysis Summary
## Complete Feature Set and Capabilities

---

## OVERVIEW

**Project Name**: entext (fork of cntext)
**Current Version**: 2.2.0
**Primary Language Support**: English & Chinese (Bilingual)
**Main Use Case**: Text analysis for social science research, especially financial/corporate documents
**Repository Status**: Active development with recent enhancements for English text analysis

---

## MAIN MODULES AND PURPOSES

### 1. **IO Module** - Data Import/Export and Preprocessing
**Location**: `/cntext/io/`

#### Core Functions:
- **File Reading**:
  - `read_pdf(file)` - Extract text from PDF files
  - `read_docx(file)` - Extract text from DOCX files
  - `read_file(file, encoding)` - Read plain text files
  - `read_files(fformat, encoding)` - Batch read multiple files into DataFrame

- **Dictionary Management**:
  - `read_yaml_dict(yfile)` - Load built-in sentiment/semantic dictionaries
  - `build_yaml_dict(data, yfile)` - Create custom YAML dictionaries
  - `get_dict_list()` - List all available built-in dictionaries

- **Text Preprocessing**:
  - `fix_text(text)` - Fix encoding issues and normalize text
  - `fix_contractions(text)` - Expand English contractions ("don't" → "do not")
  - `clean_text(text, lang='english')` - Comprehensive text cleaning
  - `extract_mda(file)` - Extract MD&A sections from financial documents

---

### 2. **STATS Module** - Statistical Text Analysis
**Location**: `/cntext/stats/`

#### Available Metrics:

**A. Sentiment Analysis**:
- `sentiment(text, diction, lang='english')` 
  - Unweighted sentiment counting by category
  - Returns counts for each dictionary category
  
- `sentiment_by_valence(text, diction, lang='english')`
  - Weighted sentiment analysis (with valence scores)
  - Returns numeric scores per category

**B. Readability Metrics** (`readability(text, lang='english')`):
- **Flesch Reading Ease Index** - Measures how easy text is to understand (0-100 scale)
- **Flesch-Kincaid Grade Level** - US grade level needed to understand the text
- **Gunning Fog Index** - Years of education needed to understand
- **SMOG Index** - Simplified Measure of Gobbledygook (reading difficulty)
- **Coleman-Liau Index** - Based on characters per word and words per sentence
- **Automated Readability Index (ARI)** - Based on characters and sentence length
- **Rix Index** - Based on word length and sentence frequency

**C. Word Analysis**:
- `word_count(text, lang='english', lemmatize=False)`
  - Returns Counter object with word frequencies
  - Optional lemmatization (companies→company, running→run)
  - Auto-removes stopwords
  
- `word_in_context(text, keywords, window=3, lang='english')`
  - Finds keywords with their context windows
  - Returns DataFrame with keyword and surrounding words

- `word_hhi(text, lang='english')`
  - Herfindahl-Hirschman Index for vocabulary concentration
  - Measures lexical diversity/richness
  - Higher = more repetitive; Lower = more diverse vocabulary

**D. Text Similarity**:
- `cosine_sim(text1, text2, lang='english')` - Vector space cosine similarity
- `jaccard_sim(text1, text2, lang='english')` - Set-based Jaccard similarity
- `minedit_sim(text1, text2, lang='english')` - Minimum edit distance similarity
- `simple_sim(text1, text2, lang='english')` - Diff-based similarity

**E. Specialized Metrics** (Chinese focus, but English support via custom dictionaries):
- `fepu(text)` - Firm-level perceived Economic Policy Uncertainty
- `epu(df, freq='Y')` - Economic Policy Uncertainty index (time-series)
- `semantic_brand_score(text, brands)` - Brand importance via semantic network analysis

---

### 3. **MODEL Module** - Word Embedding Training
**Location**: `/cntext/model/`

#### Word Embedding Models:

**A. Word2Vec Training**:
```python
wv = ct.Word2Vec(
    corpus_file='path/to/corpus.txt',
    lang='english',
    vector_size=100,      # Dimensions of word vectors
    window_size=6,        # Context window
    min_count=5,          # Minimum word frequency
    max_iter=5,           # Training iterations
    chunksize=10000       # Memory management
)
```
- Automatic preprocessing (tokenization, stopword removal)
- Memory-efficient processing for large corpora
- Returns trained KeyedVectors model

**B. GloVe Training**:
```python
wv = ct.GloVe(corpus_file='path/to/corpus.txt', lang='english', ...)
```
- Global Vectors for Word Representation
- Uses Stanford NLP implementation
- Similar interface to Word2Vec

**C. FastText**:
```python
wv = ct.FastText(corpus_file='path/to/corpus.txt', lang='english', ...)
```
- Handles subword information
- Better for rare words and misspellings

**D. Co-occurrence Methods**:
- `SoPmi(corpus_file, seed_file, lang='english')` - Semantic word expansion

**E. Dictionary Expansion**:
- `expand_dictionary(wv, seeddict, topn=100)` - Expand sentiment dictionaries using embeddings
- `co_occurrence_matrix(corpus_file)` - Build co-occurrence matrices

**F. Model Evaluation**:
- `evaluate_similarity(wv, file=None)` - Test word similarity understanding
- `evaluate_analogy(wv, file=None)` - Test word analogy reasoning
- `load_w2v(wv_path)` - Load pre-trained models

---

### 4. **MIND Module** - Semantic and Cognitive Analysis
**Location**: `/cntext/mind.py`

#### Advanced Semantic Projection:

**A. Concept Axis Generation**:
```python
axis = ct.generate_concept_axis(
    wv, 
    poswords=['innovation', 'creative', 'novel'],
    negwords=['traditional', 'conservative']
)
```
- Creates directional semantic axes
- Measures abstract concepts in text (innovation, risk, growth, etc.)

**B. Text Projection**:
```python
score = ct.project_text(wv, text, axis, lang='english')
# Returns float: positive=toward poswords, negative=toward negwords
```
- Measures where a document falls on a concept spectrum
- Core feature for measuring organizational culture, attitudes, bias

**C. Word-Level Projection**:
- `project_word(wv, a, b)` - Project single word onto another
- `sematic_projection(wv, words, poswords, negwords)` - Project word list onto axis
- `sematic_distance(wv, words1, words2)` - Semantic distance between word sets

**D. Semantic Centroid**:
```python
centroid = ct.semantic_centroid(wv, words)
```
- Computes semantic center of multiple words
- Useful for measuring concept centrality

**E. Creativity & Cognitive Diversity**:
```python
dat_score = ct.divergent_association_task(wv, words, minimum=7)
# Measures semantic diversity (creativity proxy)

diversity = ct.discursive_diversity_score(wv, words)
# Measures cognitive diversity in language use
```

**F. Temporal Semantic Change**:
```python
wv_2020_aligned = ct.procrustes_align(wv_2000, wv_2020)
# Aligns embedding spaces to track meaning shift over time
```

**G. Optimization Function**:
```python
score = ct.wepa(wv, text, poswords, negwords, lang='english')
# Fast projection with internal caching
```

---

### 5. **LLM Module** - Large Language Model Integration
**Location**: `/cntext/llm.py`

#### LLM-Powered Text Analysis:

```python
result = ct.llm(
    text='Company reported strong earnings...',
    task='sentiment',              # or custom task
    backend='ollama',              # or 'openai', custom
    model_name='qwen2.5:3b',      # Local or remote model
    base_url='http://localhost:11434',  # Custom endpoint
    api_key='...',
    temperature=0,
    rate_limit=100                 # Requests per minute
)
```

**Supported Backends**:
- Local: Ollama (port 11434), LM Studio (port 1234)
- Remote: OpenAI, Aliyun, Baidu Qianfan, custom APIs

**Task Types**:
- `sentiment` - Financial sentiment analysis
- `classification` - Document classification
- Custom prompts with structured JSON output

**Features**:
- Async batch processing with rate limiting
- Structured output with Pydantic models
- Retry logic with configurable attempts

---

### 6. **ENGLISH_NLP Module** - Enhanced English Processing
**Location**: `/cntext/english_nlp.py`

#### English-Specific Utilities:

**A. Tokenization**:
```python
tokens = ct.tokenize_english(text, lemmatize=False, remove_punct=True)
```
- spaCy if available (better quality)
- Fallback to NLTK
- Automatic punctuation handling

**B. Preprocessing Pipeline**:
```python
tokens = ct.preprocess_english(
    text,
    lemmatize=True,           # companies→company
    remove_punct=True,
    remove_numbers=False,      # or replace with '_num_'
    min_length=1,
    stopwords=my_stopwords
)
```

**C. Backend Detection**:
```python
info = ct.english_nlp.get_backend_info()
# {'backend': 'spacy'/'nltk', 'spacy_available': bool, ...}
```

---

### 7. **PLOT Module** - Visualization
**Location**: `/cntext/plot.py`

#### Visualization Functions:
- `lexical_dispersion_plot1(text, keywords)` - Keyword distribution across text
- `lexical_dispersion_plot2(text, keywords)` - Alternative visualization
- `matplotlib_chinese()` - Font support for Chinese visualization

---

## AVAILABLE SENTIMENT DICTIONARIES (Built-in)

### English Dictionaries:

| Dictionary | File | Categories | Size | Use Case |
|-----------|------|------------|------|----------|
| **Loughran-McDonald** | `en_common_LoughranMcDonald.yaml` | Negative, Positive, Uncertainty, Litigious, StrongModal, WeakModal, Constraining | ~2,000 words | Financial documents, SEC filings, MD&A |
| **NRC Emotion** | `en_common_NRC.yaml` | Anger, Anticipation, Disgust, Fear, Joy, Negative, Positive, Sadness, Surprise, Trust | ~14,000 words | Emotional tone analysis |
| **Lexicoder (LSD2015)** | `en_common_LSD2015.yaml` | Positive, Negative | ~4,500 words | General sentiment |
| **SentiWS** | `en_common_SentiWS.yaml` | Multiple valence scores | ~17,000 words | Fine-grained sentiment |
| **Concreteness** | `en_valence_Concreteness.yaml` | Concreteness ratings (0-5) | ~40,000 words | Abstractness measurement |
| **ANEW** | `en_common_ANEW.yaml` | Pleasure, Arousal, Dominance | ~1,000 words | Dimensional emotion |

### Multilingual:
- `enzh_common_StopWords.yaml` - English & Chinese stopwords
- `enzh_common_AdvConj.yaml` - Adverbs and conjunctions (both languages)

---

## LANGUAGE SUPPORT

### **English** ✓ (Fully Supported)
- All sentiment analysis functions with lang='english'
- All readability metrics calibrated for English
- spaCy integration for advanced NLP
- English-specific dictionaries (Loughran-McDonald, NRC, etc.)
- MD&A extraction and analysis

### **Chinese** ✓ (Fully Supported)  
- All sentiment analysis functions with lang='chinese'
- Additional Chinese dictionaries (finance, policy, digitalization)
- Jieba tokenization with Chinese-specific preprocessing
- Chinese-specific metrics (FEPU, EPU)

### **Default**: English (in current fork)

---

## KEY ENTRY POINTS FOR MD&A ANALYSIS

### **Quick Start Pattern**:
```python
import cntext as ct

# 1. Load MD&A text
text = ct.read_pdf('10-K_filing.pdf')
text = ct.clean_text(text, lang='english')
text = ct.fix_contractions(text)

# 2. Sentiment analysis (Financial)
lm_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')
sentiment = ct.sentiment(text, diction=lm_dict, lang='english')
print(sentiment)
# {'Positive_num': 45, 'Negative_num': 23, 'Uncertainty_num': 12, ...}

# 3. Readability
readability = ct.readability(text, lang='english')
print(readability)
# {'flesch_kincaid_grade_level': 14.2, 'fog_index': 12.1, ...}

# 4. Word frequency
freq = ct.word_count(text, lang='english', lemmatize=True)
print(freq.most_common(10))

# 5. Text similarity (compare with competitor)
competitor_text = ct.read_pdf('competitor_10-K.pdf')
similarity = ct.cosine_sim(text, competitor_text, lang='english')
```

### **Batch Processing Pattern** (600MB+ datasets):
```python
import pandas as pd

# Load large CSV efficiently
df = pd.read_csv('mda_data.csv', encoding='utf-8')

# Preprocess
df['text_clean'] = df['mda_text'].apply(
    lambda x: ct.clean_text(ct.fix_contractions(str(x)), lang='english')
)

# Calculate metrics
lm_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')

df['sentiment'] = df['text_clean'].apply(
    lambda x: ct.sentiment(x, diction=lm_dict, lang='english')
)

df['readability'] = df['text_clean'].apply(
    lambda x: ct.readability(x, lang='english')
)

# Export
df.to_csv('mda_analysis_results.csv')
```

### **Semantic Projection Pattern** (Measuring concepts):
```python
# Train embeddings on MD&A corpus
wv = ct.Word2Vec(
    corpus_file='all_mda_texts.txt',
    lang='english',
    vector_size=100,
    window_size=6,
    min_count=5
)

# Create concept axes
# Axis 1: Optimism vs. Pessimism
optimism_axis = ct.generate_concept_axis(
    wv,
    poswords=['growth', 'opportunity', 'strong', 'profitable', 'innovative'],
    negwords=['decline', 'risk', 'weak', 'loss', 'uncertain']
)

# Axis 2: Digitalization vs. Traditional
digital_axis = ct.generate_concept_axis(
    wv,
    poswords=['digital', 'automation', 'AI', 'technology', 'innovation'],
    negwords=['traditional', 'manual', 'legacy', 'conventional']
)

# Measure for each document
df['optimism_score'] = df['text_clean'].apply(
    lambda x: ct.project_text(wv, x, optimism_axis, lang='english')
)

df['digital_score'] = df['text_clean'].apply(
    lambda x: ct.project_text(wv, x, digital_axis, lang='english')
)
```

---

## AVAILABLE METRICS SUMMARY TABLE

| Metric Category | Function | Input | Output | Use Case |
|---|---|---|---|---|
| **Sentiment** | `sentiment()` | Text + dictionary | Category counts | Financial tone analysis |
| **Sentiment (Weighted)** | `sentiment_by_valence()` | Text + valence dict | Numeric scores | Refined sentiment scoring |
| **Readability** | `readability()` | Text | 6 indices | Document complexity |
| **Word Frequency** | `word_count()` | Text | Counter dict | Key topic identification |
| **Word Context** | `word_in_context()` | Text + keywords | DataFrame | Contextual analysis |
| **Vocabulary Diversity** | `word_hhi()` | Text | Float [0-1] | Language richness |
| **Text Similarity** | `cosine_sim()` | Text1, Text2 | Float [0-1] | Document comparison |
| **Projection Score** | `project_text()` | Text + axis | Float | Abstract concept measurement |
| **Semantic Distance** | `sematic_distance()` | Words1, Words2 | Float | Concept relatedness |
| **Creativity Score** | `divergent_association_task()` | Words | Float | Semantic divergence |
| **Diversity Score** | `discursive_diversity_score()` | Words | Float | Language diversity |
| **Semantic Brand** | `semantic_brand_score()` | Text + brands | DataFrame | Brand importance network |
| **Policy Uncertainty** | `fepu()` or `epu()` | Text/DataFrame | Float/Series | Policy sentiment (Ch) |

---

## EXAMPLE USAGE PATTERNS

### Pattern 1: Simple Sentiment Analysis
```python
import cntext as ct

text = "Strong revenue growth but uncertainty regarding regulatory changes."
lm_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')
sentiment = ct.sentiment(text, diction=lm_dict, lang='english')

# Output: {'Positive_num': 1, 'Negative_num': 0, 'Uncertainty_num': 1, ...}
```

### Pattern 2: Readability Assessment
```python
readability = ct.readability(text, lang='english')

# Output: {
#   'fog_index': 14.2,
#   'flesch_kincaid_grade_level': 13.5,
#   'smog_index': 14.8,
#   'coleman_liau_index': 12.3,
#   'ari': 15.1,
#   'rix': 45.2
# }
```

### Pattern 3: Train Custom Embeddings
```python
# Train Word2Vec on your corpus
wv = ct.Word2Vec(
    corpus_file='corporate_mda_corpus.txt',
    lang='english',
    vector_size=100,
    window_size=5,
    min_count=3
)

# Find similar words
similar = wv.most_similar('innovation', topn=5)
# [('innovative', 0.87), ('revolutionize', 0.84), ...]
```

### Pattern 4: Measure Organizational Attitudes
```python
# Define concept axes
risk_axis = ct.generate_concept_axis(
    wv,
    poswords=['risk', 'danger', 'threat', 'challenge'],
    negwords=['opportunity', 'safe', 'secure', 'strength']
)

# Measure risk perception in company statements
scores = []
for company_mda in mda_texts:
    risk_score = ct.project_text(wv, company_mda, risk_axis, lang='english')
    scores.append(risk_score)
    # Positive scores = risk-averse; Negative = risk-seeking
```

### Pattern 5: Batch Processing Large Dataset
```python
import pandas as pd
from tqdm import tqdm

# Load 600MB MD&A dataset
df = pd.read_csv('large_mda_data.csv')

# Load dictionary once (not in loop)
lm_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')

# Process in batches
results = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    text = ct.clean_text(row['mda_text'], lang='english')
    
    sentiment = ct.sentiment(text, diction=lm_dict, lang='english')
    readability = ct.readability(text, lang='english')
    word_freq = ct.word_count(text, lang='english')
    
    results.append({
        'idx': idx,
        **sentiment,
        **readability,
        'top_words': word_freq.most_common(5)
    })

results_df = pd.DataFrame(results)
```

---

## KEY FILES AND LOCATIONS

```
/home/user/cntext/
├── cntext/
│   ├── __init__.py                 # Main API exports
│   ├── stats/                      # Statistical analysis
│   │   ├── sentiment.py            # Sentiment functions
│   │   ├── readability.py          # Readability metrics
│   │   ├── similarity.py           # Text similarity
│   │   ├── index.py                # Word frequency, HHI, SBS
│   │   └── utils.py                # Helper functions
│   ├── model/                      # Word embedding training
│   │   ├── w2v.py                  # Word2Vec wrapper
│   │   ├── glove.py                # GloVe implementation
│   │   ├── fasttext.py             # FastText wrapper
│   │   └── utils.py                # Model utilities
│   ├── mind.py                     # Semantic projection & concepts
│   ├── io/
│   │   ├── file.py                 # File reading (PDF, DOCX, TXT)
│   │   ├── dict.py                 # Dictionary management
│   │   ├── mda.py                  # MD&A extraction
│   │   ├── data/                   # Built-in dictionaries
│   │   │   ├── en_common_*.yaml   # English dictionaries
│   │   │   └── zh_common_*.yaml   # Chinese dictionaries
│   │   └── utils.py                # Text preprocessing
│   ├── english_nlp.py              # Enhanced English tokenization
│   ├── llm.py                      # LLM integration
│   └── plot.py                     # Visualization
├── analyze_mda_template.py         # Template for MD&A analysis
├── guide_mda_analysis.py           # Detailed workflow guide
└── demo_simple.py                  # Simple demonstrations
```

---

## SUMMARY TABLE: METRICS FOR MD&A ANALYSIS

### Financial/Corporate Document Analysis

| Aspect | Best Metric | Dictionary | Interpretation |
|--------|-------------|------------|-----------------|
| **Tone** | `sentiment(text, lm_dict)` | Loughran-McDonald | Positive/Negative counts |
| **Tone (Weighted)** | `sentiment_by_valence()` | LM with valence | Numeric sentiment score |
| **Uncertainty** | Extract "Uncertainty" from LM | Loughran-McDonald | Risk/uncertainty perception |
| **Risk Perception** | `project_text(wv, text, risk_axis)` | Custom embeddings | Risk-averse to risk-seeking |
| **Complexity** | `readability(text)` | Built-in formulas | Fog/Flesch/SMOG indices |
| **Key Topics** | `word_count(text, lemmatize=True)` | Built-in | Most frequent meaningful words |
| **Similarity** | `cosine_sim(text1, text2)` | Word vectors | Document similarity [0-1] |
| **Concept Strength** | `project_text(text, concept_axis)` | Custom embeddings | Strength of concept in text |
| **Vocabulary Richness** | `word_hhi(text)` | Built-in formula | Vocabulary concentration |
| **Innovation Tone** | Custom axis projection | Custom embeddings | Innovation vs. traditional |
| **Growth Sentiment** | Custom axis projection | Custom embeddings | Optimistic vs. pessimistic |
| **Emotional Tone** | `sentiment(text, nrc_dict)` | NRC Emotion | Multiple emotions |

---

## NEXT STEPS FOR YOUR USE CASE

### Recommended approach for MD&A analysis:

1. **Load your MD&A texts** using `read_pdf()`, `read_files()`, or `read_csv()`

2. **Preprocess** with `clean_text()` and `fix_contractions()`

3. **Calculate core metrics**:
   - Sentiment using Loughran-McDonald dictionary
   - Readability using all 6 metrics
   - Word frequency with lemmatization
   
4. **For advanced analysis**:
   - Train Word2Vec embeddings on your corpus
   - Create custom concept axes (risk, innovation, growth, etc.)
   - Project texts onto these axes
   
5. **Batch process** large datasets with pandas groupby/apply patterns

6. **Visualize** trends over time (by company, year, sector)

---

**This fork is production-ready for English MD&A analysis with comprehensive metrics across sentiment, readability, semantic analysis, and custom concept measurement.**

