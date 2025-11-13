# English Text Analysis Enhancements

This document summarizes all enhancements made to adapt the cntext library for English text analysis.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Workflow Demonstration](#workflow-demonstration)
3. [Key Improvements](#key-improvements)
4. [Files Modified](#files-modified)
5. [Installation](#installation)
6. [Usage Examples](#usage-examples)
7. [Before & After Comparison](#before--after-comparison)

---

## Overview

This fork adapts the excellent **cntext** library (originally designed for Chinese text analysis) to provide robust English text analysis capabilities while maintaining all original Chinese functionality.

**Original Project:** https://github.com/hiDaDeng/cntext
**Original Authors:** Xudong Deng (DaDeng), Peng Nan - Harbin Institute of Technology

### What Makes This Fork Special

- ✅ Enhanced English tokenization (not just `text.split()`)
- ✅ Optional lemmatization support (running → run, companies → company)
- ✅ spaCy integration with NLTK fallback
- ✅ Bug fixes in English language handling
- ✅ All Chinese functionality preserved
- ✅ Backward compatible - all changes are opt-in

---

## Workflow Demonstration

### Run the Demo

```bash
cd /home/user/cntext
python demo_simple.py
```

### What the Demo Shows

The demonstration script showcases:

1. **Enhanced Tokenization** - Comparison of old vs. new approach
2. **Sentiment Analysis** - Using Loughran-McDonald financial dictionary
3. **Word Frequency** - Impact of lemmatization on accuracy
4. **Text Similarity** - Comparing documents
5. **Keyword in Context** - Finding keywords with surrounding text
6. **Code Improvements** - Summary of all changes
7. **Usage Examples** - How to use each feature

**Sample Output:**

```
TechInnovate Corp (Positive):
  Positive words found: 7
  Negative words found: 0
  Net sentiment: 1.00 (POSITIVE)

TraditionalManufacture Inc (Negative):
  Positive words found: 1
  Negative words found: 8
  Net sentiment: -0.78 (NEGATIVE)
```

---

## Key Improvements

### 1. New Module: `cntext/english_nlp.py`

A comprehensive English NLP utility module with:

- **Smart backend detection**: Uses spaCy if available, falls back to NLTK
- **Enhanced tokenization**: Superior to `text.split()`
- **Lemmatization support**: Converts words to base forms
- **Preprocessing pipeline**: Handles punctuation, numbers, stopwords
- **Graceful degradation**: Works without spaCy

**Key Functions:**

```python
from cntext.english_nlp import tokenize_english, preprocess_english, get_backend_info

# Check what's available
info = get_backend_info()
# Returns: {'backend': 'spacy' or 'nltk', 'recommendation': '...'}

# Tokenize
tokens = tokenize_english(text, lemmatize=True)
# Returns: ['company', 'run', 'innovative', 'program']

# Full preprocessing
tokens = preprocess_english(text, lemmatize=True, remove_numbers=True)
```

### 2. Bug Fix: `cntext/model/sopmi.py`

**Problem:** Always used jieba for all languages, breaking English processing

**Solution:** Now properly detects `lang='english'` and uses appropriate tokenization

**Impact:** SoPmi dictionary expansion now works correctly for English

### 3. Enhanced: `cntext/stats/utils.py` - `word_count()`

**New Parameter:** `lemmatize=False`

**Before:**
```python
word_count("companies are running", lang='english')
# Result: {'companies': 1, 'are': 1, 'running': 1}
```

**After:**
```python
word_count("companies are running", lang='english', lemmatize=True)
# Result: {'company': 1, 'be': 1, 'run': 1}
```

### 4. Enhanced: `cntext/model/utils.py` - `preprocess_line()`

**New Parameter:** `lemmatize=False`

**Improvements:**
- Better number handling (supports decimals: `25.5` → `_num_`)
- Uses enhanced tokenization from `english_nlp` module
- Proper punctuation removal
- Lemmatization for word embeddings

### 5. Updated: Package Dependencies

**setup.py:**
```python
extras_require={
    'english': ['spacy>=3.0.0'],  # Optional enhanced English support
}
```

**pyproject.toml:**
```toml
spacy = {version = "^3.0.0", optional = true}

[tool.poetry.extras]
english = ["spacy"]
```

---

## Files Modified

| File | Status | Changes |
|------|--------|---------|
| `cntext/english_nlp.py` | ✅ Created | New module (400+ lines) with enhanced English NLP |
| `cntext/model/sopmi.py` | ✅ Fixed | English language handling bug fixed |
| `cntext/model/utils.py` | ✅ Enhanced | Added lemmatization parameter |
| `cntext/stats/utils.py` | ✅ Enhanced | Added lemmatization parameter |
| `cntext/hello.py` | ✅ Fixed | Made IPython optional |
| `setup.py` | ✅ Updated | Added spaCy as optional dependency |
| `pyproject.toml` | ✅ Updated | Added spaCy configuration |
| `demo_simple.py` | ✅ Added | Workflow demonstration script |
| `README.md` | ✅ Rewritten | English-focused documentation |
| `CITATION.cff` | ✅ Updated | Fork attribution and references |
| `LICENSE` | ✅ Enhanced | Added fork attribution |

---

## Installation

### Basic Installation (NLTK-based)

```bash
pip install cntext
```

This provides all functionality with basic English tokenization.

### Enhanced Installation (spaCy-based)

```bash
# Install with enhanced English support
pip install cntext[english]

# Download spaCy model
python -m spacy download en_core_web_sm
```

This provides higher-quality tokenization and lemmatization.

---

## Usage Examples

### 1. Word Frequency Analysis

```python
import cntext as ct

text = "The companies are running innovative programs."

# Basic (works without spaCy)
counts = ct.word_count(text, lang='english')
# → {'companies': 1, 'running': 1, 'innovative': 1, 'programs': 1}

# With lemmatization (better with spaCy)
counts = ct.word_count(text, lang='english', lemmatize=True)
# → {'company': 1, 'run': 1, 'innovative': 1, 'program': 1}
```

### 2. Sentiment Analysis

```python
# Load Loughran-McDonald financial sentiment dictionary
lm_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')

text = """The company reported strong quarterly earnings,
exceeding expectations. However, uncertainty remains about
potential litigation risks."""

# Analyze sentiment
sentiment = ct.sentiment(text, diction=lm_dict, lang='english')
# → {'Positive': 2, 'Negative': 0, 'Uncertainty': 1, 'Litigious': 1, ...}
```

### 3. Text Similarity

```python
text1 = "Innovation drives progress and transforms industries."
text2 = "Creative thinking leads to advancement and changes markets."

similarity = ct.cosine_sim(text1, text2, lang='english')
# → 0.456 (higher = more similar)
```

### 4. Word Embeddings (Enhanced)

```python
# Train Word2Vec with improved preprocessing
wv = ct.Word2Vec(
    corpus_file='corpus.txt',
    lang='english',
    lemmatize=True,  # New parameter!
    vector_size=100,
    window=5,
    min_count=5
)

# Find similar words
similar = wv.most_similar('innovation', topn=5)
# → [('creativity', 0.85), ('breakthrough', 0.82), ...]
```

### 5. Semantic Projection (Most Powerful!)

```python
# Create a conceptual axis: "Optimistic ←→ Pessimistic"
axis = ct.generate_concept_axis(
    wv,
    poswords=['growth', 'success', 'innovative', 'opportunity'],
    negwords=['challenge', 'uncertainty', 'difficulty', 'risk']
)

# Measure where text falls on this axis
company_report = "We achieved exceptional growth through innovative solutions."
score = ct.project_text(wv, company_report, axis, lang='english')
# → 0.34 (positive = optimistic, negative = pessimistic)
```

### 6. Keyword in Context

```python
text = """Innovation drives progress. The company focuses on
innovation and creativity. Our innovation strategy succeeds."""

contexts = ct.word_in_context(text, ['innovation'], window=3, lang='english')
# Returns DataFrame with keyword contexts
```

### 7. Readability Metrics

```python
readability = ct.readability(text, lang='english')
# → {'Flesch': 45.2, 'Fog': 12.3, 'SMOG': 10.1, ...}
```

---

## Before & After Comparison

### Tokenization Quality

| Text | Before | After (No Lemma) | After (With Lemma) |
|------|--------|------------------|---------------------|
| "The companies are running innovative programs." | `['The', 'companies', 'are', 'running', 'innovative', 'programs.']` | `['companies', 'are', 'running', 'innovative', 'programs']` | `['company', 'be', 'run', 'innovative', 'program']` |

### Word Frequency Accuracy

**Text:** "companies running company runs"

| Approach | Result |
|----------|--------|
| **Before** | `{'companies': 1, 'running': 1, 'company': 1, 'runs': 1}` |
| **After (lemmatized)** | `{'company': 2, 'run': 2}` |

**Impact:** More accurate frequency counts by merging word forms

### Feature Comparison

| Feature | Before | After |
|---------|--------|-------|
| Tokenization | `text.split()` | spaCy/NLTK tokenization |
| Lemmatization | ❌ Not available | ✅ Optional parameter |
| Punctuation | Manual removal | ✅ Automatic |
| Number handling | Basic | ✅ Decimals supported |
| SoPmi English | ❌ Broken | ✅ Fixed |
| Quality | Basic | ✅ Research-grade |

---

## Git Commits

### Commit 1: Rebranding
```
Rebrand fork for English text analysis
- Updated README.md with English documentation
- Modified package metadata (v3.0.0)
- Added proper attribution to original cntext
```

### Commit 2: English NLP Enhancements
```
Implement enhanced English NLP support
- Created english_nlp.py module
- Fixed sopmi.py English handling bug
- Enhanced word_count() and preprocess_line()
- Added spaCy as optional dependency
```

### Commit 3: IPython Fix & Demo
```
Fix IPython dependency and add workflow demonstration
- Made IPython optional in hello.py
- Added demo_simple.py with comprehensive examples
```

**Branch:** `claude/modify-chinese-text-analysis-011CV4ffA8qngax7KnSMNReR`
**Status:** ✅ All changes committed and pushed

---

## Use Cases

### 1. Academic Research
Analyze English text in social science studies:
- Measure attitudes and cultural concepts
- Track semantic changes over time
- Quantify abstract constructs

### 2. Financial Analysis
Sentiment analysis of corporate documents:
- Earnings calls transcripts
- Annual reports (10-K)
- Press releases
- Analyst reports

### 3. Content Analysis
Study language patterns:
- Social media analysis
- News article sentiment
- Political discourse
- Marketing copy

### 4. Market Research
Analyze feedback and surveys:
- Customer reviews
- Product feedback
- Survey responses
- User interviews

### 5. Cultural Studies
Measure biases and attitudes:
- Implicit associations
- Stereotypes in text
- Cultural concept evolution
- Group differences

---

## Next Steps

### For Users

1. **Test with your data**
   - Try the demo: `python demo_simple.py`
   - Run analyses on your English text

2. **Install enhanced support**
   ```bash
   pip install cntext[english]
   python -m spacy download en_core_web_sm
   ```

3. **Explore built-in dictionaries**
   - Loughran-McDonald (Financial)
   - NRC (Emotions)
   - Concreteness
   - ANEW (Affective norms)

4. **Train domain embeddings**
   - Use your own corpus
   - Enable lemmatization for better quality
   - Create custom semantic axes

### For Contributors

Valuable contribution areas:

1. **English Dictionaries**
   - LIWC dictionary
   - VADER sentiment
   - Domain-specific lexicons (medical, legal, etc.)

2. **Documentation**
   - English-focused tutorials
   - Case studies with real data
   - Jupyter notebook examples

3. **Testing**
   - Unit tests for English functions
   - Benchmark against established tools
   - Performance optimizations

4. **Features**
   - Named Entity Recognition examples
   - POS tagging utilities
   - Additional preprocessing options

---

## Acknowledgments

**Huge thanks to:**

- **Xudong Deng (DaDeng)** and **Peng Nan** for creating the original cntext library
- **Harbin Institute of Technology** for supporting the original research
- All contributors to the original cntext project

This fork stands on the shoulders of their excellent work in developing innovative semantic analysis methods for social science research.

---

## License

This project maintains the MIT License of the original cntext project.

```
Original Copyright (c) 2022 DaDeng
Fork modifications (c) 2025 Fork Maintainer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

See [LICENSE](LICENSE) for full text.

---

## Support

- **Original cntext**: https://github.com/hiDaDeng/cntext
- **This fork**: [Update with your GitHub URL]
- **Report Issues**: [Your GitHub Issues URL]
- **Documentation**: [Your GitHub Wiki URL]

---

**Last Updated:** 2025-01-13
**Fork Version:** 3.0.0
**Based on:** cntext 2.2.0
