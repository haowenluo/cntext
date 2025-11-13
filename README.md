# entext: English Text Analysis for Social Science Research

**A fork of [cntext](https://github.com/hiDaDeng/cntext) adapted for English text analysis**

> **Acknowledgment**: This project is a fork of the excellent [cntext](https://github.com/hiDaDeng/cntext) library created by [DaDeng](https://github.com/hiDaDeng). The original cntext was designed for Chinese text analysis in social science research. This fork adapts the codebase to focus on English text analysis while maintaining the innovative semantic analysis capabilities of the original project.

---

## Overview

**entext** is a Python library for English text analysis designed specifically for **social science researchers**. Going beyond traditional word frequency and sentiment analysis, entext provides word embedding training, semantic projection calculations, and tools to **measure abstract constructs from large-scale unstructured text**—such as attitudes, cognition, cultural concepts, and psychological states.

## 🎯 What You Can Do With entext

### 1. Build Structured Research Datasets
- Aggregate multiple text files (txt/pdf/docx/csv) into a DataFrame: `ct.read_files()`
- Calculate text readability metrics (Flesch Index, SMOG, etc.): `ct.readability()`
- Extract and preprocess text for analysis: `ct.clean_text()`

### 2. Traditional Text Analysis
- Word frequency statistics and keyword extraction: `ct.word_count()`
- Sentiment analysis with built-in dictionaries (Loughran-McDonald, NRC, etc.): `ct.sentiment()`
- Text similarity computation (cosine distance): `ct.cosine_sim()`

### 3. Measure Implicit Attitudes & Cultural Change
- Train domain-specific word embeddings with two lines of code (Word2Vec/GloVe): `ct.Word2Vec()`
- Construct concept semantic axes (e.g., "innovation vs. tradition"): `ct.generate_concept_axis()`
- Quantify stereotypes and organizational culture shifts through semantic projection: `ct.project_text()`

### 4. LLM Integration for Structured Analysis
- Call LLMs for semantic parsing with structured output (emotions, intent classification): `ct.llm()`
- Support for Ollama, OpenAI, and custom backends

---

entext does not pursue black-box prediction but is committed to making text a theory-driven scientific measurement tool. Open source and free for academic and research use.

---

## Installation

```bash
pip3 install cntext --upgrade
```

**Note**: This package currently retains the `cntext` name for compatibility. Future releases may rebrand as `entext`.

**Requirements**: Python 3.9 ~ 3.12

For English text analysis, you may also want to install spaCy:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

---

## Quick Start

```python
import cntext as ct

print('Current version:', ct.__version__)

# Example: Analyze English text sentiment
text = """The company reported strong quarterly earnings,
exceeding market expectations. However, concerns remain
about supply chain disruptions and rising costs."""

# Load Loughran-McDonald financial sentiment dictionary
lm_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')

# Perform sentiment analysis
result = ct.sentiment(text, diction=lm_dict, lang='english')
print(result)
```

---

## Module Overview

entext contains 6 main modules:

```python
import cntext as ct
ct.hello()
```

1. **io** - Data import/export and preprocessing
2. **stats** - Statistical analysis (word count, sentiment, similarity, readability)
3. **model** - Word embedding training (Word2Vec, GloVe, FastText)
4. **mind** - Cognitive/semantic analysis (projections, associations)
5. **plot** - Visualization tools
6. **llm** - Large Language Model integration

### Module Function Summary

| Module | Function | Description |
|--------|----------|-------------|
| **io** | `ct.get_dict_list()` | List built-in dictionaries |
| **io** | `ct.read_yaml_dict(yfile)` | Load built-in YAML dictionary |
| **io** | `ct.read_pdf(file)` | Read PDF file |
| **io** | `ct.read_docx(file)` | Read DOCX file |
| **io** | `ct.read_file(file, encoding)` | Read text file |
| **io** | `ct.read_files(fformat, encoding)` | Read multiple files matching pattern, return DataFrame |
| **io** | `ct.fix_text(text)` | Fix encoding issues, normalize text |
| **io** | `ct.fix_contractions(text)` | Expand English contractions (you're → you are) |
| **io** | `ct.clean_text(text, lang='english')` | Clean and normalize text |
| **stats** | `ct.word_count(text, lang='english')` | Word frequency statistics |
| **stats** | `ct.readability(text, lang='english')` | Calculate readability metrics (Flesch, SMOG, etc.) |
| **stats** | `ct.sentiment(text, diction, lang='english')` | Sentiment analysis with unweighted dictionary |
| **stats** | `ct.sentiment_by_valence(text, diction, lang='english')` | Weighted sentiment analysis |
| **stats** | `ct.word_in_context(text, keywords, window, lang='english')` | Find keywords with context window |
| **stats** | `ct.cosine_sim(text1, text2, lang='english')` | Cosine similarity |
| **stats** | `ct.jaccard_sim(text1, text2, lang='english')` | Jaccard similarity |
| **stats** | `ct.minedit_sim(text1, text2, lang='english')` | Minimum edit distance |
| **stats** | `ct.word_hhi(text)` | Herfindahl-Hirschman Index for text |
| **model** | `ct.Word2Vec(corpus_file, lang='english', ...)` | Train Word2Vec model |
| **model** | `ct.GloVe(corpus_file, lang='english', ...)` | Train GloVe model (uses Stanford NLP) |
| **model** | `ct.evaluate_similarity(wv, file=None)` | Evaluate model with synonym test |
| **model** | `ct.evaluate_analogy(wv, file=None)` | Evaluate model with analogy test |
| **model** | `ct.load_w2v(wv_path)` | Load trained Word2Vec/GloVe model |
| **model** | `ct.expand_dictionary(wv, seeddict, topn=100)` | Expand dictionary using embeddings |
| **model** | `ct.SoPmi(corpus_file, seed_file, lang='english')` | Co-occurrence-based dictionary expansion |
| **mind** | `ct.generate_concept_axis(wv, poswords, negwords)` | Generate concept axis vector |
| **mind** | `ct.Text2Mind(wv)` | Mine implicit attitudes, biases, stereotypes |
| **mind** | `ct.semantic_projection(wv, words, poswords, negwords)` | Measure semantic projection |
| **mind** | `ct.project_word(wv, a, b)` | Project word a onto word b |
| **mind** | `ct.project_text(wv, text, axis, lang='english')` | Project text onto concept axis |
| **mind** | `ct.semantic_distance(wv, words1, words2)` | Measure semantic distance |
| **mind** | `ct.divergent_association_task(wv, words)` | Measure divergent thinking (creativity) |
| **mind** | `ct.discursive_diversity_score(wv, words)` | Measure discursive diversity (cognitive diversity) |
| **mind** | `ct.procrustes_align(base_wv, other_wv)` | Align embeddings to track semantic change over time |
| **llm** | `ct.llm(text, task, backend, model_name, ...)` | Call LLM for structured text analysis |

---

## Built-in English Dictionaries

List all available dictionaries:
```python
import cntext as ct
ct.get_dict_list()
```

### Key English Dictionaries

| Dictionary File | Description | Language | Categories |
|----------------|-------------|----------|------------|
| **en_common_LoughranMcDonald.yaml** | Loughran-McDonald Financial Sentiment Dictionary (2018) | English | Negative, Positive, Uncertainty, Litigious, StrongModal, WeakModal, Constraining |
| **en_common_NRC.yaml** | NRC Word-Emotion Association Lexicon | English | Fine-grained emotions (anger, fear, joy, sadness, etc.) |
| **en_valence_Concreteness.yaml** | English Concreteness Ratings | English | Word concreteness scores |
| **en_common_ANEW.yaml** | Affective Norms for English Words (ANEW) | English | pleasure, arousal, dominance |
| **en_common_LSD2015.yaml** | Lexicoder Sentiment Dictionary (2015) | English | Positive, Negative |
| **enzh_common_StopWords.yaml** | English & Chinese stopwords | Bilingual | Stopwords |

Load a dictionary:
```python
lm_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')
print(lm_dict['Name'])  # Dictionary name
print(lm_dict['Desc'])  # Description
print(lm_dict['Category'])  # Categories
print(lm_dict['Dictionary'])  # Actual word lists
```

---

## Usage Examples

### 1. Basic Text Processing

```python
import cntext as ct

# Read PDF
text = ct.read_pdf('research_paper.pdf')

# Clean text
text = ct.clean_text(text, lang='english')

# Fix contractions
text = ct.fix_contractions(text)  # "don't" → "do not"

# Calculate readability
readability_scores = ct.readability(text, lang='english')
print(readability_scores)
# Output: {'Flesch': 45.2, 'Fog': 12.3, 'SMOG': 10.1, ...}
```

### 2. Sentiment Analysis

```python
# Load financial sentiment dictionary
fin_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')

# Corporate annual report text
text = """Our revenues increased significantly this quarter,
demonstrating strong market demand. However, we face uncertainty
regarding regulatory changes and potential litigation risks."""

# Analyze sentiment
result = ct.sentiment(text, diction=fin_dict, lang='english')
print(result)
# Output: {'Negative': 2, 'Positive': 2, 'Uncertainty': 1, 'Litigious': 1, ...}
```

### 3. Word Frequency Analysis

```python
text = """Innovation drives progress. Innovation transforms industries.
Creativity and innovation are essential for growth."""

word_freq = ct.word_count(text, lang='english')
print(word_freq)
# Output: DataFrame with word frequencies
```

### 4. Text Similarity

```python
text1 = "Machine learning is transforming healthcare through predictive analytics."
text2 = "Artificial intelligence revolutionizes medical diagnosis using data analysis."

similarity = ct.cosine_sim(text1, text2, lang='english')
print(f"Cosine similarity: {similarity:.3f}")
```

### 5. Train Word Embeddings

```python
# Train Word2Vec on your corpus
wv = ct.Word2Vec(
    corpus_file='my_corpus.txt',
    lang='english',
    vector_size=100,
    window=5,
    min_count=5,
    workers=4
)

# Find similar words
similar_words = wv.most_similar('innovation', topn=10)
print(similar_words)

# Test word analogy: king - man + woman = ?
result = wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)  # Should find 'queen'
```

### 6. Semantic Projection Analysis

This is one of the most powerful features - measure abstract concepts in text:

```python
# Load pre-trained embeddings
wv = ct.Word2Vec('corpus.txt', lang='english', vector_size=100)

# Create a concept axis: "innovation vs. tradition"
axis = ct.generate_concept_axis(
    wv,
    poswords=['innovation', 'creative', 'novel', 'breakthrough'],
    negwords=['traditional', 'conventional', 'conservative', 'established']
)

# Measure where a text falls on this axis
company_mission = "We embrace cutting-edge technology and pioneering solutions."
score = ct.project_text(wv, company_mission, axis, lang='english')
print(f"Innovation score: {score:.3f}")
# Positive score = more innovative, negative = more traditional
```

### 7. Divergent Thinking (Creativity Measurement)

```python
# Measure semantic diversity of word associations (creativity proxy)
creative_words = ['sky', 'ocean', 'freedom', 'music', 'dance', 'quantum']
conventional_words = ['desk', 'chair', 'office', 'computer', 'meeting', 'report']

creative_score = ct.divergent_association_task(wv, creative_words)
conventional_score = ct.divergent_association_task(wv, conventional_words)

print(f"Creative set DAT score: {creative_score:.3f}")
print(f"Conventional set DAT score: {conventional_score:.3f}")
# Higher score = more divergent/creative thinking
```

### 8. Semantic Change Over Time

```python
# Train embeddings on texts from different time periods
wv_2000 = ct.Word2Vec('corpus_2000s.txt', lang='english', vector_size=100)
wv_2020 = ct.Word2Vec('corpus_2020s.txt', lang='english', vector_size=100)

# Align the embedding spaces
wv_2020_aligned = ct.procrustes_align(wv_2000, wv_2020)

# Track how word meanings shifted
word = 'startup'
neighbors_2000 = wv_2000.most_similar(word, topn=5)
neighbors_2020 = wv_2020_aligned.most_similar(word, topn=5)

print(f"'{word}' neighbors in 2000s:", neighbors_2000)
print(f"'{word}' neighbors in 2020s:", neighbors_2020)
```

### 9. LLM Integration

```python
# Use local LLM (via Ollama) for structured text analysis
text = """The company's quarterly earnings exceeded expectations,
but concerns about supply chain issues persist."""

# Sentiment analysis via LLM
result = ct.llm(
    text=text,
    task='sentiment',
    backend='ollama',
    model_name='qwen2.5:3b'
)
print(result)

# Custom prompt
custom_result = ct.llm(
    text=text,
    prompt="Extract key business risks mentioned in this text.",
    backend='ollama',
    model_name='qwen2.5:3b'
)
print(custom_result)
```

---

## Differences from Original cntext

### What's Changed
- **Documentation**: Rewritten in English for English-speaking researchers
- **Focus**: Emphasis on English text analysis examples
- **Default language**: Examples default to `lang='english'`
- **Dictionary expansion**: Plans to add more English-specific dictionaries

### What's the Same
- **Core algorithms**: All sentiment, embedding, and semantic projection code unchanged
- **Bilingual support**: Chinese functionality still works with `lang='chinese'`
- **API compatibility**: Function signatures remain identical
- **Dependencies**: Same requirements as original cntext

---

## Roadmap

Future enhancements planned for English text analysis:

- [ ] Integrate spaCy for better English NLP (lemmatization, POS tagging, NER)
- [ ] Add LIWC (Linguistic Inquiry and Word Count) dictionary support
- [ ] Include VADER sentiment analyzer
- [ ] Expand domain-specific dictionaries (medical, legal, political)
- [ ] Add named entity recognition examples
- [ ] Create English-focused tutorials and case studies
- [ ] Performance optimizations for large English corpora

---

## Citation

If you use this fork in your research, please cite both:

### This Fork (entext)
```
Fork of cntext adapted for English text analysis
Repository: [Your GitHub URL]
Original project: https://github.com/hiDaDeng/cntext
```

### Original cntext Project
**APA Style:**
```
Deng, X., & Nan, P. (2022). cntext: a Python tool for text mining (Version 1.7.9) [Computer software]. https://github.com/hiDaDeng/cntext
```

**BibTeX:**
```bibtex
@software{deng2022cntext,
  title = {cntext: a Python tool for text mining},
  author = {Deng, Xudong and Nan, Peng},
  year = {2022},
  version = {1.7.9},
  url = {https://github.com/hiDaDeng/cntext},
  note = {Harbin Institute of Technology}
}
```

---

## License

This project maintains the MIT License of the original cntext project.

```
MIT License

Original Copyright (c) 2022 DaDeng
Fork modifications (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## Contributing

This is an independent fork maintained separately from the original cntext project. Contributions focused on improving English text analysis are welcome!

Areas where contributions are especially valuable:
- English-specific dictionaries and lexicons
- spaCy integration for better English NLP
- English text analysis tutorials and examples
- Bug fixes and performance improvements
- Documentation improvements

---

## Acknowledgments

**Special thanks to:**
- **DaDeng** and **Peng Nan** for creating the original [cntext](https://github.com/hiDaDeng/cntext) library
- The Harbin Institute of Technology for supporting the original research
- All contributors to the original cntext project

This fork stands on the shoulders of their excellent work in developing innovative semantic analysis methods for social science research.

---

## Support & Resources

- **Original cntext**: https://github.com/hiDaDeng/cntext
- **Report Issues**: [Your GitHub Issues URL]
- **Documentation**: [Your GitHub Wiki URL]

---

**Note**: This project is for research and educational purposes. Please ensure your use complies with your institution's research ethics guidelines and data privacy regulations.
