# Replication Package

This folder contains all materials needed to replicate the digitalization attitudes analysis test results.

## 📁 Folder Contents

### Documentation
- **REPLICATION_GUIDE.md** - Complete step-by-step guide for replicating the analysis
- **REPLICATION_RESULTS.md** - Verification report from fresh replication (Nov 16, 2025)
- **TEST_SUMMARY.md** - Executive summary of test results and methodology

### Test Scripts
- **generate_test_mda_data.py** - Generates synthetic MD&A test dataset (45 documents)
- **run_test_analysis.py** - Runs complete analysis pipeline with all 4 approaches

### Test Data
- **test_mda_dataset.csv** - Synthetic MD&A sections dataset
  - 45 rows (15 companies × 3 years: 2022-2024)
  - 7,641 total words
  - Ground truth labels for validation

### Test Results
- **test_results/** - Complete analysis outputs
  - `test_results_complete.csv` - Full results (45 rows, 29 columns, 130KB)
  - `test_results_by_year.csv` - Temporal trends (3 years, 710 bytes)
  - `validation_report.txt` - Accuracy validation metrics
  - `word_embeddings.model` - Trained Word2Vec model (510 vocab, 50-dim)

### Intermediate Files
- **output/** - Word2Vec training cache files
  - `temp_corpus_cache.txt` - Preprocessed corpus cache
  - `temp_corpus-Word2Vec.50.10.bin` - Binary embeddings file
- **temp_corpus.txt** - Temporary corpus file for training (59KB)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Core packages
pip install pandas numpy scikit-learn matplotlib gensim nltk tqdm scipy

# CRITICAL: Install jieba with PEP517
python -m pip install --use-pep517 jieba

# REQUIRED: LLM support (not optional)
pip install openai instructor pydantic

# Additional packages
pip install PyMuPDF pyecharts python-docx aiolimiter nest-asyncio
pip install opencc-python-reimplemented contractions psutil requests beautifulsoup4 lxml
pip install ftfy chardet networkx h5py distinctiveness

# Download NLTK data
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"
```

### 2. Generate Test Data
```bash
cd /home/user/cntext/replication
python generate_test_mda_data.py
```

### 3. Run Analysis
```bash
python run_test_analysis.py
```

**Expected runtime**: ~30-60 seconds

---

## 📊 What Gets Tested

### 4 Measurement Approaches
1. **Keyword Frequency** - Raw counts of digital/AI/tech terms
2. **Normalized Frequency** - Adjusts for document length (per 1000 words)
3. **Sentiment-Aware Dictionary** - Positive vs negative digital attitudes
4. **Semantic Projection** - Latent attitudes via word embeddings

### 3 Semantic Dimensions
1. **Digital Embrace** - Embracer ↔ Skeptic axis
2. **AI Enthusiasm** - Enthusiast ↔ Cautious axis
3. **Innovation Leadership** - Leader ↔ Follower axis

---

## ✅ Expected Results

All results should match these exact values:

| Metric | Expected Value |
|--------|----------------|
| Documents analyzed | 45 |
| Average word count | 169.8 |
| Total digital frequency | 185.72 per 1000 words |
| Digital sentiment | +0.086 (net) |
| Semantic digital_embrace | -0.3036 (mean) |
| Semantic ai_enthusiasm | +0.2026 (mean) |
| Vocabulary size | 510 words |
| Keyword correlation | r = 0.954 |
| Semantic correlation | r = -0.054 |

---

## 📖 Documentation

### For Replication
1. Start with **REPLICATION_GUIDE.md** for step-by-step instructions
2. Check **REPLICATION_RESULTS.md** for known issues and solutions
3. Review **TEST_SUMMARY.md** for overview and context

### For Research
- **REPLICATION_GUIDE.md** Section: "Adapting for Your Data" (lines 560+)
- Example: Customize dictionaries, semantic axes, file formats
- Scalability: Batch processing for large datasets

---

## 🐛 Common Issues

### Issue #1: ModuleNotFoundError: openai
**Solution**: `pip install openai instructor pydantic`
**Why**: Required by cntext module (not optional)

### Issue #2: jieba installation fails
**Solution**: `python -m pip install --use-pep517 jieba`
**Why**: Avoids setuptools compatibility issues

### Issue #3: Low semantic accuracy
**Expected** for small test dataset (45 documents)
**Solution**: Use larger corpus (500+ docs) for real research

See **REPLICATION_GUIDE.md** Troubleshooting section for complete list.

---

## 📦 File Sizes

```
Total: ~250KB

Documentation:
- REPLICATION_GUIDE.md:    27KB
- REPLICATION_RESULTS.md:  14KB
- TEST_SUMMARY.md:          9KB

Scripts:
- generate_test_mda_data.py: 19KB
- run_test_analysis.py:      17KB

Data:
- test_mda_dataset.csv:      65KB
- temp_corpus.txt:           60KB

Results:
- test_results_complete.csv: 130KB
- word_embeddings.model:     120KB
- test_results_by_year.csv:  710 bytes
- validation_report.txt:     437 bytes

Cache:
- output/:                   ~500KB (Word2Vec binary)
```

---

## 🔬 Verification Status

**Replication Verified**: ✅ November 16, 2025
**Environment**: Linux 4.4.0, Python 3.11.14
**Status**: 100% metric match achieved
**Issues Found**: 3 (all documented and resolved)

---

## 📝 Version History

### November 16, 2025
- ✅ Complete replication verification
- ✅ Documentation updates based on fresh replication
- ✅ All 3 issues identified and corrected in guide
- ✅ Files organized into dedicated replication folder

---

## 📄 License

Maintains the MIT License from the original cntext repository.

---

## 🔗 Related Files

**In main repository:**
- `cntext/` - Core library code
- `demos/` - Feature demonstrations
- `ENGLISH_ENHANCEMENTS.md` - Technical documentation

**For production use:**
- `analyze_digital_attitudes_template.py` - Ready-to-use template
- `guide_digitalization_attitudes.py` - Conceptual guide

---

**Last Updated**: November 16, 2025
**Maintained by**: Research team
**Contact**: See main repository for support
