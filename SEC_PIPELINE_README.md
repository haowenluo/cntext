# SEC 10-K Sentence Labeling Pipeline

This directory contains scripts to process SEC 10-K Item extractions into sentence-level datasets ready for manual labeling and machine learning.

## 📁 Files

| File | Purpose |
|------|---------|
| `build_sentence_table.py` | Convert Item extractions → sentence-level table |
| `build_labeling_sample.py` | Create balanced sample for labeling |
| `tech_keywords.yaml` | Technology keyword dictionary (customizable) |
| `test_pipeline.py` | Test scripts with synthetic data |

## 🔄 Workflow

```
Your Item Extractions (JSON/CSV)
         ↓
[1] build_sentence_table.py
         ↓
  sentence_table.csv (all sentences)
         ↓
[2] build_labeling_sample.py
         ↓
  label_set.csv (2000-3000 sentences for labeling)
         ↓
[3] Manual Labeling (Excel/Sheets)
         ↓
  labeled_data.csv (ready for ML)
```

---

## Step 1: Build Sentence Table

### Input Format

Your Item extraction output should be JSON or CSV with these columns:

```json
{
  "cik": 1234567,
  "accession": "0000000000-20-000001",
  "fiscal_year": 2020,
  "filing_date": "2021-02-15",
  "item": "1",
  "item_text": "We develop innovative software solutions..."
}
```

### Run

```bash
python build_sentence_table.py \
  --input your_items.json \
  --output sentence_table.csv \
  --format both
```

**Options:**
- `--input`: Path to JSON or CSV with Item extractions (required)
- `--output`: Output path for sentence table (required)
- `--format`: Output format: `csv`, `parquet`, or `both` (default: both)

### Output Schema

```
cik, accession, fiscal_year, filing_date, item, sent_id, sentence_text
```

Example:
```csv
cik,accession,fiscal_year,filing_date,item,sent_id,sentence_text
1234567,0000000000-20-000001,2020,2021-02-15,1,0,We develop innovative software solutions.
1234567,0000000000-20-000001,2020,2021-02-15,1,1,Our cloud platform serves over 1000 customers.
```

### What It Does

1. **Sentence Splitting**: Uses spaCy (preferred) or NLTK fallback
2. **Text Cleaning**:
   - Fixes encoding issues (`ct.fix_text()`)
   - Expands contractions (`ct.fix_contractions()`)
   - Normalizes whitespace
3. **Quality Filtering**:
   - Removes sentences < 20 characters
   - Filters out tables (high numeric ratio)
   - Removes noise (high punctuation ratio)
4. **Export**: Saves to CSV and/or Parquet

---

## Step 2: Build Labeling Sample

### Run

```bash
python build_labeling_sample.py \
  --input sentence_table.csv \
  --output label_set.csv \
  --size 2500
```

**Options:**
- `--input`: Path to sentence_table.csv from Step 1 (required)
- `--output`: Output path for label set (required)
- `--keywords`: Path to tech keywords YAML (default: `tech_keywords.yaml`)
- `--size`: Total sentences to sample (default: 2500)

### Output Schema

```
cik, accession, fiscal_year, filing_date, item, sent_id,
sentence_text, tech_hit, source_pool,
TECH_IMPL, TECH_ADOPT, TECH_PRODUCT, NON_TECH
```

Example:
```csv
cik,accession,fiscal_year,item,sent_id,sentence_text,tech_hit,source_pool,TECH_IMPL,TECH_ADOPT,TECH_PRODUCT,NON_TECH
1234567,0000-20-001,2020,1,5,Our cloud platform uses machine learning.,True,tech_hit,,,
1234567,0000-20-001,2020,1,12,We have facilities in 20 states.,False,random,,,
```

### What It Does

1. **Keyword Matching**: Creates `tech_hit` flag using tech_keywords.yaml
   - Case-insensitive matching
   - Multi-word phrase support
   - Word boundary detection (avoids partial matches)

2. **Deduplication**: Removes exact sentence duplicates

3. **Sampling Strategy**:
   - 50% from `tech_hit==1` (likely tech sentences)
   - 50% from random pool (includes non-tech)
   - Fixed random seed (42) for reproducibility

4. **Export**: CSV ready for labeling in Excel/Sheets

---

## Step 3: Manual Labeling

### Instructions

1. Open `label_set.csv` in Excel or Google Sheets

2. For each sentence, mark **exactly ONE** category with `1`:

   | Column | Description | Examples |
   |--------|-------------|----------|
   | `TECH_IMPL` | Technology implementation/usage | "We deployed AI algorithms", "Our cloud infrastructure processes..." |
   | `TECH_ADOPT` | Technology adoption/investment | "We invested $50M in R&D", "We acquired a machine learning company" |
   | `TECH_PRODUCT` | Technology product/offering | "Our SaaS platform offers...", "We sell cybersecurity software" |
   | `NON_TECH` | Not technology-related | "We operate retail stores", "Revenue increased 10%" |

3. **Constraint**: If `NON_TECH=1`, all other columns must be `0`

4. Leave unlabeled rows blank (or use `0` for all)

### Quality Tips

- Use `tech_hit` column as a hint (not definitive)
- Check `source_pool`:
  - `tech_hit`: Sentence matched tech keywords
  - `random`: Randomly sampled (may or may not be tech)
- Focus on the **sentence's main topic**, not just keyword presence
- Be consistent across similar sentences

---

## 📝 Customizing Tech Keywords

Edit `tech_keywords.yaml` to add/remove keywords:

```yaml
Dictionary:
  ai_ml:
    - artificial intelligence
    - machine learning
    - your custom term

  industry_specific:  # Add new categories
    - telemedicine
    - blockchain
```

**Tips:**
- Add industry-specific terms (e.g., "electronic health record" for healthcare)
- Remove overly broad terms causing false positives
- Use multi-word phrases for precision
- Test with small sample first

---

## ⚙️ Configuration

### build_sentence_table.py

Edit these variables at the top of the script:

```python
MIN_SENTENCE_LENGTH = 20          # Minimum characters
MAX_SENTENCE_LENGTH = 2000        # Maximum (filters OCR errors)
MIN_WORD_COUNT = 3                # Minimum words
MAX_NUMERIC_RATIO = 0.6           # Max numeric content (filters tables)
MAX_PUNCT_RATIO = 0.4             # Max punctuation (filters noise)
```

### build_labeling_sample.py

```python
DEFAULT_SAMPLE_SIZE = 2500        # Total sentences to sample
TECH_HIT_RATIO = 0.5              # Proportion from tech_hit==1
RANDOM_SEED = 42                  # For reproducibility
ENABLE_DEDUP = True               # Remove duplicates
STRATIFY_BY_YEAR = False          # Balance across years
STRATIFY_BY_ITEM = False          # Balance across Items
```

---

## 🧪 Testing

Run the test pipeline with synthetic data:

```bash
python test_pipeline.py
```

This creates `test_pipeline_data/` with:
- Sample Item extractions
- Sentence table
- Label set

Use this to verify the pipeline before processing real data.

---

## 📊 Expected Output Sizes

Assuming 1000 10-K filings with Items 1, 1A, and 7:

| Stage | Approximate Size |
|-------|------------------|
| Item extractions | 3,000 items (1, 1A, 7 per filing) |
| Sentence table | ~150,000 sentences (50 per item avg) |
| Label set | 2,500 sentences (for labeling) |

**Disk usage:**
- `sentence_table.csv`: ~50-100 MB (depends on text length)
- `sentence_table.parquet`: ~20-40 MB (more efficient)
- `label_set.csv`: ~1-2 MB

---

## 🔧 Troubleshooting

### "spaCy model not found"

Install spaCy's English model:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

Or use NLTK fallback (automatic, lower quality)

### "Not enough tech sentences to sample"

- Reduce `--size` parameter
- Add more keywords to `tech_keywords.yaml`
- Check that input data contains tech-related content

### "Duplicate sentences"

This is normal! Deduplication is enabled by default.
- Check summary: `label_set.txt` shows # duplicates removed
- Disable: Set `ENABLE_DEDUP = False` in script

### "Tech_hit flag all False"

- Check `tech_keywords.yaml` is in same directory
- Verify keywords match your domain
- Try sample sentences manually to test matching

---

## 💡 Tips for Large Datasets

### Memory Management

For 100,000+ sentences:

1. Use Parquet format:
   ```bash
   --format parquet
   ```

2. Process in chunks (modify script):
   ```python
   for chunk in pd.read_csv(input_path, chunksize=10000):
       # Process chunk
   ```

### Sampling Strategy

For better quality:

1. **Stratify by year** for temporal balance:
   ```python
   STRATIFY_BY_YEAR = True
   ```

2. **Stratify by item** for content balance:
   ```python
   STRATIFY_BY_ITEM = True
   ```

3. **Increase tech ratio** if non-tech is too common:
   ```python
   TECH_HIT_RATIO = 0.7  # 70% tech, 30% random
   ```

---

## 📚 Repository Integration

These scripts leverage existing cntext capabilities:

| Function | Script Usage | Source |
|----------|--------------|--------|
| `ct.fix_text()` | Encoding normalization | `cntext/io/utils.py:76` |
| `ct.fix_contractions()` | Expand "you're" → "you are" | `cntext/io/utils.py:86` |
| `ct.clean_text()` | Whitespace/URL cleaning | `cntext/io/utils.py:149` |
| spaCy/NLTK tokenization | Sentence splitting | `cntext/english_nlp.py` |

---

## 🚀 Next Steps After Labeling

Once you have `labeled_data.csv`:

1. **Train classifier** (sklearn, transformers)
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(...)
   ```

2. **Apply to full dataset**
   ```python
   predictions = model.predict(sentence_table['sentence_text'])
   ```

3. **Analyze results** using cntext:
   ```python
   # Count tech mentions by year
   tech_by_year = df.groupby('fiscal_year')['tech_pred'].sum()

   # Sentiment of tech sentences
   tech_sentences = df[df['tech_pred'] == 1]
   sentiment = ct.sentiment(tech_sentences['sentence_text'])
   ```

---

## 📧 Support

For issues or questions:
- Check the generated `*_summary.txt` files for diagnostics
- Review sample outputs in terminal
- Adjust configuration variables for your use case

---

**Happy labeling! 🏷️**
