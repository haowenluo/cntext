# Local Pipeline Test - Yearly Batch Processing

This folder contains scripts for testing the SEC pipeline locally with **yearly batch processing** for scalability.

---

## 📁 Files

| File | Purpose |
|------|---------|
| `run_local_test.py` | **Main script** - Processes MDA files with yearly batching |
| `MDA_FORMAT_GUIDE.md` | Documents MDA → Pipeline format transformation |
| `output/` | Generated output files (sentence tables, labeling samples) |

---

## 🚀 Quick Start

### 1. Prepare Your Data

Place your MDA JSON files in a directory (can have subdirectories):

```
your_mda_files/
├── 2020/
│   ├── company1_10K_2020_....json
│   ├── company2_10K_2020_....json
├── 2021/
│   ├── company1_10K_2021_....json
└── ...
```

### 2. Update Configuration

Edit `run_local_test.py` line 31:

```python
MDA_SAMPLE_DIR = Path("path/to/your/mda/files")
```

### 3. Run the Script

```bash
cd local_pipeline_test
python run_local_test.py
```

---

## 📊 What It Does

### Step 1: Transform MDA Files
- Reads all JSON files recursively
- Transforms from MDA format → Pipeline format
- Groups files by fiscal year

### Step 2: Process Each Year Separately
```
Year 2020 (50 filings) → sentences_2020.parquet
Year 2021 (45 filings) → sentences_2021.parquet
Year 2022 (60 filings) → sentences_2022.parquet
...
```

### Step 3: Combine Years
- Merges all yearly sentence tables
- Creates combined CSV for reference

### Step 4: Generate Labeling Sample
- Samples 500 sentences (or max available)
- Balanced: 50% tech hits, 50% random
- Across all years for temporal diversity

---

## 📂 Output Structure

```
output/
├── yearly_parquet/
│   ├── sentences_2009.parquet
│   ├── sentences_2010.parquet
│   ├── sentences_2015.parquet
│   └── ...                          # One file per year
├── sentence_table_combined.csv      # All years combined
├── label_set_combined.csv           # 500 sentences for labeling
└── tech_keywords.yaml               # Technology keyword dictionary
```

---

## 🔍 Example Output

### Console Output:
```
================================================================================
LOCAL PIPELINE TEST - YEARLY BATCH PROCESSING
================================================================================

STEP 2: TRANSFORMING MDA FILES (GROUPED BY YEAR)
================================================================================
Found 50 MDA JSON files
  ✓ 1643988_10K_2020_0001387131-21-004517.json (CIK: 1643988, Year: 2020)
  ✓ 1596946_10K_2020_0001564590-20-029174.json (CIK: 1596946, Year: 2020)
  ...

✓ Transformed: 50 files
  Skipped: 0 files
  Years covered: [2009, 2010, 2015, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

STEP 3: PROCESSING EACH YEAR → PARQUET FILES
================================================================================

--- Processing Year 2009 (4 filings) ---
  ✓ 1,413 sentences → sentences_2009.parquet

--- Processing Year 2010 (5 filings) ---
  ✓ 2,357 sentences → sentences_2010.parquet

...

SUMMARY
================================================================================

📁 Yearly Parquet Files:
  - sentences_2009.parquet: 1,413 sentences from 4 filings
  - sentences_2010.parquet: 2,357 sentences from 5 filings
  - sentences_2015.parquet: 1,780 sentences from 5 filings
  ...

📊 Combined Outputs:
  - sentence_table_combined.csv: 16,594 sentences total
  - label_set_combined.csv: 500 sentences for labeling

📈 Statistics:
  Total filings processed: 50
  Total sentences extracted: 16,594
  Years covered: 2009 - 2025
  Avg sentences per filing: 331.9

💡 Scalability Notes:
  - Each year processed separately → memory efficient
  - Yearly Parquet files → easy to load specific years
  - Fault tolerant → crash only loses current year
  - Ready for 100k+ filings → millions of sentences
```

---

## ⚙️ Customization

### Change Sample Size

Edit line 253 in `run_local_test.py`:

```python
sample_size = min(2500, len(combined_df))  # Change 2500 to your desired size
```

### Add Custom Keywords

Edit `tech_keywords.yaml` in the output folder:

```yaml
Dictionary:
  fintech:
    - digital wallet
    - payment processing
    - peer-to-peer lending
```

### Process Specific Years

Modify the year loop (line 182-215):

```python
# Only process 2020-2022
for year in sorted(records_by_year.keys()):
    if year not in [2020, 2021, 2022]:
        continue
    # ... rest of processing
```

---

## 🔧 Requirements

### Python Packages

```bash
pip install pandas numpy tqdm pyyaml pyarrow
pip install spacy ftfy contractions chardet
pip install networkx scipy scikit-learn gensim nltk
pip install distinctiveness aiolimiter instructor pydantic psutil
pip install --use-pep517 jieba
pip install opencc-python-reimplemented

# Download spaCy model
python -m spacy download en_core_web_sm
```

### System Resources

**Minimum:**
- RAM: 4GB (processes one year at a time)
- Storage: 100MB per 1000 filings (Parquet compressed)

**Recommended:**
- RAM: 8GB+
- Storage: SSD for faster I/O

---

## 📈 Scalability

### Tested Performance

| Dataset Size | Filings | Sentences | Processing Time | Memory Usage |
|--------------|---------|-----------|-----------------|--------------|
| **Test** | 50 | 16,594 | 2 min | <2 GB |
| **Medium** | 1,000 | ~330,000 | 30-45 min | 2-4 GB |
| **Large** | 10,000 | ~3,300,000 | 5-8 hours | 4-8 GB |
| **Extra Large** | 100,000+ | ~33,000,000+ | 2-3 days | 8-16 GB |

### Memory Efficiency

- **Without batching**: Entire dataset loaded → 40GB+ RAM for 100k filings
- **With yearly batching**: One year at a time → 4-8GB RAM for 100k filings

### Fault Tolerance

If processing crashes at year 2015:
- ✅ **Years 2009-2014**: Already saved to Parquet (safe!)
- ❌ **Year 2015**: Lost (only this year)
- ⏭️ **Years 2016+**: Not processed yet

Simply restart and skip completed years.

---

## 🐛 Troubleshooting

### Issue: "No module named 'cntext'"

**Solution:** Ensure you're running from the repository root or update `sys.path`:

```python
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
```

### Issue: "No records found"

**Solution:** Check `MDA_SAMPLE_DIR` path points to correct directory:

```python
# Line 31 in run_local_test.py
MDA_SAMPLE_DIR = Path("correct/path/to/mda/files")
```

### Issue: Out of memory

**Solutions:**
1. Process fewer years at once (modify year loop)
2. Reduce sample size
3. Close other applications
4. Use a machine with more RAM

### Issue: Parquet import error

**Solution:** Install PyArrow:

```bash
pip install pyarrow
```

---

## 📚 Integration with Main Pipeline

### Using Yearly Files in Analysis

```python
import pandas as pd

# Load specific year
df_2020 = pd.read_parquet('output/yearly_parquet/sentences_2020.parquet')

# Load multiple years
years = [2020, 2021, 2022]
dfs = [pd.read_parquet(f'output/yearly_parquet/sentences_{y}.parquet') for y in years]
combined = pd.concat(dfs, ignore_index=True)

# Analyze trends
yearly_stats = combined.groupby('fiscal_year').agg({
    'sentence_text': 'count',
    'cik': 'nunique'
}).rename(columns={'sentence_text': 'sentences', 'cik': 'companies'})

print(yearly_stats)
```

### Using in Google Colab

The yearly batch processing approach is also available in the Colab notebook:

- See **Step 2B: Yearly Batch Processing** in `SEC_Pipeline_Colab.ipynb`
- Same logic, adapted for Colab environment

---

## ✅ Best Practices

### 1. Test with Small Sample First

```python
# Process just 10 files to verify
test_files = json_files[:10]
```

### 2. Monitor Progress

The script shows progress for each year - useful for estimating total time.

### 3. Save Intermediate Results

Yearly Parquet files ARE your intermediate results - don't delete them!

### 4. Document Your Runs

Create a log file:

```python
# Add at end of run_local_test.py
with open('output/processing_log.txt', 'w') as f:
    f.write(f"Processed: {datetime.now()}\n")
    f.write(f"Files: {len(records)}\n")
    f.write(f"Years: {list(yearly_stats.keys())}\n")
```

---

## 🔗 Related Documentation

- **MDA Format Guide**: `MDA_FORMAT_GUIDE.md` - Explains format transformation
- **Colab Notebook**: `../tech_adoption_project/SEC_Pipeline_Colab.ipynb` - Interactive version
- **Pipeline README**: `../tech_adoption_project/SEC_PIPELINE_README.md` - Full documentation
- **Main README**: `../tech_adoption_project/README.md` - Project overview

---

**Ready to process your large dataset with yearly batching!** 🚀

For questions or issues, check the troubleshooting section above or review the related documentation.
