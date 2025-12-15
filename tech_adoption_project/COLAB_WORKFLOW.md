# Google Colab Workflow for SEC 10-K Sentence Labeling

Quick reference guide for running the SEC pipeline in Google Colab.

---

## 🚀 **Quick Start (3 Steps)**

### **1. Setup (One-time, ~5 minutes)**

Open the Colab notebook: `SEC_Pipeline_Colab.ipynb`

Run these cells:
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install -q pandas spacy ftfy contractions
!python -m spacy download en_core_web_sm

# Clone repo
!git clone https://github.com/haowenluo/cntext.git /content/cntext
import sys
sys.path.insert(0, '/content/cntext')
```

### **2. Process Data (~10-30 minutes depending on size)**

```python
# Build sentence table
from build_sentence_table import build_sentence_table

sentence_df = build_sentence_table(
    input_path='/content/drive/MyDrive/sec_10k_project/extracted_items/items.json',
    output_path='/content/drive/MyDrive/sec_10k_project/sentence_tables/sentences.csv'
)

# Create labeling sample
from build_labeling_sample import build_labeling_sample
from tqdm import tqdm
tqdm.pandas()

label_df = build_labeling_sample(
    input_path='/content/drive/MyDrive/sec_10k_project/sentence_tables/sentences.csv',
    output_path='/content/drive/MyDrive/sec_10k_project/labeling_samples/label_set.csv',
    sample_size=2500
)
```

### **3. Label in Google Sheets (Hours to days)**

1. Navigate to: `sec_10k_project/labeling_samples/` in Google Drive
2. Right-click `label_set.csv` → Open with Google Sheets
3. Label each sentence with ONE category:
   - `TECH_IMPL` = Implementation/usage
   - `TECH_ADOPT` = Adoption/investment
   - `TECH_PRODUCT` = Product/offering
   - `NON_TECH` = Not tech-related
4. File auto-saves to Drive

---

## 📂 **Drive Folder Structure**

```
/content/drive/MyDrive/sec_10k_project/
├── extracted_items/          # [INPUT] Your Item extractions (JSON/CSV)
│   └── items_2020.json
├── sentence_tables/          # [OUTPUT] Sentence-level tables
│   ├── sentence_table_2020.csv
│   └── sentence_table_2020.parquet
├── labeling_samples/         # [OUTPUT] Samples for labeling
│   ├── label_set_2020.csv
│   └── label_set_2020.txt (summary)
└── labeled_data/             # [INPUT] After labeling
    └── labeled_set_2020.csv
```

---

## 📋 **Input Format**

Your Item extractions should be JSON or CSV with these columns:

```json
{
  "cik": 1234567,
  "accession": "0000000000-20-000001",
  "fiscal_year": 2020,
  "filing_date": "2021-02-15",
  "item": "1",
  "item_text": "We develop innovative software solutions using artificial intelligence and cloud computing..."
}
```

---

## ⚙️ **Configuration Options**

### Sentence Table Builder

Edit in notebook cell before running:

```python
# Filter settings (in build_sentence_table.py)
MIN_SENTENCE_LENGTH = 20          # Characters
MAX_SENTENCE_LENGTH = 2000        # Filter OCR errors
MIN_WORD_COUNT = 3                # Words
MAX_NUMERIC_RATIO = 0.6           # Filter tables
```

### Labeling Sample Builder

```python
# Sampling settings
SAMPLE_SIZE = 2500                # Total sentences
TECH_HIT_RATIO = 0.5              # 50% tech, 50% random
RANDOM_SEED = 42                  # Reproducibility
```

### Tech Keywords

Customize `tech_keywords.yaml`:

```python
import yaml
with open('/content/tech_keywords.yaml', 'r') as f:
    keywords = yaml.safe_load(f)

# Add custom terms
keywords['Dictionary']['fintech'] = [
    'digital wallet',
    'payment processing',
    'peer-to-peer lending'
]

with open('/content/tech_keywords.yaml', 'w') as f:
    yaml.dump(keywords, f)
```

---

## 🔍 **Quality Checks**

### After Sentence Table

```python
import pandas as pd
df = pd.read_csv('sentence_table.csv')

print(f"Total sentences: {len(df):,}")
print(f"Sentences by Item:\n{df['item'].value_counts()}")
print(f"Avg length: {df['sentence_text'].str.len().mean():.0f} chars")
```

### After Labeling Sample

```python
label_df = pd.read_csv('label_set.csv')

print(f"Tech hits: {(label_df['tech_hit']==True).sum()}")
print(f"Random: {(label_df['source_pool']=='random').sum()}")
```

### After Labeling

```python
labeled_df = pd.read_csv('labeled_set.csv')

# Check distribution
for col in ['TECH_IMPL', 'TECH_ADOPT', 'TECH_PRODUCT', 'NON_TECH']:
    print(f"{col}: {(labeled_df[col]==1).sum()}")

# Validate
label_sum = labeled_df[['TECH_IMPL','TECH_ADOPT','TECH_PRODUCT','NON_TECH']].sum(axis=1)
print(f"Valid (exactly 1 label): {(label_sum==1).sum()}")
print(f"Invalid (0 or >1 labels): {(label_sum!=1).sum()}")
```

---

## 💡 **Tips for Large Datasets**

### Memory Management

```python
# Use Parquet for large files
sentence_df = pd.read_parquet('sentence_table.parquet')

# Or process in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)
```

### Batch Processing Multiple Years

```python
years = [2018, 2019, 2020, 2021, 2022]

for year in years:
    input_file = f'items_{year}.json'
    output_file = f'sentences_{year}.csv'

    build_sentence_table(input_file, output_file)

# Combine all years
all_sentences = pd.concat([
    pd.read_csv(f'sentences_{year}.csv')
    for year in years
])
```

### Stratified Sampling

```python
# Balance across years
STRATIFY_BY_YEAR = True  # Edit in build_labeling_sample.py

# Or balance across Items
STRATIFY_BY_ITEM = True
```

---

## 🐛 **Troubleshooting**

### Issue: "No module named 'cntext'"

**Solution:**
```python
import sys
sys.path.insert(0, '/content/cntext')
```

### Issue: "spaCy model not found"

**Solution:**
```python
!python -m spacy download en_core_web_sm
```

### Issue: "Not enough tech sentences"

**Solutions:**
1. Reduce sample size: `sample_size=1500`
2. Add more keywords to `tech_keywords.yaml`
3. Lower tech ratio: `TECH_HIT_RATIO = 0.3`

### Issue: Runtime disconnects

**Solutions:**
1. Save checkpoints to Drive frequently
2. Use Colab Pro for longer runtime
3. Process in smaller batches

### Issue: "Duplicate sentences removed"

**This is normal!** Deduplication prevents labeling the same sentence twice.
Check summary file to see how many were removed.

---

## 📊 **Expected Sizes**

| Stage | Input | Output | Time |
|-------|-------|--------|------|
| Sentence table | 1000 filings (JSON) | ~150K sentences (50MB CSV) | 5-15 min |
| Labeling sample | 150K sentences | 2500 for labeling (2MB) | 2-5 min |
| Labeling | 2500 sentences | Manual work | Hours-days |

---

## 🎯 **Best Practices**

### 1. Test with Small Sample First
```python
# Use just 10 filings to test
test_df = all_items.head(10)
test_df.to_json('test_items.json')
# Then run pipeline
```

### 2. Save Intermediate Results
```python
# Always save to Drive, not /content/
OUTPUT_PATH = '/content/drive/MyDrive/...'  # ✓ Good
OUTPUT_PATH = '/content/...'                 # ✗ Lost on disconnect
```

### 3. Version Your Outputs
```python
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
output_file = f'label_set_{timestamp}.csv'
```

### 4. Document Your Decisions
```python
# Create metadata file
metadata = {
    'date': '2024-01-15',
    'input_files': ['items_2020.json', 'items_2021.json'],
    'sample_size': 2500,
    'tech_ratio': 0.5,
    'keywords_version': 'v1.0',
    'notes': 'Added fintech keywords'
}
import json
with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

## 🔄 **Typical Session Flow**

```python
# === Session Start ===

# 1. Mount & setup (run every time)
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/haowenluo/cntext.git /content/cntext
import sys; sys.path.insert(0, '/content/cntext')

# 2. Process data (run once per dataset)
from build_sentence_table import build_sentence_table
sentence_df = build_sentence_table(...)

from build_labeling_sample import build_labeling_sample
label_df = build_labeling_sample(...)

# 3. Download or open in Sheets
# (Label offline)

# 4. Validate (next session)
labeled_df = pd.read_csv('.../labeled_set.csv')
# Run quality checks

# === Session End ===
```

---

## 📚 **Additional Resources**

- Full documentation: `SEC_PIPELINE_README.md`
- Script details: Comments in `build_sentence_table.py`, `build_labeling_sample.py`
- Keyword customization: Edit `tech_keywords.yaml`
- Test pipeline: Run `test_pipeline.py` locally

---

## ✅ **Checklist**

Before starting:
- [ ] Item extractions ready in JSON/CSV format
- [ ] Google Drive folder structure created
- [ ] Colab notebook opened
- [ ] Dependencies installed

Before labeling:
- [ ] Sentence table looks correct (spot check)
- [ ] Tech_hit flag makes sense (review samples)
- [ ] Sample size is appropriate (2000-3000)
- [ ] No duplicate sentences (check summary)

After labeling:
- [ ] All sentences labeled (no blanks)
- [ ] No multi-label rows (exactly 1 per sentence)
- [ ] Distribution looks reasonable (not 99% one class)
- [ ] Saved to Drive in labeled_data folder

---

**Ready to start? Open `SEC_Pipeline_Colab.ipynb` and follow the cells!** 🚀
