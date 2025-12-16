# Tech Adoption Project

This folder contains a complete pipeline for processing SEC 10-K Item extractions into sentence-level datasets for technology adoption research.

## 📁 Files in This Folder

| File | Purpose |
|------|---------|
| `build_sentence_table.py` | Convert Item extractions → sentence-level table |
| `build_labeling_sample.py` | Create balanced sample for manual labeling |
| `tech_keywords.yaml` | Technology keyword dictionary (300+ terms) |
| `test_pipeline.py` | Test pipeline with synthetic data |
| `SEC_PIPELINE_README.md` | Complete documentation and usage guide |
| `SEC_Pipeline_Colab.ipynb` | Google Colab notebook (interactive) **w/ MDA auto-conversion** |
| `COLAB_WORKFLOW.md` | Quick reference for Colab usage |
| `COLAB_UPDATE_SUMMARY.md` | Latest updates to Colab notebook (MDA format support) |

## 🚀 Quick Start

### For Google Colab Users (Recommended)
1. Upload `SEC_Pipeline_Colab.ipynb` to your Google Drive
2. Open in Google Colab
3. Follow the notebook cells step-by-step
4. **NEW:** Notebook automatically handles MDA format conversion!

### For Local/Server Users

**Small Dataset (< 1000 filings):**
1. Install dependencies:
   ```bash
   pip install pandas numpy tqdm pyyaml spacy ftfy contractions pyarrow
   python -m spacy download en_core_web_sm
   ```

2. Build sentence table:
   ```bash
   cd tech_adoption_project
   python build_sentence_table.py --input items.json --output sentences.csv
   ```

3. Create labeling sample:
   ```bash
   python build_labeling_sample.py --input sentences.csv --output label_set.csv
   ```

**Large Dataset (1000+ filings) - Yearly Batch Processing:**
1. Use the local test script with yearly batching:
   ```bash
   cd local_pipeline_test
   python run_local_test.py
   ```

   This will:
   - Group files by fiscal year
   - Create `yearly_parquet/sentences_YYYY.parquet` for each year
   - Generate combined labeling sample across all years
   - See `local_pipeline_test/README.md` for details

## 📖 Documentation

- **Complete Guide**: See `SEC_PIPELINE_README.md` for full documentation
- **Colab Guide**: See `COLAB_WORKFLOW.md` for Google Colab workflow
- **Test Pipeline**: Run `python test_pipeline.py` to verify installation

## 🎯 Workflow Overview

### Standard Workflow (< 1000 filings):
```
Item Extractions (JSON/CSV)
         ↓
  build_sentence_table.py
         ↓
  sentence_table.csv
         ↓
  build_labeling_sample.py
         ↓
  label_set.csv (2500 sentences)
         ↓
  Manual Labeling
         ↓
  labeled_data.csv
```

### Yearly Batch Workflow (1000+ filings) - **NEW!**:
```
MDA Files (grouped by year)
         ↓
  Process Year 2020 → sentences_2020.parquet
  Process Year 2021 → sentences_2021.parquet
  Process Year 2022 → sentences_2022.parquet
         ↓
  Combine all years → sentence_table_all_years.csv
         ↓
  build_labeling_sample.py
         ↓
  label_set_combined.csv (2500 sentences)
         ↓
  Manual Labeling
```

**Scalability Benefits:**
- ✅ Memory efficient: Process one year at a time
- ✅ Fault tolerant: Crash only loses current year
- ✅ Handles 100k+ filings → millions of sentences
- ✅ Load specific years for targeted analysis

## 📊 Input Format

**Two formats supported** (Colab notebook auto-detects and converts):

### Pipeline Format (ready to use):
```json
{
  "cik": 1234567,
  "accession": "0000000000-20-000001",
  "fiscal_year": 2020,
  "filing_date": "2021-02-15",
  "item": "1",
  "item_text": "Your extracted text here..."
}
```

### MDA Format (auto-converted in Colab):
```json
{
  "cik": "1643988",
  "company": "Company_1643988",
  "filing_date": "2020-01-01",
  "period_of_report": "2020-12-31",
  "filename": "1643988_10K_2020_0001387131-21-004517.htm",
  "item_7": "ITEM 7. MANAGEMENT'S DISCUSSION..."
}
```

**Note:** For local usage, see `local_pipeline_test/MDA_FORMAT_GUIDE.md` for transformation details.

## 🛠️ Integration with cntext Repository

This pipeline leverages the parent cntext repository for:
- Text cleaning: `ct.fix_text()`, `ct.fix_contractions()`, `ct.clean_text()`
- Sentence splitting: `english_nlp.py` module
- Dictionary utilities: `ct.read_yaml_dict()`

To use in scripts:
```python
import sys
sys.path.insert(0, '/path/to/cntext')
import cntext as ct
```

## 📝 Label Schema

The pipeline creates labels for:
- **TECH_IMPL**: Technology implementation/usage
- **TECH_ADOPT**: Technology adoption/investment
- **TECH_PRODUCT**: Technology product/offering
- **NON_TECH**: Not technology-related

## ⚙️ Customization

1. **Add Keywords**: Edit `tech_keywords.yaml`
2. **Adjust Filters**: Modify constants in `build_sentence_table.py`
3. **Change Sample Size**: Use `--size` parameter in `build_labeling_sample.py`
4. **Stratify Sampling**: Edit `STRATIFY_BY_YEAR` or `STRATIFY_BY_ITEM` flags

## 🔗 Related Files in Parent Repository

This project uses:
- `/cntext/io/utils.py` - Text cleaning functions
- `/cntext/english_nlp.py` - English NLP utilities
- `/cntext/stats/utils.py` - Sentence splitting
- `/cntext/io/dict.py` - Dictionary loading

## 📧 Support

For detailed instructions, troubleshooting, and examples:
- Read `SEC_PIPELINE_README.md` (comprehensive guide)
- Read `COLAB_WORKFLOW.md` (Colab-specific guide)
- Check the notebook `SEC_Pipeline_Colab.ipynb`

---

**Ready to start?** Choose your platform:
- 💻 **Colab**: Open `SEC_Pipeline_Colab.ipynb`
- 🖥️ **Local**: Follow `SEC_PIPELINE_README.md`
