# MDA JSON Format Guide

This document explains the format differences between your MDA extraction files and what the SEC pipeline expects.

## Your MDA JSON Format

```json
{
    "cik": "1643988",
    "company": "Company_1643988",
    "filing_type": "10-K",
    "filing_date": "2020-01-01",
    "period_of_report": "2020-12-31",
    "filename": "1643988_10K_2020_0001387131-21-004517.htm",
    "item_7": "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS..."
}
```

## Pipeline Expected Format

```json
{
    "cik": 1643988,
    "accession": "0001387131-21-004517",
    "fiscal_year": 2020,
    "filing_date": "2021-02-15",
    "item": "7",
    "item_text": "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS..."
}
```

## Field Mapping

| Your Field | Pipeline Field | Transformation |
|------------|----------------|----------------|
| `cik` | `cik` | Convert string to int |
| `filing_date` | `filing_date` | Direct copy |
| `period_of_report` | `fiscal_year` | Extract year: `"2020-12-31"` → `2020` |
| `filename` | `accession` | Extract last part: `"..._0001387131-21-004517.htm"` → `"0001387131-21-004517"` |
| `item_7` | `item_text` | Rename field |
| — | `item` | Add constant: `"7"` |

## Using in Google Colab

Add this preprocessing cell after environment setup:

```python
import json
import glob

def transform_mda_to_pipeline_format(mda_json_path):
    with open(mda_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filename = data.get('filename', '')
    parts = filename.replace('.htm', '').split('_')
    accession = parts[-1] if len(parts) >= 4 else 'unknown'
    
    period = data.get('period_of_report', '')
    fiscal_year = int(period[:4]) if period and len(period) >= 4 else 0
    
    return {
        'cik': int(data['cik']) if data.get('cik') else 0,
        'accession': accession,
        'fiscal_year': fiscal_year,
        'filing_date': data.get('filing_date', ''),
        'item': '7',
        'item_text': data.get('item_7', '')
    }

# Process all your MDA JSON files
all_records = []
for filepath in glob.glob(f'{BASE_PATH}/extracted_items/**/*.json', recursive=True):
    record = transform_mda_to_pipeline_format(filepath)
    if record['item_text']:
        all_records.append(record)

# Save combined file for pipeline
with open(f'{BASE_PATH}/extracted_items/items_combined.json', 'w') as f:
    json.dump(all_records, f)

INPUT_FILENAME = 'items_combined.json'
```
