"""
Local Pipeline Test for SEC MDA Analysis

This script:
1. Reads MDA JSON files from the sample folder
2. Transforms them to pipeline-expected format
3. Runs build_sentence_table.py and build_labeling_sample.py
4. Outputs results to local_pipeline_test/output/

Usage:
    conda activate <your_env_name>
    cd F:\github\cntext\local_pipeline_test
    python run_local_test.py
"""

import sys
import json
import glob
from pathlib import Path
from datetime import datetime

# Add cntext to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input: Your MDA sample files
MDA_SAMPLE_DIR = REPO_ROOT / "random_samples_2025-12-16_10-09-22"

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"

# Pipeline scripts location
PIPELINE_DIR = REPO_ROOT / "tech_adoption_project"

# ============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================

def transform_mda_to_pipeline_format(mda_json_path):
    """
    Transform a single MDA JSON file to pipeline format.
    
    Your MDA format:
        cik, company, filing_type, filing_date, period_of_report, filename, item_7
    
    Pipeline format:
        cik, accession, fiscal_year, filing_date, item, item_text
    """
    with open(mda_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract accession from filename
    # e.g., "1643988_10K_2020_0001387131-21-004517.htm" -> "0001387131-21-004517"
    filename = data.get('filename', '')
    parts = filename.replace('.htm', '').replace('.json', '').split('_')
    accession = parts[-1] if len(parts) >= 4 else 'unknown'
    
    # Extract fiscal year from period_of_report
    # e.g., "2020-12-31" -> 2020
    period = data.get('period_of_report', '')
    fiscal_year = int(period[:4]) if period and len(period) >= 4 else 0
    
    # Get CIK as integer
    cik = data.get('cik', '0')
    cik = int(cik) if isinstance(cik, str) and cik.isdigit() else cik
    
    return {
        'cik': cik,
        'accession': accession,
        'fiscal_year': fiscal_year,
        'filing_date': data.get('filing_date', ''),
        'item': '7',  # MD&A is always Item 7
        'item_text': data.get('item_7', '')
    }


def collect_and_transform_mda_files(sample_dir):
    """
    Find all MDA JSON files and transform them to pipeline format.
    """
    all_records = []
    json_files = list(Path(sample_dir).rglob("*.json"))
    
    # Filter out non-MDA files (like sampling_log.csv, etc.)
    json_files = [f for f in json_files if f.name != 'sampling_log.csv']
    
    print(f"Found {len(json_files)} MDA JSON files")
    
    for filepath in json_files:
        try:
            record = transform_mda_to_pipeline_format(filepath)
            if record['item_text']:  # Skip empty MD&A
                all_records.append(record)
                print(f"  ✓ {filepath.name} (CIK: {record['cik']}, Year: {record['fiscal_year']})")
            else:
                print(f"  ⚠ {filepath.name} - Empty item_7, skipped")
        except Exception as e:
            print(f"  ✗ Error processing {filepath.name}: {e}")
    
    return all_records


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("LOCAL PIPELINE TEST FOR SEC MDA ANALYSIS")
    print("=" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # STEP 1: Create output directory
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: SETUP")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"✓ Output directory: {OUTPUT_DIR}")
    
    # ========================================================================
    # STEP 2: Transform MDA files
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: TRANSFORMING MDA FILES TO PIPELINE FORMAT")
    print("=" * 80)
    
    records = collect_and_transform_mda_files(MDA_SAMPLE_DIR)
    
    if not records:
        print("✗ No records found! Check MDA_SAMPLE_DIR path.")
        return
    
    # Save combined file
    combined_json_path = OUTPUT_DIR / "combined_items.json"
    with open(combined_json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Created {combined_json_path}")
    print(f"  Total records: {len(records)}")
    
    # ========================================================================
    # STEP 3: Run sentence table builder
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: BUILDING SENTENCE TABLE")
    print("=" * 80)
    
    # Import pipeline module
    sys.path.insert(0, str(PIPELINE_DIR))
    from build_sentence_table import build_sentence_table
    
    sentence_table_path = OUTPUT_DIR / "sentence_table.csv"
    sentence_df = build_sentence_table(
        input_path=str(combined_json_path),
        output_path=str(sentence_table_path),
        output_format='both'
    )
    
    print(f"\n✓ Sentence table created: {len(sentence_df)} sentences")
    
    # ========================================================================
    # STEP 4: Run labeling sample builder
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: BUILDING LABELING SAMPLE")
    print("=" * 80)
    
    from build_labeling_sample import build_labeling_sample
    import warnings
    warnings.filterwarnings('ignore')
    
    from tqdm import tqdm
    tqdm.pandas()
    
    # Copy tech_keywords.yaml to output if needed
    keywords_src = PIPELINE_DIR / "tech_keywords.yaml"
    keywords_dst = OUTPUT_DIR / "tech_keywords.yaml"
    if keywords_src.exists() and not keywords_dst.exists():
        import shutil
        shutil.copy(keywords_src, keywords_dst)
    
    label_set_path = OUTPUT_DIR / "label_set.csv"
    sample_size = min(500, len(sentence_df))  # Sample up to 500 or all available
    
    label_df = build_labeling_sample(
        input_path=str(sentence_table_path),
        output_path=str(label_set_path),
        keywords_file=str(keywords_dst),
        sample_size=sample_size
    )
    
    print(f"\n✓ Label set created: {len(label_df)} sentences")
    
    # ========================================================================
    # STEP 5: Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: SUMMARY")
    print("=" * 80)
    
    print(f"\nOutput files in: {OUTPUT_DIR}")
    print(f"  - combined_items.json: {len(records)} MDA records")
    print(f"  - sentence_table.csv: {len(sentence_df)} sentences")
    print(f"  - sentence_table.parquet: {len(sentence_df)} sentences")
    print(f"  - label_set.csv: {len(label_df)} sentences for labeling")
    
    # Show sample results
    print(f"\nSample sentences from label set:")
    print("-" * 80)
    
    for idx, row in label_df.head(5).iterrows():
        print(f"\n[{idx+1}] CIK: {row['cik']} | Year: {row['fiscal_year']} | Item: {row['item']}")
        print(f"    Tech hit: {row['tech_hit']} | Pool: {row['source_pool']}")
        text = row['sentence_text'][:150] + "..." if len(row['sentence_text']) > 150 else row['sentence_text']
        print(f"    {text}")
    
    print("\n" + "=" * 80)
    print("LOCAL PIPELINE TEST COMPLETE! ✓")
    print("=" * 80)
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
