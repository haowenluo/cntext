"""
Local Pipeline Test for SEC MDA Analysis - Batch Processing Version

This script:
1. Reads MDA JSON files from the sample folder
2. Groups them by fiscal year
3. Processes each year separately → yearly Parquet files
4. Creates a combined labeling sample from all years

Usage:
    conda activate <your_env_name>
    cd F:\github\cntext\local_pipeline_test
    python run_local_test.py

For large-scale processing (100k+ files), this approach:
- Prevents memory crashes by processing one year at a time
- Enables parallel processing of different years
- Provides fault tolerance (if one year fails, others are saved)
- Makes incremental updates easy (just process new year)
"""

import sys
import json
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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

# Output format: 'parquet' (recommended), 'csv', or 'both'
OUTPUT_FORMAT = 'parquet'

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


def collect_and_group_mda_files_by_year(sample_dir):
    """
    Find all MDA JSON files and group them by fiscal year.
    
    Returns:
        dict: {year: [list of transformed records]}
    """
    records_by_year = defaultdict(list)
    json_files = list(Path(sample_dir).rglob("*.json"))
    
    # Filter out non-MDA files
    json_files = [f for f in json_files if f.name != 'sampling_log.csv']
    
    print(f"Found {len(json_files)} MDA JSON files")
    
    for filepath in json_files:
        try:
            record = transform_mda_to_pipeline_format(filepath)
            if record['item_text']:  # Skip empty MD&A
                year = record['fiscal_year']
                records_by_year[year].append(record)
                print(f"  ✓ {filepath.name} → Year {year}")
            else:
                print(f"  ⚠ {filepath.name} - Empty item_7, skipped")
        except Exception as e:
            print(f"  ✗ Error processing {filepath.name}: {e}")
    
    return dict(records_by_year)


# ============================================================================
# MAIN PIPELINE - BATCH PROCESSING BY YEAR
# ============================================================================

def main():
    print("=" * 80)
    print("LOCAL PIPELINE TEST - BATCH PROCESSING BY YEAR")
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
    # STEP 2: Group MDA files by year
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: GROUPING MDA FILES BY YEAR")
    print("=" * 80)
    
    records_by_year = collect_and_group_mda_files_by_year(MDA_SAMPLE_DIR)
    
    if not records_by_year:
        print("✗ No records found! Check MDA_SAMPLE_DIR path.")
        return
    
    print(f"\n✓ Grouped into {len(records_by_year)} years:")
    for year in sorted(records_by_year.keys()):
        print(f"  {year}: {len(records_by_year[year])} files")
    
    # ========================================================================
    # STEP 3: Process each year separately
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: PROCESSING EACH YEAR (BATCH MODE)")
    print("=" * 80)
    
    # Import pipeline modules
    sys.path.insert(0, str(PIPELINE_DIR))
    from build_sentence_table import build_sentence_table
    
    all_sentence_dfs = []
    year_stats = {}
    
    for year in sorted(records_by_year.keys()):
        print(f"\n{'─' * 60}")
        print(f"Processing Year {year} ({len(records_by_year[year])} files)")
        print(f"{'─' * 60}")
        
        # Save year's records to temp JSON
        year_json_path = OUTPUT_DIR / f"temp_items_{year}.json"
        with open(year_json_path, 'w', encoding='utf-8') as f:
            json.dump(records_by_year[year], f, indent=2, ensure_ascii=False)
        
        # Process this year
        year_output_path = OUTPUT_DIR / f"sentences_{year}.parquet"
        
        try:
            sentence_df = build_sentence_table(
                input_path=str(year_json_path),
                output_path=str(year_output_path),
                output_format=OUTPUT_FORMAT
            )
            
            all_sentence_dfs.append(sentence_df)
            year_stats[year] = len(sentence_df)
            print(f"✓ Year {year}: {len(sentence_df)} sentences → sentences_{year}.parquet")
            
        except Exception as e:
            print(f"✗ Error processing year {year}: {e}")
            continue
        
        finally:
            # Clean up temp file
            if year_json_path.exists():
                year_json_path.unlink()
    
    # ========================================================================
    # STEP 4: Create combined labeling sample
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: CREATING COMBINED LABELING SAMPLE")
    print("=" * 80)
    
    import pandas as pd
    from build_labeling_sample import build_labeling_sample
    import warnings
    warnings.filterwarnings('ignore')
    
    from tqdm import tqdm
    tqdm.pandas()
    
    # Combine all years for sampling
    if all_sentence_dfs:
        combined_df = pd.concat(all_sentence_dfs, ignore_index=True)
        combined_csv_path = OUTPUT_DIR / "all_sentences_combined.csv"
        combined_df.to_csv(combined_csv_path, index=False)
        
        print(f"✓ Combined all years: {len(combined_df)} total sentences")
        
        # Copy tech_keywords.yaml to output
        import shutil
        keywords_src = PIPELINE_DIR / "tech_keywords.yaml"
        keywords_dst = OUTPUT_DIR / "tech_keywords.yaml"
        if keywords_src.exists() and not keywords_dst.exists():
            shutil.copy(keywords_src, keywords_dst)
        
        # Create balanced labeling sample
        label_set_path = OUTPUT_DIR / "label_set.csv"
        sample_size = min(500, len(combined_df))
        
        label_df = build_labeling_sample(
            input_path=str(combined_csv_path),
            output_path=str(label_set_path),
            keywords_file=str(keywords_dst),
            sample_size=sample_size
        )
        
        print(f"✓ Label set created: {len(label_df)} sentences")
        
        # Clean up combined CSV (keep parquet files)
        if combined_csv_path.exists():
            combined_csv_path.unlink()
    
    # ========================================================================
    # STEP 5: Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: SUMMARY")
    print("=" * 80)
    
    print(f"\nOutput files in: {OUTPUT_DIR}")
    print(f"\nYearly Parquet files:")
    total_sentences = 0
    for year in sorted(year_stats.keys()):
        print(f"  sentences_{year}.parquet: {year_stats[year]:,} sentences")
        total_sentences += year_stats[year]
    
    print(f"\n  Total: {total_sentences:,} sentences across {len(year_stats)} years")
    print(f"\nLabeling sample:")
    print(f"  label_set.csv: {len(label_df) if 'label_df' in dir() else 0} sentences")
    
    # Show sample results
    if 'label_df' in dir() and len(label_df) > 0:
        print(f"\nSample sentences from label set:")
        print("-" * 80)
        
        for idx, row in label_df.head(3).iterrows():
            print(f"\n[{idx+1}] CIK: {row['cik']} | Year: {row['fiscal_year']}")
            print(f"    Tech hit: {row['tech_hit']} | Pool: {row['source_pool']}")
            text = row['sentence_text'][:120] + "..." if len(row['sentence_text']) > 120 else row['sentence_text']
            print(f"    {text}")
    
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE! ✓")
    print("=" * 80)
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"""
Benefits of this approach for large-scale processing:
  ✓ Memory safe: One year at a time
  ✓ Fault tolerant: If one year fails, others are saved
  ✓ Parallelizable: Run different years on different machines
  ✓ Incremental: Easy to add new years later
  ✓ Compact: Parquet files are 3-4x smaller than CSV
    """)


if __name__ == '__main__':
    main()
