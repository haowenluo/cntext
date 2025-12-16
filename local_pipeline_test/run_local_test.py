"""
Local Pipeline Test for SEC MDA Analysis - Yearly Batch Processing

This script:
1. Reads MDA JSON files from the sample folder
2. Transforms them to pipeline-expected format
3. Groups by fiscal year for scalable processing
4. Processes each year → separate Parquet file
5. Creates a combined labeling sample across all years

Scalability benefits:
- Memory efficient: Process one year at a time
- Fault tolerant: Crash only loses current year
- Large dataset ready: Handles 100k+ files → millions of sentences

Usage:
    conda activate <your_env_name>
    cd local_pipeline_test
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


def collect_and_group_by_year(sample_dir):
    """
    Find all MDA JSON files, transform them, and group by fiscal year.

    Returns:
        dict: {fiscal_year: [records]}
    """
    records_by_year = defaultdict(list)
    json_files = list(Path(sample_dir).rglob("*.json"))

    # Filter out non-MDA files
    json_files = [f for f in json_files if f.name not in ['sampling_log.csv', 'sampling_summary.txt']]

    print(f"Found {len(json_files)} MDA JSON files")

    transformed_count = 0
    skipped_count = 0

    for filepath in json_files:
        try:
            record = transform_mda_to_pipeline_format(filepath)
            if record['item_text']:  # Skip empty MD&A
                records_by_year[record['fiscal_year']].append(record)
                transformed_count += 1
                if transformed_count <= 5:  # Show first 5
                    print(f"  ✓ {filepath.name} (CIK: {record['cik']}, Year: {record['fiscal_year']})")
            else:
                print(f"  ⚠ {filepath.name} - Empty item_7, skipped")
                skipped_count += 1
        except Exception as e:
            print(f"  ✗ Error processing {filepath.name}: {e}")
            skipped_count += 1

    if transformed_count > 5:
        print(f"  ... ({transformed_count - 5} more files processed)")

    print(f"\n✓ Transformed: {transformed_count} files")
    print(f"  Skipped: {skipped_count} files")
    print(f"  Years covered: {sorted(records_by_year.keys())}")

    return records_by_year


# ============================================================================
# MAIN PIPELINE - BATCH PROCESSING BY YEAR
# ============================================================================

def main():
    print("=" * 80)
    print("LOCAL PIPELINE TEST - YEARLY BATCH PROCESSING")
    print("=" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================================================================
    # STEP 1: Create output directory
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: SETUP")
    print("=" * 80)

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    yearly_dir = OUTPUT_DIR / "yearly_parquet"
    yearly_dir.mkdir(exist_ok=True, parents=True)

    print(f"✓ Output directory: {OUTPUT_DIR}")
    print(f"✓ Yearly Parquet directory: {yearly_dir}")

    # ========================================================================
    # STEP 2: Transform and group MDA files by year
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: TRANSFORMING MDA FILES (GROUPED BY YEAR)")
    print("=" * 80)

    records_by_year = collect_and_group_by_year(MDA_SAMPLE_DIR)

    if not records_by_year:
        print("✗ No records found! Check MDA_SAMPLE_DIR path.")
        return

    # ========================================================================
    # STEP 3: Process each year separately
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: PROCESSING EACH YEAR → PARQUET FILES")
    print("=" * 80)

    # Import pipeline module
    sys.path.insert(0, str(PIPELINE_DIR))
    from build_sentence_table import build_sentence_table

    yearly_stats = {}
    all_sentence_tables = []

    for year in sorted(records_by_year.keys()):
        records = records_by_year[year]
        print(f"\n--- Processing Year {year} ({len(records)} filings) ---")

        # Save year's records to temporary JSON
        year_json_path = OUTPUT_DIR / f"items_{year}.json"
        with open(year_json_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        # Process year's sentences
        year_parquet_path = yearly_dir / f"sentences_{year}.parquet"

        try:
            sentence_df = build_sentence_table(
                input_path=str(year_json_path),
                output_path=str(year_parquet_path),
                output_format='parquet'  # Only save Parquet for yearly files
            )

            yearly_stats[year] = {
                'filings': len(records),
                'sentences': len(sentence_df)
            }

            all_sentence_tables.append(sentence_df)

            print(f"  ✓ {len(sentence_df):,} sentences → {year_parquet_path.name}")

            # Clean up temporary JSON
            year_json_path.unlink()

        except Exception as e:
            print(f"  ✗ Error processing year {year}: {e}")
            continue

    if not all_sentence_tables:
        print("\n✗ No sentence tables generated!")
        return

    # ========================================================================
    # STEP 4: Combine all years for labeling sample
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: CREATING COMBINED LABELING SAMPLE")
    print("=" * 80)

    import pandas as pd
    combined_df = pd.concat(all_sentence_tables, ignore_index=True)

    # Save combined table for reference
    combined_csv_path = OUTPUT_DIR / "sentence_table_combined.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"✓ Combined sentence table: {len(combined_df):,} sentences")
    print(f"  Saved to: {combined_csv_path}")

    # Build labeling sample from combined data
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

    label_set_path = OUTPUT_DIR / "label_set_combined.csv"
    sample_size = min(500, len(combined_df))  # Sample up to 500 or all available

    label_df = build_labeling_sample(
        input_path=str(combined_csv_path),
        output_path=str(label_set_path),
        keywords_file=str(keywords_dst),
        sample_size=sample_size
    )

    print(f"\n✓ Labeling sample: {len(label_df):,} sentences")
    print(f"  Sampled across {len(records_by_year)} years")

    # ========================================================================
    # STEP 5: Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n📁 Yearly Parquet Files ({yearly_dir}):")
    for year in sorted(yearly_stats.keys()):
        stats = yearly_stats[year]
        print(f"  - sentences_{year}.parquet: {stats['sentences']:,} sentences from {stats['filings']} filings")

    print(f"\n📊 Combined Outputs ({OUTPUT_DIR}):")
    print(f"  - sentence_table_combined.csv: {len(combined_df):,} sentences total")
    print(f"  - label_set_combined.csv: {len(label_df):,} sentences for labeling")

    print(f"\n📈 Statistics:")
    print(f"  Total filings processed: {sum(s['filings'] for s in yearly_stats.values())}")
    print(f"  Total sentences extracted: {len(combined_df):,}")
    print(f"  Years covered: {min(yearly_stats.keys())} - {max(yearly_stats.keys())}")
    print(f"  Avg sentences per filing: {len(combined_df) / sum(s['filings'] for s in yearly_stats.values()):.1f}")

    # Show sample from label set
    print(f"\n📝 Sample sentences from labeling set:")
    print("-" * 80)

    for idx, row in label_df.head(3).iterrows():
        print(f"\n[{idx+1}] CIK: {row['cik']} | Year: {row['fiscal_year']} | Tech hit: {row['tech_hit']}")
        text = row['sentence_text'][:150] + "..." if len(row['sentence_text']) > 150 else row['sentence_text']
        print(f"    {text}")

    print("\n" + "=" * 80)
    print("✓ YEARLY BATCH PROCESSING COMPLETE!")
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

    print(f"\n💡 Scalability Notes:")
    print(f"  - Each year processed separately → memory efficient")
    print(f"  - Yearly Parquet files → easy to load specific years")
    print(f"  - Fault tolerant → crash only loses current year")
    print(f"  - Ready for 100k+ filings → millions of sentences")


if __name__ == '__main__':
    main()
