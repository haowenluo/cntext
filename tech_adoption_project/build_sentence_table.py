"""
Build Sentence Table from SEC 10-K Item Extractions

Takes extracted 10-K Items (1, 1A, 7) and creates a sentence-level dataset
ready for sampling and labeling.

Input Format (JSON or CSV):
    - cik: Company CIK number
    - accession: SEC accession number
    - fiscal_year: Fiscal year
    - filing_date: Filing date
    - item: Item number (1, 1A, or 7)
    - item_text: Extracted item text content

Output Format (CSV + Parquet):
    - cik, accession, fiscal_year, filing_date
    - item: Item number
    - sent_id: Unique sentence ID within filing+item
    - sentence_text: Cleaned sentence text

Usage:
    python build_sentence_table.py --input filings.json --output sentence_table.csv
"""

import sys
sys.path.insert(0, '/home/user/cntext')

import pandas as pd
import cntext as ct
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings('ignore')

# Try to import spaCy for better sentence splitting
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        print("✓ Using spaCy for sentence splitting (high quality)")
    except OSError:
        SPACY_AVAILABLE = False
        print("⚠ spaCy model not found. Using NLTK fallback.")
        print("  Install with: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠ spaCy not installed. Using NLTK fallback.")

# Fallback to NLTK
if not SPACY_AVAILABLE:
    from cntext.stats.utils import en_split_sentences

# ============================================================================
# CONFIGURATION
# ============================================================================

# Sentence filtering parameters
MIN_SENTENCE_LENGTH = 20          # Minimum characters per sentence
MAX_SENTENCE_LENGTH = 2000        # Maximum characters (filter run-on OCR errors)
MIN_WORD_COUNT = 3                # Minimum words per sentence
MAX_NUMERIC_RATIO = 0.6           # Maximum ratio of numeric characters (filter tables)
MAX_PUNCT_RATIO = 0.4             # Maximum ratio of punctuation (filter noise)

# Processing
BATCH_SIZE = 100                  # Process N filings at a time
ENABLE_CLEANING = True            # Apply ct.clean_text() normalization

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def split_sentences(text):
    """
    Split text into sentences using best available method.

    Priority: spaCy > NLTK regex
    """
    if not text or not isinstance(text, str):
        return []

    if SPACY_AVAILABLE:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    else:
        # Use repository's regex-based splitter
        return [s.strip() for s in en_split_sentences(text) if s.strip()]


def is_valid_sentence(sentence):
    """
    Filter out low-quality sentences (tables, noise, headers).

    Returns:
        bool: True if sentence should be kept
    """
    if not sentence or len(sentence) < MIN_SENTENCE_LENGTH:
        return False

    if len(sentence) > MAX_SENTENCE_LENGTH:
        return False

    # Word count check
    word_count = len(sentence.split())
    if word_count < MIN_WORD_COUNT:
        return False

    # Numeric content check (filter tables)
    numeric_chars = sum(c.isdigit() for c in sentence)
    if numeric_chars / len(sentence) > MAX_NUMERIC_RATIO:
        return False

    # Punctuation check (filter noise)
    punct_chars = sum(c in '.,;:!?-_=+*/\\|[]{}()<>' for c in sentence)
    if punct_chars / len(sentence) > MAX_PUNCT_RATIO:
        return False

    # Filter obvious headers/footers (all caps lines)
    if sentence.isupper() and word_count < 10:
        return False

    return True


def clean_sentence(sentence):
    """
    Apply light-touch cleaning to sentence.

    Uses repository utilities for normalization.
    """
    # Fix encoding issues
    sentence = ct.fix_text(sentence)

    # Expand contractions
    sentence = ct.fix_contractions(sentence)

    # Optional: Apply full cleaning (configurable)
    if ENABLE_CLEANING:
        # Note: clean_text() lowercases and does heavy normalization
        # You may want to skip this for labeling (preserve original)
        # Uncomment if you want aggressive cleaning:
        # sentence = ct.clean_text(sentence, lang='english')
        pass

    # Light normalization: whitespace only
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    return sentence


def load_input_data(input_path):
    """
    Load Item extractions from JSON or CSV.

    Expected columns:
        - cik, accession, fiscal_year, filing_date, item, item_text
    """
    input_path = Path(input_path)

    if input_path.suffix == '.json':
        print(f"Loading JSON: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle both list of dicts and single dict
        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

    elif input_path.suffix == '.csv':
        print(f"Loading CSV: {input_path}")
        df = pd.read_csv(input_path)

    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    # Validate required columns
    required_cols = ['cik', 'accession', 'fiscal_year', 'filing_date', 'item', 'item_text']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"✓ Loaded {len(df)} Item extractions")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Items: {sorted(df['item'].unique())}")
    print(f"  Years: {df['fiscal_year'].min()} - {df['fiscal_year'].max()}")

    return df


def process_filing_to_sentences(row):
    """
    Convert a single Item extraction to sentence-level records.

    Args:
        row: DataFrame row with columns: cik, accession, fiscal_year, filing_date, item, item_text

    Returns:
        List of dicts with sentence-level data
    """
    sentences = []

    # Split into sentences
    raw_sentences = split_sentences(row['item_text'])

    # Process each sentence
    sent_id = 0
    for raw_sent in raw_sentences:
        # Clean
        cleaned = clean_sentence(raw_sent)

        # Validate
        if not is_valid_sentence(cleaned):
            continue

        # Create record
        sentences.append({
            'cik': row['cik'],
            'accession': row['accession'],
            'fiscal_year': row['fiscal_year'],
            'filing_date': row['filing_date'],
            'item': row['item'],
            'sent_id': sent_id,
            'sentence_text': cleaned
        })

        sent_id += 1

    return sentences


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def build_sentence_table(input_path, output_path, output_format='both'):
    """
    Main pipeline: Item extractions → sentence table.

    Args:
        input_path: Path to JSON/CSV with Item extractions
        output_path: Path for output CSV
        output_format: 'csv', 'parquet', or 'both'
    """

    print("=" * 80)
    print("SENTENCE TABLE BUILDER FOR SEC 10-K ITEMS")
    print("=" * 80)

    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 1: LOADING ITEM EXTRACTIONS")
    print("=" * 80)

    df = load_input_data(input_path)

    # ========================================================================
    # STEP 2: PROCESS TO SENTENCES
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 2: SPLITTING INTO SENTENCES")
    print("=" * 80)

    all_sentences = []

    # Process with progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing filings"):
        try:
            sentences = process_filing_to_sentences(row)
            all_sentences.extend(sentences)
        except Exception as e:
            print(f"\n⚠ Error processing {row['accession']}: {e}")
            continue

    # Create DataFrame
    sentence_df = pd.DataFrame(all_sentences)

    print(f"\n✓ Created sentence table with {len(sentence_df)} sentences")
    print(f"  From {len(df)} Item extractions")
    print(f"  Average: {len(sentence_df) / len(df):.1f} sentences per Item")

    # ========================================================================
    # STEP 3: SUMMARY STATISTICS
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 3: SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nSentence counts by Item:")
    item_counts = sentence_df.groupby('item').size()
    for item, count in item_counts.items():
        print(f"  Item {item}: {count:,} sentences")

    print(f"\nSentence counts by year:")
    year_counts = sentence_df.groupby('fiscal_year').size().sort_index()
    for year, count in year_counts.head(10).items():
        print(f"  {year}: {count:,} sentences")
    if len(year_counts) > 10:
        print(f"  ... ({len(year_counts) - 10} more years)")

    # Length statistics
    sentence_df['sent_length'] = sentence_df['sentence_text'].str.len()
    sentence_df['sent_words'] = sentence_df['sentence_text'].str.split().str.len()

    print(f"\nSentence length statistics:")
    print(f"  Mean length: {sentence_df['sent_length'].mean():.0f} characters")
    print(f"  Median length: {sentence_df['sent_length'].median():.0f} characters")
    print(f"  Mean words: {sentence_df['sent_words'].mean():.1f} words")

    # ========================================================================
    # STEP 4: EXPORT
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 4: EXPORTING RESULTS")
    print("=" * 80)

    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(exist_ok=True, parents=True)

    # Drop temporary columns
    export_df = sentence_df.drop(columns=['sent_length', 'sent_words'], errors='ignore')

    # Export to CSV
    if output_format in ['csv', 'both']:
        csv_path = output_path.with_suffix('.csv')
        export_df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")

    # Export to Parquet (more efficient for large datasets)
    if output_format in ['parquet', 'both']:
        parquet_path = output_path.with_suffix('.parquet')
        export_df.to_parquet(parquet_path, index=False)
        print(f"✓ Saved Parquet: {parquet_path} ({parquet_path.stat().st_size / 1e6:.1f} MB)")

    # Save processing summary
    summary_path = output_dir / f"{output_path.stem}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("SENTENCE TABLE BUILD SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Output: {output_path}\n\n")
        f.write(f"Total sentences: {len(sentence_df):,}\n")
        f.write(f"Total filings processed: {len(df):,}\n")
        f.write(f"Sentences per filing: {len(sentence_df) / len(df):.1f}\n\n")
        f.write("Sentences by Item:\n")
        for item, count in item_counts.items():
            f.write(f"  Item {item}: {count:,}\n")
        f.write(f"\nYears covered: {sentence_df['fiscal_year'].min()} - {sentence_df['fiscal_year'].max()}\n")
        f.write(f"Unique CIKs: {sentence_df['cik'].nunique():,}\n")

    print(f"✓ Saved summary: {summary_path}")

    # ========================================================================
    # DONE
    # ========================================================================

    print("\n" + "=" * 80)
    print("BUILD COMPLETE! ✓")
    print("=" * 80)
    print(f"""
Next steps:
1. Review {output_path} for sentence quality
2. Run build_labeling_sample.py to create labeling dataset
3. Check {summary_path} for detailed statistics

Example preview (first 3 sentences):
""")

    for idx, row in export_df.head(3).iterrows():
        print(f"\n[{row['cik']} | {row['fiscal_year']} | Item {row['item']} | Sent {row['sent_id']}]")
        print(f"  {row['sentence_text'][:150]}...")

    return export_df


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build sentence-level table from SEC 10-K Item extractions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python build_sentence_table.py --input filings.json --output sentence_table.csv
    python build_sentence_table.py --input filings.csv --output data/sentences.parquet --format parquet

Input format (JSON or CSV with columns):
    cik, accession, fiscal_year, filing_date, item, item_text

Output format:
    cik, accession, fiscal_year, filing_date, item, sent_id, sentence_text
        """
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Path to input JSON or CSV file with Item extractions'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Path for output sentence table (e.g., sentence_table.csv)'
    )

    parser.add_argument(
        '--format',
        choices=['csv', 'parquet', 'both'],
        default='both',
        help='Output format (default: both)'
    )

    args = parser.parse_args()

    # Run pipeline
    build_sentence_table(
        input_path=args.input,
        output_path=args.output,
        output_format=args.format
    )
