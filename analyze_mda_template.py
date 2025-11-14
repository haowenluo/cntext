"""
Quick-Start Template: Analyze Your MD&A Data

Customize this template for your specific JSON/CSV structure.
This provides a ready-to-run script for your 600MB+ dataset.
"""

import sys
sys.path.insert(0, '/home/user/cntext')

import pandas as pd
import cntext as ct
import json
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - CUSTOMIZE THESE
# ============================================================================

# Your data location
DATA_SOURCE = '/path/to/your/mda_data.csv'  # Or directory with JSON files
OUTPUT_DIR = 'mda_analysis_results'

# Column names in your data (adjust to match your structure)
COLUMNS = {
    'company_id': 'cik',          # Company identifier
    'company_name': 'company_name',
    'date': 'filing_date',         # Filing date
    'year': 'fiscal_year',         # Fiscal year
    'text': 'mda_text'            # MD&A text content
}

# Processing settings
BATCH_SIZE = 100                   # Process N documents at a time
MIN_WORD_COUNT = 100              # Filter out very short texts
USE_LEMMATIZATION = True          # Better quality but slower

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("=" * 80)
print("STEP 1: LOADING MD&A DATA")
print("=" * 80)

def load_data(source):
    """Load data from CSV or JSON files."""
    source_path = Path(source)

    if source_path.suffix == '.csv':
        print(f"Loading CSV: {source}")
        # For large files, you can add chunksize parameter
        df = pd.read_csv(source)

    elif source_path.is_dir():
        print(f"Loading JSON files from: {source}")
        json_files = list(source_path.glob('*.json'))
        print(f"Found {len(json_files)} JSON files")

        records = []
        for json_file in tqdm(json_files, desc="Loading"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                records.append(data)

        df = pd.DataFrame(records)

    else:
        raise ValueError(f"Unknown data source type: {source}")

    print(f"✓ Loaded {len(df)} MD&A sections")
    print(f"  Columns: {list(df.columns)}")

    return df

df = load_data(DATA_SOURCE)

# ============================================================================
# STEP 2: PREPROCESS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: PREPROCESSING")
print("=" * 80)

def preprocess(df):
    """Clean and prepare MD&A text."""

    # Get text column
    text_col = COLUMNS['text']

    # Remove nulls
    initial_count = len(df)
    df = df.dropna(subset=[text_col])
    print(f"Removed {initial_count - len(df)} null texts")

    # Clean text
    print("Cleaning text (contractions, encoding)...")
    df['mda_clean'] = df[text_col].progress_apply(
        lambda x: ct.clean_text(ct.fix_contractions(str(x)), lang='english')
    )

    # Add metrics
    df['word_count'] = df['mda_clean'].str.split().str.len()
    df['char_count'] = df['mda_clean'].str.len()

    # Filter short texts
    before = len(df)
    df = df[df['word_count'] >= MIN_WORD_COUNT]
    print(f"Filtered {before - len(df)} short texts (<{MIN_WORD_COUNT} words)")

    print(f"✓ Preprocessed: {len(df)} MD&A sections ready")
    print(f"  Average length: {df['word_count'].mean():.0f} words")

    return df

tqdm.pandas()  # Enable progress_apply
df = preprocess(df)

# ============================================================================
# STEP 3: SENTIMENT ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: SENTIMENT ANALYSIS (Loughran-McDonald)")
print("=" * 80)

def analyze_sentiment(df):
    """Analyze sentiment using financial dictionary."""

    # Load dictionary once
    print("Loading Loughran-McDonald financial sentiment dictionary...")
    lm_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')

    # Process in batches
    results = []

    for start_idx in tqdm(range(0, len(df), BATCH_SIZE), desc="Sentiment"):
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        batch = df.iloc[start_idx:end_idx]

        for idx, row in batch.iterrows():
            try:
                sentiment = ct.sentiment(
                    row['mda_clean'],
                    diction=lm_dict,
                    lang='english'
                )
                sentiment['idx'] = idx
                results.append(sentiment)

            except Exception as e:
                print(f"Error at {idx}: {e}")
                continue

    # Merge results
    sentiment_df = pd.DataFrame(results).set_index('idx')
    df = df.join(sentiment_df)

    # Calculate derived metrics
    df['lm_positive_pct'] = df['Positive'] / df['word_count'] * 100
    df['lm_negative_pct'] = df['Negative'] / df['word_count'] * 100
    df['lm_net_sentiment'] = (
        (df['Positive'] - df['Negative']) /
        (df['Positive'] + df['Negative'] + 1)
    )

    print(f"✓ Sentiment analysis complete!")
    print(f"\nSentiment distribution:")
    print(f"  Mean Positive words: {df['Positive'].mean():.1f}")
    print(f"  Mean Negative words: {df['Negative'].mean():.1f}")
    print(f"  Mean Uncertainty words: {df['Uncertainty'].mean():.1f}")

    return df

df = analyze_sentiment(df)

# ============================================================================
# STEP 4: READABILITY ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: READABILITY ANALYSIS")
print("=" * 80)

def analyze_readability(df):
    """Calculate readability metrics."""

    results = []

    for start_idx in tqdm(range(0, len(df), BATCH_SIZE), desc="Readability"):
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        batch = df.iloc[start_idx:end_idx]

        for idx, row in batch.iterrows():
            try:
                readability = ct.readability(row['mda_clean'], lang='english')
                readability['idx'] = idx
                results.append(readability)

            except Exception as e:
                print(f"Error at {idx}: {e}")
                continue

    # Merge results
    readability_df = pd.DataFrame(results).set_index('idx')
    df = df.join(readability_df)

    # Calculate average
    readability_cols = ['Flesch', 'Fog', 'SMOG', 'Coleman_Liau', 'ARI']
    df['readability_avg'] = df[readability_cols].mean(axis=1)

    print(f"✓ Readability analysis complete!")
    print(f"\nAverage readability scores:")
    print(f"  Flesch: {df['Flesch'].mean():.1f}")
    print(f"  Fog Index: {df['Fog'].mean():.1f}")
    print(f"  Average: {df['readability_avg'].mean():.1f}")

    return df

df = analyze_readability(df)

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: SAVING RESULTS")
print("=" * 80)

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Save complete results
output_file = f'{OUTPUT_DIR}/mda_analysis_complete.csv'
df.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# Save summary by year
if COLUMNS['year'] in df.columns:
    summary_by_year = df.groupby(COLUMNS['year']).agg({
        'lm_net_sentiment': ['mean', 'std', 'count'],
        'Positive': 'mean',
        'Negative': 'mean',
        'Uncertainty': 'mean',
        'readability_avg': 'mean',
        'word_count': ['mean', 'median']
    }).round(3)

    summary_file = f'{OUTPUT_DIR}/mda_summary_by_year.csv'
    summary_by_year.to_csv(summary_file)
    print(f"✓ Saved: {summary_file}")

# Save summary by company (top 20)
if COLUMNS['company_name'] in df.columns:
    company_summary = df.groupby(COLUMNS['company_name']).agg({
        'lm_net_sentiment': 'mean',
        'readability_avg': 'mean',
        'word_count': 'mean',
        COLUMNS['company_id']: 'first'
    }).round(3).sort_values('lm_net_sentiment', ascending=False)

    top_companies = company_summary.head(20)
    company_file = f'{OUTPUT_DIR}/mda_top_companies.csv'
    top_companies.to_csv(company_file)
    print(f"✓ Saved: {company_file}")

# Save column documentation
with open(f'{OUTPUT_DIR}/README.txt', 'w') as f:
    f.write("MD&A Analysis Results\\n")
    f.write("=" * 50 + "\\n\\n")
    f.write("Columns:\\n\\n")

    f.write("Loughran-McDonald Sentiment:\\n")
    f.write("  Positive - Count of positive words\\n")
    f.write("  Negative - Count of negative words\\n")
    f.write("  Uncertainty - Count of uncertainty words\\n")
    f.write("  Litigious - Count of litigation-related words\\n")
    f.write("  lm_net_sentiment - Net sentiment score (-1 to 1)\\n\\n")

    f.write("Readability Metrics (higher = more complex):\\n")
    f.write("  Flesch - Flesch Reading Ease\\n")
    f.write("  Fog - Gunning Fog Index\\n")
    f.write("  SMOG - SMOG Index\\n")
    f.write("  Coleman_Liau - Coleman-Liau Index\\n")
    f.write("  ARI - Automated Readability Index\\n")
    f.write("  readability_avg - Average of all metrics\\n")

print(f"✓ Saved: {OUTPUT_DIR}/README.txt")

# ============================================================================
# STEP 6: GENERATE SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

print(f"\nTotal MD&A sections analyzed: {len(df)}")

print("\nSentiment Statistics:")
print(f"  Average net sentiment: {df['lm_net_sentiment'].mean():+.3f}")
print(f"  Positive documents (>0.1): {(df['lm_net_sentiment'] > 0.1).sum()} ({(df['lm_net_sentiment'] > 0.1).mean()*100:.1f}%)")
print(f"  Negative documents (<-0.1): {(df['lm_net_sentiment'] < -0.1).sum()} ({(df['lm_net_sentiment'] < -0.1).mean()*100:.1f}%)")

print("\nReadability Statistics:")
print(f"  Average Flesch score: {df['Flesch'].mean():.1f}")
print(f"  Average Fog Index: {df['Fog'].mean():.1f}")
print(f"  Average complexity: {df['readability_avg'].mean():.1f}")

if COLUMNS['year'] in df.columns:
    print(f"\nYears covered: {df[COLUMNS['year']].min()} - {df[COLUMNS['year']].max()}")
    print(f"Documents per year: {len(df) / df[COLUMNS['year']].nunique():.0f} average")

print(f"\nResults saved to: {OUTPUT_DIR}/")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE! ✓")
print("=" * 80)
print("""
Next steps:
1. Review mda_analysis_complete.csv for full results
2. Check mda_summary_by_year.csv for trends
3. Visualize trends in Excel or Python
4. Run additional analyses if needed

For advanced analysis (semantic projection, LLM), see:
- guide_mda_analysis.py
- demos/demo_mind.py
- demos/demo_llm.py
""")
