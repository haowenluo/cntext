"""
Practical Guide: Analyzing MD&A Sections from 10-K Filings

This guide shows how to analyze large-scale MD&A (Management Discussion & Analysis)
data from SEC 10-K filings using the enhanced cntext fork.

Dataset characteristics:
- Source: 10-K filings
- Format: JSON files (extracted MD&A sections)
- Combined size: 600MB+ CSV
- Analysis needs: Financial sentiment, trends, readability, semantic patterns

This script provides memory-efficient approaches for large datasets.
"""

import sys
sys.path.insert(0, '/home/user/cntext')

import json
import pandas as pd
from pathlib import Path
import numpy as np
from collections import defaultdict

print("=" * 80)
print("MD&A ANALYSIS WORKFLOW - Large-Scale Financial Text Analysis")
print("=" * 80)
print()

# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================

print("PART 1: DATA LOADING AND PREPROCESSING")
print("-" * 80)
print()

print("""
Step 1.1: Loading JSON Files
-----------------------------

If your MD&A data is in JSON format, here's how to load it efficiently:
""")

example_json_code = '''
import json
import pandas as pd
from pathlib import Path

def load_mda_from_json(json_file):
    """Load MD&A data from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_all_mda_files(data_dir):
    """
    Load all JSON files from directory into DataFrame.

    Expected JSON structure:
    {
        "cik": "0000320193",
        "company_name": "Apple Inc.",
        "filing_date": "2023-10-27",
        "fiscal_year": 2023,
        "mda_text": "Business Overview...",
        "form_type": "10-K"
    }
    """
    json_files = list(Path(data_dir).glob('*.json'))
    print(f"Found {len(json_files)} JSON files")

    records = []
    for json_file in json_files:
        try:
            data = load_mda_from_json(json_file)
            records.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} MD&A sections")

    return df

# Load your data
df = load_all_mda_files('/path/to/your/json/files/')
'''

print(example_json_code)

print("""
Step 1.2: Loading from Large CSV (Memory-Efficient)
----------------------------------------------------

For 600MB+ CSV files, use chunked reading to avoid memory issues:
""")

example_csv_code = '''
import pandas as pd

def load_mda_csv_chunked(csv_file, chunksize=1000):
    """
    Load large CSV in chunks.

    Expected CSV columns:
    - cik: Company CIK number
    - company_name: Company name
    - filing_date: Filing date
    - fiscal_year: Fiscal year
    - mda_text: MD&A section text
    """
    chunks = []
    total_rows = 0

    for chunk in pd.read_csv(csv_file, chunksize=chunksize):
        chunks.append(chunk)
        total_rows += len(chunk)
        print(f"Loaded {total_rows} rows...", end='\\r')

    df = pd.concat(chunks, ignore_index=True)
    print(f"\\nTotal: {len(df)} MD&A sections loaded")

    return df

# Or load directly if you have enough RAM
df = pd.read_csv('mda_data.csv')
'''

print(example_csv_code)

print("""
Step 1.3: Data Cleaning and Preprocessing
------------------------------------------
""")

preprocessing_code = '''
import cntext as ct

def preprocess_mda(df):
    """
    Clean and preprocess MD&A text.
    """
    print("Preprocessing MD&A text...")

    # 1. Remove rows with missing text
    df = df.dropna(subset=['mda_text'])
    print(f"  After removing nulls: {len(df)} sections")

    # 2. Clean text
    print("  Cleaning text (fix contractions, encoding issues)...")
    df['mda_clean'] = df['mda_text'].apply(
        lambda x: ct.clean_text(
            ct.fix_contractions(str(x)),
            lang='english'
        )
    )

    # 3. Add text length metrics
    df['text_length'] = df['mda_clean'].str.len()
    df['word_count'] = df['mda_clean'].str.split().str.len()

    # 4. Filter out very short sections (likely extraction errors)
    min_words = 100
    df = df[df['word_count'] >= min_words]
    print(f"  After filtering short texts (<{min_words} words): {len(df)} sections")

    return df

df = preprocess_mda(df)
'''

print(preprocessing_code)

# ============================================================================
# PART 2: SENTIMENT ANALYSIS (Financial Focus)
# ============================================================================

print("\n\n" + "=" * 80)
print("PART 2: SENTIMENT ANALYSIS - Loughran-McDonald Financial Dictionary")
print("=" * 80)
print()

sentiment_code = '''
import cntext as ct
import pandas as pd
from tqdm import tqdm

def analyze_mda_sentiment(df, text_column='mda_clean'):
    """
    Analyze sentiment using Loughran-McDonald financial dictionary.
    This dictionary is specifically designed for financial text!
    """
    print("Analyzing MD&A sentiment...")

    # Load Loughran-McDonald dictionary
    lm_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')

    # Initialize result columns
    sentiment_categories = ['Positive', 'Negative', 'Uncertainty',
                           'Litigious', 'StrongModal', 'WeakModal',
                           'Constraining']

    for cat in sentiment_categories:
        df[f'lm_{cat.lower()}'] = 0

    # Process each MD&A section (with progress bar)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row[text_column]

        # Get sentiment scores
        sentiment = ct.sentiment(text, diction=lm_dict, lang='english')

        # Store results
        for cat in sentiment_categories:
            df.at[idx, f'lm_{cat.lower()}'] = sentiment.get(cat, 0)

    # Calculate derived metrics
    df['lm_net_sentiment'] = (
        (df['lm_positive'] - df['lm_negative']) /
        (df['lm_positive'] + df['lm_negative'] + 1)  # +1 to avoid division by zero
    )

    df['lm_tone'] = df['lm_net_sentiment'].apply(
        lambda x: 'positive' if x > 0.1 else 'negative' if x < -0.1 else 'neutral'
    )

    print("✓ Sentiment analysis complete!")

    return df

# Run sentiment analysis
df = analyze_mda_sentiment(df)

# Summary statistics
print("\\nSentiment Summary:")
print(df['lm_tone'].value_counts())
print("\\nAverage sentiment by year:")
print(df.groupby('fiscal_year')['lm_net_sentiment'].mean())
'''

print(sentiment_code)

# ============================================================================
# PART 3: READABILITY ANALYSIS
# ============================================================================

print("\n\n" + "=" * 80)
print("PART 3: READABILITY ANALYSIS")
print("=" * 80)
print()

readability_code = '''
import cntext as ct
from tqdm import tqdm

def analyze_mda_readability(df, text_column='mda_clean'):
    """
    Calculate readability metrics for MD&A sections.
    Higher scores = more complex/difficult to read.
    """
    print("Analyzing MD&A readability...")

    # Initialize columns
    readability_metrics = ['Flesch', 'Fog', 'SMOG', 'Coleman_Liau', 'ARI']
    for metric in readability_metrics:
        df[f'readability_{metric.lower()}'] = 0.0

    # Calculate readability for each section
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row[text_column]

        try:
            scores = ct.readability(text, lang='english')

            for metric in readability_metrics:
                df.at[idx, f'readability_{metric.lower()}'] = scores.get(metric, 0)

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    # Calculate average readability score
    df['readability_avg'] = df[[f'readability_{m.lower()}' for m in readability_metrics]].mean(axis=1)

    print("✓ Readability analysis complete!")

    return df

# Run readability analysis
df = analyze_mda_readability(df)

# Analyze trends
print("\\nReadability by year:")
print(df.groupby('fiscal_year')['readability_avg'].agg(['mean', 'median', 'std']))
'''

print(readability_code)

# ============================================================================
# PART 4: KEYWORD AND TOPIC ANALYSIS
# ============================================================================

print("\n\n" + "=" * 80)
print("PART 4: KEYWORD AND TOPIC ANALYSIS")
print("=" * 80)
print()

keyword_code = '''
import cntext as ct
from collections import Counter

def extract_mda_keywords(df, text_column='mda_clean', top_n=20):
    """
    Extract most frequent keywords across all MD&A sections.
    Uses lemmatization for better accuracy.
    """
    print("Extracting keywords from MD&A corpus...")

    # Combine all text
    all_text = ' '.join(df[text_column].astype(str).tolist())

    # Get word frequencies with lemmatization
    word_freq = ct.word_count(all_text, lang='english', lemmatize=True)

    # Get top keywords
    top_keywords = word_freq.most_common(top_n)

    print(f"\\nTop {top_n} keywords across all MD&A sections:")
    for word, count in top_keywords:
        print(f"  {word}: {count:,}")

    return dict(top_keywords)

def analyze_keyword_by_year(df, keywords, text_column='mda_clean'):
    """
    Track how keyword usage changes over time.
    """
    print("\\nTracking keyword usage over time...")

    results = []

    for year in sorted(df['fiscal_year'].unique()):
        year_df = df[df['fiscal_year'] == year]
        year_text = ' '.join(year_df[text_column].astype(str).tolist())

        # Count keywords in this year
        year_counts = ct.word_count(year_text, lang='english', lemmatize=True)

        year_data = {'year': year}
        for keyword in keywords:
            year_data[keyword] = year_counts.get(keyword, 0)

        results.append(year_data)

    trend_df = pd.DataFrame(results)
    return trend_df

# Extract keywords
keywords = extract_mda_keywords(df, top_n=20)

# Track keyword trends
keyword_trends = analyze_keyword_by_year(df, list(keywords.keys())[:10])
print("\\nKeyword trends:")
print(keyword_trends)
'''

print(keyword_code)

# ============================================================================
# PART 5: SEMANTIC ANALYSIS (Advanced)
# ============================================================================

print("\n\n" + "=" * 80)
print("PART 5: SEMANTIC ANALYSIS - Measuring Abstract Concepts")
print("=" * 80)
print()

semantic_code = '''
import cntext as ct
import tempfile
import os

def train_mda_embeddings(df, text_column='mda_clean'):
    """
    Train domain-specific word embeddings on your MD&A corpus.
    This captures the semantic relationships in YOUR data.
    """
    print("Training Word2Vec on MD&A corpus...")

    # Create temporary corpus file
    corpus_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')

    # Write each MD&A section as a separate document
    for text in df[text_column]:
        # Clean and write
        cleaned = ' '.join(str(text).split())  # Normalize whitespace
        corpus_file.write(cleaned + '\\n')

    corpus_file.close()

    # Train Word2Vec
    wv = ct.Word2Vec(
        corpus_file=corpus_file.name,
        lang='english',
        lemmatize=True,  # Better quality
        vector_size=100,
        window=10,  # Larger window for paragraph-level context
        min_count=5,
        workers=4
    )

    # Clean up
    os.unlink(corpus_file.name)

    print(f"✓ Embeddings trained! Vocabulary: {len(wv.index_to_key)} words")

    return wv

def create_financial_concept_axes(wv):
    """
    Create semantic axes for financial concepts.
    These measure abstract dimensions in your MD&A text.
    """
    print("\\nCreating financial concept axes...")

    axes = {}

    # Axis 1: Optimistic vs Pessimistic
    axes['optimism'] = ct.generate_concept_axis(
        wv,
        poswords=['growth', 'opportunity', 'success', 'strong', 'positive', 'increase'],
        negwords=['decline', 'risk', 'concern', 'weakness', 'negative', 'decrease']
    )

    # Axis 2: Innovative vs Traditional
    axes['innovation'] = ct.generate_concept_axis(
        wv,
        poswords=['innovation', 'technology', 'digital', 'new', 'advanced', 'cutting-edge'],
        negwords=['traditional', 'conventional', 'legacy', 'established', 'historical']
    )

    # Axis 3: Expansion vs Contraction
    axes['expansion'] = ct.generate_concept_axis(
        wv,
        poswords=['expand', 'growth', 'increase', 'acquisition', 'market', 'development'],
        negwords=['reduce', 'decline', 'decrease', 'contraction', 'closure', 'exit']
    )

    print("✓ Created 3 semantic axes: optimism, innovation, expansion")

    return axes

def project_mda_onto_axes(df, wv, axes, text_column='mda_clean'):
    """
    Project each MD&A section onto semantic axes.
    This quantifies abstract concepts in each document.
    """
    print("\\nProjecting MD&A sections onto semantic axes...")

    from tqdm import tqdm

    for axis_name, axis in axes.items():
        df[f'semantic_{axis_name}'] = 0.0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{axis_name}"):
            text = row[text_column]

            try:
                score = ct.project_text(wv, text, axis, lang='english')
                df.at[idx, f'semantic_{axis_name}'] = score
            except Exception as e:
                continue

    print("✓ Semantic projection complete!")

    return df

# Train embeddings on your corpus
wv = train_mda_embeddings(df)

# Create semantic axes
axes = create_financial_concept_axes(wv)

# Project MD&A sections
df = project_mda_onto_axes(df, wv, axes)

# Analyze semantic scores by year
print("\\nSemantic scores by year:")
semantic_cols = [col for col in df.columns if col.startswith('semantic_')]
print(df.groupby('fiscal_year')[semantic_cols].mean())
'''

print(semantic_code)

# ============================================================================
# PART 6: BATCH PROCESSING FOR LARGE DATASETS
# ============================================================================

print("\n\n" + "=" * 80)
print("PART 6: MEMORY-EFFICIENT BATCH PROCESSING")
print("=" * 80)
print()

batch_code = '''
def process_mda_in_batches(df, batch_size=100):
    """
    Process large datasets in batches to manage memory.
    Useful for 600MB+ datasets.
    """
    import cntext as ct
    from tqdm import tqdm

    print(f"Processing {len(df)} MD&A sections in batches of {batch_size}...")

    # Load dictionary once
    lm_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')

    results = []

    # Process in batches
    for start_idx in tqdm(range(0, len(df), batch_size)):
        end_idx = min(start_idx + batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]

        batch_results = []

        for idx, row in batch.iterrows():
            try:
                # Sentiment analysis
                sentiment = ct.sentiment(row['mda_clean'], diction=lm_dict, lang='english')

                # Readability
                readability = ct.readability(row['mda_clean'], lang='english')

                # Combine results
                result = {
                    'index': idx,
                    **sentiment,
                    **readability
                }

                batch_results.append(result)

            except Exception as e:
                print(f"Error processing {idx}: {e}")
                continue

        results.extend(batch_results)

        # Optional: Save intermediate results
        if (start_idx // batch_size) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(f'mda_results_checkpoint_{start_idx}.csv', index=False)

    # Merge results back
    results_df = pd.DataFrame(results)
    df = df.merge(results_df, left_index=True, right_on='index', how='left')

    return df

# For very large datasets
df_processed = process_mda_in_batches(df, batch_size=100)
'''

print(batch_code)

# ============================================================================
# PART 7: COMPLETE WORKFLOW EXAMPLE
# ============================================================================

print("\n\n" + "=" * 80)
print("PART 7: COMPLETE END-TO-END WORKFLOW")
print("=" * 80)
print()

complete_workflow = '''
"""
Complete workflow for analyzing MD&A data from 10-K filings.
"""

import pandas as pd
import cntext as ct
from pathlib import Path
import json

def complete_mda_analysis_workflow(data_source, output_dir='mda_results'):
    """
    Complete analysis pipeline for MD&A data.

    Args:
        data_source: Path to CSV file or directory with JSON files
        output_dir: Directory to save results
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # ========================================
    # STEP 1: Load Data
    # ========================================
    print("STEP 1: Loading data...")

    if Path(data_source).suffix == '.csv':
        df = pd.read_csv(data_source)
    else:
        # Load from JSON files
        json_files = list(Path(data_source).glob('*.json'))
        records = [json.load(open(f)) for f in json_files]
        df = pd.DataFrame(records)

    print(f"Loaded {len(df)} MD&A sections")

    # ========================================
    # STEP 2: Preprocessing
    # ========================================
    print("\\nSTEP 2: Preprocessing...")

    df = df.dropna(subset=['mda_text'])
    df['mda_clean'] = df['mda_text'].apply(
        lambda x: ct.clean_text(ct.fix_contractions(str(x)), lang='english')
    )
    df['word_count'] = df['mda_clean'].str.split().str.len()
    df = df[df['word_count'] >= 100]

    print(f"After preprocessing: {len(df)} sections")

    # ========================================
    # STEP 3: Sentiment Analysis
    # ========================================
    print("\\nSTEP 3: Sentiment analysis...")

    lm_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')

    sentiment_results = []
    for idx, row in df.iterrows():
        sentiment = ct.sentiment(row['mda_clean'], diction=lm_dict, lang='english')
        sentiment['index'] = idx
        sentiment_results.append(sentiment)

    sentiment_df = pd.DataFrame(sentiment_results)
    df = df.merge(sentiment_df, left_index=True, right_on='index')

    # Calculate net sentiment
    df['net_sentiment'] = (
        (df['Positive'] - df['Negative']) /
        (df['Positive'] + df['Negative'] + 1)
    )

    # ========================================
    # STEP 4: Readability Analysis
    # ========================================
    print("\\nSTEP 4: Readability analysis...")

    readability_results = []
    for idx, row in df.iterrows():
        readability = ct.readability(row['mda_clean'], lang='english')
        readability['index'] = idx
        readability_results.append(readability)

    readability_df = pd.DataFrame(readability_results)
    df = df.merge(readability_df, left_index=True, right_on='index')

    # ========================================
    # STEP 5: Save Results
    # ========================================
    print("\\nSTEP 5: Saving results...")

    # Save full results
    df.to_csv(f'{output_dir}/mda_analysis_complete.csv', index=False)

    # Save summary statistics
    summary = df.groupby('fiscal_year').agg({
        'net_sentiment': ['mean', 'std'],
        'Flesch': ['mean', 'std'],
        'word_count': ['mean', 'median']
    }).round(3)

    summary.to_csv(f'{output_dir}/mda_summary_by_year.csv')

    print(f"\\n✓ Analysis complete! Results saved to {output_dir}/")

    return df

# Run complete workflow
df_results = complete_mda_analysis_workflow(
    data_source='/path/to/your/mda_data.csv',
    output_dir='mda_analysis_results'
)
'''

print(complete_workflow)

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("\n\n" + "=" * 80)
print("SUMMARY: Key Recommendations for Your MD&A Analysis")
print("=" * 80)
print()

recommendations = """
1. START SMALL
--------------
Before processing all 600MB:
- Test workflow on a sample (100-1000 documents)
- Validate results
- Tune parameters
- Then scale up

2. LOUGHRAN-MCDONALD DICTIONARY
--------------------------------
This is the GOLD STANDARD for financial text:
- Specifically designed for 10-K, 10-Q filings
- Accounts for financial context (e.g., "negative" as in "negative results")
- Provides 7 categories: Positive, Negative, Uncertainty, Litigious, etc.
- Already included in cntext!

3. MEMORY MANAGEMENT
--------------------
For 600MB+ datasets:
- Process in batches (100-1000 documents at a time)
- Save intermediate results
- Use chunked CSV reading
- Consider sampling for initial exploration

4. KEY ANALYSES FOR MD&A
-------------------------
Priority 1 (Essential):
  ✓ Sentiment analysis (Loughran-McDonald)
  ✓ Readability metrics
  ✓ Word frequency / keyword extraction

Priority 2 (Valuable):
  ✓ Semantic projection (optimism, innovation, risk)
  ✓ Keyword trends over time
  ✓ Company-level aggregation

Priority 3 (Advanced):
  ✓ LLM analysis for specific questions
  ✓ Topic modeling
  ✓ Comparative analysis across industries

5. TYPICAL MD&A INSIGHTS
-------------------------
What you can discover:
- Tone shifts over time (pre/post crisis)
- Readability complexity trends
- Risk disclosure patterns
- Innovation language adoption
- Industry-specific terminology
- Optimism vs pessimism in forward-looking statements

6. PERFORMANCE OPTIMIZATION
----------------------------
Speed up processing:
- Use lemmatize=False for faster processing (less accurate)
- Enable lemmatize=True for better quality (slower)
- Process batches in parallel if you have multiple cores
- Cache dictionary loadings (load once, use many times)

7. OUTPUT FORMATS
-----------------
Recommended outputs:
- Full CSV with all metrics
- Summary statistics by year
- Summary statistics by company
- Visualization data (for plotting trends)
- Semantic scores (if using embeddings)

8. VALIDATION
-------------
Validate your results:
- Manually review a sample
- Compare with known events (e.g., 2008 financial crisis)
- Check for extraction errors (very short/long texts)
- Verify sentiment scores make intuitive sense
"""

print(recommendations)

print("\n" + "=" * 80)
print("READY TO ANALYZE YOUR MD&A DATA!")
print("=" * 80)
print()

print("""
Next Steps:

1. Save this guide as a reference
2. Adapt the code to your JSON/CSV structure
3. Start with a small sample (e.g., 100 documents)
4. Run the complete workflow
5. Examine results and tune parameters
6. Scale up to full dataset

Need help? Check:
- demos/demo_stats.py for sentiment analysis details
- demos/demo_mind.py for semantic projection
- ENGLISH_ENHANCEMENTS.md for complete documentation

Good luck with your MD&A analysis! 🚀
""")
