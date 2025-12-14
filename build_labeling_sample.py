"""
Build Labeling Sample from Sentence Table

Creates a balanced sample of sentences for manual tech classification labeling.
Samples both "likely tech" (keyword hits) and random background sentences.

Input:
    - sentence_table.csv (from build_sentence_table.py)
    - tech_keywords.yaml (technology-related keyword patterns)

Output:
    - label_set.csv: 2000-3000 sentences ready for manual labeling
      Columns: cik, accession, fiscal_year, filing_date, item, sent_id,
               sentence_text, tech_hit, source_pool

Usage:
    python build_labeling_sample.py --input sentence_table.csv --output label_set.csv --size 2500
"""

import sys
sys.path.insert(0, '/home/user/cntext')

import pandas as pd
import cntext as ct
import argparse
import re
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Sampling parameters
DEFAULT_SAMPLE_SIZE = 2500        # Total sentences to sample
TECH_HIT_RATIO = 0.5              # Proportion of tech_hit==1 sentences
RANDOM_SEED = 42                  # For reproducibility

# Deduplication
ENABLE_DEDUP = True               # Remove exact duplicates

# Stratification (optional)
STRATIFY_BY_YEAR = False          # Balance samples across years
STRATIFY_BY_ITEM = False          # Balance samples across Items

# Filtering
MIN_SENTENCE_LENGTH = 30          # Slightly higher than build_sentence_table
EXCLUDE_VERY_SHORT = True

# ============================================================================
# TECH KEYWORD MATCHING
# ============================================================================

def load_tech_keywords(keywords_file='tech_keywords.yaml'):
    """
    Load technology keyword patterns from YAML file.

    Returns:
        dict: Dictionary with keyword categories
    """
    try:
        keywords_dict = ct.read_yaml_dict(keywords_file)
        print(f"✓ Loaded tech keywords from: {keywords_file}")
        return keywords_dict
    except Exception as e:
        print(f"⚠ Could not load {keywords_file}: {e}")
        print("  Using default keyword list")
        return get_default_tech_keywords()


def get_default_tech_keywords():
    """
    Default tech keyword patterns if YAML not found.

    Categories:
        - Core tech terms
        - Digital/IT terms
        - Innovation terms
        - Automation terms
    """
    return {
        'Dictionary': {
            'core_tech': [
                'artificial intelligence', 'machine learning', 'deep learning',
                'neural network', 'algorithm', 'automation', 'robot', 'robotics',
                'blockchain', 'cryptocurrency', 'cloud computing', 'edge computing',
                'quantum computing', 'data science', 'big data', 'analytics',
                'internet of things', 'iot', 'sensor', 'smart device'
            ],
            'digital_it': [
                'digital platform', 'digitalization', 'digitization', 'digital transformation',
                'software', 'application', 'mobile app', 'api', 'database',
                'server', 'network', 'cybersecurity', 'encryption', 'firewall',
                'saas', 'paas', 'iaas', 'cloud-based', 'web-based',
                'e-commerce', 'online platform', 'digital infrastructure'
            ],
            'innovation': [
                'innovative technology', 'proprietary technology', 'patent',
                'research and development', 'r&d investment', 'technological innovation',
                'breakthrough', 'cutting-edge', 'state-of-the-art', 'next-generation',
                'advanced technology', 'emerging technology', 'disruptive technology'
            ],
            'automation_ai': [
                'automated', 'autonomous', 'self-driving', 'computer vision',
                'natural language processing', 'nlp', 'speech recognition',
                'predictive analytics', 'recommendation system', 'optimization algorithm',
                'process automation', 'intelligent system', 'cognitive computing'
            ]
        }
    }


def create_tech_hit_flag(sentence_df, keywords_dict):
    """
    Create tech_hit flag based on keyword matching.

    Uses case-insensitive regex matching for flexibility.

    Args:
        sentence_df: DataFrame with sentence_text column
        keywords_dict: Dictionary of keyword categories

    Returns:
        DataFrame with tech_hit column
    """
    print("\n" + "=" * 80)
    print("CREATING TECH_HIT FLAGS")
    print("=" * 80)

    # Flatten all keywords into one list
    all_keywords = []
    for category, keywords in keywords_dict['Dictionary'].items():
        all_keywords.extend(keywords)

    print(f"Using {len(all_keywords)} keyword patterns")
    print(f"Categories: {list(keywords_dict['Dictionary'].keys())}")

    # Create regex pattern (word boundary matching)
    # Escape special regex characters and create pattern
    pattern_parts = []
    for keyword in all_keywords:
        # Handle multi-word phrases
        escaped = re.escape(keyword)
        # Replace spaces with flexible whitespace matcher
        escaped = escaped.replace(r'\ ', r'\s+')
        pattern_parts.append(escaped)

    # Combine into single pattern with OR
    combined_pattern = '|'.join(pattern_parts)
    regex = re.compile(r'\b(' + combined_pattern + r')\b', re.IGNORECASE)

    # Apply matching
    print("Matching keywords against sentences...")

    def has_tech_keyword(text):
        if not isinstance(text, str):
            return False
        return bool(regex.search(text))

    tqdm.pandas(desc="Matching")
    sentence_df['tech_hit'] = sentence_df['sentence_text'].progress_apply(has_tech_keyword)

    # Statistics
    tech_count = sentence_df['tech_hit'].sum()
    tech_pct = tech_count / len(sentence_df) * 100

    print(f"\n✓ Tech keyword matching complete")
    print(f"  Tech hits: {tech_count:,} ({tech_pct:.1f}%)")
    print(f"  Non-tech: {len(sentence_df) - tech_count:,} ({100 - tech_pct:.1f}%)")

    return sentence_df


# ============================================================================
# SAMPLING FUNCTIONS
# ============================================================================

def deduplicate_sentences(sentence_df):
    """
    Remove exact duplicate sentences.

    Huge efficiency gain for labeling - avoid labeling same sentence twice.
    """
    if not ENABLE_DEDUP:
        return sentence_df

    print("\n" + "=" * 80)
    print("DEDUPLICATING SENTENCES")
    print("=" * 80)

    before = len(sentence_df)

    # Keep first occurrence
    sentence_df = sentence_df.drop_duplicates(subset=['sentence_text'], keep='first')

    after = len(sentence_df)
    removed = before - after
    removed_pct = removed / before * 100

    print(f"✓ Removed {removed:,} duplicate sentences ({removed_pct:.1f}%)")
    print(f"  Unique sentences: {after:,}")

    return sentence_df


def stratified_sample(df, n, stratify_col, random_state):
    """
    Sample with stratification by a column (year or item).

    Ensures balanced representation across strata.
    """
    if stratify_col not in df.columns:
        return df.sample(n=min(n, len(df)), random_state=random_state)

    # Calculate samples per stratum
    strata = df[stratify_col].value_counts()
    n_strata = len(strata)
    per_stratum = max(1, n // n_strata)

    samples = []
    for stratum_value in strata.index:
        stratum_df = df[df[stratify_col] == stratum_value]
        sample_size = min(per_stratum, len(stratum_df))
        samples.append(stratum_df.sample(n=sample_size, random_state=random_state))

    result = pd.concat(samples, ignore_index=True)

    # If we didn't get enough, top up with random sampling
    if len(result) < n:
        remaining = n - len(result)
        remaining_df = df[~df.index.isin(result.index)]
        if len(remaining_df) > 0:
            top_up = remaining_df.sample(n=min(remaining, len(remaining_df)), random_state=random_state)
            result = pd.concat([result, top_up], ignore_index=True)

    return result.head(n)


def create_labeling_sample(sentence_df, sample_size):
    """
    Create balanced labeling sample.

    Strategy:
        - 50% from tech_hit==1 (likely tech)
        - 50% from random (including non-tech)

    Args:
        sentence_df: Full sentence DataFrame
        sample_size: Total number of sentences to sample

    Returns:
        DataFrame with sampled sentences + source_pool column
    """
    print("\n" + "=" * 80)
    print("CREATING LABELING SAMPLE")
    print("=" * 80)

    # Calculate split
    n_tech = int(sample_size * TECH_HIT_RATIO)
    n_random = sample_size - n_tech

    print(f"Target sample size: {sample_size}")
    print(f"  Tech hits (tech_hit==1): {n_tech}")
    print(f"  Random background: {n_random}")

    samples = []

    # Sample 1: Tech hits
    tech_df = sentence_df[sentence_df['tech_hit'] == True].copy()
    print(f"\nSampling from tech hits pool: {len(tech_df):,} sentences")

    if len(tech_df) < n_tech:
        print(f"  ⚠ Warning: Only {len(tech_df)} tech sentences available (requested {n_tech})")
        n_tech_actual = len(tech_df)
        tech_sample = tech_df
    else:
        if STRATIFY_BY_YEAR:
            tech_sample = stratified_sample(tech_df, n_tech, 'fiscal_year', RANDOM_SEED)
        elif STRATIFY_BY_ITEM:
            tech_sample = stratified_sample(tech_df, n_tech, 'item', RANDOM_SEED)
        else:
            tech_sample = tech_df.sample(n=n_tech, random_state=RANDOM_SEED)
        n_tech_actual = len(tech_sample)

    tech_sample['source_pool'] = 'tech_hit'
    samples.append(tech_sample)
    print(f"  ✓ Sampled {n_tech_actual} tech sentences")

    # Sample 2: Random (from full population)
    print(f"\nSampling random background: {n_random} sentences")

    # Exclude already-sampled tech sentences
    random_pool = sentence_df[~sentence_df.index.isin(tech_sample.index)].copy()
    print(f"  Random pool size: {len(random_pool):,} sentences")

    if len(random_pool) < n_random:
        print(f"  ⚠ Warning: Only {len(random_pool)} sentences available (requested {n_random})")
        random_sample = random_pool
    else:
        if STRATIFY_BY_YEAR:
            random_sample = stratified_sample(random_pool, n_random, 'fiscal_year', RANDOM_SEED)
        elif STRATIFY_BY_ITEM:
            random_sample = stratified_sample(random_pool, n_random, 'item', RANDOM_SEED)
        else:
            random_sample = random_pool.sample(n=n_random, random_state=RANDOM_SEED)

    random_sample['source_pool'] = 'random'
    samples.append(random_sample)
    print(f"  ✓ Sampled {len(random_sample)} random sentences")

    # Combine
    label_set = pd.concat(samples, ignore_index=True)

    # Shuffle
    label_set = label_set.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\n✓ Created labeling sample: {len(label_set)} sentences")

    # Summary stats
    print(f"\nSample composition:")
    print(f"  Source pool distribution:")
    print(label_set['source_pool'].value_counts())
    print(f"\n  Tech_hit distribution:")
    print(label_set['tech_hit'].value_counts())

    if 'fiscal_year' in label_set.columns:
        print(f"\n  Year distribution:")
        year_dist = label_set['fiscal_year'].value_counts().sort_index()
        for year, count in year_dist.head(10).items():
            print(f"    {year}: {count}")

    if 'item' in label_set.columns:
        print(f"\n  Item distribution:")
        print(label_set['item'].value_counts().sort_index())

    return label_set


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_labeling_sample(input_path, output_path, keywords_file, sample_size):
    """
    Main pipeline: sentence_table → label_set.

    Args:
        input_path: Path to sentence_table.csv
        output_path: Path for label_set.csv
        keywords_file: Path to tech_keywords.yaml
        sample_size: Total sentences to sample
    """
    print("=" * 80)
    print("LABELING SAMPLE BUILDER FOR TECH CLASSIFICATION")
    print("=" * 80)

    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 1: LOADING SENTENCE TABLE")
    print("=" * 80)

    input_path = Path(input_path)

    if input_path.suffix == '.parquet':
        sentence_df = pd.read_parquet(input_path)
    elif input_path.suffix == '.csv':
        sentence_df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")

    print(f"✓ Loaded {len(sentence_df):,} sentences from {input_path}")

    # Validate columns
    required_cols = ['cik', 'accession', 'fiscal_year', 'filing_date', 'item', 'sent_id', 'sentence_text']
    missing = [col for col in required_cols if col not in sentence_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ========================================================================
    # STEP 2: LOAD KEYWORDS
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 2: LOADING TECH KEYWORDS")
    print("=" * 80)

    keywords_dict = load_tech_keywords(keywords_file)

    # ========================================================================
    # STEP 3: CREATE TECH_HIT FLAG
    # ========================================================================

    sentence_df = create_tech_hit_flag(sentence_df, keywords_dict)

    # ========================================================================
    # STEP 4: DEDUPLICATE
    # ========================================================================

    sentence_df = deduplicate_sentences(sentence_df)

    # ========================================================================
    # STEP 5: SAMPLE
    # ========================================================================

    label_set = create_labeling_sample(sentence_df, sample_size)

    # ========================================================================
    # STEP 6: EXPORT
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 6: EXPORTING LABEL SET")
    print("=" * 80)

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Add label columns (empty for now - to be filled during labeling)
    label_set['TECH_IMPL'] = ''
    label_set['TECH_ADOPT'] = ''
    label_set['TECH_PRODUCT'] = ''
    label_set['NON_TECH'] = ''

    # Reorder columns for labeling workflow
    label_columns = [
        'cik', 'accession', 'fiscal_year', 'filing_date', 'item', 'sent_id',
        'sentence_text', 'tech_hit', 'source_pool',
        'TECH_IMPL', 'TECH_ADOPT', 'TECH_PRODUCT', 'NON_TECH'
    ]

    label_set = label_set[label_columns]

    # Export
    label_set.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path} ({output_path.stat().st_size / 1e6:.2f} MB)")

    # Save summary
    summary_path = output_path.with_suffix('.txt')
    with open(summary_path, 'w') as f:
        f.write("LABELING SAMPLE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Output: {output_path}\n")
        f.write(f"Sample size: {len(label_set):,}\n\n")
        f.write("Source pool distribution:\n")
        for pool, count in label_set['source_pool'].value_counts().items():
            f.write(f"  {pool}: {count:,} ({count/len(label_set)*100:.1f}%)\n")
        f.write("\nTech_hit distribution:\n")
        for hit, count in label_set['tech_hit'].value_counts().items():
            f.write(f"  {hit}: {count:,} ({count/len(label_set)*100:.1f}%)\n")
        f.write(f"\nLabeling instructions:\n")
        f.write("1. Open label_set.csv in Excel or Google Sheets\n")
        f.write("2. For each sentence, mark exactly ONE label column with '1':\n")
        f.write("   - TECH_IMPL: Technology implementation/usage\n")
        f.write("   - TECH_ADOPT: Technology adoption/investment\n")
        f.write("   - TECH_PRODUCT: Technology product/offering\n")
        f.write("   - NON_TECH: Not technology-related\n")
        f.write("3. Leave other columns as '0'\n")
        f.write("4. Constraint: If NON_TECH=1, all others must be 0\n")

    print(f"✓ Saved summary: {summary_path}")

    # ========================================================================
    # DONE
    # ========================================================================

    print("\n" + "=" * 80)
    print("BUILD COMPLETE! ✓")
    print("=" * 80)
    print(f"""
Next steps:
1. Open {output_path} in Excel or Google Sheets
2. Label sentences using the TECH_* and NON_TECH columns
3. Save and use for model training

Sample preview (first 5 sentences):
""")

    for idx, row in label_set.head(5).iterrows():
        print(f"\n[{idx+1}] Tech hit: {row['tech_hit']} | Pool: {row['source_pool']}")
        print(f"    {row['sentence_text'][:120]}...")

    return label_set


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build labeling sample from sentence table',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python build_labeling_sample.py --input sentence_table.csv --output label_set.csv
    python build_labeling_sample.py --input sentence_table.parquet --output data/labels.csv --size 3000

Output columns:
    cik, accession, fiscal_year, filing_date, item, sent_id, sentence_text,
    tech_hit, source_pool, TECH_IMPL, TECH_ADOPT, TECH_PRODUCT, NON_TECH
        """
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Path to sentence table (CSV or Parquet from build_sentence_table.py)'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Path for output label set CSV'
    )

    parser.add_argument(
        '--keywords',
        default='tech_keywords.yaml',
        help='Path to tech keywords YAML file (default: tech_keywords.yaml)'
    )

    parser.add_argument(
        '--size',
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f'Total sample size (default: {DEFAULT_SAMPLE_SIZE})'
    )

    args = parser.parse_args()

    # Run pipeline
    tqdm.pandas()
    build_labeling_sample(
        input_path=args.input,
        output_path=args.output,
        keywords_file=args.keywords,
        sample_size=args.size
    )
