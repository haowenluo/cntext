"""
Quick-Start Template: Measure Corporate Attitudes Toward Digitalization & AI

This template analyzes MD&A sections (or full 10-K filings) to quantify companies'
attitudes toward digitalization, technology adoption, and AI use.

Implements 4 complementary approaches:
1. Keyword frequency (baseline)
2. Normalized frequency (accounts for length)
3. Dictionary-based sentiment (positive vs skeptical)
4. Semantic projection (BEST - measures latent attitudes)

Customize the CONFIGURATION section for your specific data structure.
"""

import sys
sys.path.insert(0, '/home/user/cntext')

import pandas as pd
import cntext as ct
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ============================================================================
# CONFIGURATION - CUSTOMIZE THESE
# ============================================================================

# Your data location
DATA_SOURCE = '/path/to/your/mda_data.csv'  # Or directory with JSON files
OUTPUT_DIR = 'digital_attitudes_results'

# Column names in your data (adjust to match your structure)
COLUMNS = {
    'company_id': 'cik',          # Company identifier
    'company_name': 'company_name',
    'date': 'filing_date',         # Filing date
    'year': 'fiscal_year',         # Fiscal year
    'text': 'mda_text'            # MD&A text content (or full 10-K)
}

# Processing settings
BATCH_SIZE = 100                   # Process N documents at a time
MIN_WORD_COUNT = 100              # Filter out very short texts
USE_LEMMATIZATION = True          # Better quality, slower processing

# Embedding settings (for semantic projection)
TRAIN_EMBEDDINGS = True           # Train domain-specific embeddings
EMBEDDING_DIM = 100               # Dimensionality of word vectors
MIN_WORD_FREQ = 5                 # Minimum word frequency for embeddings

# ============================================================================
# DIGITALIZATION DICTIONARIES
# ============================================================================

# Define keyword dictionaries for different aspects
DIGITAL_KEYWORDS = {
    'core_digital': [
        'digital', 'digitalization', 'digitization', 'digitalize', 'digitize',
        'online', 'e-commerce', 'internet', 'web', 'cloud', 'platform'
    ],

    'technology': [
        'technology', 'technological', 'innovation', 'innovative', 'automation',
        'automate', 'transform', 'transformation', 'upgrade', 'modernize',
        'software', 'hardware', 'system', 'infrastructure'
    ],

    'ai_ml': [
        'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning',
        'neural network', 'algorithm', 'predictive', 'intelligent', 'cognitive',
        'analytics', 'data science', 'big data'
    ],

    'emerging_tech': [
        'blockchain', 'cryptocurrency', 'iot', 'internet of things', '5g',
        'robotics', 'drone', 'augmented reality', 'virtual reality', 'ar', 'vr',
        'quantum', 'edge computing'
    ]
}

# Sentiment-aware dictionaries
POSITIVE_DIGITAL_TERMS = [
    'adopt', 'adoption', 'implement', 'deploy', 'invest', 'investment',
    'opportunity', 'growth', 'enhance', 'improve', 'advantage', 'competitive',
    'efficiency', 'optimize', 'streamline', 'leverage', 'enable', 'empower',
    'accelerate', 'advance', 'pioneer', 'leader', 'strategic', 'potential'
]

NEGATIVE_DIGITAL_TERMS = [
    'risk', 'concern', 'challenge', 'threat', 'vulnerability', 'disruption',
    'liability', 'uncertain', 'uncertainty', 'compliance', 'regulatory',
    'cost', 'expense', 'barrier', 'limitation', 'constraint', 'difficulty',
    'security', 'privacy', 'breach', 'cyberattack', 'failure'
]

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

    print(f"✓ Loaded {len(df)} documents")
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
    """Clean and prepare text."""

    text_col = COLUMNS['text']

    # Remove nulls
    initial_count = len(df)
    df = df.dropna(subset=[text_col])
    print(f"Removed {initial_count - len(df)} null texts")

    # Clean text
    print("Cleaning text (contractions, encoding)...")
    df['text_clean'] = df[text_col].progress_apply(
        lambda x: ct.clean_text(ct.fix_contractions(str(x)), lang='english')
    )

    # Add metrics
    df['word_count'] = df['text_clean'].str.split().str.len()
    df['char_count'] = df['text_clean'].str.len()

    # Filter short texts
    before = len(df)
    df = df[df['word_count'] >= MIN_WORD_COUNT]
    print(f"Filtered {before - len(df)} short texts (<{MIN_WORD_COUNT} words)")

    print(f"✓ Preprocessed: {len(df)} documents ready")
    print(f"  Average length: {df['word_count'].mean():.0f} words")
    print(f"  Total corpus size: {df['word_count'].sum():,} words")

    return df

tqdm.pandas()  # Enable progress_apply
df = preprocess(df)

# ============================================================================
# STEP 3: APPROACH 1 - SIMPLE KEYWORD FREQUENCY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: APPROACH 1 - KEYWORD FREQUENCY ANALYSIS")
print("=" * 80)

def count_keywords(text, keywords):
    """Count occurrences of keywords in text."""
    text_lower = text.lower()
    count = 0
    for keyword in keywords:
        # Handle multi-word phrases
        count += text_lower.count(keyword.lower())
    return count

print("Counting digital, tech, and AI keywords...")

# Count each category
for category, keywords in DIGITAL_KEYWORDS.items():
    df[f'count_{category}'] = df['text_clean'].progress_apply(
        lambda x: count_keywords(x, keywords)
    )

# Total digital mentions
digital_cols = [f'count_{cat}' for cat in DIGITAL_KEYWORDS.keys()]
df['count_total_digital'] = df[digital_cols].sum(axis=1)

print(f"✓ Keyword counting complete!")
print(f"\nAverage mentions per document:")
for category in DIGITAL_KEYWORDS.keys():
    avg = df[f'count_{category}'].mean()
    print(f"  {category}: {avg:.2f}")
print(f"  Total digital: {df['count_total_digital'].mean():.2f}")

# ============================================================================
# STEP 4: APPROACH 2 - NORMALIZED FREQUENCY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: APPROACH 2 - NORMALIZED FREQUENCY (per 1000 words)")
print("=" * 80)

# Normalize by document length
for category in DIGITAL_KEYWORDS.keys():
    df[f'freq_{category}'] = (
        df[f'count_{category}'] / df['word_count'] * 1000
    )

df['freq_total_digital'] = df['count_total_digital'] / df['word_count'] * 1000

print(f"✓ Frequency normalization complete!")
print(f"\nAverage frequency per 1000 words:")
for category in DIGITAL_KEYWORDS.keys():
    avg = df[f'freq_{category}'].mean()
    print(f"  {category}: {avg:.2f}")
print(f"  Total digital: {df['freq_total_digital'].mean():.2f}")

# ============================================================================
# STEP 5: APPROACH 3 - SENTIMENT-AWARE DICTIONARY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: APPROACH 3 - SENTIMENT-AWARE ANALYSIS")
print("=" * 80)

print("Analyzing positive vs negative digital sentiment...")

# Count positive and negative terms
df['digital_positive'] = df['text_clean'].progress_apply(
    lambda x: count_keywords(x, POSITIVE_DIGITAL_TERMS)
)

df['digital_negative'] = df['text_clean'].progress_apply(
    lambda x: count_keywords(x, NEGATIVE_DIGITAL_TERMS)
)

# Calculate net sentiment and tone
df['digital_net_sentiment'] = (
    (df['digital_positive'] - df['digital_negative']) /
    (df['digital_positive'] + df['digital_negative'] + 1)
)

df['digital_tone'] = df['digital_net_sentiment'].apply(
    lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
)

print(f"✓ Sentiment analysis complete!")
print(f"\nDigital sentiment distribution:")
print(f"  Mean positive terms: {df['digital_positive'].mean():.2f}")
print(f"  Mean negative terms: {df['digital_negative'].mean():.2f}")
print(f"  Mean net sentiment: {df['digital_net_sentiment'].mean():+.3f}")
print(f"\nTone distribution:")
print(df['digital_tone'].value_counts())

# ============================================================================
# STEP 6: APPROACH 4 - SEMANTIC PROJECTION (BEST!)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: APPROACH 4 - SEMANTIC PROJECTION ANALYSIS")
print("=" * 80)

if TRAIN_EMBEDDINGS and len(df) >= 50:  # Need minimum corpus size
    print("Training domain-specific word embeddings...")
    print(f"  Corpus size: {len(df)} documents, {df['word_count'].sum():,} words")
    print(f"  This may take several minutes...")

    # Train Word2Vec on the corpus
    wv = ct.Word2Vec(
        texts=df['text_clean'].tolist(),
        sg=1,                    # Skip-gram
        vector_size=EMBEDDING_DIM,
        window=10,               # Larger window for semantic relationships
        min_count=MIN_WORD_FREQ,
        workers=4,
        epochs=10,
        lang='english',
        lemmatize=USE_LEMMATIZATION
    )

    print(f"✓ Embeddings trained! Vocabulary size: {len(wv.key_to_index)}")

    # Create semantic axes
    print("\nGenerating semantic concept axes...")

    axes = {}

    # Axis 1: Digital Embracer ←→ Digital Skeptic
    try:
        axes['digital_embrace'] = ct.generate_concept_axis(
            wv,
            poswords=['digital', 'innovation', 'transform', 'opportunity',
                     'leverage', 'strategic', 'adopt', 'invest'],
            negwords=['risk', 'concern', 'threat', 'challenge', 'uncertain',
                     'liability', 'constraint', 'barrier']
        )
        print("  ✓ Digital Embrace axis created")
    except Exception as e:
        print(f"  ✗ Could not create Digital Embrace axis: {e}")
        axes['digital_embrace'] = None

    # Axis 2: AI Enthusiast ←→ AI Cautious
    try:
        axes['ai_enthusiasm'] = ct.generate_concept_axis(
            wv,
            poswords=['ai', 'intelligent', 'automation', 'analytics',
                     'predictive', 'optimize', 'enhance', 'potential'],
            negwords=['risk', 'bias', 'privacy', 'security', 'regulatory',
                     'compliance', 'failure', 'uncertain']
        )
        print("  ✓ AI Enthusiasm axis created")
    except Exception as e:
        print(f"  ✗ Could not create AI Enthusiasm axis: {e}")
        axes['ai_enthusiasm'] = None

    # Axis 3: Innovation Leader ←→ Cautious Follower
    try:
        axes['innovation_leadership'] = ct.generate_concept_axis(
            wv,
            poswords=['pioneer', 'leader', 'innovate', 'breakthrough',
                     'advance', 'competitive', 'disruptive', 'transform'],
            negwords=['traditional', 'established', 'conservative', 'stable',
                     'gradual', 'cautious', 'proven', 'conventional']
        )
        print("  ✓ Innovation Leadership axis created")
    except Exception as e:
        print(f"  ✗ Could not create Innovation Leadership axis: {e}")
        axes['innovation_leadership'] = None

    # Project documents onto axes
    print("\nProjecting documents onto semantic axes...")
    print("  This measures LATENT ATTITUDES beyond keyword matching")

    for axis_name, axis in axes.items():
        if axis is not None:
            df[f'sem_{axis_name}'] = df['text_clean'].progress_apply(
                lambda x: ct.project_text(wv, x, axis, lang='english')
            )

            mean_score = df[f'sem_{axis_name}'].mean()
            std_score = df[f'sem_{axis_name}'].std()

            print(f"\n  {axis_name}:")
            print(f"    Mean score: {mean_score:+.4f}")
            print(f"    Std dev: {std_score:.4f}")
            print(f"    Range: [{df[f'sem_{axis_name}'].min():.4f}, {df[f'sem_{axis_name}'].max():.4f}]")

    print("\n✓ Semantic projection complete!")
    print("\nInterpretation guide:")
    print("  digital_embrace: +score = embracing, -score = skeptical")
    print("  ai_enthusiasm: +score = enthusiastic, -score = cautious")
    print("  innovation_leadership: +score = leader, -score = follower")

    EMBEDDINGS_TRAINED = True

else:
    print("Skipping embedding training (insufficient corpus size or disabled)")
    print(f"  Need at least 50 documents, have {len(df)}")
    EMBEDDINGS_TRAINED = False

# ============================================================================
# STEP 7: TEMPORAL ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: TEMPORAL ANALYSIS")
print("=" * 80)

if COLUMNS['year'] in df.columns and df[COLUMNS['year']].notna().sum() > 0:
    print("Analyzing trends over time...")

    # Group by year
    year_col = COLUMNS['year']

    # Ensure year is numeric
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    df_temporal = df.dropna(subset=[year_col])

    agg_dict = {
        'freq_total_digital': ['mean', 'median', 'std'],
        'freq_core_digital': 'mean',
        'freq_technology': 'mean',
        'freq_ai_ml': 'mean',
        'digital_net_sentiment': ['mean', 'std'],
        COLUMNS['company_id']: 'count'
    }

    # Add semantic scores if available
    if EMBEDDINGS_TRAINED:
        for axis_name in ['digital_embrace', 'ai_enthusiasm', 'innovation_leadership']:
            col = f'sem_{axis_name}'
            if col in df_temporal.columns:
                agg_dict[col] = ['mean', 'std']

    temporal_summary = df_temporal.groupby(year_col).agg(agg_dict).round(4)
    temporal_summary.columns = ['_'.join(col).strip('_') for col in temporal_summary.columns]

    print(f"✓ Temporal analysis complete!")
    print(f"\nYears covered: {df_temporal[year_col].min():.0f} - {df_temporal[year_col].max():.0f}")
    print(f"Documents per year: {len(df_temporal) / df_temporal[year_col].nunique():.0f} average")

    TEMPORAL_AVAILABLE = True
else:
    print("Skipping temporal analysis (no year column or all null)")
    TEMPORAL_AVAILABLE = False

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: SAVING RESULTS")
print("=" * 80)

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# 1. Save complete results
output_file = f'{OUTPUT_DIR}/digital_attitudes_complete.csv'
df.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# 2. Save temporal summary
if TEMPORAL_AVAILABLE:
    temporal_file = f'{OUTPUT_DIR}/digital_attitudes_by_year.csv'
    temporal_summary.to_csv(temporal_file)
    print(f"✓ Saved: {temporal_file}")

# 3. Save company-level summary (top 30)
if COLUMNS['company_name'] in df.columns:
    company_agg = {
        'freq_total_digital': 'mean',
        'digital_net_sentiment': 'mean',
        COLUMNS['company_id']: 'first'
    }

    if EMBEDDINGS_TRAINED and 'sem_digital_embrace' in df.columns:
        company_agg['sem_digital_embrace'] = 'mean'
        company_agg['sem_ai_enthusiasm'] = 'mean'

    company_summary = df.groupby(COLUMNS['company_name']).agg(company_agg).round(4)
    company_summary = company_summary.sort_values('freq_total_digital', ascending=False)

    top_companies = company_summary.head(30)
    company_file = f'{OUTPUT_DIR}/digital_attitudes_top_companies.csv'
    top_companies.to_csv(company_file)
    print(f"✓ Saved: {company_file}")

# 4. Save column documentation
with open(f'{OUTPUT_DIR}/README.txt', 'w') as f:
    f.write("Digital Attitudes Analysis Results\n")
    f.write("=" * 50 + "\n\n")
    f.write("Columns:\n\n")

    f.write("APPROACH 1: Keyword Frequency (Raw Counts)\n")
    f.write("  count_core_digital - Count of digital/online keywords\n")
    f.write("  count_technology - Count of technology keywords\n")
    f.write("  count_ai_ml - Count of AI/ML keywords\n")
    f.write("  count_emerging_tech - Count of blockchain/IoT/5G keywords\n")
    f.write("  count_total_digital - Total digital mentions\n\n")

    f.write("APPROACH 2: Normalized Frequency (per 1000 words)\n")
    f.write("  freq_core_digital - Digital keyword frequency\n")
    f.write("  freq_technology - Technology keyword frequency\n")
    f.write("  freq_ai_ml - AI/ML keyword frequency\n")
    f.write("  freq_emerging_tech - Emerging tech keyword frequency\n")
    f.write("  freq_total_digital - Total digital frequency\n\n")

    f.write("APPROACH 3: Sentiment-Aware Dictionary\n")
    f.write("  digital_positive - Count of positive digital terms\n")
    f.write("  digital_negative - Count of negative digital terms\n")
    f.write("  digital_net_sentiment - Net sentiment score (-1 to 1)\n")
    f.write("  digital_tone - Overall tone (positive/neutral/negative)\n\n")

    if EMBEDDINGS_TRAINED:
        f.write("APPROACH 4: Semantic Projection (Most Sophisticated!)\n")
        f.write("  sem_digital_embrace - Digital embracer (+) vs skeptic (-)\n")
        f.write("  sem_ai_enthusiasm - AI enthusiast (+) vs cautious (-)\n")
        f.write("  sem_innovation_leadership - Leader (+) vs follower (-)\n\n")

    f.write("Note: Semantic projection captures LATENT ATTITUDES that go\n")
    f.write("beyond simple keyword matching. This is the most nuanced measure.\n")

print(f"✓ Saved: {OUTPUT_DIR}/README.txt")

# 5. Save embeddings if trained
if EMBEDDINGS_TRAINED:
    embedding_file = f'{OUTPUT_DIR}/word_embeddings.model'
    wv.save(embedding_file)
    print(f"✓ Saved: {embedding_file}")

# ============================================================================
# STEP 9: GENERATE ANALYSIS SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

print(f"\nTotal documents analyzed: {len(df)}")
print(f"Average document length: {df['word_count'].mean():.0f} words")

print("\n--- APPROACH 1 & 2: Keyword Analysis ---")
print(f"Average digital mentions per document: {df['count_total_digital'].mean():.2f}")
print(f"Average digital frequency (per 1000 words): {df['freq_total_digital'].mean():.2f}")
print(f"\nBreakdown:")
print(f"  Core digital terms: {df['freq_core_digital'].mean():.2f} per 1000 words")
print(f"  General technology: {df['freq_technology'].mean():.2f} per 1000 words")
print(f"  AI/ML terms: {df['freq_ai_ml'].mean():.2f} per 1000 words")
print(f"  Emerging tech: {df['freq_emerging_tech'].mean():.2f} per 1000 words")

print("\n--- APPROACH 3: Sentiment Analysis ---")
print(f"Average net digital sentiment: {df['digital_net_sentiment'].mean():+.3f}")
print(f"Tone distribution:")
for tone in ['positive', 'neutral', 'negative']:
    count = (df['digital_tone'] == tone).sum()
    pct = count / len(df) * 100
    print(f"  {tone.capitalize()}: {count} ({pct:.1f}%)")

if EMBEDDINGS_TRAINED:
    print("\n--- APPROACH 4: Semantic Projection ---")
    print("(Most sophisticated measure of attitudes!)")

    for axis_name in ['digital_embrace', 'ai_enthusiasm', 'innovation_leadership']:
        col = f'sem_{axis_name}'
        if col in df.columns:
            mean_score = df[col].mean()
            positive_pct = (df[col] > 0.02).mean() * 100
            negative_pct = (df[col] < -0.02).mean() * 100

            print(f"\n  {axis_name.replace('_', ' ').title()}:")
            print(f"    Mean score: {mean_score:+.4f}")
            print(f"    High positive: {positive_pct:.1f}%")
            print(f"    High negative: {negative_pct:.1f}%")

if TEMPORAL_AVAILABLE:
    print(f"\n--- TEMPORAL TRENDS ---")
    print(f"Years covered: {df_temporal[year_col].min():.0f} - {df_temporal[year_col].max():.0f}")

    # Calculate trend
    first_year = df_temporal[year_col].min()
    last_year = df_temporal[year_col].max()

    first_freq = df_temporal[df_temporal[year_col] == first_year]['freq_total_digital'].mean()
    last_freq = df_temporal[df_temporal[year_col] == last_year]['freq_total_digital'].mean()

    change = ((last_freq - first_freq) / first_freq * 100) if first_freq > 0 else 0

    print(f"Digital frequency trend: {first_freq:.2f} ({first_year:.0f}) → {last_freq:.2f} ({last_year:.0f})")
    print(f"Change: {change:+.1f}%")

print(f"\n\nResults saved to: {OUTPUT_DIR}/")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE! ✓")
print("=" * 80)

print("""
Next Steps for Research:

1. Review Results:
   - digital_attitudes_complete.csv: Full dataset with all metrics
   - digital_attitudes_by_year.csv: Temporal trends
   - digital_attitudes_top_companies.csv: Company-level patterns

2. Statistical Analysis:
   - Correlation between semantic scores and keyword frequency
   - Regression analysis with financial performance
   - Time-series analysis of attitude evolution
   - Industry comparisons (add industry classification)

3. Validation:
   - Manual review of high/low scoring documents
   - Compare semantic projection with dictionary methods
   - Check for face validity with domain experts

4. Research Questions to Explore:
   - Are attitudes becoming more positive over time?
   - Do digital embracers outperform skeptics financially?
   - Industry differences in AI enthusiasm?
   - Does digital adoption correlate with innovation?
   - Market reactions to changes in digital attitudes?

5. Advanced Extensions:
   - Add industry fixed effects
   - Control for company size, age, sector
   - Examine specific technologies (cloud, blockchain, etc.)
   - Analyze Q&A sections separately
   - Compare MD&A vs risk factors sections

Publication Strategy:
- Semantic projection is a PUBLISHED methodology (unique!)
- Emphasize theory-driven approach
- Show validation across multiple methods
- Highlight temporal dynamics
- Consider accounting, finance, or strategy journals

For implementation questions, see:
- guide_digitalization_attitudes.py (conceptual guide)
- guide_mda_analysis.py (general MD&A analysis guide)
- ENGLISH_ENHANCEMENTS.md (technical documentation)
""")
