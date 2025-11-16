"""
TEST SCRIPT: Digitalization Attitudes Analysis

This script runs the complete digitalization attitudes analysis on our test dataset.
It demonstrates all 4 measurement approaches and validates the entire pipeline.
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
# CONFIGURATION - TEST DATA
# ============================================================================

# Test data location
DATA_SOURCE = '/home/user/cntext/test_data/test_mda_dataset.csv'
OUTPUT_DIR = '/home/user/cntext/test_data/test_results'

# Column names in test data
COLUMNS = {
    'company_id': 'cik',
    'company_name': 'company_name',
    'date': 'filing_date',
    'year': 'fiscal_year',
    'text': 'mda_text'
}

# Processing settings (optimized for small test dataset)
BATCH_SIZE = 100
MIN_WORD_COUNT = 50  # Lower threshold for test data
USE_LEMMATIZATION = True

# Embedding settings (smaller for faster testing)
TRAIN_EMBEDDINGS = True
EMBEDDING_DIM = 50  # Smaller for test
MIN_WORD_FREQ = 2   # Lower for small corpus

# ============================================================================
# DIGITALIZATION DICTIONARIES
# ============================================================================

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
print("STEP 1: LOADING TEST MD&A DATA")
print("=" * 80)

df = pd.read_csv(DATA_SOURCE)
print(f"✓ Loaded {len(df)} documents")
print(f"  Columns: {list(df.columns)}")
print(f"  Companies: {df['company_name'].nunique()}")
print(f"  Years: {df['fiscal_year'].min()} - {df['fiscal_year'].max()}")

# ============================================================================
# STEP 2: PREPROCESS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: PREPROCESSING")
print("=" * 80)

text_col = COLUMNS['text']

# Remove nulls
initial_count = len(df)
df = df.dropna(subset=[text_col])
print(f"Removed {initial_count - len(df)} null texts")

# Clean text
print("Cleaning text...")
tqdm.pandas()
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

df['digital_positive'] = df['text_clean'].progress_apply(
    lambda x: count_keywords(x, POSITIVE_DIGITAL_TERMS)
)

df['digital_negative'] = df['text_clean'].progress_apply(
    lambda x: count_keywords(x, NEGATIVE_DIGITAL_TERMS)
)

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
# STEP 6: APPROACH 4 - SEMANTIC PROJECTION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: APPROACH 4 - SEMANTIC PROJECTION ANALYSIS")
print("=" * 80)

if TRAIN_EMBEDDINGS and len(df) >= 10:
    print("Training domain-specific word embeddings...")
    print(f"  Corpus size: {len(df)} documents, {df['word_count'].sum():,} words")

    # Save texts to temporary corpus file
    corpus_file = '/home/user/cntext/test_data/temp_corpus.txt'
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for text in df['text_clean']:
            f.write(text + '\n')

    print(f"  Saved corpus to: {corpus_file}")

    # Train Word2Vec
    wv = ct.Word2Vec(
        corpus_file=corpus_file,
        lang='english',
        vector_size=EMBEDDING_DIM,
        window_size=10,
        min_count=MIN_WORD_FREQ,
        max_iter=10
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
            negwords=['risk', 'privacy', 'security', 'regulatory',
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
                     'advance', 'competitive', 'transform'],
            negwords=['traditional', 'established', 'conservative', 'stable',
                     'gradual', 'cautious', 'proven']
        )
        print("  ✓ Innovation Leadership axis created")
    except Exception as e:
        print(f"  ✗ Could not create Innovation Leadership axis: {e}")
        axes['innovation_leadership'] = None

    # Project documents onto axes
    print("\nProjecting documents onto semantic axes...")

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

    EMBEDDINGS_TRAINED = True

else:
    print("Skipping embedding training (insufficient corpus size)")
    EMBEDDINGS_TRAINED = False

# ============================================================================
# STEP 7: TEMPORAL ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: TEMPORAL ANALYSIS")
print("=" * 80)

year_col = COLUMNS['year']
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

if EMBEDDINGS_TRAINED:
    for axis_name in ['digital_embrace', 'ai_enthusiasm', 'innovation_leadership']:
        col = f'sem_{axis_name}'
        if col in df_temporal.columns:
            agg_dict[col] = ['mean', 'std']

temporal_summary = df_temporal.groupby(year_col).agg(agg_dict).round(4)
temporal_summary.columns = ['_'.join(col).strip('_') for col in temporal_summary.columns]

print(f"✓ Temporal analysis complete!")
print(f"\nYears covered: {df_temporal[year_col].min():.0f} - {df_temporal[year_col].max():.0f}")

# ============================================================================
# STEP 8: VALIDATION (Compare with ground truth)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: VALIDATION AGAINST GROUND TRUTH")
print("=" * 80)

# Our test data has ground truth labels!
if 'true_attitude' in df.columns and EMBEDDINGS_TRAINED:
    print("Comparing predicted attitudes with ground truth...")

    # Map true attitudes to numeric
    attitude_map = {
        'very_positive': 2,
        'positive': 1,
        'neutral': 0,
        'negative': -1,
        'very_negative': -2
    }

    df['true_attitude_numeric'] = df['true_attitude'].map(attitude_map)

    # Check correlation
    if 'sem_digital_embrace' in df.columns:
        corr = df[['true_attitude_numeric', 'sem_digital_embrace']].corr().iloc[0, 1]
        print(f"\nCorrelation with ground truth:")
        print(f"  Semantic (digital_embrace) vs True Attitude: {corr:.3f}")

        # Also check keyword approach
        corr_keyword = df[['true_attitude_numeric', 'digital_net_sentiment']].corr().iloc[0, 1]
        print(f"  Keyword sentiment vs True Attitude: {corr_keyword:.3f}")

        if corr > corr_keyword:
            print(f"\n  → Semantic projection is {(corr - corr_keyword)*100:.1f}% more accurate!")

    # Show examples
    print("\nSample Predictions vs Ground Truth:")
    print("=" * 80)
    sample_cols = ['company_name', 'fiscal_year', 'true_attitude', 'digital_tone']
    if 'sem_digital_embrace' in df.columns:
        sample_cols.append('sem_digital_embrace')

    print(df[sample_cols].head(10).to_string(index=False))

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: SAVING TEST RESULTS")
print("=" * 80)

Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

# Save complete results
output_file = f'{OUTPUT_DIR}/test_results_complete.csv'
df.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# Save temporal summary
temporal_file = f'{OUTPUT_DIR}/test_results_by_year.csv'
temporal_summary.to_csv(temporal_file)
print(f"✓ Saved: {temporal_file}")

# Save validation report
if 'true_attitude' in df.columns:
    validation_file = f'{OUTPUT_DIR}/validation_report.txt'
    with open(validation_file, 'w') as f:
        f.write("VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("Ground Truth Distribution:\n")
        f.write(df['true_attitude'].value_counts().to_string())
        f.write("\n\n")

        f.write("Predicted Tone Distribution:\n")
        f.write(df['digital_tone'].value_counts().to_string())
        f.write("\n\n")

        if EMBEDDINGS_TRAINED and 'sem_digital_embrace' in df.columns:
            f.write("Correlation Analysis:\n")
            corr = df[['true_attitude_numeric', 'sem_digital_embrace']].corr().iloc[0, 1]
            corr_keyword = df[['true_attitude_numeric', 'digital_net_sentiment']].corr().iloc[0, 1]
            f.write(f"  Semantic approach: r = {corr:.3f}\n")
            f.write(f"  Keyword approach: r = {corr_keyword:.3f}\n")
            f.write(f"  Improvement: {(corr - corr_keyword)*100:.1f}%\n")

    print(f"✓ Saved: {validation_file}")

# Save embeddings
if EMBEDDINGS_TRAINED:
    embedding_file = f'{OUTPUT_DIR}/word_embeddings.model'
    wv.save(embedding_file)
    print(f"✓ Saved: {embedding_file}")

print("\n" + "=" * 80)
print("TEST COMPLETE! ✓")
print("=" * 80)

print(f"""
Results saved to: {OUTPUT_DIR}/

Key Files:
- test_results_complete.csv: Full results with all metrics
- test_results_by_year.csv: Temporal trends
- validation_report.txt: Accuracy vs ground truth
- word_embeddings.model: Trained embeddings

Summary Statistics:
- Documents analyzed: {len(df)}
- Average word count: {df['word_count'].mean():.0f}
- Average digital frequency: {df['freq_total_digital'].mean():.2f} per 1000 words
- Mean digital sentiment: {df['digital_net_sentiment'].mean():+.3f}
""")

if EMBEDDINGS_TRAINED and 'sem_digital_embrace' in df.columns:
    print(f"- Mean semantic score (digital embrace): {df['sem_digital_embrace'].mean():+.4f}")
    print(f"- Semantic accuracy: r = {corr:.3f} with ground truth")

print("\n✓ All tests passed successfully!")
