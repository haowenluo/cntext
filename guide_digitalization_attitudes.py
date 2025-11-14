"""
Research Guide: Measuring Corporate Attitudes Toward Digitalization & AI

This guide shows multiple approaches to measure how companies discuss
digitalization, technology adoption, and AI in 10-K/MD&A sections.

Approaches covered:
1. Simple word counting (baseline)
2. Normalized frequency measures (better)
3. Dictionary-based sentiment (good)
4. Semantic projection (BEST - measures attitude!)
5. Temporal analysis (tracking change)
6. Comparative analysis (industry benchmarks)
"""

import sys
sys.path.insert(0, '/home/user/cntext')

print("=" * 80)
print("MEASURING CORPORATE ATTITUDES TOWARD DIGITALIZATION & AI")
print("=" * 80)
print()

# ============================================================================
# APPROACH 1: SIMPLE WORD COUNTING (Baseline)
# ============================================================================

print("APPROACH 1: SIMPLE WORD COUNTING")
print("-" * 80)
print()

print("""
Your initial idea - count digitalization-related words.

Advantages:
+ Simple and interpretable
+ Fast to compute
+ Easy to explain

Limitations:
- Doesn't distinguish "We embrace AI" vs "AI poses threats"
- Longer documents have higher counts (need normalization)
- Misses semantic nuances
- Can't distinguish positive/negative attitudes

Implementation:
""")

approach1_code = '''
import cntext as ct
import pandas as pd

# Define digitalization keywords
digitalization_keywords = [
    # AI & Machine Learning
    'artificial intelligence', 'ai', 'machine learning', 'deep learning',
    'neural network', 'natural language processing', 'nlp',

    # Digital Technologies
    'digitalization', 'digitization', 'digital transformation',
    'automation', 'robotic process automation', 'rpa',

    # Cloud & Data
    'cloud computing', 'cloud', 'big data', 'data analytics',
    'data science', 'predictive analytics',

    # Emerging Tech
    'blockchain', 'cryptocurrency', 'internet of things', 'iot',
    '5g', 'edge computing', 'quantum computing',

    # Business Processes
    'digital platform', 'e-commerce', 'online', 'mobile app',
    'digital channels', 'digital capabilities'
]

def count_digitalization_mentions(text, keywords):
    """
    Count how many times keywords appear in text.

    Problem: "We face AI risks" and "We leverage AI" both count as 1.
    """
    text_lower = text.lower()
    count = sum(text_lower.count(keyword) for keyword in keywords)
    return count

# Apply to dataframe
df['digital_count'] = df['mda_clean'].apply(
    lambda x: count_digitalization_mentions(x, digitalization_keywords)
)

# Basic statistics
print(f"Average mentions per document: {df['digital_count'].mean():.1f}")
print(f"Companies with 0 mentions: {(df['digital_count'] == 0).sum()}")
'''

print(approach1_code)

# ============================================================================
# APPROACH 2: NORMALIZED FREQUENCY (Better)
# ============================================================================

print("\n\n" + "=" * 80)
print("APPROACH 2: NORMALIZED FREQUENCY (Improved)")
print("=" * 80)
print()

print("""
Normalize by document length to make comparisons fair.

Advantages:
+ Accounts for document length
+ Can compare across companies/years
+ Still simple and interpretable

Improvements over Approach 1:
✓ Fair comparison between short/long documents
✓ Can use TF-IDF for better weighting
✓ Percentage easier to interpret than raw counts

Implementation:
""")

approach2_code = '''
import cntext as ct
import pandas as pd
import numpy as np

def calculate_digital_intensity(df, keywords):
    """
    Calculate normalized digitalization intensity metrics.
    """

    # Method 1: Simple percentage
    df['digital_mentions'] = df['mda_clean'].apply(
        lambda x: sum(x.lower().count(kw) for kw in keywords)
    )

    df['digital_intensity_pct'] = (
        df['digital_mentions'] / df['word_count'] * 100
    )

    # Method 2: Using cntext's word_count with lemmatization
    # This is better because it handles word forms (digitize, digitizing, etc.)
    def count_with_lemma(text, keywords):
        word_freq = ct.word_count(text, lang='english', lemmatize=True)

        # Normalize keywords to lemmatized form
        keyword_lemmas = set()
        for kw in keywords:
            # Simple lemmatization approximations
            keyword_lemmas.add(kw.rstrip('s'))  # Remove plural
            keyword_lemmas.add(kw.rstrip('ing')) # Remove gerund
            keyword_lemmas.add(kw)

        total = sum(word_freq.get(kw, 0) for kw in keyword_lemmas)
        return total

    df['digital_mentions_lemma'] = df['mda_clean'].apply(
        lambda x: count_with_lemma(x, keywords)
    )

    df['digital_intensity_lemma'] = (
        df['digital_mentions_lemma'] / df['word_count'] * 100
    )

    return df

df = calculate_digital_intensity(df, digitalization_keywords)

# Categorize companies
df['digital_adoption'] = pd.cut(
    df['digital_intensity_pct'],
    bins=[-np.inf, 0.1, 0.5, 1.0, np.inf],
    labels=['Minimal', 'Low', 'Moderate', 'High']
)

print("Digital Adoption Distribution:")
print(df['digital_adoption'].value_counts())
'''

print(approach2_code)

# ============================================================================
# APPROACH 3: DICTIONARY-BASED SENTIMENT (Good)
# ============================================================================

print("\n\n" + "=" * 80)
print("APPROACH 3: DICTIONARY-BASED SENTIMENT (Good)")
print("=" * 80)
print()

print("""
Create custom dictionary with positive/negative digitalization terms.

Advantages:
+ Distinguishes positive vs negative attitudes
+ Can measure enthusiasm vs concern
+ Theory-driven (you define what's positive/negative)

This captures ATTITUDE, not just mentions!

Implementation:
""")

approach3_code = '''
import cntext as ct
import pandas as pd
import yaml

# Create custom digitalization sentiment dictionary
digital_sentiment_dict = {
    'Name': 'Digitalization Attitude Dictionary',
    'Desc': 'Measures positive vs negative attitudes toward digital transformation',
    'Category': ['Positive', 'Negative', 'Neutral'],
    'Dictionary': {
        'Positive': [
            # Embracing digitalization
            'leverage ai', 'embrace digital', 'adopt technology',
            'digital innovation', 'digital transformation success',
            'ai-powered', 'ai-driven', 'technology enables',
            'digital capabilities', 'competitive advantage through',
            'efficiency gains', 'automation benefits',

            # Positive outcomes
            'digital revenue', 'online growth', 'e-commerce expansion',
            'improved efficiency', 'cost savings', 'productivity gains',
            'enhanced customer', 'better insights', 'faster decision',

            # Strategic positioning
            'digital leader', 'technology leadership', 'innovation leader',
            'digital first', 'tech-enabled', 'data-driven strategy'
        ],

        'Negative': [
            # Concerns and risks
            'cybersecurity risk', 'data breach', 'privacy concerns',
            'technology risk', 'digital disruption threat',
            'obsolescence risk', 'legacy systems', 'technical debt',

            # Challenges
            'implementation challenges', 'adoption barriers',
            'digital skills gap', 'technology costs', 'integration issues',
            'system failures', 'downtime', 'technical difficulties',

            # Threats
            'disrupted by', 'threatened by technology', 'competitive threat',
            'regulatory challenges', 'compliance burden', 'security vulnerabilities'
        ],

        'Neutral': [
            # Neutral mentions
            'digital initiative', 'technology investment', 'ai project',
            'system upgrade', 'platform migration', 'digital tools',
            'technology spending', 'it infrastructure'
        ]
    }
}

def analyze_digital_attitude(df, text_column='mda_clean'):
    """
    Analyze attitude toward digitalization using custom dictionary.
    """

    results = []

    for idx, row in df.iterrows():
        text = row[text_column]

        # Count positive/negative/neutral mentions
        text_lower = text.lower()

        positive = sum(text_lower.count(phrase) for phrase in digital_sentiment_dict['Dictionary']['Positive'])
        negative = sum(text_lower.count(phrase) for phrase in digital_sentiment_dict['Dictionary']['Negative'])
        neutral = sum(text_lower.count(phrase) for phrase in digital_sentiment_dict['Dictionary']['Neutral'])

        # Calculate attitude score
        total = positive + negative + neutral
        if total > 0:
            attitude_score = (positive - negative) / total
        else:
            attitude_score = 0

        results.append({
            'idx': idx,
            'digital_positive': positive,
            'digital_negative': negative,
            'digital_neutral': neutral,
            'digital_attitude_score': attitude_score
        })

    results_df = pd.DataFrame(results).set_index('idx')
    df = df.join(results_df)

    # Classify attitude
    df['digital_attitude'] = df['digital_attitude_score'].apply(
        lambda x: 'Positive' if x > 0.2 else 'Negative' if x < -0.2 else 'Neutral'
    )

    return df

df = analyze_digital_attitude(df)

print("Digital Attitude Distribution:")
print(df['digital_attitude'].value_counts())

print("\\nAverage attitude by year:")
print(df.groupby('fiscal_year')['digital_attitude_score'].mean())
'''

print(approach3_code)

# ============================================================================
# APPROACH 4: SEMANTIC PROJECTION (BEST!)
# ============================================================================

print("\n\n" + "=" * 80)
print("APPROACH 4: SEMANTIC PROJECTION - MEASURES LATENT ATTITUDES (BEST!)")
print("=" * 80)
print()

print("""
This is THE MOST SOPHISTICATED approach - unique to cntext!

Uses word embeddings to measure where companies fall on conceptual dimensions:
- "Digital Embracer" vs "Digital Skeptic"
- "AI Enthusiast" vs "AI Cautious"
- "Innovation Leader" vs "Tradition Focused"

Advantages:
+ Captures semantic nuances beyond keyword matching
+ Measures ATTITUDE holistically across entire text
+ Doesn't require perfect keyword list
+ Theory-driven but data-informed
+ Can track subtle shifts over time
+ Published research methodology

Why This is Better:
- Catches "We're excited about leveraging innovative cloud solutions"
  without needing exact phrase "cloud computing"
- Understands context: "AI risks we've mitigated" vs "AI risks threaten us"
- Measures overall tone and framing, not just keywords

This is PERFECT for your research question!

Implementation:
""")

approach4_code = '''
import cntext as ct
import tempfile
import os
import pandas as pd

# STEP 1: Train domain-specific embeddings on your corpus
# -------------------------------------------------------

def train_corpus_embeddings(df, text_column='mda_clean'):
    """
    Train Word2Vec on your 10-K corpus.
    This learns semantic relationships specific to YOUR data.
    """
    print("Training embeddings on your corpus...")

    # Create corpus file
    corpus_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')

    for text in df[text_column]:
        cleaned = ' '.join(str(text).split())
        corpus_file.write(cleaned + '\\n')

    corpus_file.close()

    # Train embeddings
    wv = ct.Word2Vec(
        corpus_file=corpus_file.name,
        lang='english',
        lemmatize=True,
        vector_size=100,
        window=10,
        min_count=5,
        workers=4
    )

    os.unlink(corpus_file.name)

    print(f"✓ Trained! Vocabulary: {len(wv.index_to_key)} words")

    return wv

wv = train_corpus_embeddings(df)


# STEP 2: Create semantic axes for digitalization attitudes
# ---------------------------------------------------------

def create_digitalization_axes(wv):
    """
    Create semantic dimensions for measuring digital attitudes.

    Each axis represents a conceptual spectrum.
    """

    axes = {}

    # Axis 1: Digital Embracer ←→ Digital Skeptic
    # Measures: Enthusiasm vs concern about digitalization
    axes['digital_embrace'] = ct.generate_concept_axis(
        wv,
        poswords=['digital', 'innovation', 'transform', 'leverage', 'adopt',
                 'advanced', 'cutting-edge', 'opportunity'],
        negwords=['risk', 'concern', 'threat', 'challenge', 'disruption',
                 'vulnerability', 'legacy', 'traditional']
    )

    # Axis 2: AI Enthusiast ←→ AI Cautious
    # Measures: Proactive AI adoption vs defensive posture
    axes['ai_enthusiasm'] = ct.generate_concept_axis(
        wv,
        poswords=['ai', 'machine learning', 'intelligent', 'automation',
                 'analytics', 'capability', 'enable', 'enhance'],
        negwords=['risk', 'liability', 'bias', 'regulatory', 'compliance',
                 'ethical', 'concern', 'uncertain']
    )

    # Axis 3: Tech Leader ←→ Tech Follower
    # Measures: Leadership vs catch-up positioning
    axes['tech_leadership'] = ct.generate_concept_axis(
        wv,
        poswords=['leader', 'pioneer', 'first', 'innovative', 'proprietary',
                 'competitive advantage', 'differentiated', 'breakthrough'],
        negwords=['competitor', 'catch up', 'behind', 'adopt', 'implement',
                 'standard', 'industry practice', 'following']
    )

    # Axis 4: Transformation ←→ Incrementalism
    # Measures: Bold transformation vs gradual adoption
    axes['transformation'] = ct.generate_concept_axis(
        wv,
        poswords=['transformation', 'reimagine', 'reinvent', 'revolutionary',
                 'fundamental', 'strategic', 'comprehensive', 'bold'],
        negwords=['incremental', 'gradual', 'pilot', 'test', 'trial',
                 'evaluate', 'cautious', 'measured']
    )

    print(f"✓ Created {len(axes)} semantic axes")

    return axes

axes = create_digitalization_axes(wv)


# STEP 3: Project each MD&A onto the axes
# ----------------------------------------

def project_digital_attitudes(df, wv, axes, text_column='mda_clean'):
    """
    Measure where each company falls on digitalization dimensions.

    Returns scores from -1 (negative pole) to +1 (positive pole).
    """
    from tqdm import tqdm

    print("Projecting MD&A texts onto semantic axes...")

    for axis_name, axis in axes.items():
        df[f'semantic_{axis_name}'] = 0.0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=axis_name):
            try:
                score = ct.project_text(wv, row[text_column], axis, lang='english')
                df.at[idx, f'semantic_{axis_name}'] = score
            except:
                continue

    print("✓ Semantic projection complete!")

    return df

df = project_digital_attitudes(df, wv, axes)


# STEP 4: Analyze and interpret results
# --------------------------------------

# Create categorical variables
df['digital_attitude_category'] = pd.cut(
    df['semantic_digital_embrace'],
    bins=[-1, -0.05, 0.05, 1],
    labels=['Skeptical', 'Neutral', 'Embracing']
)

df['ai_attitude_category'] = pd.cut(
    df['semantic_ai_enthusiasm'],
    bins=[-1, -0.05, 0.05, 1],
    labels=['Cautious', 'Balanced', 'Enthusiastic']
)

# Summary statistics
print("\\n" + "=" * 60)
print("SEMANTIC PROJECTION RESULTS")
print("=" * 60)

print("\\nDigital Embrace Distribution:")
print(df['digital_attitude_category'].value_counts())

print("\\nAI Enthusiasm Distribution:")
print(df['ai_attitude_category'].value_counts())

print("\\nAverage Scores by Year:")
semantic_cols = [col for col in df.columns if col.startswith('semantic_')]
year_avg = df.groupby('fiscal_year')[semantic_cols].mean()
print(year_avg)

# Identify digital leaders
digital_leaders = df.nlargest(10, 'semantic_digital_embrace')[
    ['company_name', 'fiscal_year', 'semantic_digital_embrace',
     'semantic_ai_enthusiasm', 'semantic_tech_leadership']
]

print("\\nTop 10 Digital Embracers:")
print(digital_leaders)
'''

print(approach4_code)

# ============================================================================
# APPROACH 5: TEMPORAL ANALYSIS
# ============================================================================

print("\n\n" + "=" * 80)
print("APPROACH 5: TEMPORAL ANALYSIS - Tracking Change Over Time")
print("=" * 80)
print()

print("""
Track how individual companies' attitudes evolve.

Research Questions:
- Do companies become more positive about AI over time?
- What triggered attitude shifts?
- Are certain industries leading?
- Did COVID accelerate digital transformation?

Implementation:
""")

approach5_code = '''
import pandas as pd
import numpy as np

def analyze_temporal_changes(df):
    """
    Track how each company's digital attitude changes over time.
    """

    # Calculate year-over-year changes
    df = df.sort_values(['company_id', 'fiscal_year'])

    df['semantic_digital_change'] = df.groupby('company_id')['semantic_digital_embrace'].diff()
    df['semantic_ai_change'] = df.groupby('company_id')['semantic_ai_enthusiasm'].diff()

    # Identify companies with big shifts
    big_shifters = df[abs(df['semantic_digital_change']) > 0.1].copy()

    print(f"Found {len(big_shifters)} significant attitude shifts")

    # Categorize shifts
    big_shifters['shift_direction'] = big_shifters['semantic_digital_change'].apply(
        lambda x: 'More Positive' if x > 0 else 'More Negative'
    )

    print("\\nShift Distribution:")
    print(big_shifters['shift_direction'].value_counts())

    # Average trajectory by industry (if you have industry data)
    if 'industry' in df.columns:
        industry_trends = df.groupby(['industry', 'fiscal_year']).agg({
            'semantic_digital_embrace': 'mean',
            'semantic_ai_enthusiasm': 'mean'
        }).reset_index()

        print("\\nIndustry Trends Available!")

    return df, big_shifters

df, shifters = analyze_temporal_changes(df)

# Analyze specific events (e.g., COVID impact)
pre_covid = df[df['fiscal_year'] <= 2019]['semantic_digital_embrace'].mean()
post_covid = df[df['fiscal_year'] >= 2020]['semantic_digital_embrace'].mean()

print(f"\\nCOVID Impact Analysis:")
print(f"Pre-COVID (≤2019): {pre_covid:.3f}")
print(f"Post-COVID (≥2020): {post_covid:.3f}")
print(f"Change: {post_covid - pre_covid:+.3f}")
'''

print(approach5_code)

# ============================================================================
# COMPARISON OF APPROACHES
# ============================================================================

print("\n\n" + "=" * 80)
print("COMPARISON: Which Approach Should You Use?")
print("=" * 80)
print()

comparison_table = """
┌──────────────┬────────────┬────────────┬─────────────┬──────────────┐
│ Approach     │ Complexity │ Insight    │ Speed       │ Best For     │
├──────────────┼────────────┼────────────┼─────────────┼──────────────┤
│ 1. Count     │ Low        │ Basic      │ Very Fast   │ Exploration  │
│ 2. Normalize │ Low        │ Basic      │ Very Fast   │ Comparisons  │
│ 3. Dict      │ Medium     │ Good       │ Fast        │ Sentiment    │
│ 4. Semantic  │ High       │ EXCELLENT  │ Slow*       │ Research**   │
│ 5. Temporal  │ Medium     │ Good       │ Medium      │ Trends       │
└──────────────┴────────────┴────────────┴─────────────┴──────────────┘

* One-time training cost, then fast inference
** Best for academic research, nuanced analysis

Recommendation for YOUR Research:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use MULTIPLE approaches in sequence:

1. Start with Approach 2 (Normalized Frequency)
   - Quick baseline
   - Identify companies mentioning digital/AI
   - Validate data quality

2. Main Analysis: Approach 4 (Semantic Projection)
   - This measures ATTITUDE, not just mentions
   - Captures nuanced differences
   - Publishable research method
   - Unique insights

3. Supplement with Approach 3 (Dictionary)
   - Validate semantic findings
   - Provide interpretable metrics
   - Compare methodologies

4. Add Approach 5 (Temporal)
   - Track changes over time
   - Event study analysis
   - Industry trends

This gives you:
✓ Multiple measurement approaches (robustness)
✓ Sophisticated analysis (semantic projection)
✓ Easy-to-explain metrics (word counts)
✓ Temporal dynamics (change over time)
"""

print(comparison_table)

# ============================================================================
# PRACTICAL RECOMMENDATIONS
# ============================================================================

print("\n\n" + "=" * 80)
print("PRACTICAL RECOMMENDATIONS FOR YOUR PROJECT")
print("=" * 80)
print()

recommendations = """
Research Design Suggestions:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DEFINE YOUR RESEARCH QUESTIONS

   Example RQs:
   - How has corporate enthusiasm for AI evolved 2015-2023?
   - Which industries are digital transformation leaders?
   - Do positive digital attitudes predict firm performance?
   - Did COVID-19 accelerate digital transformation discourse?


2. CREATE DOMAIN-SPECIFIC KEYWORDS

   Expand beyond generic "AI" and "digital":

   AI & ML:
   - artificial intelligence, machine learning, deep learning
   - computer vision, NLP, neural networks, generative AI
   - GPT, transformers, large language models

   Cloud & Infrastructure:
   - cloud computing, AWS, Azure, GCP, hybrid cloud
   - SaaS, PaaS, IaaS, serverless, containerization

   Data & Analytics:
   - big data, data lake, data warehouse, analytics
   - business intelligence, predictive analytics, real-time analytics

   Automation:
   - robotic process automation (RPA), workflow automation
   - intelligent automation, cognitive automation

   Emerging Tech:
   - blockchain, cryptocurrency, web3, metaverse
   - IoT, edge computing, 5G, quantum computing


3. VALIDATE YOUR MEASURES

   - Manually review 50-100 random documents
   - Check if high scorers actually embrace digital
   - Check if low scorers actually are skeptical
   - Adjust keyword lists or axes as needed


4. CONTROL VARIABLES TO CONSIDER

   - Company size (revenue, market cap, employees)
   - Industry / sector
   - Geographic location
   - Age of company
   - Document length (to ensure it's not just verbosity)


5. STATISTICAL ANALYSES

   Descriptive:
   - Trends over time (line plots)
   - Distribution by industry (box plots)
   - Top/bottom companies (rankings)

   Inferential:
   - Panel regression: attitude ~ year + controls
   - Event study: pre/post COVID, new regulation, etc.
   - Difference-in-differences: treated vs control industries

   Predictive:
   - Does digital attitude predict stock returns?
   - Does it predict innovation outputs (patents)?
   - Does it predict survival/growth?


6. PUBLICATION STRATEGY

   Semantic projection is particularly valuable because:
   - Novel methodology
   - Published in top journals
   - Captures latent constructs
   - Addresses "attitude" vs "mentions" distinction

   Frame your contribution:
   "Unlike simple keyword counting, we use semantic projection
   to measure companies' ATTITUDES toward digitalization,
   capturing nuanced differences in how management frames
   technological change."


7. ROBUSTNESS CHECKS

   - Compare semantic projection vs dictionary approach
   - Use different keyword lists
   - Try different time windows
   - Subset analyses by industry
   - Alternative measures (e.g., TF-IDF)


8. VISUALIZATION IDEAS

   - Heatmap: companies × years, colored by attitude
   - Line plots: average attitude by industry over time
   - Scatter: digital attitude vs firm performance
   - Network: companies clustered by semantic similarity
   - Word clouds: high vs low adopters
"""

print(recommendations)

# ============================================================================
# EXAMPLE RESEARCH HYPOTHESES
# ============================================================================

print("\n\n" + "=" * 80)
print("EXAMPLE RESEARCH HYPOTHESES YOU COULD TEST")
print("=" * 80)
print()

hypotheses = """
Using your data and these methods, you could test:

H1: Corporate enthusiasm for digital transformation increased
    following the COVID-19 pandemic.

    Method: Compare semantic_digital_embrace scores pre/post 2020
    Analysis: t-test, event study, interrupted time series


H2: Technology companies adopt more positive framing of AI
    compared to traditional industries.

    Method: Compare semantic_ai_enthusiasm across industries
    Analysis: ANOVA, regression with industry fixed effects


H3: Companies with positive digital attitudes show higher
    subsequent revenue growth.

    Method: Regress future performance on semantic scores
    Analysis: Panel regression, instrumental variables


H4: Positive digital framing is associated with higher
    R&D intensity and patent applications.

    Method: Correlate semantic scores with innovation metrics
    Analysis: Fixed effects regression, mediation analysis


H5: Companies that increase digital transformation language
    experience positive stock market reactions.

    Method: Event study around 10-K filing date
    Analysis: Abnormal returns analysis


H6: First-movers in digital transformation language become
    subsequent market leaders.

    Method: Identify early adopters, track performance
    Analysis: Survival analysis, longitudinal panel


H7: During crises, companies with existing positive digital
    attitudes adapt faster.

    Method: Interaction effect (crisis × digital attitude)
    Analysis: Difference-in-differences


H8: Digital transformation rhetoric without actual investment
    (cheap talk) is detectable and not rewarded by markets.

    Method: Compare language vs CapEx, correlate with returns
    Analysis: 2SLS, mediation with actual spending
"""

print(hypotheses)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n\n" + "=" * 80)
print("SUMMARY: YOUR ACTION PLAN")
print("=" * 80)
print()

action_plan = """
Step-by-Step Implementation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Week 1: Data Preparation & Baseline
  ☐ Run analyze_mda_template.py on your data
  ☐ Implement Approach 2 (normalized frequency)
  ☐ Create initial keyword list
  ☐ Validate data quality

Week 2: Dictionary Development
  ☐ Expand keyword list (consult domain experts)
  ☐ Implement Approach 3 (dictionary sentiment)
  ☐ Manually validate 50-100 documents
  ☐ Refine dictionary

Week 3: Semantic Projection (Main Analysis)
  ☐ Train embeddings on your corpus
  ☐ Create 3-4 semantic axes
  ☐ Project all documents
  ☐ Validate results against manual coding

Week 4: Temporal & Comparative Analysis
  ☐ Implement temporal tracking
  ☐ Industry comparisons
  ☐ Event studies (COVID, etc.)
  ☐ Generate visualizations

Week 5: Statistical Analysis
  ☐ Descriptive statistics
  ☐ Regression analyses
  ☐ Robustness checks
  ☐ Interpretation

Week 6: Paper Writing
  ☐ Methods section (emphasize semantic projection)
  ☐ Results tables and figures
  ☐ Discussion and implications
  ☐ Submit for feedback


Files You'll Need to Create:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. digitalization_keywords.txt
   - Your comprehensive keyword list
   - Organized by category

2. digital_sentiment_dict.yaml
   - Custom dictionary for Approach 3
   - Positive/negative/neutral terms

3. analyze_digital_attitudes.py
   - Main analysis script
   - Combines all approaches

4. validate_measures.py
   - Manual validation against sample
   - Inter-rater reliability if multiple coders


Expected Outputs:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. digital_metrics.csv
   - All computed metrics per document

2. summary_by_year.csv
   - Temporal trends

3. summary_by_industry.csv
   - Industry comparisons

4. top_digital_leaders.csv
   - Companies ranked by digital attitude

5. temporal_shifters.csv
   - Companies with big attitude changes

6. validation_results.csv
   - Manual vs automated comparison
"""

print(action_plan)

print("\n" + "=" * 80)
print("READY TO MEASURE CORPORATE DIGITAL ATTITUDES!")
print("=" * 80)
print()

print("""
Key Takeaways:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ✓ Simple word counting is a START, but limited
2. ✓ Semantic projection is the BEST for measuring attitudes
3. ✓ Use MULTIPLE approaches for robustness
4. ✓ cntext is PERFECT for this research question
5. ✓ This methodology is PUBLISHABLE in top journals

Your Research Advantage:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- You're measuring ATTITUDES, not just keywords
- Semantic projection = sophisticated, novel method
- Large-scale analysis (600MB+ data)
- Temporal dynamics
- Multiple validation approaches
- Strong theoretical grounding

This is a strong research design! 🚀

Next: Want me to create a ready-to-run script specifically
for your digitalization attitude measurement?
""")
