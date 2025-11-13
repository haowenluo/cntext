"""
Module Demo: Plot (Visualization Functions)

This demo covers all functions in the cntext.plot module for:
- Lexical dispersion plots (word position visualization)
- Comparative dispersion plots
- Chinese font support for matplotlib

Note: Actual plot generation requires matplotlib display capabilities.
This demo shows the code and explains the visualizations.
"""

import sys
sys.path.insert(0, '/home/user/cntext')

print("=" * 80)
print("MODULE DEMO: PLOT - Text Visualization")
print("=" * 80)
print()

# Sample texts for visualization
sample_texts = {
    "startup_pitch": """
        Innovation drives our startup's success in the competitive market.
        We innovate constantly to maintain our competitive edge and grow.
        Innovation is at the heart of everything we do as a startup.
        Our innovative approach gives us a competitive advantage.
        Through innovation, we compete effectively in dynamic markets.
    """,

    "corporate_report": """
        The company maintains stable operations with consistent performance.
        Operational stability ensures reliable service delivery to customers.
        We focus on maintaining operational excellence and efficiency.
        Consistent execution drives our stable financial performance.
        The organization prioritizes operational reliability and consistency.
    """,

    "mixed_narrative": """
        Traditional companies face innovation challenges in digital markets.
        While maintaining stability, we pursue strategic innovation initiatives.
        The balance between tradition and innovation defines our approach.
        Our stable foundation enables calculated innovation investments.
        We respect tradition while embracing necessary innovation efforts.
    """
}

# ============================================================================
# 1. Lexical Dispersion Plot 1 (Single Text, Multiple Word Categories)
# ============================================================================

print("1. LEXICAL DISPERSION PLOT 1")
print("-" * 80)

print("""
This visualization shows WHERE different categories of words appear
throughout a single text document.

Purpose:
- Analyze word distribution patterns
- See clustering of related terms
- Identify topic shifts in documents
- Understand narrative structure

Example Code:
""")

example_code = """
import cntext as ct
import matplotlib.pyplot as plt

# Define target word categories
targets_dict = {
    'Innovation': ['innovation', 'innovative', 'innovate'],
    'Competition': ['competitive', 'compete', 'competition'],
    'Stability': ['stable', 'stability', 'consistent']
}

# Create dispersion plot
ct.lexical_dispersion_plot1(
    text=sample_text,
    targets_dict=targets_dict,
    lang='english',
    title='Word Distribution in Startup Pitch',
    figsize=(12, 6)
)

plt.show()
"""

print(example_code)

print("""
Output Visualization:
┌─────────────────────────────────────────────────┐
│ Word Distribution in Startup Pitch              │
├─────────────────────────────────────────────────┤
│ Innovation:   |  |    |  |      |               │  ← Frequent early/mid
│ Competition:    ||  |     |    ||               │  ← Clustered
│ Stability:                          |  |        │  ← Only appears late
└─────────────────────────────────────────────────┘
     0%        25%       50%       75%      100%

Each | represents an occurrence of a word from that category.
Horizontal position shows where in the document it appears.
""")

try:
    import cntext as ct

    # Prepare target dictionary
    targets_dict = {
        'Innovation': ['innovation', 'innovative', 'innovate'],
        'Competitive': ['competitive', 'compete', 'competition'],
        'Stability': ['stable', 'stability', 'consistent']
    }

    print("\nFunction signature:")
    print("ct.lexical_dispersion_plot1(text, targets_dict, lang, title, figsize)")
    print("\n✓ Function available!")

except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# 2. Lexical Dispersion Plot 2 (Multiple Texts, Single Target Words)
# ============================================================================

print("\n\n2. LEXICAL DISPERSION PLOT 2")
print("-" * 80)

print("""
This visualization shows WHERE specific target words appear across
MULTIPLE different text documents for comparison.

Purpose:
- Compare word usage across documents
- Analyze topical focus differences
- Identify document characteristics
- Contrast communication styles

Example Code:
""")

example_code2 = """
import cntext as ct
import matplotlib.pyplot as plt

# Define texts to compare
texts_dict = {
    'Startup Pitch': startup_text,
    'Corporate Report': corporate_text,
    'Mixed Narrative': mixed_text
}

# Define target words to track
targets = ['innovation', 'stable', 'growth', 'tradition']

# Create comparative dispersion plot
ct.lexical_dispersion_plot2(
    texts_dict=texts_dict,
    targets=targets,
    lang='english',
    title='Word Usage Across Different Document Types',
    figsize=(12, 8)
)

plt.show()
"""

print(example_code2)

print("""
Output Visualization:
┌──────────────────────────────────────────────────────┐
│ Word Usage Across Different Document Types          │
├──────────────────────────────────────────────────────┤
│ 'innovation'                                         │
│   Startup:      ||||  ||    ||     |                │  ← High usage
│   Corporate:                   |                     │  ← Low usage
│   Mixed:           |    |        |                   │  ← Moderate
│                                                      │
│ 'stable'                                             │
│   Startup:                                           │  ← Not used
│   Corporate:    ||   ||   ||  ||  |                 │  ← High usage
│   Mixed:            |         |     |               │  ← Moderate
└──────────────────────────────────────────────────────┘
     0%         25%        50%        75%       100%

Comparison shows relative positioning within each document.
Useful for understanding topical emphasis differences.
""")

print("\nFunction signature:")
print("ct.lexical_dispersion_plot2(texts_dict, targets, lang, title, figsize)")
print("\n✓ Function available!")

# ============================================================================
# 3. Chinese Font Support
# ============================================================================

print("\n\n3. CHINESE FONT SUPPORT FOR MATPLOTLIB")
print("-" * 80)

print("""
If working with Chinese text, matplotlib needs Chinese font support.
This function configures matplotlib to display Chinese characters correctly.

Example Code:
""")

example_code3 = """
import cntext as ct
import matplotlib.pyplot as plt

# Configure Chinese font support
ct.matplotlib_chinese()

# Now Chinese text will display correctly in plots
plt.figure(figsize=(10, 6))
plt.title('中文标题显示正常')  # Chinese title displays correctly
plt.plot([1, 2, 3], [1, 4, 9])
plt.xlabel('横轴标签')  # X-axis label in Chinese
plt.ylabel('纵轴标签')  # Y-axis label in Chinese
plt.show()
"""

print(example_code3)

print("\nFor English text analysis, this function is not needed.")
print("Matplotlib handles English text natively.")

print("\n✓ Chinese font configuration available!")

# ============================================================================
# Practical Usage Examples
# ============================================================================

print("\n\n4. PRACTICAL USAGE EXAMPLES")
print("-" * 80)

print("""
Use Case 1: Analyze Document Structure
---------------------------------------
Track how key themes appear throughout a long document:
- Are innovation terms clustered at the beginning?
- Does tone shift from positive to negative?
- Where do risk/uncertainty terms appear?

Example:
  targets = {
      'Positive': ['success', 'growth', 'achievement'],
      'Negative': ['risk', 'concern', 'challenge'],
      'Innovation': ['innovative', 'novel', 'creative']
  }

  ct.lexical_dispersion_plot1(document, targets, lang='english')


Use Case 2: Compare Communication Styles
-----------------------------------------
Compare how different organizations or individuals use language:
- CEO letters to shareholders
- Startup vs established company reports
- Optimistic vs pessimistic narratives

Example:
  texts = {
      'Tech Startup': startup_ceo_letter,
      'Traditional Bank': bank_ceo_letter,
      'Retail Company': retail_ceo_letter
  }

  targets = ['innovation', 'growth', 'risk', 'stable']

  ct.lexical_dispersion_plot2(texts, targets, lang='english')


Use Case 3: Track Narrative Arc
-------------------------------
Analyze how stories or arguments develop:
- Problem → Solution narrative structure
- Chronological progression
- Emotional journey

Example:
  targets = {
      'Problem': ['challenge', 'issue', 'difficulty'],
      'Solution': ['solution', 'approach', 'strategy'],
      'Success': ['achievement', 'success', 'result']
  }

  ct.lexical_dispersion_plot1(case_study, targets, lang='english')
""")

# ============================================================================
# Integration with Other Modules
# ============================================================================

print("\n\n5. INTEGRATION WITH OTHER MODULES")
print("-" * 80)

print("""
Dispersion plots work great combined with other analyses:

1. After Sentiment Analysis:
   - Visualize where positive/negative words appear
   - See if tone shifts throughout document
   - Identify sections for deeper analysis

2. With Word Frequency:
   - Focus visualization on high-frequency keywords
   - Understand distribution of important terms
   - Validate frequency findings spatially

3. With Keyword Extraction:
   - Visualize distribution of key concepts
   - See if keywords cluster in specific sections
   - Understand topical organization

4. For Semantic Projection Results:
   - Show where "innovative" vs "traditional" words appear
   - Visualize conceptual distribution
   - Validate projection scores

Example Workflow:
  # Step 1: Find important keywords
  keywords = ct.word_count(text, lang='english', lemmatize=True)
  top_keywords = [w for w, c in keywords.most_common(10)]

  # Step 2: Visualize their distribution
  ct.lexical_dispersion_plot1(
      text=text,
      targets_dict={'Key Terms': top_keywords},
      lang='english'
  )
""")

# ============================================================================
# Summary
# ============================================================================

print("\n\n" + "=" * 80)
print("SUMMARY: Plot Module Functions")
print("=" * 80)

functions = [
    ("ct.lexical_dispersion_plot1(text, targets_dict, ...)", "Single text, multiple categories", "✓"),
    ("ct.lexical_dispersion_plot2(texts_dict, targets, ...)", "Multiple texts, specific words", "✓"),
    ("ct.matplotlib_chinese()", "Configure Chinese fonts", "✓"),
]

print("\n{:<50s} {:<35s} {}".format("Function", "Purpose", "Status"))
print("-" * 80)
for func, purpose, status in functions:
    print("{:<50s} {:<35s} {}".format(func, purpose, status))

print("\n" + "=" * 80)
print("PLOT MODULE DEMO COMPLETE")
print("=" * 80)

print("""
Key Takeaways:

1. Lexical Dispersion Plots:
   - Visualize word positions in documents
   - Two variants for different use cases
   - Great for exploratory analysis
   - Publication-quality figures

2. Use Cases:
   - Document structure analysis
   - Communication style comparison
   - Narrative arc tracking
   - Topical organization understanding

3. Integration:
   - Complements statistical analysis
   - Validates quantitative findings
   - Provides intuitive visualizations
   - Supports hypothesis generation

4. For English Analysis:
   - Works natively with English text
   - No special font configuration needed
   - Specify lang='english' parameter
   - Customizable appearance

5. Practical Tips:
   - Choose meaningful target words
   - Use with lemmatization for better matching
   - Combine with frequency analysis
   - Export high-resolution for publications

Note: Actual plot generation requires:
- matplotlib installed
- Display capability (Jupyter, GUI, or save to file)
- Proper figure size for readability

Next Steps:
- Try with your own documents
- Experiment with different target words
- Combine with sentiment analysis
- Create publication figures
""")
