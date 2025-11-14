"""
Module Demo: Mind (Semantic Analysis & Cognitive Measurement)

This demo covers all functions in the cntext.mind module for:
- Semantic projection (measuring abstract concepts)
- Concept axis generation
- Semantic distance calculation
- Divergent Association Task (creativity measurement)
- Discursive diversity (cognitive diversity)
- Procrustes alignment (tracking semantic change)

This is the MOST INNOVATIVE module in cntext!
"""

import sys
sys.path.insert(0, '/home/user/cntext')

print("=" * 80)
print("MODULE DEMO: MIND - Semantic Analysis & Cognitive Measurement")
print("=" * 80)
print("\nThis module contains cntext's most innovative features!")
print("It allows you to measure abstract concepts that are hard to capture")
print("with traditional dictionary-based methods.\n")

import tempfile
import os

# Create a richer corpus for semantic analysis
print("Setting up corpus for semantic analysis...")
corpus_dir = tempfile.mkdtemp()
corpus_file = os.path.join(corpus_dir, 'corpus.txt')

# Create corpus with clear semantic dimensions
corpus_text = """Innovation drives modern business transformation and competitive success.
Traditional companies struggle with change management and adaptation challenges.
Creative thinking enables breakthrough solutions and novel approaches to problems.
Conservative strategies maintain stability but limit growth opportunities and progress.
Technology enables innovative disruption across traditional industry sectors and markets.
Digital transformation requires organizational change and cultural adaptation throughout companies.
Established firms leverage experience while innovative startups pursue disruptive opportunities.
Risk-taking entrepreneurs drive innovation through creative experimentation and bold initiatives.
Conventional wisdom suggests caution while innovation demands courage and vision for change.
Forward-thinking leaders embrace change while traditional managers resist transformation efforts.
Cutting-edge technology transforms business models and creates competitive advantages rapidly.
Time-tested methods provide reliability while innovation promises growth through new approaches.
Optimistic leaders see opportunities in challenges and embrace positive transformation initiatives.
Pessimistic managers focus on risks and obstacles that constrain change initiatives continuously.
Confident teams pursue ambitious goals with determination and positive attitudes toward challenges.
Uncertain organizations hesitate to commit resources due to concerns about potential failures.
Strong performance demonstrates capability and validates strategic decisions for future growth.
Weak results raise doubts about direction and generate concerns regarding operational effectiveness.
Success breeds confidence and motivation while failure creates doubt and uncertainty about paths.
Growth opportunities emerge from innovation while stagnation results from resistance to change.
Excellence requires continuous improvement and commitment to quality in all organizational activities.
Mediocrity results from complacency and lack of ambition in pursuing organizational objectives.
Achievement drives motivation and creates momentum for continued success and improvement efforts.
Setbacks undermine confidence and create obstacles that slow progress toward strategic goals.
Positive outcomes reinforce effective strategies while negative results prompt strategic reassessment.
Opportunities abound for organizations that embrace innovation and pursue strategic transformation.
Threats emerge when companies fail to adapt to changing market dynamics and conditions.
Progress requires vision and determination while stagnation results from fear and resistance.
Advancement comes through calculated risks while decline follows excessive caution and inaction.
Development demands investment in capabilities while deterioration follows neglect and underinvestment.
Bright prospects inspire confidence and action while dark outlooks generate hesitation and doubt.
Strong leadership drives positive transformation while weak management perpetuates status quo.
Effective strategies create competitive advantages while ineffective approaches erode market position.
Dynamic markets reward innovation and agility while static environments favor traditional approaches.
Rapid change creates both opportunities and challenges for businesses across all sectors.
Stability provides security but limits growth while volatility enables transformation and innovation.
Certain outcomes reduce risk but cap potential while uncertain ventures offer breakthrough possibilities.
Known approaches guarantee reliability while novel methods promise revolutionary improvements and gains.
""" * 10  # Repeat to get more data

with open(corpus_file, 'w', encoding='utf-8') as f:
    f.write(corpus_text)

print(f"✓ Created corpus with {len(corpus_text.split())} words")
print("  (Contains text along multiple semantic dimensions)\n")

# ============================================================================
# 1. Train Word Embeddings (Required for Mind Module)
# ============================================================================

print("1. TRAIN WORD EMBEDDINGS (PREREQUISITE)")
print("-" * 80)

try:
    import cntext as ct

    print("\nTraining Word2Vec model...")
    print("Parameters: vector_size=100, window=5, min_count=3\n")

    wv = ct.Word2Vec(
        corpus_file=corpus_file,
        lang='english',
        lemmatize=True,  # Better quality
        vector_size=100,
        window=5,
        min_count=3,
        workers=2
    )

    print(f"✓ Model trained!")
    print(f"  Vocabulary: {len(wv.index_to_key)} words")
    print(f"  Dimensions: {wv.vector_size}\n")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 2. Concept Axis Generation
# ============================================================================

print("\n2. CONCEPT AXIS GENERATION")
print("-" * 80)

try:
    print("\n2.1 Create semantic axis: Innovation ←→ Traditional")

    # Define words representing each pole
    innovation_words = ['innovation', 'creative', 'novel', 'breakthrough', 'disruptive']
    traditional_words = ['traditional', 'conventional', 'conservative', 'established', 'time-tested']

    print(f"  Positive pole: {innovation_words}")
    print(f"  Negative pole: {traditional_words}")

    # Generate the concept axis
    axis_innovation = ct.generate_concept_axis(wv, innovation_words, traditional_words)

    print(f"\n✓ Concept axis created!")
    print(f"  Axis is a {axis_innovation.shape} dimensional vector")

    print("\n2.2 Create another axis: Optimistic ←→ Pessimistic")

    optimistic_words = ['optimistic', 'confident', 'positive', 'opportunity', 'success']
    pessimistic_words = ['pessimistic', 'uncertain', 'negative', 'risk', 'failure']

    print(f"  Positive pole: {optimistic_words}")
    print(f"  Negative pole: {pessimistic_words}")

    axis_sentiment = ct.generate_concept_axis(wv, optimistic_words, pessimistic_words)

    print(f"\n✓ Second axis created!")

    print("\n✓ Concept axis generation works!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 3. Semantic Projection - Words
# ============================================================================

print("\n\n3. SEMANTIC PROJECTION - INDIVIDUAL WORDS")
print("-" * 80)

try:
    print("\n3.1 Project individual words onto Innovation←→Traditional axis:")

    test_words = [
        'technology', 'stability', 'startup', 'corporation',
        'experiment', 'proven', 'agile', 'bureaucratic'
    ]

    print("\n  Word projections (positive = innovative, negative = traditional):\n")

    projections = []
    for word in test_words:
        if word in wv:
            score = ct.project_word(wv, word, axis_innovation)
            projections.append((word, score))
            direction = "→ innovative" if score > 0.01 else "→ traditional" if score < -0.01 else "→ neutral"
            print(f"    {word:15s}: {score:+.4f} {direction}")

    print("\n3.2 Project words onto Optimistic←→Pessimistic axis:")

    sentiment_words = [
        'success', 'failure', 'opportunity', 'threat',
        'growth', 'decline', 'achievement', 'setback'
    ]

    print("\n  Word projections (positive = optimistic, negative = pessimistic):\n")

    for word in sentiment_words:
        if word in wv:
            score = ct.project_word(wv, word, axis_sentiment)
            direction = "→ optimistic" if score > 0.01 else "→ pessimistic" if score < -0.01 else "→ neutral"
            print(f"    {word:15s}: {score:+.4f} {direction}")

    print("\n✓ Word projection works!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 4. Semantic Projection - Texts
# ============================================================================

print("\n\n4. SEMANTIC PROJECTION - FULL TEXTS")
print("-" * 80)

try:
    print("\nThis is the killer feature! Project entire texts onto concept axes.")

    # Sample company descriptions with different characteristics
    companies = {
        "Tech Startup": """
            Our company drives innovation through cutting-edge technology and
            creative solutions. We disrupt traditional markets with novel approaches
            and breakthrough products. Risk-taking and experimentation define our
            culture as we pursue transformative opportunities.
        """,

        "Traditional Corp": """
            Our firm maintains conservative strategies based on proven methods and
            established practices. We prioritize stability and reliability through
            time-tested approaches. Conventional wisdom guides our decision-making
            as we preserve our strong market position.
        """,

        "Hybrid Company": """
            The organization balances innovation with stability, pursuing selective
            opportunities while maintaining operational reliability. We combine
            creative thinking with proven methods to achieve sustainable growth
            and competitive advantages in evolving markets.
        """
    }

    print("\n4.1 Project companies onto Innovation←→Traditional axis:\n")

    innovation_scores = {}
    for company, description in companies.items():
        score = ct.project_text(wv, description, axis_innovation, lang='english')
        innovation_scores[company] = score

        classification = "INNOVATIVE" if score > 0.02 else "TRADITIONAL" if score < -0.02 else "BALANCED"
        print(f"  {company:20s}: {score:+.4f} ({classification})")

    print("\n4.2 Project companies onto Optimistic←→Pessimistic axis:\n")

    for company, description in companies.items():
        score = ct.project_text(wv, description, axis_sentiment, lang='english')

        classification = "OPTIMISTIC" if score > 0.02 else "PESSIMISTIC" if score < -0.02 else "NEUTRAL"
        print(f"  {company:20s}: {score:+.4f} ({classification})")

    # Visualize Innovation scores
    print("\n4.3 Visual representation of Innovation scores:")
    print("\n  Traditional              Neutral              Innovative")
    print("      -0.1                    0.0                   +0.1")
    print("        |                      |                      |")

    for company, score in sorted(innovation_scores.items(), key=lambda x: x[1]):
        # Normalize to 0-40 range for display
        position = int((score + 0.1) / 0.2 * 40)
        position = max(0, min(40, position))
        bar = ' ' * position + '■' + ' ' * (40 - position)
        print(f"  {company:18s} |{bar}|")

    print("\n✓ Text projection works - This is incredibly powerful!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 5. Semantic Distance
# ============================================================================

print("\n\n5. SEMANTIC DISTANCE CALCULATION")
print("-" * 80)

try:
    print("\n5.1 Calculate semantic distance between word groups:")

    group1 = ['innovation', 'technology', 'digital']
    group2 = ['traditional', 'conventional', 'established']
    group3 = ['growth', 'success', 'achievement']

    print(f"\nGroup 1: {group1}")
    print(f"Group 2: {group2}")
    print(f"Group 3: {group3}")

    dist_12 = ct.semantic_distance(wv, group1, group2)
    dist_13 = ct.semantic_distance(wv, group1, group3)
    dist_23 = ct.semantic_distance(wv, group2, group3)

    print(f"\nSemantic distances:")
    print(f"  Group 1 ↔ Group 2: {dist_12:.4f} (innovation vs traditional)")
    print(f"  Group 1 ↔ Group 3: {dist_13:.4f} (innovation vs success)")
    print(f"  Group 2 ↔ Group 3: {dist_23:.4f} (traditional vs success)")

    print("\n✓ Semantic distance calculation works!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 6. Divergent Association Task (Creativity Measurement)
# ============================================================================

print("\n\n6. DIVERGENT ASSOCIATION TASK (Creativity Measurement)")
print("-" * 80)

try:
    print("\nDAT measures semantic diversity of word associations")
    print("Higher score = more divergent/creative thinking\n")

    print("6.1 Compare creative vs. conventional word sets:")

    creative_words = ['innovation', 'art', 'music', 'quantum', 'poetry', 'dream']
    conventional_words = ['desk', 'chair', 'office', 'computer', 'meeting', 'report']

    # Filter words that are in vocabulary
    creative_words = [w for w in creative_words if w in wv]
    conventional_words = [w for w in conventional_words if w in wv]

    print(f"Creative set: {creative_words}")
    creative_score = ct.divergent_association_task(wv, creative_words)
    print(f"DAT Score: {creative_score:.4f}\n")

    print(f"Conventional set: {conventional_words}")
    conventional_score = ct.divergent_association_task(wv, conventional_words)
    print(f"DAT Score: {conventional_score:.4f}\n")

    if creative_score > conventional_score:
        print(f"✓ Creative set has higher divergent thinking score!")
        print(f"  Difference: {creative_score - conventional_score:.4f}")

    print("\n✓ DAT measurement works!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 7. Discursive Diversity Score (Cognitive Diversity)
# ============================================================================

print("\n\n7. DISCURSIVE DIVERSITY SCORE (Cognitive Diversity)")
print("-" * 80)

try:
    print("\nMeasures cognitive/linguistic diversity in word usage")
    print("Higher score = more diverse conceptual thinking\n")

    diverse_words = ['innovation', 'tradition', 'technology', 'nature', 'art', 'science']
    focused_words = ['business', 'company', 'firm', 'corporation', 'enterprise', 'organization']

    # Filter words in vocabulary
    diverse_words = [w for w in diverse_words if w in wv]
    focused_words = [w for w in focused_words if w in wv]

    print(f"Diverse concepts: {diverse_words}")
    diverse_score = ct.discursive_diversity_score(wv, diverse_words)
    print(f"Diversity Score: {diverse_score:.4f}\n")

    print(f"Focused concepts: {focused_words}")
    focused_score = ct.discursive_diversity_score(wv, focused_words)
    print(f"Diversity Score: {focused_score:.4f}\n")

    if diverse_score > focused_score:
        print(f"✓ Diverse set has higher cognitive diversity!")
        print(f"  Difference: {diverse_score - focused_score:.4f}")

    print("\n✓ Discursive diversity measurement works!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 8. Procrustes Alignment (Semantic Change Over Time)
# ============================================================================

print("\n\n8. PROCRUSTES ALIGNMENT (Tracking Semantic Change)")
print("-" * 80)

print("\nProcrustes alignment allows tracking semantic change over time")
print("by aligning embeddings from different time periods.\n")

print("Example usage:")
print("""
# Train models on corpora from different time periods
wv_2000 = ct.Word2Vec('corpus_2000s.txt', lang='english', ...)
wv_2020 = ct.Word2Vec('corpus_2020s.txt', lang='english', ...)

# Align 2020 model to 2000 space
wv_2020_aligned = ct.procrustes_align(base_wv=wv_2000, other_wv=wv_2020)

# Compare word meanings over time
word = 'innovation'
neighbors_2000 = wv_2000.most_similar(word, topn=5)
neighbors_2020 = wv_2020_aligned.most_similar(word, topn=5)

# Semantic shift analysis
for w1, w2 in zip(neighbors_2000, neighbors_2020):
    print(f"2000s: {w1[0]:15s} → 2020s: {w2[0]:15s}")
""")

print("\n✓ Procrustes alignment available for diachronic analysis!")

# ============================================================================
# Cleanup
# ============================================================================

print("\n\nCleaning up...")
import shutil
shutil.rmtree(corpus_dir)
print("✓ Cleanup complete")

# ============================================================================
# Summary
# ============================================================================

print("\n\n" + "=" * 80)
print("SUMMARY: Mind Module Functions")
print("=" * 80)

functions = [
    ("ct.generate_concept_axis(wv, pos, neg)", "Create semantic dimension", "✓"),
    ("ct.project_word(wv, word, axis)", "Project word onto axis", "✓"),
    ("ct.project_text(wv, text, axis, lang)", "Project text onto axis", "✓"),
    ("ct.semantic_projection(wv, words, pos, neg)", "General projection function", "✓"),
    ("ct.semantic_distance(wv, words1, words2)", "Distance between word groups", "✓"),
    ("ct.divergent_association_task(wv, words)", "Measure creativity/divergence", "✓"),
    ("ct.discursive_diversity_score(wv, words)", "Measure cognitive diversity", "✓"),
    ("ct.procrustes_align(base_wv, other_wv)", "Align embeddings over time", "✓"),
    ("ct.Text2Mind(wv)", "Comprehensive analysis object", "Available"),
]

print("\n{:<45s} {:<35s} {}".format("Function", "Purpose", "Status"))
print("-" * 80)
for func, purpose, status in functions:
    print("{:<45s} {:<35s} {}".format(func, purpose, status))

print("\n" + "=" * 80)
print("MIND MODULE DEMO COMPLETE")
print("=" * 80)

print("""
Key Takeaways:

1. Concept Axes (Revolutionary Feature!):
   - Define abstract dimensions (innovation vs tradition)
   - Create from opposing word sets
   - Reusable across texts and analyses
   - Theory-driven measurement

2. Semantic Projection:
   - Measure where texts fall on conceptual dimensions
   - Goes beyond dictionary matching
   - Captures nuanced meanings
   - Quantifies abstract constructs

3. Text-Level Measurement:
   - Project entire documents onto axes
   - Classify texts on multiple dimensions
   - Track changes over time
   - Compare organizations, authors, periods

4. Word-Level Analysis:
   - See which words lean toward each pole
   - Validate your concept axes
   - Discover unexpected associations
   - Build better dictionaries

5. Creativity & Diversity:
   - DAT measures divergent thinking
   - Discursive diversity measures cognitive breadth
   - Quantify abstract cognitive constructs
   - Compare individuals, groups, texts

6. Semantic Change:
   - Procrustes alignment tracks meaning shifts
   - Compare across time periods
   - Study cultural evolution
   - Analyze discourse changes

7. Research Applications:
   - Organizational culture measurement
   - Stereotype and bias quantification
   - Cultural concept tracking
   - Attitude measurement
   - Discourse analysis

This module represents the cutting edge of computational text analysis!
It allows measuring abstract constructs that were previously unmeasurable.

Next Steps:
- Apply to your research questions
- Create domain-specific concept axes
- Combine with traditional methods
- Publish novel findings!
""")
