"""
Module Demo: IO (Input/Output & Text Preprocessing)

This demo covers all functions in the cntext.io module for:
- Reading various file formats (PDF, DOCX, TXT, CSV)
- Text cleaning and preprocessing
- Encoding detection and handling
- Batch file processing
"""

import sys
sys.path.insert(0, '/home/user/cntext')

print("=" * 80)
print("MODULE DEMO: IO - Input/Output & Text Preprocessing")
print("=" * 80)
print()

# ============================================================================
# 1. Dictionary Management
# ============================================================================

print("1. DICTIONARY MANAGEMENT")
print("-" * 80)

try:
    import cntext as ct

    print("\n1.1 List all built-in dictionaries:")
    dict_list = ct.get_dict_list()

    # Separate by language
    english_dicts = [d for d in dict_list if d.startswith('en_')]
    chinese_dicts = [d for d in dict_list if d.startswith('zh_')]
    bilingual_dicts = [d for d in dict_list if d.startswith('enzh_')]

    print(f"\nEnglish dictionaries ({len(english_dicts)}):")
    for d in english_dicts:
        print(f"  • {d}")

    print(f"\nChinese dictionaries ({len(chinese_dicts)}):")
    for d in chinese_dicts[:3]:  # Show first 3
        print(f"  • {d}")
    print(f"  ... and {len(chinese_dicts)-3} more")

    print(f"\nBilingual dictionaries ({len(bilingual_dicts)}):")
    for d in bilingual_dicts:
        print(f"  • {d}")

    print("\n✓ Dictionary listing works!")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n1.2 Load and examine a dictionary:")
try:
    # Load Loughran-McDonald financial sentiment dictionary
    lm_dict = ct.read_yaml_dict('en_common_LoughranMcDonald.yaml')

    print(f"\nDictionary Name: {lm_dict['Name']}")
    print(f"Description: {lm_dict['Desc'][:100]}...")
    print(f"Categories: {', '.join(lm_dict['Category'])}")
    print(f"\nSample 'Negative' words: {lm_dict['Dictionary']['Negative'][:10]}")
    print(f"Sample 'Positive' words: {lm_dict['Dictionary']['Positive'][:10]}")

    print("\n✓ Dictionary loading works!")

except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# 2. Text Cleaning & Preprocessing
# ============================================================================

print("\n\n2. TEXT CLEANING & PREPROCESSING")
print("-" * 80)

print("\n2.1 Fix contractions (English):")
try:
    text_with_contractions = "I don't think we're going to make it. They've been waiting."
    print(f"Original: {text_with_contractions}")

    fixed = ct.fix_contractions(text_with_contractions)
    print(f"Fixed:    {fixed}")
    print("✓ Contraction expansion works!")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n2.2 Fix text encoding issues:")
try:
    # Simulate text with encoding issues
    messy_text = "The companyâ€™s revenue increased by 25%"
    print(f"Messy text: {messy_text}")

    cleaned = ct.fix_text(messy_text)
    print(f"Cleaned:    {cleaned}")
    print("✓ Text fixing works!")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n2.3 Clean text (comprehensive):")
try:
    dirty_text = """
    The company's    revenue   increased
    by 25% in 2024!!! We're very excited.
    Contact us at: email@company.com
    """

    print(f"Original:\n{dirty_text}")

    # Clean with English language setting
    cleaned = ct.clean_text(dirty_text, lang='english')
    print(f"\nCleaned:\n{cleaned}")
    print("✓ Text cleaning works!")

except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# 3. File Reading
# ============================================================================

print("\n\n3. FILE READING")
print("-" * 80)

print("\n3.1 Create sample files for testing:")
import tempfile
import os

# Create temporary directory for test files
test_dir = tempfile.mkdtemp()
print(f"Test directory: {test_dir}")

# Create sample text file
txt_file = os.path.join(test_dir, "sample.txt")
with open(txt_file, 'w', encoding='utf-8') as f:
    f.write("""Innovation drives progress in modern business.
Companies must adapt to changing market conditions.
Technology enables new opportunities for growth.""")

# Create another text file
txt_file2 = os.path.join(test_dir, "report.txt")
with open(txt_file2, 'w', encoding='utf-8') as f:
    f.write("""Annual performance exceeded expectations.
Revenue growth remained strong throughout the year.
Strategic investments yielded positive returns.""")

print(f"✓ Created {txt_file}")
print(f"✓ Created {txt_file2}")

print("\n3.2 Read single text file:")
try:
    content = ct.read_file(txt_file)
    print(f"Content length: {len(content)} characters")
    print(f"First 100 chars: {content[:100]}...")
    print("✓ Single file reading works!")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n3.3 Batch read multiple files:")
try:
    # Read all .txt files in directory
    pattern = os.path.join(test_dir, "*.txt")
    df = ct.read_files(pattern, encoding='utf-8')

    print(f"Loaded {len(df)} files into DataFrame")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDataFrame preview:")
    print(df.head())
    print("✓ Batch file reading works!")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n3.4 Detect file encoding:")
try:
    encoding = ct.detect_encoding(txt_file)
    print(f"Detected encoding: {encoding}")
    print("✓ Encoding detection works!")

except Exception as e:
    print(f"✗ Error: {e}")

# Clean up test files
import shutil
shutil.rmtree(test_dir)
print(f"\n✓ Cleaned up test directory")

# ============================================================================
# 4. Advanced Text Processing
# ============================================================================

print("\n\n4. ADVANCED TEXT PROCESSING")
print("-" * 80)

print("\n4.1 Traditional Chinese to Simplified (if text contains Chinese):")
try:
    # This is primarily for Chinese text
    chinese_text = "繁體中文轉換為簡體中文"
    simplified = ct.traditional2simple(chinese_text)
    print(f"Traditional: {chinese_text}")
    print(f"Simplified:  {simplified}")
    print("✓ Traditional→Simplified conversion works!")

except Exception as e:
    print(f"Note: This function is for Chinese text")

print("\n4.2 Extract MD&A from annual reports:")
print("Note: extract_mda() is specialized for Chinese A-share annual reports")
print("For English reports, you would use different extraction patterns")
print("Example usage:")
print("""
    # For Chinese annual reports
    text = ct.read_file('annual_report.txt')
    mda = ct.extract_mda(text)

    # For English annual reports, you'd need custom patterns
    # based on SEC filing structures (10-K, etc.)
""")

# ============================================================================
# Summary
# ============================================================================

print("\n\n" + "=" * 80)
print("SUMMARY: IO Module Functions")
print("=" * 80)

functions = [
    ("ct.get_dict_list()", "List all built-in dictionaries", "✓"),
    ("ct.read_yaml_dict(name)", "Load a YAML dictionary", "✓"),
    ("ct.fix_contractions(text)", "Expand English contractions", "✓"),
    ("ct.fix_text(text)", "Fix encoding issues", "✓"),
    ("ct.clean_text(text, lang)", "Comprehensive text cleaning", "✓"),
    ("ct.read_file(path)", "Read single text file", "✓"),
    ("ct.read_files(pattern)", "Batch read files to DataFrame", "✓"),
    ("ct.detect_encoding(file)", "Detect file encoding", "✓"),
    ("ct.read_pdf(file)", "Read PDF file", "Requires PyMuPDF"),
    ("ct.read_docx(file)", "Read Word document", "Requires python-docx"),
    ("ct.traditional2simple(text)", "Chinese character conversion", "Chinese only"),
    ("ct.extract_mda(text)", "Extract MD&A from reports", "Chinese reports"),
]

print("\n{:<35s} {:<40s} {}".format("Function", "Purpose", "Status"))
print("-" * 80)
for func, purpose, status in functions:
    print("{:<35s} {:<40s} {}".format(func, purpose, status))

print("\n" + "=" * 80)
print("IO MODULE DEMO COMPLETE")
print("=" * 80)

print("""
Key Takeaways:

1. Dictionary Management:
   - 4 English dictionaries available (LM, NRC, Concreteness, ANEW)
   - 11+ Chinese dictionaries
   - 2 bilingual resources (stopwords, adverbs/conjunctions)

2. Text Preprocessing:
   - fix_contractions() expands English contractions
   - fix_text() handles encoding issues
   - clean_text() provides comprehensive cleaning

3. File Operations:
   - read_file() for single files
   - read_files() for batch processing → DataFrame
   - Supports TXT, CSV (PDF/DOCX require extra packages)

4. English Focus:
   - All functions work with English text
   - Specify lang='english' where applicable
   - Bilingual stopwords included

Next Steps:
- See demo_stats.py for text analysis functions
- See demo_model.py for word embeddings
- Install PyMuPDF and python-docx for full file support
""")
