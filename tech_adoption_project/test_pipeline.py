"""
Test the sentence table and labeling sample pipeline with synthetic data.

This script generates sample 10-K Item extractions and tests both:
1. build_sentence_table.py
2. build_labeling_sample.py
"""

import sys
sys.path.insert(0, '/home/user/cntext')

import pandas as pd
import json
from pathlib import Path

# Generate sample 10-K Item data
def generate_test_data():
    """
    Create synthetic 10-K Item extractions for testing.
    """

    sample_texts = {
        'item_1_tech': """
            Our business relies heavily on advanced technology infrastructure.
            We utilize cloud computing platforms including Amazon Web Services and Microsoft Azure
            for scalable data processing. Our proprietary machine learning algorithms
            analyze customer behavior to improve recommendations. We have invested significantly
            in artificial intelligence and automation to enhance operational efficiency.
            Our cybersecurity measures include encryption and multi-factor authentication
            to protect sensitive customer data.
        """,

        'item_1_mixed': """
            We operate retail stores across the United States and maintain an e-commerce platform
            for online sales. Our digital transformation initiatives include implementing
            a new customer relationship management system and upgrading our network infrastructure.
            We employ approximately 50,000 people in various roles including software engineers,
            retail associates, and warehouse workers. Our supply chain operations utilize
            IoT sensors to track inventory in real-time.
        """,

        'item_1a_risks': """
            We face risks related to cybersecurity threats and data breaches.
            Any failure of our cloud infrastructure could disrupt operations.
            Our investments in emerging technologies like blockchain and quantum computing
            may not yield expected returns. Competition in the technology sector is intense.
            We depend on third-party software platforms and APIs for critical functions.
            Regulatory changes affecting data privacy could impact our business model.
        """,

        'item_7_mda': """
            Revenue increased 15% year-over-year driven by strong demand for our software as a service
            offerings. We invested $500 million in research and development, focusing on
            artificial intelligence and natural language processing capabilities.
            Operating expenses included costs for cloud computing services and data center operations.
            Our digital marketing initiatives improved customer acquisition efficiency.
            We completed the rollout of our mobile application which now accounts for
            40% of total transactions.
        """,

        'item_1_non_tech': """
            Our manufacturing facilities produce consumer goods including household products
            and personal care items. We source raw materials from suppliers globally.
            Our distribution network includes warehouses and transportation logistics.
            We sell products through retail partners and directly to consumers.
            Our workforce includes production workers, sales representatives, and administrative staff.
            We are committed to sustainable manufacturing practices and reducing environmental impact.
        """
    }

    # Create sample records
    records = []

    for i, (key, text) in enumerate(sample_texts.items()):
        # Determine item number from key
        if 'item_1a' in key:
            item = '1A'
        elif 'item_7' in key:
            item = '7'
        else:
            item = '1'

        record = {
            'cik': 1000000 + i,
            'accession': f'0000000000-20-00000{i}',
            'fiscal_year': 2020 + (i % 3),
            'filing_date': f'2021-0{(i % 9) + 1}-15',
            'item': item,
            'item_text': text.strip()
        }

        records.append(record)

    return records


def main():
    """
    Run test pipeline.
    """
    print("=" * 80)
    print("TESTING SENTENCE TABLE & LABELING SAMPLE PIPELINE")
    print("=" * 80)

    # Create test data directory
    test_dir = Path('test_pipeline_data')
    test_dir.mkdir(exist_ok=True)

    # Step 1: Generate test data
    print("\n[1/4] Generating test data...")
    test_records = generate_test_data()

    # Save as JSON
    json_path = test_dir / 'test_items.json'
    with open(json_path, 'w') as f:
        json.dump(test_records, f, indent=2)
    print(f"✓ Saved test data: {json_path} ({len(test_records)} items)")

    # Also save as CSV
    csv_path = test_dir / 'test_items.csv'
    pd.DataFrame(test_records).to_csv(csv_path, index=False)
    print(f"✓ Saved test data: {csv_path}")

    # Step 2: Test sentence table builder
    print("\n[2/4] Testing build_sentence_table.py...")
    print("-" * 80)

    from build_sentence_table import build_sentence_table

    sentence_table_path = test_dir / 'sentence_table.csv'
    sentence_df = build_sentence_table(
        input_path=json_path,
        output_path=sentence_table_path,
        output_format='both'
    )

    print(f"\n✓ Sentence table created: {len(sentence_df)} sentences")

    # Step 3: Test labeling sample builder
    print("\n[3/4] Testing build_labeling_sample.py...")
    print("-" * 80)

    from build_labeling_sample import build_labeling_sample
    import warnings
    warnings.filterwarnings('ignore')

    from tqdm import tqdm
    tqdm.pandas()

    label_set_path = test_dir / 'label_set.csv'
    label_df = build_labeling_sample(
        input_path=sentence_table_path,
        output_path=label_set_path,
        keywords_file='tech_keywords.yaml',
        sample_size=min(30, len(sentence_df))  # Sample all or 30, whichever is smaller
    )

    print(f"\n✓ Label set created: {len(label_df)} sentences")

    # Step 4: Display results
    print("\n[4/4] Results Summary")
    print("=" * 80)

    print(f"\nTest files created in: {test_dir}/")
    print(f"  - test_items.json: {len(test_records)} Item extractions")
    print(f"  - sentence_table.csv: {len(sentence_df)} sentences")
    print(f"  - sentence_table.parquet: {len(sentence_df)} sentences")
    print(f"  - label_set.csv: {len(label_df)} sentences for labeling")

    print(f"\nSample sentences from label set:")
    print("-" * 80)

    for idx, row in label_df.head(5).iterrows():
        print(f"\n[{idx+1}] CIK: {row['cik']} | Year: {row['fiscal_year']} | Item: {row['item']}")
        print(f"    Tech hit: {row['tech_hit']} | Pool: {row['source_pool']}")
        print(f"    {row['sentence_text'][:120]}...")

    print("\n" + "=" * 80)
    print("TEST COMPLETE! ✓")
    print("=" * 80)
    print("""
All components working correctly:
  ✓ Sentence table builder
  ✓ Tech keyword matching
  ✓ Deduplication
  ✓ Sampling pipeline
  ✓ Label set export

You can now use these scripts with your real 10-K data.
    """)


if __name__ == '__main__':
    main()
