"""
CEO Narcissism (FPSP) Pipeline
===============================
End-to-end pipeline that processes a folder of Refinitiv StreetEvents
earnings call transcripts (.txt) and produces a firm-year-level panel
dataset of CEO narcissism scores.

Workflow:
    1. Discover all .txt transcript files in input directory.
    2. Parse each transcript (speaker diarization, section splitting).
    3. Isolate CEO speech from the Q&A section (unscripted remarks).
    4. Compute the FPSP ratio and per-1000-word metrics.
    5. Output a CSV panel keyed by (company, ceo_name, quarter, date).

Usage:
    python ceo_narcissism_pipeline.py \\
        --input_dir  ./transcripts \\
        --output     ./output/ceo_narcissism_panel.csv

References:
    Aktas et al. (2016). CEO narcissism and the takeover process.
    Capalbo et al. (2018). CEO narcissism and earnings management.
"""

import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm

from cntext.io.earnings_call import (
    parse_earnings_call,
    get_ceo_qa_text,
    extract_header_metadata,
)
from cntext.stats.narcissism import fpsp_ratio


def process_single_transcript(filepath: str) -> dict:
    """
    Process one transcript file and return a result row.

    Returns:
        dict with metadata + FPSP metrics, or None if parsing fails
        or no CEO Q&A speech is found.
    """
    filename = os.path.basename(filepath)

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    except Exception as e:
        print(f"  [WARN] Could not read {filename}: {e}")
        return None

    # Parse transcript structure
    parsed = parse_earnings_call(text)

    # Extract header metadata (quarter, company, date)
    metadata = extract_header_metadata(text)

    # Find CEO name from corporate participants
    ceo_name = ''
    ceo_title = ''
    for p in parsed['corporate_participants']:
        title_lower = p.get('title', '').lower()
        if any(kw in title_lower for kw in [
            'chief executive officer', 'ceo',
            'president & ceo', 'president and ceo',
            'president & chief executive officer',
            'president and chief executive officer',
        ]):
            ceo_name = p['name']
            ceo_title = p['title']
            break

    # Isolate CEO speech from Q&A section
    ceo_qa_text = get_ceo_qa_text(parsed)

    if not ceo_qa_text.strip():
        print(f"  [INFO] No CEO Q&A speech found in {filename}")
        return None

    # Compute FPSP metrics
    metrics = fpsp_ratio(ceo_qa_text)

    # Also compute presentation-section metrics for comparison
    ceo_pres_turns = [t['text'] for t in parsed['presentation']
                      if t['is_ceo']]
    ceo_pres_text = ' '.join(ceo_pres_turns)
    pres_metrics = fpsp_ratio(ceo_pres_text) if ceo_pres_text.strip() else {}

    # Count turns for diagnostics
    total_qa_turns = len(parsed['qa'])
    ceo_qa_turns = len([t for t in parsed['qa'] if t['is_ceo']])

    return {
        # Identifiers
        'file': filename,
        'company_name': metadata.get('company_name', ''),
        'quarter': metadata.get('quarter', ''),
        'date': metadata.get('date', ''),
        'ceo_name': ceo_name,
        'ceo_title': ceo_title,

        # Q&A metrics (primary — unscripted)
        'qa_singular_count': metrics['singular_count'],
        'qa_plural_count': metrics['plural_count'],
        'qa_fpsp_ratio': metrics['fpsp_ratio'],
        'qa_fpsp_per_1000': metrics['fpsp_per_1000'],
        'qa_fppp_per_1000': metrics['fppp_per_1000'],
        'qa_word_count': metrics['word_count'],

        # Presentation metrics (secondary — scripted)
        'pres_singular_count': pres_metrics.get('singular_count', 0),
        'pres_plural_count': pres_metrics.get('plural_count', 0),
        'pres_fpsp_ratio': pres_metrics.get('fpsp_ratio', float('nan')),
        'pres_word_count': pres_metrics.get('word_count', 0),

        # Diagnostics
        'total_qa_turns': total_qa_turns,
        'ceo_qa_turns': ceo_qa_turns,
        'num_corporate_participants': len(parsed['corporate_participants']),
        'num_conference_participants': len(parsed['conference_participants']),
    }


def run_pipeline(input_dir: str,
                 output_path: str,
                 file_pattern: str = '*.txt') -> pd.DataFrame:
    """
    Process all transcript files in a directory and produce a panel dataset.

    Args:
        input_dir (str): Directory containing .txt transcript files.
        output_path (str): Path for the output CSV file.
        file_pattern (str): Glob pattern for transcript files.

    Returns:
        pd.DataFrame: Panel dataset with one row per transcript.
    """
    # Discover files
    pattern = os.path.join(input_dir, '**', file_pattern)
    files = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(files)} transcript files in {input_dir}")

    if not files:
        print("No files found. Check input_dir and file_pattern.")
        return pd.DataFrame()

    # Process each file
    results = []
    for filepath in tqdm(files, desc="Processing transcripts"):
        row = process_single_transcript(filepath)
        if row is not None:
            results.append(row)

    if not results:
        print("No valid results produced.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Sort by company and quarter
    df = df.sort_values(['company_name', 'quarter']).reset_index(drop=True)

    # Save output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nPanel dataset saved to {output_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Companies: {df['company_name'].nunique()}")
    print(f"  CEOs: {df['ceo_name'].nunique()}")
    print(f"\nSummary statistics (Q&A FPSP Ratio):")
    print(df['qa_fpsp_ratio'].describe().to_string())

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CEO Narcissism (FPSP) pipeline for earnings call transcripts.'
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Directory containing .txt transcript files.'
    )
    parser.add_argument(
        '--output', type=str, default='./output/ceo_narcissism_panel.csv',
        help='Output CSV path (default: ./output/ceo_narcissism_panel.csv).'
    )
    parser.add_argument(
        '--pattern', type=str, default='*.txt',
        help='File glob pattern (default: *.txt).'
    )
    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        output_path=args.output,
        file_pattern=args.pattern,
    )
