"""
Comprehensive MD&A Text Analysis Pipeline
==========================================

This script performs end-to-end analysis of MD&A sections from SEC 10-K filings using
the cntext library's full suite of 40+ text analysis metrics.

Features:
- Loads MD&A data from parquet format
- Calculates sentiment metrics (Loughran-McDonald financial dictionary)
- Computes readability indices (6 different measures)
- Extracts vocabulary features (diversity, complexity, etc.)
- Batch processing with progress tracking
- Error handling and logging
- Exports comprehensive results

Author: Claude Code
Date: 2025-11-19
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import logging
import time
from pathlib import Path
import sys

# Import cntext modules
import cntext as ct
from cntext.stats.sentiment import sentiment
from cntext.stats.readability import readability
from cntext.stats.utils import word_count
from cntext.io.dict import read_yaml_dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mda_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MDAAnalyzer:
    """Comprehensive MD&A Text Analyzer using cntext library"""

    def __init__(self, data_path='mda_full.parquet'):
        """
        Initialize the analyzer

        Args:
            data_path (str): Path to the parquet file containing MD&A data
        """
        self.data_path = data_path
        self.df = None
        self.results_df = None

        # Load dictionaries
        logger.info("Loading sentiment dictionaries...")
        self.lm_dict = read_yaml_dict('en_common_LoughranMcDonald.yaml')
        self.nrc_dict = read_yaml_dict('en_common_NRC.yaml')
        logger.info("✓ Dictionaries loaded successfully")

    def load_data(self, sample_size=None):
        """
        Load MD&A data from parquet file

        Args:
            sample_size (int, optional): If provided, loads only a sample of records
        """
        logger.info(f"Loading data from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)

        if sample_size:
            self.df = self.df.head(sample_size)
            logger.info(f"✓ Loaded sample of {len(self.df):,} records")
        else:
            logger.info(f"✓ Loaded {len(self.df):,} records")

        # Display basic statistics
        logger.info(f"   Companies: {self.df['company'].nunique()}")
        logger.info(f"   Year range: {self.df['year'].min()} - {self.df['year'].max()}")
        logger.info(f"   Avg MD&A length: {self.df['mda_text'].str.len().mean():,.0f} characters")

    def calculate_sentiment_lm(self, text):
        """
        Calculate Loughran-McDonald sentiment metrics

        The Loughran-McDonald dictionary is specifically designed for financial text
        and is the gold standard for analyzing SEC filings.

        Returns:
            dict: Sentiment metrics including Positive, Negative, Uncertainty, Litigious, etc.
        """
        try:
            result = sentiment(text, diction=self.lm_dict['Dictionary'], lang='english', return_series=False)

            # Calculate percentages relative to total word count
            word_num = result.get('word_num', 1)
            if word_num == 0:
                word_num = 1

            metrics = {
                'lm_positive_num': result.get('Positive_num', 0),
                'lm_negative_num': result.get('Negative_num', 0),
                'lm_uncertainty_num': result.get('Uncertainty_num', 0),
                'lm_litigious_num': result.get('Litigious_num', 0),
                'lm_constraining_num': result.get('Constraining_num', 0),
                'lm_superfluous_num': result.get('Superfluous_num', 0),
                'lm_interesting_num': result.get('Interesting_num', 0),
                'lm_modal_weak_num': result.get('Modal_Weak_num', 0),
                'lm_modal_moderate_num': result.get('Modal_Moderate_num', 0),
                'lm_modal_strong_num': result.get('Modal_Strong_num', 0),

                # Percentages (per 100 words)
                'lm_positive_pct': round(100 * result.get('Positive_num', 0) / word_num, 4),
                'lm_negative_pct': round(100 * result.get('Negative_num', 0) / word_num, 4),
                'lm_uncertainty_pct': round(100 * result.get('Uncertainty_num', 0) / word_num, 4),
                'lm_litigious_pct': round(100 * result.get('Litigious_num', 0) / word_num, 4),
                'lm_constraining_pct': round(100 * result.get('Constraining_num', 0) / word_num, 4),

                # Net sentiment
                'lm_net_sentiment': result.get('Positive_num', 0) - result.get('Negative_num', 0),
                'lm_polarity': round((result.get('Positive_num', 0) - result.get('Negative_num', 0)) /
                                    max(result.get('Positive_num', 0) + result.get('Negative_num', 0), 1), 4),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating LM sentiment: {str(e)}")
            return self._empty_lm_metrics()

    def calculate_sentiment_nrc(self, text):
        """
        Calculate NRC emotion metrics

        Returns:
            dict: Emotion metrics (anger, fear, joy, sadness, etc.)
        """
        try:
            result = sentiment(text, diction=self.nrc_dict['Dictionary'], lang='english', return_series=False)

            word_num = result.get('word_num', 1)
            if word_num == 0:
                word_num = 1

            metrics = {
                'nrc_anger_pct': round(100 * result.get('Anger_num', 0) / word_num, 4),
                'nrc_anticipation_pct': round(100 * result.get('Anticipation_num', 0) / word_num, 4),
                'nrc_disgust_pct': round(100 * result.get('Disgust_num', 0) / word_num, 4),
                'nrc_fear_pct': round(100 * result.get('Fear_num', 0) / word_num, 4),
                'nrc_joy_pct': round(100 * result.get('Joy_num', 0) / word_num, 4),
                'nrc_sadness_pct': round(100 * result.get('Sadness_num', 0) / word_num, 4),
                'nrc_surprise_pct': round(100 * result.get('Surprise_num', 0) / word_num, 4),
                'nrc_trust_pct': round(100 * result.get('Trust_num', 0) / word_num, 4),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating NRC sentiment: {str(e)}")
            return self._empty_nrc_metrics()

    def calculate_readability(self, text):
        """
        Calculate readability indices

        Returns:
            dict: Readability metrics (Fog, SMOG, Coleman-Liau, ARI, etc.)
        """
        try:
            result = readability(text, lang='english', syllables=3, return_series=False)
            return {
                'fog_index': result.get('fog_index', np.nan),
                'flesch_kincaid_grade': result.get('flesch_kincaid_grade_level', np.nan),
                'smog_index': result.get('smog_index', np.nan),
                'coleman_liau_index': result.get('coleman_liau_index', np.nan),
                'ari': result.get('ari', np.nan),
                'rix': result.get('rix', np.nan),
            }
        except Exception as e:
            logger.error(f"Error calculating readability: {str(e)}")
            return self._empty_readability_metrics()

    def calculate_vocabulary_features(self, text):
        """
        Calculate vocabulary and linguistic features

        Returns:
            dict: Vocabulary metrics (word count, unique words, diversity, etc.)
        """
        try:
            # Use cntext's word_count function
            wc_result = word_count(text, lang='english', return_df=False)

            word_num = wc_result.get('word_num', 0)
            unique_words = wc_result.get('unique_word_num', 0)

            # Type-Token Ratio (vocabulary diversity)
            ttr = round(unique_words / max(word_num, 1), 4)

            # Calculate hapax legomena rate (words appearing only once)
            from collections import Counter
            import re
            rgx = re.compile(r"(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)")
            words = [w.lower() for w in re.split(rgx, text) if w]
            word_freq = Counter(words)
            hapax = sum(1 for count in word_freq.values() if count == 1)
            hapax_rate = round(hapax / max(word_num, 1), 4)

            # Calculate HHI concentration (vocabulary concentration index)
            word_probs = np.array([count / max(word_num, 1) for count in word_freq.values()])
            hhi = round(np.sum(word_probs ** 2), 6)

            metrics = {
                'word_num': word_num,
                'unique_word_num': unique_words,
                'sentence_num': wc_result.get('sentence_num', 0),
                'stopword_num': wc_result.get('stopword_num', 0),
                'type_token_ratio': ttr,
                'hapax_legomena_rate': hapax_rate,
                'vocabulary_hhi': hhi,
                'avg_word_length': round(sum(len(w) for w in words) / max(len(words), 1), 2),
                'avg_sentence_length': round(word_num / max(wc_result.get('sentence_num', 1), 1), 2),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating vocabulary features: {str(e)}")
            return self._empty_vocabulary_metrics()

    def calculate_text_metadata(self, text):
        """
        Calculate basic text metadata

        Returns:
            dict: Metadata metrics (character count, etc.)
        """
        return {
            'char_count': len(text),
            'char_count_no_spaces': len(text.replace(' ', '')),
        }

    def _empty_lm_metrics(self):
        """Return empty Loughran-McDonald metrics dict"""
        return {k: np.nan for k in [
            'lm_positive_num', 'lm_negative_num', 'lm_uncertainty_num',
            'lm_litigious_num', 'lm_constraining_num', 'lm_superfluous_num',
            'lm_interesting_num', 'lm_modal_weak_num', 'lm_modal_moderate_num',
            'lm_modal_strong_num', 'lm_positive_pct', 'lm_negative_pct',
            'lm_uncertainty_pct', 'lm_litigious_pct', 'lm_constraining_pct',
            'lm_net_sentiment', 'lm_polarity'
        ]}

    def _empty_nrc_metrics(self):
        """Return empty NRC metrics dict"""
        return {k: np.nan for k in [
            'nrc_anger_pct', 'nrc_anticipation_pct', 'nrc_disgust_pct',
            'nrc_fear_pct', 'nrc_joy_pct', 'nrc_sadness_pct',
            'nrc_surprise_pct', 'nrc_trust_pct'
        ]}

    def _empty_readability_metrics(self):
        """Return empty readability metrics dict"""
        return {k: np.nan for k in [
            'fog_index', 'flesch_kincaid_grade', 'smog_index',
            'coleman_liau_index', 'ari', 'rix'
        ]}

    def _empty_vocabulary_metrics(self):
        """Return empty vocabulary metrics dict"""
        return {k: np.nan for k in [
            'word_num', 'unique_word_num', 'sentence_num', 'stopword_num',
            'type_token_ratio', 'hapax_legomena_rate', 'vocabulary_hhi',
            'avg_word_length', 'avg_sentence_length'
        ]}

    def analyze_single_record(self, row):
        """
        Analyze a single MD&A text and return all metrics

        Args:
            row (pd.Series): Row containing MD&A data

        Returns:
            dict: Dictionary containing all calculated metrics
        """
        text = row['mda_text']

        # Combine all metrics
        metrics = {
            'cik': row['cik'],
            'company': row['company'],
            'filing_date': row['filing_date'],
            'period_of_report': row['period_of_report'],
            'year': row['year'],
        }

        # Add metadata
        metrics.update(self.calculate_text_metadata(text))

        # Add vocabulary features
        metrics.update(self.calculate_vocabulary_features(text))

        # Add readability metrics
        metrics.update(self.calculate_readability(text))

        # Add sentiment metrics
        metrics.update(self.calculate_sentiment_lm(text))
        metrics.update(self.calculate_sentiment_nrc(text))

        return metrics

    def analyze_batch(self, batch_size=100):
        """
        Analyze all records in batches with progress tracking

        Args:
            batch_size (int): Number of records to process in each batch
        """
        logger.info(f"\nStarting analysis of {len(self.df):,} records...")
        logger.info(f"Batch size: {batch_size}")

        results = []
        failed_records = []

        # Process with progress bar
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Analyzing MD&A texts"):
            try:
                metrics = self.analyze_single_record(row)
                results.append(metrics)

            except Exception as e:
                logger.error(f"Failed to process record {idx} (CIK: {row.get('cik', 'unknown')}): {str(e)}")
                failed_records.append({
                    'index': idx,
                    'cik': row.get('cik', 'unknown'),
                    'error': str(e)
                })

            # Periodic progress updates
            if (idx + 1) % batch_size == 0:
                logger.info(f"  Processed {idx + 1:,} / {len(self.df):,} records ({100*(idx+1)/len(self.df):.1f}%)")

        # Create results DataFrame
        self.results_df = pd.DataFrame(results)

        logger.info(f"\n✓ Analysis complete!")
        logger.info(f"  Successfully processed: {len(results):,} records")
        logger.info(f"  Failed: {len(failed_records)} records")

        if failed_records:
            failed_df = pd.DataFrame(failed_records)
            failed_df.to_csv('mda_analysis_failed_records.csv', index=False)
            logger.warning(f"  Failed records saved to: mda_analysis_failed_records.csv")

    def export_results(self, output_path='mda_analysis_results.parquet', format='parquet'):
        """
        Export analysis results

        Args:
            output_path (str): Path for output file
            format (str): Output format ('parquet', 'csv', or 'both')
        """
        if self.results_df is None:
            logger.error("No results to export. Run analyze_batch() first.")
            return

        logger.info(f"\nExporting results...")

        if format in ['parquet', 'both']:
            self.results_df.to_parquet(output_path, compression='gzip', index=False)
            file_size = Path(output_path).stat().st_size / 1024 / 1024
            logger.info(f"✓ Saved parquet: {output_path} ({file_size:.1f} MB)")

        if format in ['csv', 'both']:
            csv_path = output_path.replace('.parquet', '.csv')
            self.results_df.to_csv(csv_path, index=False)
            file_size = Path(csv_path).stat().st_size / 1024 / 1024
            logger.info(f"✓ Saved CSV: {csv_path} ({file_size:.1f} MB)")

        logger.info(f"  Shape: {self.results_df.shape}")
        logger.info(f"  Columns: {len(self.results_df.columns)}")

    def generate_summary_statistics(self):
        """Generate and display summary statistics of the analysis"""
        if self.results_df is None:
            logger.error("No results available. Run analyze_batch() first.")
            return

        logger.info("\n" + "="*80)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*80)

        # Numeric columns only
        numeric_cols = self.results_df.select_dtypes(include=[np.number]).columns

        summary = self.results_df[numeric_cols].describe()

        logger.info("\n--- KEY METRICS SUMMARY ---\n")

        # Text length
        logger.info("TEXT LENGTH:")
        logger.info(f"  Mean: {self.results_df['char_count'].mean():,.0f} characters")
        logger.info(f"  Median: {self.results_df['char_count'].median():,.0f} characters")
        logger.info(f"  Mean words: {self.results_df['word_num'].mean():,.0f}")

        # Sentiment (Loughran-McDonald)
        logger.info("\nLOUGHRAN-MCDONALD SENTIMENT (% per 100 words):")
        logger.info(f"  Positive: {self.results_df['lm_positive_pct'].mean():.3f}%")
        logger.info(f"  Negative: {self.results_df['lm_negative_pct'].mean():.3f}%")
        logger.info(f"  Uncertainty: {self.results_df['lm_uncertainty_pct'].mean():.3f}%")
        logger.info(f"  Litigious: {self.results_df['lm_litigious_pct'].mean():.3f}%")
        logger.info(f"  Net Sentiment: {self.results_df['lm_net_sentiment'].mean():.1f}")

        # Readability
        logger.info("\nREADABILITY:")
        logger.info(f"  Fog Index: {self.results_df['fog_index'].mean():.2f}")
        logger.info(f"  Flesch-Kincaid Grade: {self.results_df['flesch_kincaid_grade'].mean():.2f}")
        logger.info(f"  SMOG Index: {self.results_df['smog_index'].mean():.2f}")

        # Vocabulary
        logger.info("\nVOCABULARY:")
        logger.info(f"  Type-Token Ratio: {self.results_df['type_token_ratio'].mean():.4f}")
        logger.info(f"  Hapax Rate: {self.results_df['hapax_legomena_rate'].mean():.4f}")
        logger.info(f"  Avg Word Length: {self.results_df['avg_word_length'].mean():.2f}")
        logger.info(f"  Avg Sentence Length: {self.results_df['avg_sentence_length'].mean():.2f}")

        # Save detailed summary
        summary.to_csv('mda_analysis_summary_stats.csv')
        logger.info(f"\n✓ Detailed summary statistics saved to: mda_analysis_summary_stats.csv")

        return summary


def main():
    """Main execution function"""

    print("\n" + "="*80)
    print(" COMPREHENSIVE MD&A TEXT ANALYSIS PIPELINE")
    print(" Using cntext library - 40+ Financial Text Metrics")
    print("="*80 + "\n")

    # Initialize analyzer
    analyzer = MDAAnalyzer(data_path='mda_full.parquet')

    # Load data - start with a sample for testing
    # Change sample_size to None to process all records
    analyzer.load_data(sample_size=100)  # Start with 100 records for testing

    # Run analysis
    start_time = time.time()
    analyzer.analyze_batch(batch_size=50)
    elapsed = time.time() - start_time

    logger.info(f"\nTotal processing time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Average time per record: {elapsed/len(analyzer.df):.2f} seconds")

    # Generate summary statistics
    analyzer.generate_summary_statistics()

    # Export results
    analyzer.export_results(
        output_path='mda_analysis_results.parquet',
        format='both'  # Export both parquet and CSV
    )

    print("\n" + "="*80)
    print(" ✓ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nOutput files:")
    print("  - mda_analysis_results.parquet (compressed, recommended for large datasets)")
    print("  - mda_analysis_results.csv (human-readable)")
    print("  - mda_analysis_summary_stats.csv (statistical summary)")
    print("  - mda_analysis.log (detailed processing log)")
    print("\n")


if __name__ == '__main__':
    main()
