#!/usr/bin/env python3
"""
GDELT Data Analyzer
Utilities for analyzing and monitoring scraped GDELT data.
"""

import pandas as pd
import glob
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GDELTAnalyzer:
    """Analyze scraped GDELT data."""

    def __init__(self, data_dir: str = "./company_news"):
        self.data_dir = Path(data_dir)

    def load_company_data(self, company: str) -> pd.DataFrame:
        """
        Load all CSV files for a company.

        Args:
            company: Company name (e.g., 'tesla', 'nvidia')

        Returns:
            Combined DataFrame
        """
        pattern = str(self.data_dir / company / f"{company}_*.csv")
        files = glob.glob(pattern)

        if not files:
            logger.warning(f"No files found for {company}")
            return pd.DataFrame()

        logger.info(f"Loading {len(files)} files for {company}")

        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)

        # Parse dates
        combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
        combined['scrape_date'] = pd.to_datetime(combined['scrape_date'], errors='coerce')

        return combined

    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for a dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_articles': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'unique_domains': df['domain'].nunique() if 'domain' in df.columns else 0,
            'unique_countries': df['sourcecountry'].nunique() if 'sourcecountry' in df.columns else 0,
            'duplicates': df.duplicated(subset=['title', 'url']).sum(),
            'missing_dates': df['date'].isna().sum(),
        }

        return stats

    def analyze_coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze temporal coverage (articles per day).

        Args:
            df: DataFrame to analyze

        Returns:
            DataFrame with daily article counts
        """
        if 'date' not in df.columns or df.empty:
            return pd.DataFrame()

        # Group by date
        daily = df.groupby(df['date'].dt.date).size().reset_index()
        daily.columns = ['date', 'article_count']

        # Identify gaps
        date_range = pd.date_range(
            start=daily['date'].min(),
            end=daily['date'].max(),
            freq='D'
        )

        # Find missing dates
        full_range = pd.DataFrame({'date': date_range.date})
        coverage = full_range.merge(daily, on='date', how='left')
        coverage['article_count'] = coverage['article_count'].fillna(0).astype(int)

        return coverage

    def find_gaps(self, coverage_df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
        """
        Find dates with suspiciously low article counts.

        Args:
            coverage_df: Output from analyze_coverage()
            threshold: Minimum expected articles per day

        Returns:
            DataFrame of dates below threshold
        """
        gaps = coverage_df[coverage_df['article_count'] < threshold].copy()
        return gaps.sort_values('date')

    def analyze_sources(self, df: pd.DataFrame, top_n: int = 20) -> Dict:
        """
        Analyze article sources.

        Args:
            df: DataFrame to analyze
            top_n: Number of top sources to return

        Returns:
            Dictionary with source analysis
        """
        analysis = {}

        if 'domain' in df.columns:
            domain_counts = df['domain'].value_counts()
            analysis['top_domains'] = domain_counts.head(top_n).to_dict()
            analysis['total_domains'] = len(domain_counts)

        if 'sourcecountry' in df.columns:
            country_counts = df['sourcecountry'].value_counts()
            analysis['top_countries'] = country_counts.head(top_n).to_dict()
            analysis['total_countries'] = len(country_counts)

        if 'language' in df.columns:
            lang_counts = df['language'].value_counts()
            analysis['languages'] = lang_counts.to_dict()

        return analysis

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate articles.

        Args:
            df: DataFrame to deduplicate

        Returns:
            Deduplicated DataFrame
        """
        initial_count = len(df)

        # Remove exact duplicates
        df_clean = df.drop_duplicates(subset=['title', 'url'], keep='first')

        # Remove near-duplicates (same title, different URL)
        df_clean = df_clean.drop_duplicates(subset=['title'], keep='first')

        removed = initial_count - len(df_clean)
        logger.info(f"Removed {removed} duplicate articles ({removed / initial_count:.1%})")

        return df_clean

    def export_clean_data(self, company: str, output_file: str = None):
        """
        Load, clean, and export data for a company.

        Args:
            company: Company name
            output_file: Output filename (optional)
        """
        # Load data
        df = self.load_company_data(company)

        if df.empty:
            logger.error(f"No data found for {company}")
            return

        # Clean
        df_clean = self.remove_duplicates(df)

        # Sort by date
        df_clean = df_clean.sort_values('date')

        # Export
        if output_file is None:
            output_file = f"{company}_all_cleaned.csv"

        output_path = self.data_dir / output_file
        df_clean.to_csv(output_path, index=False)

        logger.info(f"Exported {len(df_clean)} articles to {output_path}")

        return df_clean

    def generate_report(self, company: str):
        """
        Generate comprehensive report for a company.

        Args:
            company: Company name
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Report for: {company.upper()}")
        logger.info(f"{'=' * 60}\n")

        # Load data
        df = self.load_company_data(company)

        if df.empty:
            logger.error("No data found")
            return

        # Summary stats
        stats = self.get_summary_stats(df)
        logger.info("Summary Statistics:")
        logger.info(f"  Total articles: {stats['total_articles']:,}")
        logger.info(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        logger.info(f"  Unique domains: {stats['unique_domains']}")
        logger.info(f"  Unique countries: {stats['unique_countries']}")
        logger.info(f"  Duplicates: {stats['duplicates']} ({stats['duplicates'] / len(df):.1%})")
        logger.info(f"  Missing dates: {stats['missing_dates']}")

        # Coverage analysis
        logger.info(f"\nCoverage Analysis:")
        coverage = self.analyze_coverage(df)
        logger.info(f"  Average articles/day: {coverage['article_count'].mean():.1f}")
        logger.info(f"  Median articles/day: {coverage['article_count'].median():.0f}")
        logger.info(f"  Max articles/day: {coverage['article_count'].max()}")

        gaps = self.find_gaps(coverage, threshold=5)
        logger.info(f"  Days with <5 articles: {len(gaps)}")

        if not gaps.empty:
            logger.info(f"\n  Sample gaps:")
            for _, row in gaps.head(10).iterrows():
                logger.info(f"    {row['date']}: {row['article_count']} articles")

        # Source analysis
        logger.info(f"\nTop 10 Sources:")
        source_analysis = self.analyze_sources(df, top_n=10)

        if 'top_domains' in source_analysis:
            for domain, count in list(source_analysis['top_domains'].items())[:10]:
                logger.info(f"  {domain}: {count:,} articles")

        if 'top_countries' in source_analysis:
            logger.info(f"\nTop 10 Countries:")
            for country, count in list(source_analysis['top_countries'].items())[:10]:
                logger.info(f"  {country}: {count:,} articles")

        logger.info(f"\n{'=' * 60}\n")


def main():
    """Generate reports for all companies."""
    analyzer = GDELTAnalyzer('../../data/news_data')

    companies = ['nvidia', 'tesla', 'intel', 'meta', 'boeing']

    for company in companies:
        try:
            analyzer.generate_report(company)
        except Exception as e:
            logger.error(f"Error analyzing {company}: {e}")

    # Export cleaned data
    logger.info("\nExporting cleaned datasets...")
    for company in companies:
        try:
            analyzer.export_clean_data(company)
        except Exception as e:
            logger.error(f"Error exporting {company}: {e}")


if __name__ == "__main__":
    main()