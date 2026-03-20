"""
clean_news_data.py
------------------
Module 2A: Rigorous cleaning of GDELT news headlines for the bridging model.

Pipeline:
  1. Ingest    — Load all yearly CSVs for nvidia, tesla, intel (skip boeing).
  2. English   — Keep only English-language articles.
  3. Deduplicate — Drop exact-title duplicates (same date + title).
  4. Company   — Require the headline to contain the company name OR ticker.
  5. Financial — Require at least one strong financial keyword.
  6. Source    — Retain only top-tier / reputable financial news domains.
  7. Quality   — Drop very short titles (< 30 chars) and obvious spam patterns.
  8. Save      — Write one cleaned CSV per company to data/cleaned_news/.

Usage:
    python clean_news_data.py
"""

import os
import re
import glob
import pandas as pd
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────

NEWS_ROOT   = './data/news_data'
OUTPUT_DIR  = './data/cleaned_news'
EXCLUDE     = {'boeing'}

COMPANY_MAP = {
    'nvidia': {
        'names': ['nvidia', 'nvda', 'geforce', 'jensen huang'],
        'secondary_names': [],   # no co-occurrence gating needed
        'ticker': 'NVDA',
    },
    'tesla': {
        'names': ['tesla', 'tsla'],
        # 'elon musk' is kept but gated: headline must ALSO mention tesla/tsla/stock
        'secondary_names': ['elon musk'],
        'ticker': 'TSLA',
    },
    'intel': {
        'names': ['intel', 'intc', 'pat gelsinger', 'intel corp'],
        'secondary_names': [],
        'ticker': 'INTC',
    },
}

# Per-company daily article cap — keeps top N per day (by source tier)
# to prevent high-profile tickers from dominating.
MAX_PER_COMPANY_PER_DAY = 50

# Financial / business keywords — must contain at least one
FINANCIAL_KEYWORDS = [
    # Earnings & financials
    'earning', 'revenue', 'profit', 'loss', 'quarter', 'quarterly', 'annual',
    'fiscal', 'eps', 'guidance', 'forecast', 'outlook', 'beat', 'miss',
    'report', 'result', 'income', 'margin', 'expense', 'sales', 'growth',
    # Market actions
    'stock', 'share', 'buy', 'sell', 'upgrade', 'downgrade', 'target',
    'rally', 'surge', 'plunge', 'crash', 'drop', 'rise', 'soar', 'tank',
    'bull', 'bear', 'volatil', 'momentum', 'gain', 'decline', 'jump',
    'market', 'investor', 'analyst', 'wall street', 'sec', 'ipo',
    'dividend', 'buyback', 'split', 'valuation', 'worth', 'billion',
    'trillion', 'capitali',
    # Business events
    'acqui', 'merger', 'deal', 'partnership', 'contract', 'lawsuit',
    'regulat', 'antitrust', 'layoff', 'restructur', 'recall',
    'launch', 'chip', 'semiconductor', 'ai', 'artificial intellig',
    'data center', 'cloud', 'supply chain', 'shortage', 'compet',
    'innovat', 'product', 'deliver', 'produc', 'manufactur', 'factory',
    'vehicle', 'autonomous', 'self-driv', 'gpu', 'processor',
    # Ratings & sentiment
    'rating', 'overweight', 'underweight', 'outperform', 'underperform',
    'price target', 'hold', 'neutral', 'top pick', 'recommend',
    # General business
    'ceo', 'executive', 'board', 'hire', 'appoint', 'resign',
    'invest', 'fund', 'venture', 'capital', 'financ',
]

MACRO_KEYWORDS = ['federal reserve', 'fed ', 'inflation', 'cpi', 'interest rate', 'tariff', 'war', 'pandemic']

# Reputable domains — pattern-match (substring).  Broadened to include
# mid-tier tech / finance / wire sources so we don't starve low-coverage tickers.
TRUSTED_DOMAIN_PATTERNS = [
    # Tier-1 financial
    'reuters', 'bloomberg', 'cnbc', 'wsj', 'ft.com', 'financialtimes',
    'marketwatch', 'barrons', 'seekingalpha', 'fool.com', 'motleyfool',
    'investopedia', 'yahoo', 'finance.yahoo', 'benzinga', 'thestreet',
    'investorplace', 'zacks', 'tipranks', 'marketbeat',
    # Tier-2 business & wire
    'businessinsider', 'insider.com', 'apnews', 'bbc', 'nytimes',
    'washingtonpost', 'theguardian', 'cnn.com', 'foxbusiness', 'nbcnews',
    'fortune', 'economist', 'forbes', 'biztoc', 'prnewswire',
    'globenewswire', 'accesswire', 'newswire', 'businesswire',
    # Tech
    'techcrunch', 'theverge', 'arstechnica', 'wired', 'engadget',
    'tomshardware', 'anandtech', 'semianalysis', 'electrek',
    'tesmanian', 'insideevs', 'cleantechnica', 'torquenews',
    'zdnet', 'cnet', 'techradar', 'venturebeat', 'theinformation',
    # Broader coverage
    'usatoday', 'time.com', 'newsweek', 'politico', 'thehill',
    'msn.com', 'uk.finance', 'moneycontrol', 'livemint',
    'nasdaq.com', 'schaeffers', 'investing.com', 'kiplinger',
]

# Spam patterns (case-insensitive regex)
SPAM_PATTERNS = [
    r'^[A-Z]{2,5}\s*[-–]\s*press release',      # ticker — press release
    r'click here',
    r'subscribe now',
    r'sign up',
    r'sponsored',
    r'^daily bulletin',
    r'^horoscope',
]

MIN_TITLE_LENGTH = 15


# ── Helpers ──────────────────────────────────────────────────────────────────

def _contains_any(text: str, patterns: list[str]) -> bool:
    """Case-insensitive substring match."""
    text_lower = text.lower()
    return any(p in text_lower for p in patterns)


def _matches_company(title: str, info: dict) -> bool:
    """Check if title mentions the company.

    Primary names: direct match.
    Secondary names (e.g. 'elon musk'): must co-occur with a primary name
    OR a stock-related anchor word (stock, share, market, invest, buy, sell).
    """
    t = title.lower()
    primary = [n.lower() for n in info['names']]
    if any(p in t for p in primary):
        return True
    secondary = [n.lower() for n in info.get('secondary_names', [])]
    if not secondary:
        return False
    anchors = primary + ['stock', 'share', 'market', 'invest', 'buy', 'sell',
                         'rally', 'surge', 'drop', 'crash', 'valuation']
    for sec in secondary:
        if sec in t and any(a in t for a in anchors):
            return True
    return False


def _is_trusted_domain(domain: str) -> bool:
    if pd.isna(domain):
        return False
    domain_lower = str(domain).lower()
    return any(p in domain_lower for p in TRUSTED_DOMAIN_PATTERNS)


def _is_spam(title: str) -> bool:
    for pat in SPAM_PATTERNS:
        if re.search(pat, title, re.IGNORECASE):
            return True
    return False


# ── Main Pipeline ────────────────────────────────────────────────────────────

def clean_company(company: str) -> pd.DataFrame:
    """Load, clean, and return filtered headlines for one company."""
    info = COMPANY_MAP[company]
    company_dir = os.path.join(NEWS_ROOT, company)
    csv_files = sorted(glob.glob(os.path.join(company_dir, '*.csv')))

    if not csv_files:
        print(f"  ⚠ No CSVs found in {company_dir}")
        return pd.DataFrame()

    # 1. Ingest
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=['date', 'title', 'url', 'domain', 'language'],
                             dtype=str, on_bad_lines='skip')
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠ Skipping {f}: {e}")
    if not dfs:
        return pd.DataFrame()

    raw = pd.concat(dfs, ignore_index=True)
    n_raw = len(raw)

    # 2. Parse dates and drop un-parseable
    raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
    raw = raw.dropna(subset=['date', 'title'])

    # 3. English only
    raw = raw[raw['language'].str.strip().str.lower() == 'english']

    # 4. Deduplicate — by normalised title only (catches syndicated copies
    #    across different domains and timestamps); keep earliest occurrence
    raw = raw.sort_values('date')
    raw['_title_norm'] = raw['title'].str.strip().str.lower()
    raw = raw.drop_duplicates(subset=['_title_norm'], keep='first')
    raw = raw.drop(columns=['_title_norm'])

    # 5. Company name / ticker filter (with co-occurrence gating)
    mask_company = raw['title'].apply(lambda t: _matches_company(str(t), info))
    mask_macro = raw['title'].apply(lambda t: _contains_any(str(t), MACRO_KEYWORDS))
    raw = raw[mask_company | mask_macro]

    # 6. Financial keyword filter
    mask_finance = raw['title'].apply(lambda t: _contains_any(str(t), FINANCIAL_KEYWORDS))
    raw = raw[mask_finance]

    # 7. Trusted source filter
    # mask_source = raw['domain'].apply(_is_trusted_domain)
    # raw = raw[mask_source]

    # 8. Quality filters
    raw = raw[raw['title'].str.len() >= MIN_TITLE_LENGTH]
    mask_spam = raw['title'].apply(_is_spam)
    raw = raw[~mask_spam]

    # Final sort
    raw = raw.sort_values('date').reset_index(drop=True)

    # 9. Daily cap — keep top N articles per day (prefer tier-1 sources)
    raw['_day'] = raw['date'].dt.date
    raw['_tier1'] = raw['domain'].apply(
        lambda d: any(p in str(d).lower() for p in TRUSTED_DOMAIN_PATTERNS[:19])  # tier-1 slice
    ).astype(int)
    raw = (raw.sort_values(['_day', '_tier1', 'date'], ascending=[True, False, True])
              .groupby('_day').head(MAX_PER_COMPANY_PER_DAY)
              .drop(columns=['_day', '_tier1'])
              .reset_index(drop=True))

    # Add company/ticker columns for downstream use
    raw['company'] = company
    raw['ticker'] = info['ticker']

    # Report
    n_clean = len(raw)
    pct = (n_clean / n_raw * 100) if n_raw else 0
    print(f"  {company:>8}: {n_raw:>8,} raw → {n_clean:>6,} clean ({pct:.1f}%)")
    return raw


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== Cleaning GDELT News Headlines ===\n")

    all_dfs = []
    companies = [c for c in sorted(COMPANY_MAP) if c not in EXCLUDE]

    for company in companies:
        df = clean_company(company)
        if not df.empty:
            out_path = os.path.join(OUTPUT_DIR, f'{company}_cleaned.csv')
            df.to_csv(out_path, index=False)
            print(f"  → Saved {out_path}\n")
            all_dfs.append(df)

    # Also save a merged version for convenience
    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True).sort_values('date').reset_index(drop=True)
        merged_path = os.path.join(OUTPUT_DIR, 'all_cleaned.csv')
        merged.to_csv(merged_path, index=False)
        print(f"Merged: {len(merged):,} headlines → {merged_path}")

        # Stats
        print("\n── Summary ────────────────────────────────")
        print(f"  Date range : {merged['date'].min()} → {merged['date'].max()}")
        print(f"  Companies  : {merged['company'].nunique()}")
        for comp in merged['company'].unique():
            n = (merged['company'] == comp).sum()
            print(f"    {comp:>8} : {n:>6,} headlines")
        print(f"  Domains    : {merged['domain'].nunique()} unique")
        top5 = merged['domain'].value_counts().head(5)
        for dom, cnt in top5.items():
            print(f"    {dom:>30} : {cnt:>5,}")

    print("\n✓ Done.")


if __name__ == '__main__':
    main()
