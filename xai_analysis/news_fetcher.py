"""
xai_analysis/news_fetcher.py
-----------------------------
Lightweight GDELT headline fetcher for window-based retrieval.

One API call per event (250 results, pre-ranked by relevance).
Headlines cached to data/news_cache/{TICKER}/{event_date}.csv.
"""

import json
import logging
import os
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Retry / backoff settings
MAX_RETRIES = 3
INITIAL_WAIT = 3
RATE_LIMIT_WAIT = 30
BETWEEN_REQUESTS = 8  # seconds between API calls — be gentle with free API

# GDELT struggles with long queries; keep it under this character count
MAX_QUERY_LENGTH = 200


def _make_session() -> requests.Session:
    """Create a session with connection pooling and transport retries."""
    session = requests.Session()
    retry = Retry(
        total=1,
        backoff_factor=1.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=3, pool_maxsize=3)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ── GDELT Query Generation via LLM ───────────────────────────────────────────

# Company name lookup — covers the common tickers, avoids an LLM call
_TICKER_COMPANY = {
    'NVDA': 'Nvidia',
    'TSLA': 'Tesla',
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'GOOGL': 'Google',
    'GOOG': 'Google',
    'AMZN': 'Amazon',
    'META': 'Meta',
    'INTC': 'Intel',
    'AMD': 'AMD',
    'NFLX': 'Netflix',
    'JPM': 'JPMorgan',
    'BAC': 'Bank of America',
    'V': 'Visa',
    'MA': 'Mastercard',
    'DIS': 'Disney',
    'CRM': 'Salesforce',
    'PYPL': 'PayPal',
    'UBER': 'Uber',
    'BA': 'Boeing',
    'WMT': 'Walmart',
    'PFE': 'Pfizer',
    'JNJ': 'Johnson & Johnson',
    'UNH': 'UnitedHealth',
    'XOM': 'ExxonMobil',
    'CVX': 'Chevron',
}


def _simple_query(ticker: str) -> str:
    """Build a short, reliable GDELT query without LLM."""
    company = _TICKER_COMPANY.get(ticker.upper(), ticker)
    return f'({company} OR {ticker}) (stock OR earnings OR market) sourcelang:english -forum -thread'


def generate_gdelt_query(ticker: str, client=None, model: str = "gemini-3.1-flash-lite") -> str:
    """Generate a GDELT search query for a ticker.

    Uses a simple template-based approach first (reliable).
    Falls back to LLM only if client is provided and ticker is unknown.

    Args:
        ticker: Stock ticker symbol.
        client: Optional google.genai.Client instance.
        model:  Gemini model to use.

    Returns:
        GDELT query string (kept short for reliability).
    """
    ticker_upper = ticker.upper()

    # Fast path: known ticker → skip LLM entirely
    if ticker_upper in _TICKER_COMPANY:
        query = _simple_query(ticker_upper)
        logger.info("Using template query for %s: %s", ticker, query)
        return query

    # Unknown ticker: try LLM for company name, fallback to simple
    if client is None:
        return _simple_query(ticker_upper)

    from google.genai import types

    prompt = f"""What is the company name for stock ticker {ticker}?
Reply with ONLY the company name, nothing else. Example: Nvidia"""

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        company = response.text.strip().strip('"').strip("'")
        # Keep query short and simple
        query = f'({company} OR {ticker}) (stock OR earnings OR market) sourcelang:english -forum -thread'
        if len(query) > MAX_QUERY_LENGTH:
            query = _simple_query(ticker_upper)
        logger.info("Generated GDELT query for %s: %s", ticker, query)
        return query
    except Exception as e:
        logger.error("LLM query gen failed for %s: %s — using fallback", ticker, e)
        return _simple_query(ticker_upper)


# ── GDELT Headline Fetching ──────────────────────────────────────────────────

def _build_url(query: str, start_dt: str, end_dt: str) -> str:
    """Build a GDELT API URL.

    Args:
        query:    Search query string.
        start_dt: Start datetime as YYYYMMDDHHMMSS.
        end_dt:   End datetime as YYYYMMDDHHMMSS.
    """
    import urllib.parse
    encoded = urllib.parse.quote(query)
    return (
        f"{GDELT_BASE_URL}?"
        f"query={encoded}&"
        f"mode=artlist&"
        f"format=json&"
        f"maxrecords=250&"
        f"sort=HybridRel&"
        f"startdatetime={start_dt}&"
        f"enddatetime={end_dt}"
    )


def _fetch_with_retry(url: str, session: requests.Session,
                      last_request_time: float) -> tuple[dict | None, float]:
    """GET a GDELT URL with exponential backoff.

    Returns:
        (parsed_json_or_None, updated_last_request_time)
    """
    for attempt in range(MAX_RETRIES):
        # Rate limiting
        wait = BETWEEN_REQUESTS - (time.time() - last_request_time)
        if wait > 0:
            time.sleep(wait)
        last_request_time = time.time()

        try:
            response = session.get(url, timeout=15)

            if response.status_code == 200:
                # Check content type — GDELT returns HTML on errors
                content_type = response.headers.get('content-type', '')
                if 'json' not in content_type and 'text/html' in content_type:
                    logger.warning(
                        "GDELT returned HTML instead of JSON on attempt %d "
                        "(query may be too complex)", attempt + 1
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(INITIAL_WAIT * (attempt + 1))
                    continue

                try:
                    data = response.json()
                    return data, last_request_time
                except json.JSONDecodeError:
                    # Log first 200 chars to help debug
                    snippet = response.text[:200] if response.text else '(empty)'
                    logger.warning(
                        "JSON decode error on attempt %d — response starts with: %s",
                        attempt + 1, snippet
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(INITIAL_WAIT * (attempt + 1))
                    continue

            elif response.status_code == 429:
                logger.warning("Rate limited — waiting %ds...", RATE_LIMIT_WAIT)
                time.sleep(RATE_LIMIT_WAIT)

            else:
                logger.warning("HTTP %d on attempt %d", response.status_code, attempt + 1)

        except requests.exceptions.Timeout:
            logger.warning("Timeout on attempt %d/%d", attempt + 1, MAX_RETRIES)
        except requests.exceptions.RequestException as e:
            logger.warning("Request error on attempt %d: %s", attempt + 1, e)

        if attempt < MAX_RETRIES - 1:
            wait_time = INITIAL_WAIT * (attempt + 1) + random.uniform(0, 2)
            logger.info("  Waiting %.1fs before retry...", wait_time)
            time.sleep(wait_time)

    return None, last_request_time


def fetch_event_headlines(
    query: str,
    event_date: str | pd.Timestamp,
    lookback_hours: int = 72,
    lookahead_hours: int = 24,
    session: requests.Session | None = None,
    last_request_time: float = 0.0,
) -> tuple[list[dict], float]:
    """Fetch headlines from GDELT for a single event window.

    One API call — GDELT returns up to 250 results ranked by relevance.

    Args:
        query:           GDELT search query string.
        event_date:      Event date (str YYYY-MM-DD or Timestamp).
        lookback_hours:  Hours before the event to search.
        lookahead_hours: Hours after the event to search.
        session:         Reusable requests session.
        last_request_time: Time of last API call (for rate limiting).

    Returns:
        (list_of_headline_dicts, updated_last_request_time)
    """
    if session is None:
        session = _make_session()

    if isinstance(event_date, str):
        event_dt = datetime.strptime(event_date[:10], "%Y-%m-%d")
    else:
        event_dt = pd.Timestamp(event_date).to_pydatetime()

    start_dt = event_dt - timedelta(hours=lookback_hours)
    end_dt = event_dt + timedelta(hours=lookahead_hours)

    start_str = start_dt.strftime("%Y%m%d%H%M%S")
    end_str = end_dt.strftime("%Y%m%d%H%M%S")

    url = _build_url(query, start_str, end_str)
    data, last_request_time = _fetch_with_retry(url, session, last_request_time)

    if data is None:
        logger.warning("Failed to fetch headlines for %s", event_date)
        return [], last_request_time

    articles = data.get("articles", [])
    headlines = [
        {
            "date": article.get("seendate", ""),
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "domain": article.get("domain", ""),
            "language": article.get("language", ""),
        }
        for article in articles
    ]

    logger.info("  Event %s: fetched %d headlines", str(event_date)[:10], len(headlines))
    return headlines, last_request_time


def fetch_all_event_headlines(
    query: str,
    event_dates: list,
    lookback_hours: int = 72,
    lookahead_hours: int = 24,
    cache_dir: str | Path | None = None,
    ticker: str = "",
) -> dict[str, list[dict]]:
    """Fetch headlines for multiple event dates, with disk caching.

    Args:
        query:           GDELT search query.
        event_dates:     List of event dates (str or Timestamp).
        lookback_hours:  Hours before event.
        lookahead_hours: Hours after event.
        cache_dir:       Directory for caching. If None, no caching.
        ticker:          Ticker name (for logging).

    Returns:
        {event_date_str: [headline_dicts]}
    """
    session = _make_session()
    last_request_time = 0.0
    results = {}

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    total = len(event_dates)
    for i, event_date in enumerate(event_dates, 1):
        date_str = str(event_date)[:10]

        # Check cache
        if cache_dir:
            cache_file = cache_dir / f"{date_str}.csv"
            if cache_file.exists():
                print(f"    [{i}/{total}] {date_str}: cached ✓")
                try:
                    cached_df = pd.read_csv(cache_file)
                    results[date_str] = cached_df.to_dict('records')
                    continue
                except Exception:
                    pass  # Cache corrupt, re-fetch

        # Fetch from API
        print(f"    [{i}/{total}] {date_str}: fetching...", end=' ')
        headlines, last_request_time = fetch_event_headlines(
            query, event_date, lookback_hours, lookahead_hours,
            session, last_request_time
        )
        results[date_str] = headlines
        print(f"{len(headlines)} headlines")

        # Save to cache
        if cache_dir and headlines:
            try:
                pd.DataFrame(headlines).to_csv(
                    cache_dir / f"{date_str}.csv", index=False
                )
            except Exception as e:
                logger.warning("Failed to cache headlines for %s: %s", date_str, e)

    return results
