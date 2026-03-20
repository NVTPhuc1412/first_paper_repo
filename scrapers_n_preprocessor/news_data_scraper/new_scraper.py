import json
import logging
import time
import random
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def setup_logging(log_dir: str = "./logs", log_name: str = None) -> logging.Logger:
    """Configure logging to both file (DEBUG+) and console (INFO+)."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    if log_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"gdelt_scraper_{timestamp}.log"

    log_file = log_path / log_name
    fmt = "%(asctime)s - %(levelname)s - %(message)s"

    logger = logging.getLogger("gdelt_scraper")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f"Logging to: {log_file}")

    return logger


def _make_session() -> requests.Session:
    """Create a requests Session with connection pooling and transport-level retries."""
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class GDELTScraper:
    """GDELT API scraper with rate limiting and incremental CSV saving."""

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Retry / backoff
    MAX_RETRIES = 5
    INITIAL_WAIT = 2
    RATE_LIMIT_WAIT = 60
    SERVER_ERROR_WAIT = 45
    BETWEEN_REQUESTS_WAIT = 6

    def __init__(self, output_dir: str = ".", log_dir: str = "./logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger: Optional[logging.Logger] = None
        self.consecutive_errors = 0
        self.total_rate_limits = 0

        self._session = _make_session()
        self._last_request_time = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_limited_get(self, url: str, timeout: int = 30) -> requests.Response:
        """Enforce minimum gap between requests to honour the API rate limit."""
        wait = self.BETWEEN_REQUESTS_WAIT - (time.time() - self._last_request_time)
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()
        return self._session.get(url, timeout=timeout)

    def _build_request_url(self, query: str, start_time: str, end_time: str) -> str:
        encoded_query = urllib.parse.quote(query)
        return (
            f"{self.BASE_URL}?"
            f"query={encoded_query}&"
            f"mode=artlist&"
            f"format=json&"
            f"maxrecords=100&"
            f"sort=HybridRel&"
            f"startdatetime={start_time}&"
            f"enddatetime={end_time}"
        )

    def _get_time_windows(self, date: datetime) -> List[Tuple[str, str, str]]:
        """Return four 6-hour windows covering the full day."""
        d = date.strftime("%Y%m%d")
        return [
            (f"{d}000000", f"{d}115959", "W1"),
            (f"{d}120000", f"{d}235959", "W2"),
        ]

    def _fetch_with_retry(self, url: str, date_str: str, suffix: str) -> Optional[Dict]:
        """GET a GDELT URL with exponential backoff. Returns parsed JSON or None."""
        tag = f"[{date_str} {suffix}]"

        for attempt in range(self.MAX_RETRIES):
            try:
                self.logger.debug(f"{tag} Attempt {attempt + 1}: {url[:100]}...")
                response = self._rate_limited_get(url)

                if response.status_code == 200:
                    try:
                        data = response.json()
                        self.consecutive_errors = 0
                        self.logger.debug(f"{tag} Got {len(data.get('articles', []))} articles")
                        return data
                    except json.JSONDecodeError as e:
                        self.logger.error(f"{tag} JSON decode error: {e}")
                        is_server_error = (
                            response.text.startswith("{Content-type:")
                            or "unknown error occurred" in response.text.lower()
                        )
                        if is_server_error:
                            self.consecutive_errors += 1
                            wait = self.SERVER_ERROR_WAIT * self.consecutive_errors
                            self.logger.warning(f"{tag} Server error — waiting {wait}s...")
                            time.sleep(wait)
                            continue
                        return None

                elif response.status_code == 429:
                    self.total_rate_limits += 1
                    self.logger.warning(
                        f"{tag} Rate limited (#{self.total_rate_limits}, "
                        f"attempt {attempt + 1}/{self.MAX_RETRIES}) — waiting {self.RATE_LIMIT_WAIT}s..."
                    )
                    time.sleep(self.RATE_LIMIT_WAIT)

                else:
                    self.logger.warning(
                        f"{tag} HTTP {response.status_code} (attempt {attempt + 1}/{self.MAX_RETRIES})"
                    )

            except requests.exceptions.Timeout:
                self.logger.warning(f"{tag} Timeout (attempt {attempt + 1})")
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"{tag} Request error: {e} (attempt {attempt + 1})")

            if attempt < self.MAX_RETRIES - 1:
                wait = self.INITIAL_WAIT ** (attempt + 1) + random.uniform(0, 1.5)
                self.logger.debug(f"Retrying in {wait:.1f}s...")
                time.sleep(wait)

        return None

    def _fetch_window(
        self, query: str, start_time: str, end_time: str, date_str: str, suffix: str
    ) -> Tuple[str, str, Optional[List[Dict]]]:
        """Fetch and process one time window."""
        url = self._build_request_url(query, start_time, end_time)
        data = self._fetch_with_retry(url, date_str, suffix)
        articles = self._process_articles(data, date_str) if data is not None else None
        return date_str, suffix, articles

    def _process_articles(self, data: Dict, scrape_date: str) -> List[Dict]:
        return [
            {
                "date": article.get("seendate"),
                "title": article.get("title"),
                "url": article.get("url"),
                "domain": article.get("domain"),
                "language": article.get("language"),
                "sourcecountry": article.get("sourcecountry"),
                "scrape_date": scrape_date,
            }
            for article in data.get("articles", [])
        ]

    def _save_batch(self, articles: List[Dict], output_file: Path, file_exists: bool) -> None:
        """Append a batch of articles to the output CSV."""
        if not articles:
            return
        df = pd.DataFrame(articles)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%SZ", errors="coerce")
        df.to_csv(output_file, mode="a", header=not file_exists, index=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_headlines(
        self,
        query: str,
        start_date: str,
        end_date: str,
        output_file: str,
        log_name: Optional[str] = None,
    ) -> Dict:
        """
        Fetch GDELT headlines for a date range, one time window at a time.

        Args:
            query:       GDELT search query string.
            start_date:  Start date in YYYY-MM-DD format.
            end_date:    End date in YYYY-MM-DD format.
            output_file: Output CSV path (relative to output_dir).
            log_name:    Log filename — auto-generated from query + timestamp if omitted.

        Returns:
            Dict with keys: total_articles, successful_windows, failed_windows,
            rate_limits_hit, output_file, log_file, window_stats.
        """
        if log_name is None:
            safe_query = "".join(
                c for c in query[:30] if c.isalnum() or c in (" ", "-", "_")
            ).strip()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"scrape_{safe_query}_{start_date}_{timestamp}.log"

        self.logger = setup_logging(self.log_dir, log_name)

        self.logger.info("=" * 80)
        self.logger.info("GDELT SCRAPING SESSION START")
        self.logger.info("=" * 80)
        self.logger.info(f"Query:           {query}")
        self.logger.info(f"Date range:      {start_date} → {end_date}")

        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

        output_path = self.output_dir / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = output_path.exists()

        self.logger.info(f"Output file:     {output_path}")
        self.logger.info("=" * 80)

        all_tasks: List[Tuple[datetime, str, str, str]] = []
        d = current_date
        while d <= end_date_dt:
            for start_t, end_t, suffix in self._get_time_windows(d):
                all_tasks.append((d, start_t, end_t, suffix))
            d += timedelta(days=1)

        self.logger.info(f"Total windows to fetch: {len(all_tasks)}")

        total_articles = 0
        failed_windows: List[str] = []
        successful_windows = 0
        window_stats: List[Dict] = []

        for d, start_t, end_t, suffix in all_tasks:
            date_str = d.strftime("%Y-%m-%d")
            date_str_res, suffix_res, articles = self._fetch_window(
                query, start_t, end_t, date_str, suffix
            )
            window_label = f"{date_str_res} {suffix_res}"

            if articles is None:
                self.logger.error(f"[{window_label}] Failed after {self.MAX_RETRIES} retries")
                failed_windows.append(window_label)
                window_stats.append({"window": window_label, "status": "FAILED", "articles": 0})

                if self.consecutive_errors >= 3:
                    extended_wait = self.SERVER_ERROR_WAIT * 2
                    self.logger.warning(
                        f"{self.consecutive_errors} consecutive errors — "
                        f"pausing for {extended_wait}s"
                    )
                    time.sleep(extended_wait)

            elif not articles:
                self.logger.info(f"[{window_label}] No articles found")
                window_stats.append({"window": window_label, "status": "SUCCESS", "articles": 0})

            else:
                self.logger.info(f"[{window_label}] {len(articles)} articles")
                self._save_batch(articles, output_path, file_exists)
                file_exists = True
                successful_windows += 1
                total_articles += len(articles)
                window_stats.append({"window": window_label, "status": "SUCCESS", "articles": len(articles)})

        # Summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("SCRAPING SESSION COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Total articles:     {total_articles:,}")
        self.logger.info(f"Successful windows: {successful_windows}")
        self.logger.info(f"Failed windows:     {len(failed_windows)}")

        if failed_windows:
            self.logger.warning(f"\nFailed windows ({len(failed_windows)}):")
            for w in failed_windows[:20]:
                self.logger.warning(f"  - {w}")
            if len(failed_windows) > 20:
                self.logger.warning(f"  ... and {len(failed_windows) - 20} more")

        if window_stats:
            success_rate = successful_windows / len(window_stats) * 100
            self.logger.info(f"\nSuccess rate: {success_rate:.1f}%")
            if successful_windows > 0:
                self.logger.info(f"Avg articles/window: {total_articles / successful_windows:.1f}")

        if self.total_rate_limits > 10:
            self.logger.warning(
                f"\n⚠️  High rate-limit count ({self.total_rate_limits}) — "
                "consider increasing RATE_LIMIT_WAIT."
            )

        if window_stats and len(failed_windows) > len(window_stats) * 0.2:
            self.logger.warning(
                f"\n⚠️  High failure rate ({len(failed_windows) / len(window_stats):.1%}) — "
                "GDELT may be experiencing issues."
            )

        self.logger.info("=" * 80)

        return {
            "total_articles": total_articles,
            "successful_windows": successful_windows,
            "failed_windows": failed_windows,
            "rate_limits_hit": self.total_rate_limits,
            "output_file": str(output_path),
            "log_file": str(self.log_dir / log_name),
            "window_stats": window_stats,
        }


# ------------------------------------------------------------------
# Company query presets
# ------------------------------------------------------------------

COMPANY_QUERIES = {
    "nvidia": (
        '(Nvidia OR NVDA OR "Jensen Huang") '
        "(stock OR earnings OR GPU OR AI OR data center OR demand OR H100 OR China OR AMD OR export) "
        "-forum -thread -discussion sourcelang:english"
    ),
    "tesla": (
        '(Tesla OR TSLA OR "Elon Musk") '
        "(stock OR earnings OR deliveries OR production OR analyst OR China OR recall OR FSD OR competition OR price) "
        "sourcelang:english"
    ),
    "meta": (
        '(Meta OR Facebook "Mark Zuckerberg") '
        "(stock OR earnings OR revenue OR users OR advertising OR AI OR metaverse OR regulation OR TikTok OR Reels) "
        "sourcelang:english"
    ),
    "boeing": (
        '(Boeing OR BA OR "Dave Calhoun") '
        "(stock OR earnings OR deliveries OR 737 OR 787 OR safety OR FAA OR incident OR defense OR Airbus OR strike) "
        "sourcelang:english"
    ),
    "intel": (
        '(Intel OR INTC OR "Pat Gelsinger") '
        "(stock OR earnings OR chip OR foundry OR AMD OR Nvidia OR TSMC OR AI OR delays OR CHIPS Act OR layoffs) "
        "sourcelang:english"
    ),
}


def main():
    scraper = GDELTScraper(output_dir="../../data/news_data", log_dir="./logs")
    company = "boeing"

    for year in range(2018, 2026):
        stats = scraper.fetch_headlines(
            query=COMPANY_QUERIES[company],
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            output_file=f"{company}/{company}_{year}.csv",
        )

        print(f"\n✅ Done!")
        print(f"   Articles:    {stats['total_articles']:,}")
        print(f"   Success rate:{stats['successful_windows'] / len(stats['window_stats']):.1%}")
        print(f"   Rate limits: {stats['rate_limits_hit']}")
        print(f"   Log:         {stats['log_file']}")

        time.sleep(30)


if __name__ == "__main__":
    main()