import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from google.cloud import bigquery
import time
from pathlib import Path

# ------------------------------------------------------------------
# 1. Configuration & Regex Presets
# ------------------------------------------------------------------
PROJECT_ID = "google-cloud-project-id"
OUTPUT_DIR = "../../data/market_news"

COMPANY_REGEX = {
    "nvidia": {"org_pattern": r"nvidia|nvda"},
    "tesla": {"org_pattern": r"tesla|tsla"},
    "meta": {"org_pattern": r"meta|facebook"},
    "boeing": {"org_pattern": r"boeing"},
    "intel": {"org_pattern": r"intel|intc"},
}


# ------------------------------------------------------------------
# 2. Asynchronous Title Scraper
# ------------------------------------------------------------------
async def fetch_title(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch a single URL asynchronously and extract its <title>."""
    try:
        # A short timeout ensures one hanging server doesn't stall the batch
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                html = await response.text()
                # Parse the HTML to find the headline
                soup = BeautifulSoup(html, "html.parser")

                # Extract the text string from the <title> tag
                if soup.title and soup.title.string:
                    return soup.title.string.strip()
    except Exception:
        # Fails gracefully on timeouts, connection errors, or anti-bot blocks
        pass

    return "Title Not Found"


async def scrape_all_titles(urls: list) -> list:
    """Manage the concurrent fetching of a list of URLs."""
    # We use a custom connector to limit concurrent connections and avoid overloading our own machine
    connector = aiohttp.TCPConnector(limit=50)

    # Customize the User-Agent so news sites are less likely to block the request
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        print(f"    -> Asynchronously scraping {len(urls)} headlines...")
        tasks = [fetch_title(session, url) for url in urls]

        # asyncio.gather runs all tasks concurrently and returns the results in the exact same order
        titles = await asyncio.gather(*tasks)
        return titles


# ------------------------------------------------------------------
# 3. BigQuery Orchestrator
# ------------------------------------------------------------------
def fetch_and_scrape_year(client: bigquery.Client, company: str, year: int):
    patterns = COMPANY_REGEX.get(company)

    # Advanced query targeting Organizations, Economic Themes, and Extreme Tone
    query = f"""
        SELECT 
            PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) as date,
            DocumentIdentifier as url,
            V2Organizations as organizations,
            V2Themes as themes,
            CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) as overall_tone
        FROM 
            `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE 
            _PARTITIONTIME BETWEEN TIMESTAMP('{year}-01-01') AND TIMESTAMP('{year}-12-31')
            AND REGEXP_CONTAINS(LOWER(V2Organizations), r'{patterns["org_pattern"]}')
            -- Target market-moving business/economic themes
            AND REGEXP_CONTAINS(LOWER(V2Themes), r'econ_|business_|finance')
            -- Filter for articles with highly positive or negative sentiment (absolute value > 3.0)
            AND ABS(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) > 3.0
    """

    print(f"[{year}] Executing BigQuery for {company.upper()}...")
    bq_start = time.time()

    try:
        df = client.query(query).to_dataframe()
        print(f"    -> Found {len(df):,} market-moving articles in {time.time() - bq_start:.1f}s")

        if not df.empty:
            # Drop duplicates in case publishers updated the same URL multiple times
            df = df.drop_duplicates(subset=['url'])

            # Execute the async scraping function
            scrape_start = time.time()
            titles = asyncio.run(scrape_all_titles(df['url'].tolist()))

            # Add the scraped titles back to our DataFrame
            df['title'] = titles
            print(f"    -> Scraped titles in {time.time() - scrape_start:.1f}s")

            # Save the final dataset
            out_dir = Path(OUTPUT_DIR) / company
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{company}_market_news_{year}.csv"

            df.to_csv(out_file, index=False)
            print(f"  ✅ Saved -> {out_file}\n")

    except Exception as e:
        print(f"  ❌ Error processing {year}: {e}\n")


# ------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------
def main():
    client = bigquery.Client(project=PROJECT_ID)
    company = "nvidia"

    print("=" * 60)
    print(f"STARTING GDELT PIPELINE: {company.upper()} (2020-2023)")
    print("=" * 60)

    for year in range(2020, 2024):
        fetch_and_scrape_year(client, company, year)


if __name__ == "__main__":
    main()