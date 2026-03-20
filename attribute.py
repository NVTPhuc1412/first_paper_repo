"""
attribute.py
------------------
Zero-shot LLM attribution of extreme anomaly events.

Runs asynchronously using the Gemini 2.5 Flash free tier. Processes requests 
in concurrent batches of 10, pausing between batches to respect the 
10 Requests Per Minute (RPM) rate limit.

Requirements:
    pip install google-genai pandas numpy
    export GEMINI_API_KEY=your_api_key_here
"""

import os
import json
import textwrap
import asyncio
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd 
from google import genai
from google.genai import types

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT        = './data'
CLEANED_NEWS_DIR = os.path.join(DATA_ROOT, 'cleaned_news')
SCORES_DIR       = os.path.join(DATA_ROOT, 'extracted_scores')

TICKERS = ['NVDA', 'TSLA', 'INTC']

TICKER_MAP = {'NVDA': 0, 'TSLA': 1, 'INTC': 2}

COMPANY_TO_TICKER = {'nvidia': 'NVDA', 'tesla': 'TSLA', 'intel': 'INTC'}
TICKER_TO_COMPANY = {v: k for k, v in COMPANY_TO_TICKER.items()}

LOOKBACK_HOURS   = 72
LOOKAHEAD_HOURS  = 24

MODEL          = 'gemini-3.1-flash-lite-preview'
TOP_N_EVENTS   = 10             # extreme anomaly events per ticker per detector
OUTPUT_FILE    = os.path.join(os.path.dirname(__file__), 'attribution_report.md')

LOOKBACK  = timedelta(hours=LOOKBACK_HOURS)
LOOKAHEAD = timedelta(hours=LOOKAHEAD_HOURS)

# Detector configuration: (display name, score column, prediction column)
@dataclass
class DetectorConfig:
    name:       str
    score_col:  str
    pred_col:   str

DETECTORS = [
    DetectorConfig('Anomaly Transformer', 'anomaly_score_AT',     'prediction_AT'),
    DetectorConfig('TranAD',              'anomaly_score_TranAD', 'prediction_TranAD'),
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_scores(ticker: str) -> pd.DataFrame:
    path = os.path.join(SCORES_DIR, f'{ticker}_scores.csv')
    df   = pd.read_csv(path, parse_dates=['Date'])
    required = [d.score_col for d in DETECTORS] + [d.pred_col for d in DETECTORS]
    return df.dropna(subset=required).reset_index(drop=True)


def load_news(ticker: str) -> pd.DataFrame:
    company = TICKER_TO_COMPANY[ticker]
    path    = os.path.join(CLEANED_NEWS_DIR, f'{company}_cleaned.csv')
    df      = pd.read_csv(path, parse_dates=['date'])
    df.columns = [c.lower() for c in df.columns]   # normalise column names
    return df.sort_values('date').reset_index(drop=True)


def get_headlines_for_event(
    news_df: pd.DataFrame,
    event_date: pd.Timestamp,
) -> list[dict]:
    """Return all headlines in [event_date - lookback, event_date + lookahead]."""
    mask   = (
        (news_df['date'] >= event_date - LOOKBACK) &
        (news_df['date'] <= event_date + LOOKAHEAD)
    )
    window = news_df.loc[mask].sort_values('date')
    return [
        {
            'date':  row['date'].strftime('%Y-%m-%d %H:%M'),
            'title': row['title'],
            'url':   row.get('url', 'N/A'),
        }
        for _, row in window.iterrows()
    ]


def top_n_events(
    score_df: pd.DataFrame,
    detector: DetectorConfig,
    n: int,
) -> pd.DataFrame:
    """Return top-N rows by detector score descending."""
    return (
        score_df
        .nlargest(n, detector.score_col)
        [['Date', detector.score_col, detector.pred_col]]
        .reset_index(drop=True)
    )


# ── Prompt ────────────────────────────────────────────────────────────────────

def build_prompt(
    ticker: str,
    detector: DetectorConfig,
    event_date: pd.Timestamp,
    score: float,
    headlines: list[dict],
) -> str:
    headlines_block = '\n'.join(
        f"[{i+1}] {h['date']} | {h['title']}\n     URL: {h['url']}"
        for i, h in enumerate(headlines)
    )
    return textwrap.dedent(f"""
        You are a financial analyst specialising in equity market microstructure
        and news-driven price anomalies.

        TASK
        ----
        The {detector.name} anomaly detection model flagged {ticker} stock on
        {event_date.date()} with an anomaly score of {score:.6f}.

        Below are all news headlines collected in the {LOOKBACK_HOURS}h before
        and {LOOKAHEAD_HOURS}h after this date. Rank them by their likely
        contribution to the anomaly detected in {ticker}'s price behaviour.

        For EACH headline provide:
          - rank        (1 = highest contribution)
          - score       (0-10, where 10 = almost certainly the direct cause)
          - direction   (POSITIVE / NEGATIVE / NEUTRAL price impact)
          - explanation (1-2 sentences: why this headline is or isn't relevant)

        Then provide a SUMMARY (3-5 sentences) synthesising what likely drove
        the anomaly based on the top headlines.

        Respond in this exact JSON format:
        {{
          "rankings": [
            {{
              "rank": 1,
              "headline_index": <1-based index from the list below>,
              "score": <0-10>,
              "direction": "POSITIVE|NEGATIVE|NEUTRAL",
              "explanation": "..."
            }}
          ],
          "summary": "..."
        }}

        HEADLINES
        ---------
        {headlines_block}
    """).strip()


# ── LLM call (Async) ──────────────────────────────────────────────────────────

async def fetch_attribution(client: genai.Client, task_data: dict) -> dict:
    """Fires a single async request to the Gemini API."""
    try:
        # Access the async client via client.aio
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=task_data['prompt'],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                temperature=0.7,
            ),
        )
        task_data['result'] = json.loads(response.text)
        task_data['error'] = None
    except Exception as e:
        task_data['result'] = None
        task_data['error'] = str(e)
        
    return task_data


# ── Report rendering ──────────────────────────────────────────────────────────

def render_event_section(
    ticker: str,
    detector: DetectorConfig,
    event_date: pd.Timestamp,
    score: float,
    headlines: list[dict],
    result: dict,
) -> str:
    lines = [
        f"#### {event_date.date()} — score: {score:.6f}",
        f"**Headlines in window**: {len(headlines)}",
        "",
        "**Summary**",
        f"> {result['summary']}",
        "",
        "**Ranked Headlines**",
        "",
        "| Rank | Score | Direction | Date | Headline | Source |",
        "|------|-------|-----------|------|----------|--------|",
    ]

    for r in sorted(result['rankings'], key=lambda x: x['rank']):
        idx = r['headline_index'] - 1
        h   = headlines[idx] if idx < len(headlines) else {'date': '?', 'title': '?', 'url': '?'}
        url_cell = f"[link]({h['url']})" if h['url'] != 'N/A' else 'N/A'
        lines.append(
            f"| {r['rank']} | {r['score']}/10 | {r['direction']} "
            f"| {h['date']} | {h['title']} | {url_cell} |"
        )

    lines += ["", "**Explanations**", ""]
    for r in sorted(result['rankings'], key=lambda x: x['rank']):
        idx = r['headline_index'] - 1
        h   = headlines[idx] if idx < len(headlines) else {'title': '?'}
        lines.append(f"**#{r['rank']}** _{h['title']}_  ")
        lines.append(f"{r['explanation']}  ")
        lines.append("")

    lines.append("")
    return '\n'.join(lines)


def render_skipped_event(
    event_date: pd.Timestamp,
    score: float,
    reason: str,
) -> str:
    return (
        f"#### {event_date.date()} — score: {score:.6f}\n"
        f"_{reason}_\n\n"
    )


# ── Main Async Workflow ───────────────────────────────────────────────────────

async def amain() -> None:
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Export it before running:  export GEMINI_API_KEY=your_key_here"
        )

    client = genai.Client(api_key=api_key)

    print("Gathering data and building prompts...")
    tasks = []
    
    # 1. Collect all events that need processing
    for ticker in TICKERS:
        score_df = load_scores(ticker)
        news_df  = load_news(ticker)

        for detector in DETECTORS:
            events = top_n_events(score_df, detector, TOP_N_EVENTS)

            for _, row in events.iterrows():
                event_date = row['Date']
                score      = float(row[detector.score_col])
                headlines  = get_headlines_for_event(news_df, event_date)

                if headlines:
                    prompt = build_prompt(ticker, detector, event_date, score, headlines)
                    tasks.append({
                        'ticker': ticker,
                        'detector': detector,
                        'event_date': event_date,
                        'score': score,
                        'headlines': headlines,
                        'prompt': prompt,
                        'result': None,
                        'error': None
                    })
                else:
                    # We will handle empty headlines later during report rendering
                    tasks.append({
                        'ticker': ticker,
                        'detector': detector,
                        'event_date': event_date,
                        'score': score,
                        'headlines': [],
                        'prompt': None,
                        'result': None,
                        'error': "No headlines found in window — skipped."
                    })

    # Filter out the tasks that actually need API calls
    api_tasks = [t for t in tasks if t['prompt'] is not None]
    
    print(f"Total events to process via API: {len(api_tasks)}")

    # 2. Process in exact batches of 10 to maximize the 10 RPM limit
    batch_size = 15
    total_batches = (len(api_tasks) + batch_size - 1) // batch_size
    
    for i in range(0, len(api_tasks), batch_size):
        batch = api_tasks[i:i+batch_size]
        current_batch = (i // batch_size) + 1
        print(f"\nProcessing batch {current_batch} of {total_batches} concurrently...")
        
        # Fire 10 requests at the exact same time
        await asyncio.gather(*(fetch_attribution(client, task) for task in batch))
        
        # If there are more batches left, sleep for 65 seconds to refresh the quota
        if i + batch_size < len(api_tasks):
            print("  -> Batch complete. Sleeping for 65 seconds to respect the 10 RPM free tier limit...")
            await asyncio.sleep(65)

    # 3. Render the report in order
    print("\nRendering report...")
    report_sections = [
        "# Anomaly Attribution Report",
        f"`{MODEL}` | top {TOP_N_EVENTS} events per detector per ticker",
        "",
    ]
    
    # Group tasks back by ticker and detector to maintain the expected file structure
    grouped = {}
    for t in tasks:
        grouped.setdefault(t['ticker'], {}).setdefault(t['detector'].name, []).append(t)
        
    for ticker in TICKERS:
        report_sections.append(f"## {ticker}\n")
        
        for detector in DETECTORS:
            report_sections.append(f"### {detector.name}\n")
            
            event_list = grouped.get(ticker, {}).get(detector.name, [])
            for task in event_list:
                if task['error']:
                    report_sections.append(render_skipped_event(task['event_date'], task['score'], task['error']))
                else:
                    section = render_event_section(
                        task['ticker'], task['detector'], task['event_date'], 
                        task['score'], task['headlines'], task['result']
                    )
                    report_sections.append(section)
            
            report_sections.append("---\n")

    report_md = '\n'.join(report_sections)
    Path(OUTPUT_FILE).write_text(report_md, encoding='utf-8')
    print(f"\nReport saved -> {OUTPUT_FILE}")


if __name__ == '__main__':
    # Run the async loop
    asyncio.run(amain())