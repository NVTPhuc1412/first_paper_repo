"""
xai_analysis/attribute.py
-------------------------
LLM-powered anomaly attribution using Gemini.
Consumes headlines from news_fetcher.py.
"""

import os
import json
import textwrap
import asyncio
import logging
from pathlib import Path
from datetime import timedelta

import pandas as pd

logger = logging.getLogger(__name__)


# ── Prompt Construction ───────────────────────────────────────────────────────

def build_attribution_prompt(
    ticker: str,
    detector_name: str,
    event_date: str,
    score: float,
    headlines: list[dict],
    lookback_hours: int = 72,
    lookahead_hours: int = 24,
) -> str:
    """Construct the Gemini attribution prompt.

    Args:
        ticker:        Stock ticker symbol.
        detector_name: Model name (e.g. 'Anomaly Transformer').
        event_date:    Event date string.
        score:         Anomaly score.
        headlines:     List of headline dicts with date, title, url.
        lookback_hours: Window lookback in hours.
        lookahead_hours: Window lookahead in hours.

    Returns:
        Prompt string.
    """
    headlines_block = '\n'.join(
        f"[{i+1}] {h.get('date', '?')} | {h.get('title', '?')}\n     URL: {h.get('url', 'N/A')}"
        for i, h in enumerate(headlines)
    )

    return textwrap.dedent(f"""
        You are a financial analyst specialising in equity market microstructure
        and news-driven price anomalies.

        TASK
        ----
        The {detector_name} anomaly detection model flagged {ticker} stock on
        {event_date} with an anomaly score of {score:.6f}.

        Below are all news headlines collected in the {lookback_hours}h before
        and {lookahead_hours}h after this date. Rank them by their likely
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


# ── LLM Call ──────────────────────────────────────────────────────────────────

async def _fetch_attribution_async(client, model: str, prompt: str) -> dict | None:
    """Send a single attribution request to Gemini (async)."""
    from google.genai import types

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                temperature=0.7,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error("LLM attribution failed: %s", e)
        return None


def _fetch_attribution_sync(client, model: str, prompt: str) -> dict | None:
    """Send a single attribution request to Gemini (sync)."""
    from google.genai import types

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                temperature=0.7,
            ),
        )
        if response.text is None:
            logger.warning("LLM returned empty response (blocked or no content)")
            return None
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        logger.error("LLM response not valid JSON: %s — raw: %s",
                      e, getattr(response, 'text', '')[:200])
        return None
    except Exception as e:
        logger.error("LLM attribution failed: %s", e)
        return None


# ── Report Rendering ──────────────────────────────────────────────────────────

def _render_event_section(
    ticker: str, detector_name: str, event_date: str,
    score: float, headlines: list[dict], result: dict,
) -> str:
    """Render one event's attribution as markdown."""
    lines = [
        f"#### {event_date} — score: {score:.6f}",
        f"**Headlines in window**: {len(headlines)}",
        "",
        "**Summary**",
        f"> {result.get('summary', 'No summary available.')}",
        "",
        "**Ranked Headlines**",
        "",
        "| Rank | Score | Direction | Date | Headline |",
        "|------|-------|-----------|------|----------|",
    ]

    for r in sorted(result.get('rankings', []), key=lambda x: x.get('rank', 99)):
        idx = r.get('headline_index', 1) - 1
        h = headlines[idx] if 0 <= idx < len(headlines) else {'date': '?', 'title': '?'}
        lines.append(
            f"| {r.get('rank', '?')} | {r.get('score', '?')}/10 | {r.get('direction', '?')} "
            f"| {h.get('date', '?')} | {h.get('title', '?')} |"
        )

    lines += ["", "**Explanations**", ""]
    for r in sorted(result.get('rankings', []), key=lambda x: x.get('rank', 99)):
        idx = r.get('headline_index', 1) - 1
        h = headlines[idx] if 0 <= idx < len(headlines) else {'title': '?'}
        lines.append(f"**#{r.get('rank', '?')}** _{h.get('title', '?')}_  ")
        lines.append(f"{r.get('explanation', '')}  ")
        lines.append("")

    return '\n'.join(lines)


# ── Main Attribution Pipeline ─────────────────────────────────────────────────

def run_attribution_pipeline(
    ticker: str,
    event_results: dict,
    headlines_by_date: dict[str, list[dict]],
    out_dir: str | Path,
    models: list[dict],
    llm_model: str = "gemini-3.1-flash-lite",
) -> str | None:
    """Run LLM attribution for all anomaly events.

    Args:
        ticker:            Ticker symbol.
        event_results:     Dict from run_xai_for_ticker: {label: [event_dicts]}.
        headlines_by_date: Dict from fetch_all_event_headlines: {date: [headlines]}.
        out_dir:           Directory for the attribution report.
        models:            Model config list.
        llm_model:         Gemini model name.

    Returns:
        Path to the generated report, or None on failure.
    """
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        logger.warning(
            "GEMINI_API_KEY not set — skipping LLM attribution. "
            "Set it with: set GEMINI_API_KEY=your_key_here"
        )
        return None

    from google import genai
    client = genai.Client(api_key=api_key)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name_map = {m['label']: m.get('name', m['label']) for m in models}

    report_lines = [
        "# Anomaly Attribution Report",
        f"**Ticker**: {ticker} | **Model**: `{llm_model}`",
        "",
        "---",
        "",
    ]

    # Structured data for JSON cache (dashboard reads this)
    attr_data = {}  # key: "{label}_{date}" -> {top_headline, summary, ...}

    for label, events in event_results.items():
        detector_name = model_name_map.get(label, label)
        report_lines.append(f"## {detector_name} (`{label}`)")
        report_lines.append("")

        for ev in events:
            event_date = ev['date']
            score_val = ev['score']
            headlines = headlines_by_date.get(event_date, [])

            if not headlines:
                report_lines.append(f"#### {event_date} — score: {score_val:.6f}")
                report_lines.append("_No headlines found in window — skipped._\n")
                continue

            print(f"    Attribution: {event_date} ({len(headlines)} headlines)...")
            prompt = build_attribution_prompt(
                ticker, detector_name, event_date, score_val, headlines
            )

            result = _fetch_attribution_sync(client, llm_model, prompt)
            if result:
                section = _render_event_section(
                    ticker, detector_name, event_date, score_val, headlines, result
                )
                report_lines.append(section)

                # Extract #1 ranked headline for JSON cache
                rankings = sorted(result.get('rankings', []),
                                   key=lambda x: x.get('rank', 99))
                top_hl = ''
                top_dir = ''
                if rankings:
                    idx = rankings[0].get('headline_index', 1) - 1
                    if 0 <= idx < len(headlines):
                        top_hl = headlines[idx].get('title', '')
                    top_dir = rankings[0].get('direction', '')

                attr_data[f"{label}_{event_date}"] = {
                    'label': label,
                    'date': event_date,
                    'score': score_val,
                    'top_headline': top_hl,
                    'top_direction': top_dir,
                    'summary': result.get('summary', ''),
                }
            else:
                report_lines.append(f"#### {event_date} — score: {score_val:.6f}")
                report_lines.append("_LLM attribution failed._\n")

            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")

    report_path = out_dir / f"{ticker}_attribution_report.md"
    report_path.write_text('\n'.join(report_lines), encoding='utf-8')
    print(f"  Attribution report saved → {report_path}")

    # Save structured JSON cache
    json_path = out_dir / f"{ticker}_attribution_data.json"
    json_path.write_text(json.dumps(attr_data, indent=2, ensure_ascii=False),
                          encoding='utf-8')
    print(f"  Attribution data cache  → {json_path}")

    return str(report_path)
