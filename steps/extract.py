"""Step 3: Extract investment calls from tweets using LLM."""

import logging
from datetime import datetime
from collections import Counter
from rich.console import Console
from rich.progress import Progress

from config import THEME_ETF_MAP, ERROR_LOG
from db import get_unprocessed_tweets, get_actionable_tweets, insert_call

logger = logging.getLogger(__name__)
console = Console()


def validate_ticker(ticker):
    """Check if a ticker is valid using yfinance."""
    if not ticker or len(ticker) > 10:
        return False
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).fast_info
        return info.get("lastPrice") is not None or info.get("regularMarketPrice") is not None
    except Exception:
        return False


def classify_actionable(llm, tweet_text):
    """Pass 1: fast yes/no classification."""
    prompt = (
        "Is this tweet an actionable investment idea — a stock pick, sector view, or market theme?\n"
        "Reply only yes or no.\n\n"
        f"Tweet: {tweet_text}"
    )
    response = llm.classify(prompt)
    return response.startswith("yes")


def extract_call(llm, tweet_text, username, date):
    """Pass 2: structured extraction."""
    prompt = f"""Extract the investment idea from this tweet. Return JSON only. Use null for unknown fields.

Tweet: {tweet_text}
Author: @{username} on {date}

{{
  "call_type": "stock" | "sector" | "theme" | "watchlist",
  "ticker": "$TICKER or null",
  "theme": "short label e.g. AI infrastructure or null",
  "etf_proxy": "e.g. XLK or null",
  "direction": "long" | "short" | "neutral" | null,
  "conviction": "high" | "medium" | "low",
  "time_horizon": "days" | "weeks" | "months" | "years" | null,
  "thesis": "1-2 sentence summary of WHY. Null if no reasoning given.",
  "call_confidence": 0.0-1.0,
  "has_reasoning": true or false,
  "is_vague": true or false,
  "is_post_hoc": true or false
}}"""

    return llm.extract_json(prompt)


def run(db_conn, llm):
    console.print("\n[bold cyan]═══ Step 3: Extract Calls ═══[/bold cyan]\n")

    # Pass 1: Classify all unprocessed tweets
    unprocessed = get_unprocessed_tweets(db_conn)
    console.print(f"Pass 1: Classifying {len(unprocessed)} unprocessed tweets...")

    actionable_count = 0
    with Progress(console=console) as progress:
        task = progress.add_task("Classifying...", total=len(unprocessed))
        for tweet in unprocessed:
            try:
                is_actionable = classify_actionable(llm, tweet["text"])
                db_conn.execute(
                    "UPDATE tweets SET is_actionable = ?, processed = 1 WHERE tweet_id = ?",
                    (1 if is_actionable else 0, tweet["tweet_id"]),
                )
                db_conn.commit()
                if is_actionable:
                    actionable_count += 1
            except Exception as e:
                logger.debug(f"Classification failed for {tweet['tweet_id']}: {e}")
                db_conn.execute(
                    "UPDATE tweets SET processed = 1 WHERE tweet_id = ?",
                    (tweet["tweet_id"],),
                )
                db_conn.commit()
                _log_error(f"Classification failed for {tweet['tweet_id']}: {e}")
            progress.advance(task)

    pass_rate = actionable_count / max(len(unprocessed), 1) * 100
    console.print(f"  Actionable: {actionable_count}/{len(unprocessed)} ({pass_rate:.1f}%)")
    if pass_rate < 5 or pass_rate > 40:
        console.print(f"  [yellow]⚠ Pass rate {pass_rate:.1f}% outside expected 5-40% range[/yellow]")

    # Pass 2: Extract structured data from actionable tweets
    actionable = get_actionable_tweets(db_conn)
    # Filter to only those without existing calls
    existing_tweet_ids = set(
        row[0] for row in db_conn.execute("SELECT DISTINCT tweet_id FROM calls").fetchall()
    )
    to_extract = [t for t in actionable if t["tweet_id"] not in existing_tweet_ids]

    console.print(f"\nPass 2: Extracting from {len(to_extract)} actionable tweets...")

    extracted = 0
    type_counts = Counter()
    ticker_counts = Counter()

    with Progress(console=console) as progress:
        task = progress.add_task("Extracting...", total=len(to_extract))
        for tweet in to_extract:
            try:
                result = extract_call(
                    llm,
                    tweet["text"],
                    tweet["username"],
                    tweet["created_at"][:10] if tweet["created_at"] else "",
                )

                if not result:
                    progress.advance(task)
                    continue

                # Discard post-hoc
                if result.get("is_post_hoc"):
                    progress.advance(task)
                    continue

                # Discard low confidence
                confidence = result.get("call_confidence", 0)
                if isinstance(confidence, (int, float)) and confidence < 0.5:
                    progress.advance(task)
                    continue

                # Clean ticker
                ticker = result.get("ticker")
                if ticker:
                    ticker = ticker.replace("$", "").upper().strip()
                    if not validate_ticker(ticker):
                        ticker = None

                # Map theme to ETF proxy
                theme = result.get("theme", "")
                etf_proxy = result.get("etf_proxy")
                if theme and not etf_proxy:
                    theme_lower = theme.lower()
                    for key, etf in THEME_ETF_MAP.items():
                        if key in theme_lower:
                            etf_proxy = etf
                            break

                call_type = result.get("call_type", "stock")

                # Need either ticker or theme
                if not ticker and not theme:
                    progress.advance(task)
                    continue

                insert_call(
                    db_conn,
                    tweet_id=tweet["tweet_id"],
                    username=tweet["username"],
                    call_date=tweet["created_at"][:10] if tweet["created_at"] else None,
                    call_type=call_type,
                    ticker=ticker,
                    theme=theme if theme else None,
                    etf_proxy=etf_proxy,
                    direction=result.get("direction"),
                    conviction=result.get("conviction", "medium"),
                    thesis=result.get("thesis"),
                    call_confidence=confidence,
                    has_reasoning=1 if result.get("has_reasoning") else 0,
                    is_vague=1 if result.get("is_vague") else 0,
                )

                extracted += 1
                type_counts[call_type] += 1
                if ticker:
                    ticker_counts[ticker] += 1

            except Exception as e:
                logger.debug(f"Extraction failed for {tweet['tweet_id']}: {e}")
                _log_error(f"Extraction failed for {tweet['tweet_id']}: {e}")

            progress.advance(task)

    # Summary
    console.print(f"\n[bold]Extraction Summary[/bold]")
    console.print(f"  Total processed:  {len(unprocessed)}")
    console.print(f"  Calls extracted:  {extracted}")
    console.print(f"  By type:")
    for ctype, count in type_counts.most_common():
        console.print(f"    {ctype}: {count}")

    has_reasoning = db_conn.execute(
        "SELECT COUNT(*) FROM calls WHERE has_reasoning = 1"
    ).fetchone()[0]
    total_calls = db_conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    reasoning_pct = has_reasoning / max(total_calls, 1) * 100

    console.print(f"  With reasoning:   {reasoning_pct:.0f}%")
    console.print(f"\n  Top 10 tickers:")
    for ticker, count in ticker_counts.most_common(10):
        console.print(f"    {ticker}: {count}")


def _log_error(message):
    try:
        with open(ERROR_LOG, "a") as f:
            f.write(f"[{datetime.utcnow().isoformat()}] [extract] {message}\n")
    except Exception:
        pass
