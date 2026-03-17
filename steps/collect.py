"""Step 2: Collect tweets for all tracked accounts."""

import logging
from datetime import datetime, timedelta
from rich.console import Console

from config import DAYS_BACK, ERROR_LOG
from db import get_accounts, insert_tweet

logger = logging.getLogger(__name__)
console = Console()


def _get_latest_tweet_date(db_conn, username):
    """Return the latest created_at ISO string for this user, or None."""
    row = db_conn.execute(
        "SELECT MAX(created_at) as latest FROM tweets WHERE username = ?",
        (username,),
    ).fetchone()
    return row["latest"] if row and row["latest"] else None


def run(db_conn, twitter_client):
    console.print("\n[bold cyan]═══ Step 2: Collect Tweets ═══[/bold cyan]\n")

    accounts = get_accounts(db_conn)
    backfill_cutoff = (datetime.utcnow() - timedelta(days=DAYS_BACK)).isoformat()

    total_tweets = 0
    fetched_count = 0
    failed_count = 0
    skipped_count = 0
    earliest_date = None
    latest_date = None

    for account in accounts:
        username = account["username"]

        # Incremental: only fetch tweets newer than what we already have
        latest_existing = _get_latest_tweet_date(db_conn, username)
        if latest_existing:
            cutoff_str = latest_existing
            fetch_count = 100  # daily runs have few new tweets
        else:
            cutoff_str = backfill_cutoff
            fetch_count = 500  # first-time backfill

        console.print(f"  Collecting tweets for @{username}...", end=" ")

        try:
            tweets = twitter_client.get_user_tweets(username, count=fetch_count)
            if not tweets:
                console.print("[yellow]no tweets[/yellow]")
                failed_count += 1
                continue

            count = 0
            all_old = True
            for tweet in tweets:
                # Skip pure retweets (no added text)
                if tweet.get("is_retweet") and not tweet.get("is_quote"):
                    continue

                created_at = tweet.get("created_at", "")
                # Filter by date
                if created_at and created_at <= cutoff_str:
                    continue

                all_old = False
                insert_tweet(
                    db_conn,
                    tweet_id=tweet["id"],
                    username=username,
                    text=tweet["text"],
                    created_at=created_at,
                    likes=tweet.get("likes", 0),
                    retweets=tweet.get("retweets", 0),
                )
                count += 1

                # Track date range
                if created_at:
                    if earliest_date is None or created_at < earliest_date:
                        earliest_date = created_at
                    if latest_date is None or created_at > latest_date:
                        latest_date = created_at

            total_tweets += count
            fetched_count += 1
            if count == 0 and latest_existing:
                console.print("[dim]up to date[/dim]")
                skipped_count += 1
            else:
                console.print(f"[green]{count} new tweets[/green]")

        except Exception as e:
            failed_count += 1
            console.print(f"[red]error: {e}[/red]")
            _log_error(f"Failed to collect @{username}: {e}")

    # Summary
    console.print(f"\n[bold]Collection Summary[/bold]")
    console.print(f"  Accounts fetched: {fetched_count}/{len(accounts)}")
    console.print(f"  Already up to date: {skipped_count}")
    console.print(f"  Accounts failed:  {failed_count}")
    console.print(f"  New tweets:       {total_tweets}")
    if earliest_date and latest_date:
        console.print(f"  Date range:       {earliest_date[:10]} → {latest_date[:10]}")


def _log_error(message):
    try:
        with open(ERROR_LOG, "a") as f:
            f.write(f"[{datetime.utcnow().isoformat()}] [collect] {message}\n")
    except Exception:
        pass
