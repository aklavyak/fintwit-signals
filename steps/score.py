"""Step 4: Score calls against price performance and compute picker scores."""

import logging
from datetime import datetime, timedelta, date
from rich.console import Console
from rich.table import Table
import yfinance as yf
from scipy.stats import binomtest

from config import SCORE_WINDOWS, BAYESIAN_PRIOR, ERROR_LOG
from db import get_unscored_calls, get_accounts, get_scored_calls_for_user, update_call_scores, update_picker_scores

logger = logging.getLogger(__name__)
console = Console()

# Cache downloaded price data to avoid redundant API calls
_price_cache = {}


def _get_prices(ticker, start_date=None):
    """Download and cache price data for a ticker. Always fetches full history."""
    cache_key = ticker
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    try:
        # Always download from 120 days ago to today to cover all scoring windows
        dl_start = date.today() - timedelta(days=120)
        data = yf.download(
            ticker, start=dl_start.strftime("%Y-%m-%d"),
            end=date.today().strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True
        )
        if data.empty:
            _price_cache[cache_key] = None
            return None

        # Handle MultiIndex columns from yfinance
        close = data["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]

        _price_cache[cache_key] = close
        return close
    except Exception as e:
        logger.debug(f"Failed to download {ticker}: {e}")
        _price_cache[cache_key] = None
        return None


def score_call(ticker, call_date_str):
    """Score a single call: excess return over SPY at 30/60/90 days."""
    try:
        call_date = datetime.strptime(call_date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None

    start = call_date - timedelta(days=2)
    stock = _get_prices(ticker, start)
    spy = _get_prices("SPY", start)

    if stock is None or spy is None or len(stock) < 2 or len(spy) < 2:
        return None

    results = {}
    for days in SCORE_WINDOWS:
        target = call_date + timedelta(days=days)
        key = f"excess_{days}d"

        if target > date.today():
            results[key] = None
            continue

        try:
            stock_at_call = stock.asof(str(call_date))
            stock_at_target = stock.asof(str(target))
            spy_at_call = spy.asof(str(call_date))
            spy_at_target = spy.asof(str(target))

            if any(v != v for v in [stock_at_call, stock_at_target, spy_at_call, spy_at_target]):
                # NaN check
                results[key] = None
                continue

            stock_ret = (stock_at_target / stock_at_call) - 1
            spy_ret = (spy_at_target / spy_at_call) - 1
            results[key] = round(float(stock_ret - spy_ret), 4)
        except Exception:
            results[key] = None

    available = [v for v in results.values() if v is not None]
    results["composite"] = round(sum(available) / len(available), 4) if available else None

    return results


def _norm(x, lo, hi):
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def calculate_picker_score(username, db_conn):
    """Calculate composite picker score for an account."""
    rows = get_scored_calls_for_user(db_conn, username)
    n = len(rows)
    if n < 5:
        return {"score": None, "tier": "insufficient_data", "n_calls": n}

    values = [row["composite_score"] for row in rows]
    wins = sum(1 for v in values if v > 0)
    mean_excess = sum(values) / n

    prior = BAYESIAN_PRIOR
    bayes_wr = (wins + prior) / (n + prior * 2)

    p_value = binomtest(wins, n, 0.5, alternative="greater").pvalue

    score = (
        0.40 * _norm(bayes_wr, 0.30, 0.75)
        + 0.35 * _norm(mean_excess, -0.10, 0.15)
        + 0.25 * _norm(1 - p_value, 0, 1)
    )

    if n < 10:
        tier = "preliminary"
    elif n < 30:
        tier = "established"
    else:
        tier = "proven"

    return {
        "score": round(score, 3),
        "tier": tier,
        "n_calls": n,
        "win_rate": round(wins / n, 3),
        "win_rate_bayesian": round(bayes_wr, 3),
        "mean_excess_return": round(mean_excess, 4),
        "p_value": round(p_value, 4),
    }


def run(db_conn):
    console.print("\n[bold cyan]═══ Step 4: Score Calls ═══[/bold cyan]\n")

    # 4a: Score individual calls
    unscored = get_unscored_calls(db_conn)
    console.print(f"Scoring {len(unscored)} unscored calls...")

    scored_count = 0
    failed_count = 0

    for call in unscored:
        ticker = call["ticker"] or call["etf_proxy"]
        if not ticker or ticker.lower() == "null":
            failed_count += 1
            continue
        # Clean ticker: strip $, whitespace
        ticker = ticker.strip().lstrip("$")

        scores = score_call(ticker, call["call_date"])
        if scores and scores.get("composite") is not None:
            update_call_scores(db_conn, call["id"], scores)
            scored_count += 1
        else:
            failed_count += 1

    console.print(f"  Scored: {scored_count}, Unscoreable: {failed_count}")

    # 4b: Picker scores
    console.print("\nCalculating picker scores...")
    accounts = get_accounts(db_conn)

    for account in accounts:
        stats = calculate_picker_score(account["username"], db_conn)
        update_picker_scores(db_conn, account["username"], stats)

    # Print leaderboard
    ranked = db_conn.execute(
        "SELECT * FROM accounts WHERE picker_score IS NOT NULL ORDER BY picker_score DESC"
    ).fetchall()

    if ranked:
        table = Table(title="Source Track Records")
        table.add_column("Rank", justify="right", style="dim")
        table.add_column("Account", style="cyan")
        table.add_column("Track Record", justify="right", style="green")
        table.add_column("Tier")
        table.add_column("Calls", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Avg Excess", justify="right")
        table.add_column("p-value", justify="right")

        for i, row in enumerate(ranked, 1):
            wr = f"{row['win_rate']*100:.0f}%" if row["win_rate"] else "-"
            excess = f"{row['mean_excess_return']*100:.1f}%" if row["mean_excess_return"] else "-"
            pval = f"{row['p_value']:.3f}" if row["p_value"] else "-"
            table.add_row(
                str(i),
                f"@{row['username']}",
                f"{row['picker_score']:.3f}",
                row["tier"] or "",
                str(row["n_calls"]),
                wr,
                excess,
                pval,
            )

        console.print(table)
    else:
        console.print("  [yellow]No accounts with enough data for picker scores yet[/yellow]")


def _log_error(message):
    try:
        with open(ERROR_LOG, "a") as f:
            f.write(f"[{datetime.utcnow().isoformat()}] [score] {message}\n")
    except Exception:
        pass
