#!/usr/bin/env python3
"""Fintwit Signal Intelligence — Idea Generation Pipeline."""

import argparse
import os
import sys
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from config import (
    RAPIDAPI_KEY, OPENAI_API_KEY,
    CHECKPOINTS_DIR, OUTPUT_DIR, DATA_DIR,
    MIN_ACCOUNTS_TO_PROCEED, MIN_CALLS_TO_PROCEED,
    DISCOVERY_INTERVAL_DAYS,
)
from db import init_db, get_db
from twitter_client import TwitterClient
from llm_client import LLMClient
from steps import discover, collect, extract, score

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ── Checkpointing ──────────────────────────────────────────────────────────

def should_run(step_name, max_age_days=None):
    """Decide whether a step should run.

    max_age_days=None  → always run (for daily steps).
    max_age_days=30    → run only if checkpoint is missing or older than 30 days.
    """
    path = Path(CHECKPOINTS_DIR, f"{step_name}.done")
    if not path.exists():
        return True
    if max_age_days is None:
        return True
    try:
        last_run = datetime.fromisoformat(path.read_text().strip())
        return (datetime.utcnow() - last_run).days >= max_age_days
    except (ValueError, OSError):
        return True


def write_checkpoint(step_name):
    Path(CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINTS_DIR, f"{step_name}.done").write_text(
        datetime.utcnow().isoformat()
    )


# ── Startup checks ────────────────────────────────────────────────────────

def check_prerequisites():
    errors = []
    if not RAPIDAPI_KEY:
        errors.append("RAPIDAPI_KEY not set. Add it to .env file.")

    if not OPENAI_API_KEY:
        errors.append("OPENAI_KEY not set. Add it to .env file.")

    if errors:
        console.print("[bold red]Startup check failed:[/bold red]")
        for e in errors:
            console.print(f"  [red]• {e}[/red]")
        sys.exit(1)


# ── Output generation ─────────────────────────────────────────────────────

def generate_leaderboard(db_conn):
    rows = db_conn.execute(
        "SELECT * FROM accounts ORDER BY picker_score DESC NULLS LAST"
    ).fetchall()

    path = os.path.join(OUTPUT_DIR, "leaderboard.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "username", "is_seed", "picker_score", "tier",
            "n_calls", "win_rate", "win_rate_bayesian",
            "mean_excess_return_pct", "p_value",
        ])
        for i, row in enumerate(rows, 1):
            writer.writerow([
                i,
                row["username"],
                row["is_seed"],
                row["picker_score"] if row["picker_score"] is not None else "",
                row["tier"] or "",
                row["n_calls"],
                row["win_rate"] if row["win_rate"] is not None else "",
                row["win_rate_bayesian"] if row["win_rate_bayesian"] is not None else "",
                f"{row['mean_excess_return']*100:.2f}" if row["mean_excess_return"] is not None else "",
                row["p_value"] if row["p_value"] is not None else "",
            ])
    return path


def generate_call_log(db_conn):
    rows = db_conn.execute("""
        SELECT c.*, a.picker_score FROM calls c
        LEFT JOIN accounts a ON c.username = a.username
        ORDER BY c.call_date DESC
    """).fetchall()

    path = os.path.join(OUTPUT_DIR, "call_log.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "call_date", "username", "picker_score", "call_type", "ticker",
            "theme", "direction", "conviction", "thesis",
            "excess_30d_pct", "excess_60d_pct", "excess_90d_pct", "composite_score",
        ])
        for row in rows:
            writer.writerow([
                row["call_date"] or "",
                row["username"],
                row["picker_score"] if row["picker_score"] is not None else "",
                row["call_type"] or "",
                row["ticker"] or "",
                row["theme"] or "",
                row["direction"] or "",
                row["conviction"] or "",
                row["thesis"] or "",
                f"{row['excess_30d']*100:.2f}" if row["excess_30d"] is not None else "",
                f"{row['excess_60d']*100:.2f}" if row["excess_60d"] is not None else "",
                f"{row['excess_90d']*100:.2f}" if row["excess_90d"] is not None else "",
                row["composite_score"] if row["composite_score"] is not None else "",
            ])
    return path


def generate_daily_briefing(db_conn):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    cutoff_14d = (datetime.utcnow() - timedelta(days=14)).strftime("%Y-%m-%d")

    n_accounts = db_conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    n_calls = db_conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    n_scored = db_conn.execute(
        "SELECT COUNT(*) FROM calls WHERE composite_score IS NOT NULL"
    ).fetchone()[0]

    lines = [
        f"# Fintwit Idea Briefing — {today}",
        f"{n_accounts} accounts tracked | {n_calls} calls extracted | {n_scored} scored vs SPY",
        "",
        "Research ideas surfaced from high-signal financial Twitter accounts.",
        "Track records measure each source's historical accuracy (excess return vs SPY over 30/60/90 days).",
        "Use this as a starting point for your own due diligence — not as trade recommendations.",
        "",
    ]

    # High Conviction: 2+ accounts with track record > 0.5 mentioning in last 14 days
    lines.append("## High Conviction Ideas")
    lines.append("*2+ credible sources (track record > 0.5) mentioning in last 14 days*")
    lines.append("")

    high_conv = db_conn.execute("""
        SELECT c.ticker, c.theme, c.direction,
               GROUP_CONCAT(DISTINCT c.username) as accounts,
               COUNT(DISTINCT c.username) as n_accounts,
               AVG(a.picker_score) as avg_score,
               GROUP_CONCAT(c.thesis, ' | ') as theses
        FROM calls c
        JOIN accounts a ON c.username = a.username
        WHERE c.call_date >= ? AND a.picker_score > 0.5
        GROUP BY COALESCE(c.ticker, c.theme)
        HAVING COUNT(DISTINCT c.username) >= 2
        ORDER BY avg_score DESC
    """, (cutoff_14d,)).fetchall()

    if high_conv:
        lines.append("| Ticker/Theme | Sources | Avg Track Record | Direction | Thesis |")
        lines.append("|---|---|---|---|---|")
        for row in high_conv:
            label = row["ticker"] or row["theme"] or "?"
            accounts = ", ".join(f"@{a}" for a in (row["accounts"] or "").split(",")[:3])
            thesis = (row["theses"] or "")[:80]
            lines.append(
                f"| {label} | {accounts} | {row['avg_score']:.2f} | {row['direction'] or '-'} | {thesis} |"
            )
    else:
        lines.append("*No high-conviction ideas found in the last 14 days.*")
    lines.append("")

    # Sector Themes
    lines.append("## Sector Themes Gaining Traction")
    themes = db_conn.execute("""
        SELECT c.theme, c.etf_proxy, COUNT(DISTINCT c.username) as n_accounts,
               GROUP_CONCAT(DISTINCT c.username) as accounts
        FROM calls c
        WHERE c.theme IS NOT NULL AND c.theme != 'null' AND c.call_date >= ?
        GROUP BY c.theme
        HAVING COUNT(DISTINCT c.username) >= 2
        ORDER BY n_accounts DESC
        LIMIT 5
    """, (cutoff_14d,)).fetchall()

    if themes:
        for row in themes:
            accts = ", ".join(f"@{a}" for a in (row["accounts"] or "").split(",")[:3])
            etf = row["etf_proxy"] if row["etf_proxy"] and row["etf_proxy"] != "null" else "—"
            lines.append(f"**{row['theme']}** — {row['n_accounts']} accounts | ETF: {etf}")
            lines.append(f"Key accounts: {accts}")
            lines.append("")
    else:
        lines.append("*No sector themes with multiple mentions found.*")
    lines.append("")

    # Watchlist: single mention from credible source
    lines.append("## Watchlist — Worth Researching")
    lines.append("*Single mention from a source with strong track record*")
    lines.append("")

    watchlist = db_conn.execute("""
        SELECT c.ticker, c.username, a.picker_score, c.thesis
        FROM calls c
        JOIN accounts a ON c.username = a.username
        WHERE c.ticker IS NOT NULL AND c.call_date >= ?
              AND a.picker_score > 0.4
        GROUP BY c.ticker
        HAVING COUNT(DISTINCT c.username) = 1
        ORDER BY a.picker_score DESC
        LIMIT 10
    """, (cutoff_14d,)).fetchall()

    if watchlist:
        lines.append("| Ticker | Source | Track Record | Thesis |")
        lines.append("|---|---|---|---|")
        for row in watchlist:
            thesis = (row["thesis"] or "")[:60]
            lines.append(
                f"| {row['ticker']} | @{row['username']} | {row['picker_score']:.2f} | {thesis} |"
            )
    else:
        lines.append("*No watchlist items found.*")
    lines.append("")

    # Source Track Records
    lines.append("## Source Track Records")
    ranked = db_conn.execute(
        "SELECT * FROM accounts WHERE picker_score IS NOT NULL ORDER BY picker_score DESC LIMIT 10"
    ).fetchall()

    if ranked:
        lines.append("| Rank | Source | Track Record | Tier | Calls | Win Rate | Avg Excess |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, row in enumerate(ranked, 1):
            wr = f"{row['win_rate']*100:.0f}%" if row["win_rate"] is not None else "-"
            excess = f"{row['mean_excess_return']*100:.1f}%" if row["mean_excess_return"] is not None else "-"
            lines.append(
                f"| {i} | @{row['username']} | {row['picker_score']:.3f} | {row['tier'] or '-'} "
                f"| {row['n_calls']} | {wr} | {excess} |"
            )
    lines.append("")

    # Data Notes
    lines.append("## Data Notes")
    low_data = db_conn.execute(
        "SELECT username FROM accounts WHERE n_calls > 0 AND n_calls < 10"
    ).fetchall()
    low_list = ", ".join(f"@{r['username']}" for r in low_data[:10])
    lines.append(f"- Accounts with <10 calls: {low_list or 'none'}")

    too_recent = db_conn.execute(
        "SELECT COUNT(*) FROM calls WHERE scored_at IS NULL AND call_date IS NOT NULL"
    ).fetchone()[0]
    lines.append(f"- Calls too recent to score: {too_recent}")

    unscoreable = db_conn.execute(
        "SELECT COUNT(*) FROM calls WHERE ticker IS NULL AND etf_proxy IS NULL"
    ).fetchone()[0]
    lines.append(f"- Calls unscoreable: {unscoreable}")
    lines.append("")
    lines.append("*This briefing surfaces ideas for further research — it is not investment advice. All ideas require independent due diligence before any action.*")

    path = os.path.join(OUTPUT_DIR, "daily_briefing.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ── Success criteria ──────────────────────────────────────────────────────

def check_success_criteria(db_conn):
    criteria = []

    n_accounts = db_conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    criteria.append((">=20 accounts tracked", n_accounts >= 20, n_accounts, "Step 1 (discover)"))

    n_calls = db_conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    criteria.append((">=200 calls extracted", n_calls >= 200, n_calls, "Step 3 (extract)"))

    n_scored_30d = db_conn.execute(
        "SELECT COUNT(*) FROM calls WHERE excess_30d IS NOT NULL"
    ).fetchone()[0]
    criteria.append((">=100 calls scored at 30d", n_scored_30d >= 100, n_scored_30d, "Step 4 (score)"))

    n_pickers = db_conn.execute(
        "SELECT COUNT(*) FROM accounts WHERE picker_score IS NOT NULL"
    ).fetchone()[0]
    criteria.append((">=5 accounts with picker_score", n_pickers >= 5, n_pickers, "Step 4 (score)"))

    outputs_exist = all(
        os.path.exists(os.path.join(OUTPUT_DIR, f)) and os.path.getsize(os.path.join(OUTPUT_DIR, f)) > 0
        for f in ["leaderboard.csv", "call_log.csv", "daily_briefing.md"]
    )
    criteria.append(("All 3 output files exist", outputs_exist, outputs_exist, "Step 5 (outputs)"))

    all_pass = True
    for desc, passed, value, step in criteria:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        if not passed:
            all_pass = False
            console.print(f"  {status} {desc} (got {value}) — check {step}")
        else:
            console.print(f"  {status} {desc} ({value})")

    return all_pass


# ── Terminal summary ──────────────────────────────────────────────────────

def print_summary(db_conn):
    n_accounts = db_conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    n_tweets = db_conn.execute("SELECT COUNT(*) FROM tweets").fetchone()[0]
    n_calls = db_conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    n_scored = db_conn.execute(
        "SELECT COUNT(*) FROM calls WHERE excess_30d IS NOT NULL"
    ).fetchone()[0]

    avg_excess = db_conn.execute(
        "SELECT AVG(composite_score) FROM calls WHERE composite_score IS NOT NULL"
    ).fetchone()[0]
    avg_pct = f"{avg_excess*100:.2f}" if avg_excess is not None else "N/A"

    top3 = db_conn.execute(
        "SELECT username, picker_score, win_rate, n_calls, tier "
        "FROM accounts WHERE picker_score IS NOT NULL "
        "ORDER BY picker_score DESC LIMIT 3"
    ).fetchall()

    summary_lines = [
        f"  Accounts tracked:      {n_accounts}",
        f"  Tweets collected:      {n_tweets}",
        f"  Calls extracted:       {n_calls}",
        f"  Calls scored (30d+):   {n_scored}",
        f"  Network avg vs SPY:    {avg_pct}%",
        "",
        "  TOP SOURCES",
    ]
    for i, row in enumerate(top3, 1):
        wr = f"{row['win_rate']*100:.0f}" if row["win_rate"] is not None else "?"
        summary_lines.append(
            f"  {i}. @{row['username']}  {row['picker_score']:.3f}  "
            f"{wr}%wr  n={row['n_calls']}  {row['tier'] or ''}"
        )

    summary_lines.extend([
        "",
        "  output/leaderboard.csv",
        "  output/call_log.csv",
        "  output/daily_briefing.md",
    ])

    console.print()
    console.print(Panel(
        "\n".join(summary_lines),
        title="[bold]FINTWIT IDEA BRIEFING COMPLETE[/bold]",
        border_style="green",
        padding=(1, 2),
    ))


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    global console

    parser = argparse.ArgumentParser(description="Fintwit Idea Generation Pipeline")
    parser.add_argument("--daily", action="store_true", help="Automated daily run (sends email)")
    parser.add_argument("--discover", action="store_true", help="Force account discovery even if not due")
    args = parser.parse_args()

    # Unattended mode: plain console + file logging
    if args.daily:
        console = Console(force_terminal=False, no_color=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            filename=os.path.join(DATA_DIR, "pipeline.log"),
            force=True,
        )

    console.print("[bold]Fintwit Idea Briefing Pipeline[/bold]\n")

    # Startup checks
    check_prerequisites()

    # Ensure directories exist
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Init
    init_db()
    db_conn = get_db()
    twitter = TwitterClient()
    llm = LLMClient()

    # Step 1: Discover — monthly (or forced with --discover)
    discover_age = 0 if args.discover else DISCOVERY_INTERVAL_DAYS
    if should_run("1_discover", max_age_days=discover_age):
        discover.run(db_conn, twitter, llm)
        write_checkpoint("1_discover")
    else:
        console.print("[dim][Step 1] Discovery ran within 30 days — skipping[/dim]")

    # Step 2: Collect — always (incremental)
    collect.run(db_conn, twitter)
    write_checkpoint("2_collect")

    # Step 3: Extract — always (processes unprocessed tweets)
    extract.run(db_conn, llm)
    write_checkpoint("3_extract")

    # Step 4: Score — always (scores unscored calls)
    score.run(db_conn)
    write_checkpoint("4_score")

    # Step 5: Generate outputs
    console.print("\n[bold cyan]═══ Step 5: Generate Outputs ═══[/bold cyan]\n")
    generate_leaderboard(db_conn)
    generate_call_log(db_conn)
    briefing_path = generate_daily_briefing(db_conn)
    console.print("  [green]Generated all output files[/green]")

    # Success criteria
    console.print("\n[bold]Success Criteria:[/bold]")
    check_success_criteria(db_conn)

    # Final summary
    print_summary(db_conn)

    # Email briefing if running in daily mode
    if args.daily:
        from email_briefing import send_briefing
        send_briefing(briefing_path)

    db_conn.close()


if __name__ == "__main__":
    main()
