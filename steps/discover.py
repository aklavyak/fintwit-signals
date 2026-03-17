"""Step 1: Discover high-signal accounts from seed tweet history."""

import logging
from collections import defaultdict
from datetime import datetime
from rich.console import Console
from rich.table import Table

from config import CANDIDATE_SCORE_CUTOFF, MIN_ACCOUNTS_TO_PROCEED, SEEDS_PATH, ERROR_LOG
from db import upsert_account

logger = logging.getLogger(__name__)
console = Console()


def load_seeds():
    seeds = []
    with open(SEEDS_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                seeds.append(line.lstrip("@"))
    return seeds


def is_spam(profile):
    if not profile:
        return True
    followers = profile.get("followers", 0)
    following = profile.get("following", 1)
    if followers / max(following, 1) < 0.05:
        return True
    if profile.get("account_age_days", 365) < 180:
        return True
    return False


def extract_candidates_from_tweets(tweets):
    """Extract candidate usernames and their signal type from a seed's tweets."""
    candidates = defaultdict(lambda: {"rt": 0, "mention": 0, "qt": 0, "mention_tweets": []})

    for tweet in tweets:
        # Retweet authors
        if tweet.get("is_retweet") and tweet.get("retweeted_user"):
            user = tweet["retweeted_user"].lower()
            candidates[user]["rt"] += 1

        # Quote tweet authors
        if tweet.get("is_quote") and tweet.get("quoted_user"):
            user = tweet["quoted_user"].lower()
            candidates[user]["qt"] += 1

        # Mentioned users (need LLM classification later)
        for mentioned in tweet.get("mentioned_users", []):
            user = mentioned.lower()
            # Skip self-mentions and RT-style mentions
            if not tweet.get("is_retweet"):
                candidates[user]["mention_tweets"].append(tweet["text"])

    return candidates


def classify_positive_mention(llm, tweet_text, mentioned_user):
    """Use LLM to classify if a mention is positive."""
    prompt = (
        f"Does this tweet express a positive view of @{mentioned_user}? "
        f"Reply only yes or no.\n\nTweet: {tweet_text}"
    )
    response = llm.classify(prompt)
    return response.startswith("yes")


def evaluate_account_quality(llm, username, sample_tweets):
    """LLM quality check on sample tweets. Returns quality dict or None."""
    tweet_text = "\n\n".join(
        f"Tweet {i+1}: {t['text']}" for i, t in enumerate(sample_tweets[:5])
    )

    prompt = f"""You are evaluating a Twitter/X account as a potential stock-picking signal source.

5 recent tweets from @{username}:
{tweet_text}

Score each dimension 1-5:
- analytical_quality: Do they explain WHY they like a stock?
- specificity: Do they name tickers, price targets, or catalysts?
- originality: Is this their own thinking or retweeting others?

Return JSON only:
{{
  "analytical_quality": 1-5,
  "specificity": 1-5,
  "originality": 1-5,
  "overall": 1-5,
  "keep": true or false,
  "reason": "one sentence"
}}"""

    result = llm.extract_json(prompt)
    if not result or "overall" not in result:
        return None
    return result


def run(db_conn, twitter_client, llm):
    console.print("\n[bold cyan]═══ Step 1: Discover Accounts ═══[/bold cyan]\n")

    # Load and register seeds
    seeds = load_seeds()
    console.print(f"Loaded {len(seeds)} seed accounts from seeds.txt")

    for seed in seeds:
        upsert_account(db_conn, seed, is_seed=1)

    # Gather candidates from seed tweet histories
    all_candidates = defaultdict(lambda: {"rt_seeds": 0, "mention_seeds": 0, "qt_seeds": 0})

    for seed in seeds:
        console.print(f"  Fetching tweets for seed: @{seed}...")
        tweets = twitter_client.get_user_tweets(seed, count=500)
        if not tweets:
            console.print(f"    [yellow]No tweets fetched for @{seed}[/yellow]")
            continue

        console.print(f"    Got {len(tweets)} tweets")
        candidates = extract_candidates_from_tweets(tweets)

        for username, signals in candidates.items():
            if username in [s.lower() for s in seeds]:
                continue  # skip other seeds

            if signals["rt"] > 0:
                all_candidates[username]["rt_seeds"] += 1
            if signals["qt"] > 0:
                all_candidates[username]["qt_seeds"] += 1

            # Classify mentions (limit to 3 per candidate per seed to control LLM calls)
            if signals["mention_tweets"]:
                for tweet_text in signals["mention_tweets"][:3]:
                    try:
                        if classify_positive_mention(llm, tweet_text, username):
                            all_candidates[username]["mention_seeds"] += 1
                            break  # one positive is enough per seed
                    except Exception as e:
                        logger.debug(f"Mention classification failed: {e}")

    console.print(f"\nFound {len(all_candidates)} unique candidates from seed networks")

    # Score, filter, and evaluate candidates
    scored = []
    for username, signals in all_candidates.items():
        candidate_score = (
            signals["rt_seeds"] * 3
            + signals["mention_seeds"] * 2
            + signals["qt_seeds"] * 2
        )
        if candidate_score == 0:
            continue

        # Spam check via profile
        profile = twitter_client.get_user_profile(username)
        if is_spam(profile):
            continue

        # LLM quality evaluation on sample tweets
        sample_tweets = twitter_client.get_user_tweets(username, count=10)
        if not sample_tweets or len(sample_tweets) < 3:
            continue

        try:
            quality = evaluate_account_quality(llm, username, sample_tweets[:5])
        except Exception as e:
            logger.warning(f"Quality eval failed for @{username}: {e}")
            _log_error(f"Quality eval failed for @{username}: {e}")
            quality = None

        llm_score = quality.get("overall", 3) if quality else 3
        final_score = candidate_score * llm_score

        if final_score >= CANDIDATE_SCORE_CUTOFF:
            scored.append({
                "username": username,
                "candidate_score": candidate_score,
                "llm_quality_score": llm_score,
                "final_score": final_score,
                "quality": quality,
            })

    # Sort by final score
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    # Save to DB
    for item in scored:
        upsert_account(
            db_conn,
            item["username"],
            candidate_score=item["candidate_score"],
            llm_quality_score=item["llm_quality_score"],
        )

    # Print results
    table = Table(title="Discovered Accounts")
    table.add_column("Username", style="cyan")
    table.add_column("Cand. Score", justify="right")
    table.add_column("LLM Quality", justify="right")
    table.add_column("Final Score", justify="right", style="green")
    table.add_column("Reason")

    for item in scored[:30]:
        reason = ""
        if item["quality"]:
            reason = item["quality"].get("reason", "")[:50]
        table.add_row(
            f"@{item['username']}",
            str(item["candidate_score"]),
            str(item["llm_quality_score"]),
            f"{item['final_score']:.1f}",
            reason,
        )

    console.print(table)

    total = len(scored) + len(seeds)
    console.print(f"\n[bold]Total accounts: {total}[/bold] ({len(seeds)} seeds + {len(scored)} discovered)")

    if total < MIN_ACCOUNTS_TO_PROCEED:
        console.print(
            f"[yellow]⚠ Only {total} accounts found (target: {MIN_ACCOUNTS_TO_PROCEED}). "
            f"Continuing anyway.[/yellow]"
        )


def _log_error(message):
    try:
        with open(ERROR_LOG, "a") as f:
            f.write(f"[{datetime.utcnow().isoformat()}] [discover] {message}\n")
    except Exception:
        pass
