import time
import logging
import requests
from datetime import datetime, timedelta
from config import RAPIDAPI_KEY, ERROR_LOG

logger = logging.getLogger(__name__)


class TwitterClient:
    """RapidAPI Twitter abstraction. Swap providers by changing _request + normalization methods."""

    BASE_URL = "https://twitter-api45.p.rapidapi.com"

    def __init__(self, api_key=None):
        self.api_key = api_key or RAPIDAPI_KEY
        if not self.api_key:
            raise ValueError("RAPIDAPI_KEY is required. Set it in .env")
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "twitter-api45.p.rapidapi.com",
        }
        self._last_request_time = 0

    def _rate_limit_wait(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)

    def _request(self, endpoint, params=None, retries=3):
        for attempt in range(retries):
            self._rate_limit_wait()
            try:
                resp = requests.get(
                    f"{self.BASE_URL}/{endpoint}",
                    headers=self.headers,
                    params=params or {},
                    timeout=30,
                )
                self._last_request_time = time.time()

                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited. Waiting {wait}s (attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error on {endpoint}: {e}")
                if attempt == retries - 1:
                    self._log_error(f"Failed after {retries} retries: {endpoint} params={params} error={e}")
                    return None
                time.sleep(2 ** attempt)

        return None

    def get_user_tweets(self, username, count=500):
        """Fetch up to `count` recent tweets for a user. Returns normalized tweet dicts."""
        tweets = []
        cursor = None
        per_page = min(count, 50)

        while len(tweets) < count:
            params = {"screenname": username, "count": str(per_page)}
            if cursor:
                params["cursor"] = cursor

            data = self._request("timeline.php", params)
            if not data or "timeline" not in data:
                break

            timeline = data["timeline"]
            if not timeline:
                break

            for raw in timeline:
                tweet = self._normalize_tweet(raw, username)
                if tweet:
                    tweets.append(tweet)

            cursor = data.get("next_cursor")
            if not cursor:
                break

        return tweets[:count]

    def get_user_profile(self, username):
        """Fetch user profile info. Returns normalized profile dict or None."""
        data = self._request("screenname.php", {"screenname": username})
        if not data:
            return None
        return self._normalize_profile(data)

    def _normalize_tweet(self, raw, fallback_username):
        """Normalize a raw tweet from twitter-api45 into our standard format."""
        try:
            text = raw.get("text", "")
            tweet_id = raw.get("rest_id") or raw.get("tweet_id") or raw.get("id_str", "")

            # Detect retweet
            is_retweet = text.startswith("RT @") or "retweeted_tweet" in raw
            retweeted_user = None
            if is_retweet and "retweeted_tweet" in raw:
                rt_data = raw["retweeted_tweet"]
                retweeted_user = (
                    rt_data.get("user", {}).get("screen_name")
                    or rt_data.get("screen_name")
                )
            elif is_retweet and text.startswith("RT @"):
                # Parse from text: "RT @username: ..."
                mention = text.split("RT @")[1].split(":")[0].split(" ")[0]
                retweeted_user = mention.strip()

            # Detect quote tweet
            is_quote = "quoted_tweet" in raw or raw.get("is_quote_status", False)
            quoted_user = None
            if "quoted_tweet" in raw:
                qt_data = raw["quoted_tweet"]
                quoted_user = (
                    qt_data.get("user", {}).get("screen_name")
                    or qt_data.get("screen_name")
                )

            # Extract mentions
            mentioned_users = []
            entities = raw.get("entities", {})
            for mention in entities.get("user_mentions", []):
                screen_name = mention.get("screen_name")
                if screen_name:
                    mentioned_users.append(screen_name)

            # Parse created_at
            created_at = raw.get("created_at", "")
            try:
                dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
                created_at = dt.isoformat()
            except (ValueError, TypeError):
                pass

            return {
                "id": str(tweet_id),
                "text": text,
                "created_at": created_at,
                "likes": int(raw.get("favorite_count", 0)),
                "retweets": int(raw.get("retweet_count", 0)),
                "is_retweet": is_retweet,
                "is_quote": is_quote,
                "quoted_user": quoted_user,
                "mentioned_users": mentioned_users,
                "retweeted_user": retweeted_user,
                "username": raw.get("user", {}).get("screen_name", fallback_username),
            }
        except Exception as e:
            logger.debug(f"Failed to normalize tweet: {e}")
            return None

    def _normalize_profile(self, raw):
        """Normalize a raw profile from twitter-api45."""
        try:
            created_at = raw.get("created_at", "")
            account_age_days = 0
            try:
                dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
                account_age_days = (datetime.now(dt.tzinfo) - dt).days
            except (ValueError, TypeError):
                account_age_days = 365  # assume not new if we can't parse

            return {
                "username": raw.get("screen_name") or raw.get("username", ""),
                "followers": int(raw.get("followers_count", raw.get("sub_count", 0))),
                "following": int(raw.get("friends_count", raw.get("following_count", 0))),
                "created_at": created_at,
                "description": raw.get("description", ""),
                "account_age_days": account_age_days,
            }
        except Exception as e:
            logger.debug(f"Failed to normalize profile: {e}")
            return None

    def _log_error(self, message):
        try:
            with open(ERROR_LOG, "a") as f:
                f.write(f"[{datetime.utcnow().isoformat()}] [twitter] {message}\n")
        except Exception:
            pass
