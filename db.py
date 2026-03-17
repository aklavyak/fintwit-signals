import sqlite3
from datetime import datetime
from config import DB_PATH


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS accounts (
            username TEXT PRIMARY KEY,
            is_seed INTEGER DEFAULT 0,
            candidate_score REAL,
            llm_quality_score REAL,
            picker_score REAL,
            tier TEXT,
            n_calls INTEGER DEFAULT 0,
            win_rate REAL,
            win_rate_bayesian REAL,
            mean_excess_return REAL,
            p_value REAL,
            added_at TEXT,
            last_updated TEXT
        );

        CREATE TABLE IF NOT EXISTS tweets (
            tweet_id TEXT PRIMARY KEY,
            username TEXT,
            text TEXT,
            created_at TEXT,
            likes INTEGER,
            retweets INTEGER,
            is_actionable INTEGER DEFAULT 0,
            processed INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tweet_id TEXT,
            username TEXT,
            call_date TEXT,
            call_type TEXT,
            ticker TEXT,
            theme TEXT,
            etf_proxy TEXT,
            direction TEXT,
            conviction TEXT,
            thesis TEXT,
            call_confidence REAL,
            has_reasoning INTEGER,
            is_vague INTEGER,
            excess_30d REAL,
            excess_60d REAL,
            excess_90d REAL,
            composite_score REAL,
            scored_at TEXT
        );
    """)
    conn.commit()
    conn.close()


def upsert_account(conn, username, **kwargs):
    now = datetime.utcnow().isoformat()
    existing = conn.execute(
        "SELECT username FROM accounts WHERE username = ?", (username,)
    ).fetchone()

    if existing:
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [now, username]
        conn.execute(
            f"UPDATE accounts SET {sets}, last_updated = ? WHERE username = ?", vals
        )
    else:
        kwargs["username"] = username
        kwargs["added_at"] = now
        kwargs["last_updated"] = now
        cols = ", ".join(kwargs.keys())
        placeholders = ", ".join("?" for _ in kwargs)
        conn.execute(
            f"INSERT INTO accounts ({cols}) VALUES ({placeholders})",
            list(kwargs.values()),
        )
    conn.commit()


def insert_tweet(conn, tweet_id, username, text, created_at, likes, retweets):
    try:
        conn.execute(
            """INSERT OR IGNORE INTO tweets
               (tweet_id, username, text, created_at, likes, retweets)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (tweet_id, username, text, created_at, likes, retweets),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass


def insert_call(conn, **kwargs):
    cols = ", ".join(kwargs.keys())
    placeholders = ", ".join("?" for _ in kwargs)
    conn.execute(
        f"INSERT INTO calls ({cols}) VALUES ({placeholders})",
        list(kwargs.values()),
    )
    conn.commit()


def get_accounts(conn, only_seeds=False):
    if only_seeds:
        return conn.execute("SELECT * FROM accounts WHERE is_seed = 1").fetchall()
    return conn.execute("SELECT * FROM accounts").fetchall()


def get_unprocessed_tweets(conn):
    return conn.execute(
        "SELECT * FROM tweets WHERE processed = 0"
    ).fetchall()


def get_actionable_tweets(conn):
    return conn.execute(
        "SELECT * FROM tweets WHERE is_actionable = 1"
    ).fetchall()


def get_unscored_calls(conn):
    return conn.execute(
        "SELECT * FROM calls WHERE scored_at IS NULL"
    ).fetchall()


def get_scored_calls_for_user(conn, username):
    return conn.execute(
        "SELECT composite_score FROM calls WHERE username = ? AND composite_score IS NOT NULL",
        (username,),
    ).fetchall()


def update_call_scores(conn, call_id, scores):
    conn.execute(
        """UPDATE calls SET excess_30d = ?, excess_60d = ?, excess_90d = ?,
           composite_score = ?, scored_at = ?
           WHERE id = ?""",
        (
            scores.get("excess_30d"),
            scores.get("excess_60d"),
            scores.get("excess_90d"),
            scores.get("composite"),
            datetime.utcnow().isoformat(),
            call_id,
        ),
    )
    conn.commit()


def update_picker_scores(conn, username, stats):
    conn.execute(
        """UPDATE accounts SET picker_score = ?, tier = ?, n_calls = ?,
           win_rate = ?, win_rate_bayesian = ?, mean_excess_return = ?,
           p_value = ?, last_updated = ?
           WHERE username = ?""",
        (
            stats["score"],
            stats["tier"],
            stats["n_calls"],
            stats.get("win_rate"),
            stats.get("win_rate_bayesian"),
            stats.get("mean_excess_return"),
            stats.get("p_value"),
            datetime.utcnow().isoformat(),
            username,
        ),
    )
    conn.commit()
