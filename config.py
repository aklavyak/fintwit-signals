import os
from dotenv import load_dotenv

load_dotenv()

# API keys
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

# OpenAI settings
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Pipeline settings
DAYS_BACK = 90
MIN_ACCOUNTS_TO_PROCEED = 20
MIN_CALLS_TO_PROCEED = 100
CANDIDATE_SCORE_CUTOFF = 3
SCORE_WINDOWS = [30, 60, 90]
BAYESIAN_PRIOR = 10

# Theme to ETF mapping
THEME_ETF_MAP = {
    "technology": "XLK",
    "financials": "XLF",
    "energy": "XLE",
    "healthcare": "XLV",
    "defense": "XAR",
    "semiconductors": "SOXX",
    "ai": "BOTZ",
    "real_estate": "XLRE",
    "consumer": "XLY",
    "utilities": "XLU",
    "industrials": "XLI",
    "small_cap": "IWM",
    "biotech": "XBI",
    "crypto": "BITO",
}

# Scheduling
DISCOVERY_INTERVAL_DAYS = 30

# Email (for --daily automated runs via Resend)
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
EMAIL_TO = os.getenv("EMAIL_TO", "")

# Paths
DB_PATH = os.path.join(os.path.dirname(__file__), "fintwit.db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SEEDS_PATH = os.path.join(os.path.dirname(__file__), "seeds.txt")
ERROR_LOG = os.path.join(DATA_DIR, "errors.log")
