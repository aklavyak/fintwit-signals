"""Email the daily briefing via Resend API."""

import logging
from pathlib import Path

import requests

from config import RESEND_API_KEY, EMAIL_TO

logger = logging.getLogger(__name__)

RESEND_URL = "https://api.resend.com/emails"


def send_briefing(briefing_path):
    """Read the markdown briefing and send it as a plain-text email."""
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not set. Skipping email.")
        print("  [skip] Email not configured — set RESEND_API_KEY in .env or GitHub secrets")
        return

    content = Path(briefing_path).read_text()

    # Extract date from first line: "# Fintwit Idea Briefing — 2026-03-14"
    first_line = content.splitlines()[0] if content else ""
    date_part = first_line.split("—")[-1].strip() if "—" in first_line else "today"

    try:
        resp = requests.post(
            RESEND_URL,
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={
                "from": "Fintwit <onboarding@resend.dev>",
                "to": EMAIL_TO,
                "subject": f"Fintwit Idea Briefing — {date_part}",
                "text": content,
            },
        )
        resp.raise_for_status()
        logger.info("Briefing emailed to %s", EMAIL_TO)
        print(f"  [ok] Briefing emailed to {EMAIL_TO}")
    except Exception as e:
        logger.error("Failed to send email: %s", e)
        print(f"  [error] Email failed: {e}")
