import json
import re
import time
import logging
from openai import OpenAI, RateLimitError, APIError
from config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI API wrapper for LLM calls."""

    def __init__(self, model=None, api_key=None):
        self.model = model or OPENAI_MODEL
        key = api_key or OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_KEY not set in .env")
        self.client = OpenAI(api_key=key)

    def _call(self, prompt, temperature=0.0, json_mode=False, retries=3):
        """Make an API call with retry logic."""
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 500,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content.strip()
            except RateLimitError:
                wait = 2 ** attempt * 5
                logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
            except APIError as e:
                logger.warning(f"API error (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    raise
        return None

    def classify(self, prompt):
        """Send a prompt expecting a short text response (e.g., yes/no)."""
        response = self._call(prompt, temperature=0.0)
        return response.lower().strip() if response else ""

    def extract_json(self, prompt):
        """Send a prompt expecting a JSON response. Returns parsed dict or None."""
        response = self._call(prompt, temperature=0.1, json_mode=True)
        if response:
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
        # Fallback: without JSON mode + robust parsing
        response = self._call(prompt, temperature=0.1, json_mode=False)
        return self._parse_json(response)

    @staticmethod
    def _parse_json(text):
        """Robustly parse JSON from LLM output."""
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.debug(f"Failed to parse JSON from: {text[:200]}")
        return None
