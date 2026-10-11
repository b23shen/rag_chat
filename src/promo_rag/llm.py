"""Gemini client setup, shared by ingestion, the agent and eval."""

from google import genai
from google.genai import types

from promo_rag.config import get_settings

# USD per 1M tokens (input, output), from ai.google.dev/gemini-api/docs/pricing, Oct 2026.
PRICES = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-embedding-2": (0.20, 0.0),
}


def gemini_client(timeout_s: float = 30) -> genai.Client:
    """The one place retries are configured, so they can't stack with another layer.

    attempts=3 is the first call plus up to 2 retries, only on rate limits and server
    errors, with exponential backoff.
    """
    return genai.Client(
        api_key=get_settings().gemini_api_key.get_secret_value(),
        http_options=types.HttpOptions(
            timeout=int(timeout_s * 1000),
            retry_options=types.HttpRetryOptions(
                attempts=3,
                initial_delay=2.0,
                max_delay=30.0,
                http_status_codes=[429, 500, 502, 503, 504],
            ),
        ),
    )


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
