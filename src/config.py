"""
Application configuration.

All configuration is sourced from environment variables (optionally loaded
from a local .env file via python-dotenv). Nothing here hard-codes secrets.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env if present. Safe no-op if the file doesn't exist.
load_dotenv()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- Cohere ---
    cohere_api_key: str | None = os.getenv("COHERE_API_KEY")
    cohere_model: str = os.getenv("COHERE_MODEL", "command-r-08-2024")
    cohere_timeout_seconds: float = _float_env("COHERE_TIMEOUT_SECONDS", 30.0)
    cohere_max_retries: int = _int_env("COHERE_MAX_RETRIES", 2)
    cohere_concurrency: int = _int_env("COHERE_CONCURRENCY", 5)
    cohere_calls_per_minute: int = _int_env("COHERE_CALLS_PER_MINUTE", 35)

    # --- GitHub ---
    github_token: str | None = os.getenv("GITHUB_TOKEN")  # optional, raises rate limit
    github_timeout_seconds: float = _float_env("GITHUB_TIMEOUT_SECONDS", 15.0)
    github_concurrency: int = _int_env("GITHUB_CONCURRENCY", 5)

    # --- Pipeline ---
    max_workers: int = _int_env("MAX_WORKERS", 5)

    @property
    def cohere_enabled(self) -> bool:
        return bool(self.cohere_api_key)


settings = Settings()
