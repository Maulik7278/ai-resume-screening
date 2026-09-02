"""
Cohere provider adapter.

This is the ONLY module that talks to the Cohere API. Everything else in
the pipeline depends on `CohereProvider.analyze_resume()`, which returns a
validated `ResumeAnalysis` or raises `LLMError`. Swapping the LLM provider
later means writing a new adapter with the same interface -- nothing else
in the codebase should need to change.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from pydantic import ValidationError

from src.config import settings
from src.models import ResumeAnalysis

try:
    import cohere
    from cohere.errors import TooManyRequestsError
except ImportError:  # pragma: no cover - cohere is a required dependency, but keep this graceful
    cohere = None  # type: ignore[assignment]
    TooManyRequestsError = None  # type: ignore[assignment]


class LLMError(Exception):
    """Raised when Cohere analysis fails after retries, or is unavailable."""


class LLMNotConfigured(LLMError):
    """Raised when no Cohere API key is configured."""


class _RateLimiter:
    """
    Simple thread-safe sliding-window rate limiter shared across all worker
    threads for a single CohereProvider instance.

    This exists because Cohere Trial keys are capped at a fixed number of
    calls per minute (40, as of writing). With bounded concurrency alone,
    a burst of requests can still exceed that cap and get 429'd. This
    limiter makes every thread wait its turn so the batch stays under the
    configured rate, instead of firing requests and hoping retries clean
    up the mess.
    """

    def __init__(self, max_calls: int, period_seconds: float = 60.0) -> None:
        self.max_calls = max(1, max_calls)
        self.period_seconds = period_seconds
        self._lock = threading.Lock()
        self._call_times: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                # Drop timestamps outside the sliding window.
                while self._call_times and now - self._call_times[0] >= self.period_seconds:
                    self._call_times.popleft()
                if len(self._call_times) < self.max_calls:
                    self._call_times.append(now)
                    return
                # Window is full -- figure out how long until the oldest
                # call ages out, then sleep (outside the lock).
                wait_time = self.period_seconds - (now - self._call_times[0]) + 0.05
            time.sleep(max(wait_time, 0.05))


_SCHEMA_INSTRUCTIONS = """You are analyzing a single resume for a hiring pipeline.

Return ONLY a single JSON object (no markdown fences, no commentary) matching
exactly this shape:

{
  "candidate_name": string or null,
  "email": string or null,
  "skills": [string, ...],
  "github_url": string or null,
  "python_evidence": [string, ...]   // direct quotes/paraphrases showing Python experience
  "ai_evidence": [string, ...]       // direct quotes/paraphrases showing AI/LLM/agentic experience
  "projects": [
    {
      "name": string,
      "technologies": [string, ...],
      "description": string,
      "depth": "shallow" | "moderate" | "deep",
      "evidence": [string, ...],
      "shallow_wrapper": boolean      // true if this looks like a thin wrapper around an API/tutorial clone
    }
  ],
  "strengths": [string, ...],
  "concerns": [string, ...],
  "summary": string
}

Base every field strictly on the resume text provided. If information is not
present, use null or an empty list/string as appropriate. Do not invent
facts not supported by the resume.
"""


@dataclass
class CohereProvider:
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    calls_per_minute: int | None = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or settings.cohere_api_key
        self.model = self.model or settings.cohere_model
        self.timeout_seconds = self.timeout_seconds or settings.cohere_timeout_seconds
        self.max_retries = self.max_retries if self.max_retries is not None else settings.cohere_max_retries
        self.calls_per_minute = self.calls_per_minute or settings.cohere_calls_per_minute
        self._client = None
        # One rate limiter per provider instance, shared across all threads
        # that use this instance (the pipeline creates one CohereProvider
        # per batch run and hands it to every worker thread).
        self._rate_limiter = _RateLimiter(self.calls_per_minute)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and cohere is not None

    def _get_client(self):
        if self._client is None:
            if cohere is None:
                raise LLMNotConfigured("cohere package is not installed")
            self._client = cohere.ClientV2(api_key=self.api_key, timeout=self.timeout_seconds)
        return self._client

    def analyze_resume(self, resume_text: str, *, source_file: str = "") -> ResumeAnalysis:
        """
        Send resume text to Cohere and return a validated ResumeAnalysis.

        Raises LLMNotConfigured if no API key is set, or LLMError if every
        attempt fails (network error, timeout, or invalid/unparsable output).
        Callers are expected to catch these and record a per-resume failure
        rather than letting them propagate and abort the whole batch.
        """
        if not self.api_key:
            raise LLMNotConfigured("COHERE_API_KEY is not set")
        if cohere is None:
            raise LLMNotConfigured("cohere package is not installed")

        client = self._get_client()
        last_error: Exception | None = None

        # max_retries=2 means up to 3 total attempts. Deliberately conservative
        # (no exponential backoff spiral) to avoid hammering the API for a
        # single resume in a 50-resume batch.
        for attempt in range(self.max_retries + 1):
            self._rate_limiter.acquire()
            try:
                response = client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SCHEMA_INSTRUCTIONS},
                        {"role": "user", "content": f"Resume text:\n\n{resume_text}"},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                raw_text = self._extract_text(response)
                payload = self._parse_json(raw_text)
                return ResumeAnalysis.model_validate(payload)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                # Malformed output is worth one retry (the model can produce
                # a clean response on a second try) but not more.
                backoff = min(2 ** attempt, 4)
            except Exception as exc:  # noqa: BLE001 - network/SDK errors, isolate and retry
                last_error = exc
                if TooManyRequestsError is not None and isinstance(exc, TooManyRequestsError):
                    # Rate limit hit despite our own limiter (e.g. another
                    # process is sharing the same key, or the window just
                    # rolled over badly). Back off well past a full minute
                    # window rather than retrying quickly and getting
                    # 429'd again immediately.
                    backoff = 20.0
                else:
                    backoff = min(2 ** attempt, 4)

            if attempt < self.max_retries:
                time.sleep(backoff)  # bounded backoff; longer specifically for 429s

        raise LLMError(
            f"Cohere analysis failed for '{source_file}' after {self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    @staticmethod
    def _extract_text(response) -> str:
        # cohere ClientV2 chat responses expose content as a list of blocks
        # on response.message.content, each with a `.text` attribute for
        # text blocks. Fall back defensively across minor SDK variations.
        try:
            content = response.message.content
            if isinstance(content, list) and content:
                text = getattr(content[0], "text", None)
                if text:
                    return text
        except AttributeError:
            pass
        # Fallbacks for older/alternate response shapes.
        text = getattr(response, "text", None)
        if text:
            return text
        raise ValueError("Could not extract text from Cohere response")

    @staticmethod
    def _parse_json(raw_text: str) -> dict:
        raw_text = raw_text.strip()
        # Strip accidental markdown fences if the model adds them anyway.
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]
        return json.loads(raw_text)
