import time

from src.llm.cohere_provider import CohereProvider, _RateLimiter


def test_rate_limiter_allows_calls_under_the_cap():
    limiter = _RateLimiter(max_calls=5, period_seconds=60)
    start = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    # All 5 calls should be effectively instant -- well under the cap.
    assert time.monotonic() - start < 1.0


def test_rate_limiter_throttles_once_cap_is_hit():
    # Small window so the test doesn't take long: 2 calls per 0.3 seconds.
    limiter = _RateLimiter(max_calls=2, period_seconds=0.3)
    start = time.monotonic()
    limiter.acquire()
    limiter.acquire()
    # Third call should block until the window rolls over.
    limiter.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.25  # allow small timing slack


def test_cohere_provider_uses_configured_calls_per_minute():
    provider = CohereProvider(api_key="fake-key", calls_per_minute=10)
    assert provider._rate_limiter.max_calls == 10


def test_cohere_provider_default_calls_per_minute_is_conservative():
    # Default should stay comfortably under Cohere's 40/minute Trial cap.
    provider = CohereProvider(api_key="fake-key")
    assert provider._rate_limiter.max_calls <= 40
