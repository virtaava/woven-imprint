import pytest
import requests

from woven_imprint.llm import resilience
from woven_imprint.llm.resilience import (  # noqa: F401
    CircuitBreaker,
    _is_retryable,
    resilient_call,
    reset_breaker,
)


@pytest.fixture(autouse=True)
def clean_breakers(monkeypatch):
    monkeypatch.setattr(resilience, "_breakers", {})
    # no real sleeping in tests
    monkeypatch.setattr(resilience.time, "sleep", lambda s: None)
    yield


def test_retries_transient_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.Timeout("slow")
        return "ok"

    assert resilient_call(flaky, provider_name="t1") == "ok"
    assert calls["n"] == 3


def test_permanent_error_not_retried():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        resilient_call(bad, provider_name="t2")
    assert calls["n"] == 1


def test_duck_typed_status_code_is_retryable():
    class SDKError(Exception):
        status_code = 429

    assert _is_retryable(SDKError()) is True

    class SDKPermanent(Exception):
        status_code = 400

    assert _is_retryable(SDKPermanent()) is False


def test_breaker_opens_after_threshold(monkeypatch):
    breaker = CircuitBreaker(threshold=2, cooldown=1000.0)
    monkeypatch.setattr(resilience, "_get_breaker", lambda name: breaker)

    def always_fails():
        raise ValueError("permanent")

    for _ in range(2):
        with pytest.raises(ValueError):
            resilient_call(always_fails, provider_name="t3")
    assert breaker.is_open
    with pytest.raises(ConnectionError, match="Circuit breaker open"):
        resilient_call(always_fails, provider_name="t3")


def test_breaker_resets_after_cooldown(monkeypatch):
    breaker = CircuitBreaker(threshold=1, cooldown=0.0)
    breaker.record_failure()
    assert breaker.is_open is False  # cooldown 0 → immediately reset


def test_gemma_edge_uses_resilience(monkeypatch):
    calls = {"n": 0}

    def flaky_post(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise requests.Timeout("slow")

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"content": "hi"}

        return R()

    monkeypatch.setattr("requests.post", flaky_post)
    monkeypatch.setenv("WOVEN_IMPRINT_GEMMA_EDGE_URL", "http://x")
    from woven_imprint.llm.gemma_edge import GemmaEdgeLLM

    llm = GemmaEdgeLLM()
    assert llm.generate([{"role": "user", "content": "hi"}]) == "hi"
    assert calls["n"] == 2
