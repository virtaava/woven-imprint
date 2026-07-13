import pytest

from tests.helpers import FakeLLM


class FlakyJSONLLM(FakeLLM):
    def __init__(self, fail_times: int):
        super().__init__()
        self.fail_times = fail_times
        self.json_calls = 0
        self.temps: list[float] = []

    def generate_json(self, messages, temperature=0.3, **kw):
        self.json_calls += 1
        self.temps.append(temperature)
        if self.json_calls <= self.fail_times:
            raise ValueError("bad json")
        return {"ok": True}


def test_retries_once_at_low_temp():
    llm = FlakyJSONLLM(fail_times=1)
    from woven_imprint.llm.base import LLMProvider

    result = LLMProvider.generate_json_robust(llm, [{"role": "user", "content": "x"}])
    assert result == {"ok": True}
    assert llm.json_calls == 2
    assert llm.temps[1] == 0.1


def test_raises_after_retry_exhausted():
    llm = FlakyJSONLLM(fail_times=2)
    from woven_imprint.llm.base import LLMProvider

    with pytest.raises(ValueError):
        LLMProvider.generate_json_robust(llm, [{"role": "user", "content": "x"}])
    assert llm.json_calls == 2
