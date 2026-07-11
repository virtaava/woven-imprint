import threading
import time

from tests.helpers import FakeEmbedder, FakeLLM, make_test_engine
from woven_imprint.background import BackgroundWorker
from woven_imprint.engine import Engine


def test_worker_runs_and_counts():
    worker = BackgroundWorker("test")
    hits = []
    worker.submit("job", hits.append, 1)
    assert worker.flush(timeout=5)
    assert hits == [1]
    assert worker.success_counts["job"] == 1
    worker.close()


def test_worker_records_failures():
    worker = BackgroundWorker("test")

    def boom():
        raise RuntimeError("nope")

    worker.submit("bad", boom)
    assert worker.flush(timeout=5)
    assert worker.failure_counts["bad"] == 1
    worker.close()


class SlowJSONLLM(FakeLLM):
    """generate_json blocks until released — proves chat() doesn't wait on it."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def generate_json(self, messages, **kw):
        self.release.wait(timeout=10)
        return super().generate_json(messages, **kw)


def test_chat_returns_before_bookkeeping():
    llm = SlowJSONLLM()
    engine = Engine(db_path=":memory:", llm=llm, embedding=FakeEmbedder())
    char = engine.create_character("Speedy")
    char.background = True
    char.parallel = False
    # Consistency stays on the synchronous hot path by design (chat() returns
    # after generation + consistency) and also calls llm.generate_json(), so
    # it would block on the same release event as the bookkeeping call this
    # test targets. Disable it here to isolate what's actually under test:
    # emotion/arc/relationship/fact-extraction deferral.
    char.enforce_consistency = False

    start = time.perf_counter()
    response = char.chat("hello", user_id="u1")
    elapsed = time.perf_counter() - start

    assert response == "I hear you."
    assert elapsed < 2.0  # did not wait for the blocked generate_json
    llm.release.set()
    assert char.flush(timeout=10)
    # Bookkeeping actually ran after flush
    assert char.get_relationship("u1") is not None


def test_flush_is_noop_when_sync():
    engine = make_test_engine()
    char = engine.create_character("Syncy")
    assert char.background is False  # helpers force sync mode
    char.chat("hi", user_id="u1")
    assert char.flush()  # trivially true
    assert char.get_relationship("u1") is not None
