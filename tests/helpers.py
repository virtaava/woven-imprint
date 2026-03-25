"""Shared test helpers for woven-imprint tests."""

from woven_imprint.engine import Engine


class FakeLLM:
    def __init__(self):
        self.call_count = 0

    def generate(self, messages, **kw):
        self.call_count += 1
        return "I hear you."

    def generate_json(self, messages, **kw):
        self.call_count += 1
        system = messages[0].get("content", "") if messages else ""
        if "extract" in system.lower() or "fact" in system.lower():
            return ["A notable fact was shared"]
        if "emotion" in system.lower():
            return {"mood": "neutral", "intensity": 0.5, "cause": ""}
        if "relationship" in system.lower():
            return {"trust": 0.01, "affection": 0.0, "respect": 0.0,
                    "familiarity": 0.02, "tension": 0.0}
        if "narrative" in system.lower() or "beat" in system.lower():
            return {"beat_type": "none"}
        if "consistency" in system.lower():
            return {"score": 1.0, "issues": []}
        if "summar" in system.lower():
            return "Session summary"
        return {}


class FakeEmbedder:
    def __init__(self):
        self._vocab = {}
        self._next = 0

    def embed(self, text):
        vec = [0.0] * 50
        for word in text.lower().split()[:10]:
            if word not in self._vocab:
                self._vocab[word] = self._next % 50
                self._next += 1
            vec[self._vocab[word]] += 1.0
        mag = sum(x * x for x in vec) ** 0.5
        if mag > 0:
            vec = [x / mag for x in vec]
        return vec

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def dimensions(self):
        return 50


def make_test_engine(db_path=":memory:"):
    """Create an Engine with fakes, parallel disabled."""
    llm = FakeLLM()
    embedder = FakeEmbedder()
    engine = Engine(db_path=db_path, llm=llm, embedding=embedder)
    orig = engine.create_character
    def _create_seq(*a, **kw):
        c = orig(*a, **kw)
        c.parallel = False
        return c
    engine.create_character = _create_seq
    return engine
