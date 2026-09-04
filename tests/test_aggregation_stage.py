"""Tier 3n: aggregation answer stage (enumerate-then-answer).

Spec: docs/superpowers/specs/2026-09-04-tier3n-aggregation-answer.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from eval.external.aggregation import (
    AGG_QA_SYSTEM,
    agg_qa_messages,
    is_aggregation_question,
    parse_agg_answer,
)
from eval.external.common import Conversation, Question
from eval.external.prompts import QA_SYSTEM
from eval.external.runner import AGG_QA_MAX_TOKENS, QA_MAX_TOKENS, RunConfig, answer_question
from woven_imprint import clock
from woven_imprint.config import get_config

FIXTURES = Path(__file__).resolve().parent.parent / "eval" / "external" / "fixtures"


@pytest.fixture(autouse=True)
def _reset_clock():
    """`answer_question` drives `clock.override()` as a plain (non-restoring) call, by design
    (see tests/test_external_runner.py's own `_reset_clock`) — reset the global clock after
    each test here so it doesn't leak a stale (2023) override into unrelated tests."""
    yield
    clock.override(None)


@pytest.fixture(autouse=True)
def _reset_query_expansion():
    """`answer_question`'s agg path mutates the global `memory.query_expansion` around a
    single retrieve call and restores it itself — but reset to a known baseline (0, the
    default) before each test here, and restore whatever was there before after, so this
    file's tests neither depend on nor leak state to the rest of the suite."""
    mem_cfg = get_config().memory
    original = mem_cfg.query_expansion
    mem_cfg.query_expansion = 0
    yield
    mem_cfg.query_expansion = original


# ── is_aggregation_question: regex positives/negatives ─────────────────────

POSITIVE_QUESTIONS = [
    "How many pets do I have?",
    "How much did I spend on groceries last week?",
    "What was the cost in total for the renovation?",
    "What is the total number of trips I took this year?",
    "What was the total amount I paid for the car?",
    "How many books did I read altogether?",
    "Can you list all the countries I visited?",
    "What order did I complete the tasks in?",
    "What is the order of the exams I have this week?",
    "What is the sequence of events that happened that day?",
    "First I went to the store, then what did I do?",
    "How many different art-related events did I attend in the past month?",
]

NEGATIVE_QUESTIONS = [
    "What time did I reach the clinic on Monday?",
    "Where did John get the idea for the new logo?",
    "What did Melanie see at the museum?",
]


@pytest.mark.parametrize("question", POSITIVE_QUESTIONS)
def test_is_aggregation_question_positives(question):
    assert is_aggregation_question(question) is True


@pytest.mark.parametrize("question", NEGATIVE_QUESTIONS)
def test_is_aggregation_question_negatives(question):
    assert is_aggregation_question(question) is False


def test_is_aggregation_question_case_insensitive():
    assert is_aggregation_question("HOW MANY dogs do I own?") is True
    assert is_aggregation_question("please LIST ALL my hobbies") is True


def test_is_aggregation_question_word_boundaried():
    """'sequence of' must match as whole words, not as a substring of a larger word — e.g.
    'consequence of' contains the literal substring 'sequence of' but is not an ordering ask."""
    assert is_aggregation_question("What was the consequence of the decision?") is False


def test_is_aggregation_question_fixture_sweep():
    """Smoke test: every question in the committed LME fixture runs through the classifier
    without error. No assertion on the outcome — this only proves the regex set never raises
    on real question text."""
    data = json.loads((FIXTURES / "longmemeval_mini.json").read_text())
    for item in data:
        result = is_aggregation_question(item["question"])
        assert isinstance(result, bool)


# ── parse_agg_answer ─────────────────────────────────────────────────────


def test_parse_agg_answer_with_marker():
    text = "Event A - Jan 1\nEvent B - Jan 5\nAnswer: 2 events"
    assert parse_agg_answer(text) == "2 events"


def test_parse_agg_answer_multiple_markers_last_wins():
    text = "Answer: wrong first guess\nOn reflection:\nAnswer: 3 events"
    assert parse_agg_answer(text) == "3 events"


def test_parse_agg_answer_no_marker_returns_whole_text_stripped():
    text = "  Not mentioned  "
    assert parse_agg_answer(text) == "Not mentioned"


def test_parse_agg_answer_trailing_whitespace():
    text = "Answer: 5 events   \n\n"
    assert parse_agg_answer(text) == "5 events"


def test_parse_agg_answer_empty_string():
    assert parse_agg_answer("") == ""


# ── AGG_QA_SYSTEM / agg_qa_messages ─────────────────────────────────────


def test_agg_qa_system_pins_answer_contract():
    assert "Answer:" in AGG_QA_SYSTEM
    assert "Not mentioned" in AGG_QA_SYSTEM
    assert AGG_QA_SYSTEM != QA_SYSTEM


def test_agg_qa_messages_mirrors_qa_messages_shape():
    messages = agg_qa_messages("Relevant memories:\nfoo", "How many?", "2023-06-10")
    assert messages[0] == {"role": "system", "content": AGG_QA_SYSTEM}
    assert messages[1]["role"] == "user"
    assert "Today is 2023-06-10." in messages[1]["content"]
    assert "Relevant memories:\nfoo" in messages[1]["content"]
    assert "Question: How many?" in messages[1]["content"]


def test_agg_qa_messages_omits_empty_memory_block():
    messages = agg_qa_messages("", "How many?", "2023-06-10")
    assert messages[1]["content"] == "Today is 2023-06-10.\n\nQuestion: How many?"


# ── runner.answer_question wiring ───────────────────────────────────────


class _FakeRetriever:
    """Records `query_expansion` (as seen by `get_config().memory` at call time) on every
    call; optionally raises instead of returning."""

    def __init__(self, mems=None, raises: Exception | None = None):
        self.mems = mems if mems is not None else []
        self.raises = raises
        self.calls: list[dict] = []

    def retrieve(self, question, limit, relationship_target):
        self.calls.append(
            {
                "question": question,
                "limit": limit,
                "relationship_target": relationship_target,
                "query_expansion": get_config().memory.query_expansion,
            }
        )
        if self.raises is not None:
            raise self.raises
        return self.mems


class _FakeChar:
    """Minimal stand-in for `Character` — just the surface `answer_question` touches."""

    def __init__(self, mems=None, raises: Exception | None = None):
        self.retriever = _FakeRetriever(mems, raises)

    def _format_pinned_block(self):
        return "", set()

    def _format_facts_block(self, user_name, pinned_ids):
        return ""

    def _format_memories(self, mems):
        return "\n".join(m["content"] for m in mems)


class _RecordingLLM:
    """Records every `generate()` call's messages+kwargs; returns a scripted response."""

    def __init__(self, response: str = "Not mentioned"):
        self.response = response
        self.calls: list[dict] = []

    def generate(self, messages, **kw):
        self.calls.append({"messages": messages, "kwargs": kw})
        return self.response


def _question(text: str, qid: str = "q1") -> Question:
    return Question(
        qid=qid,
        question=text,
        answer="",
        category="",
        evidence=[],
        asked_at=datetime(2023, 6, 10, 9, 0, tzinfo=timezone.utc),
        kind="qa",
    )


def _conversation() -> Conversation:
    return Conversation(
        conv_id="c1",
        user_name="Alice",
        character_name="Bob",
        sessions=[],
        questions=[],
        transcript_text="",
    )


def _cfg(agg_stage: bool) -> RunConfig:
    return RunConfig(bench="longmemeval_s", mode="memory", run_id="t3n-test", agg_stage=agg_stage)


def test_agg_stage_true_aggregation_question_uses_agg_prompt_and_records_raw():
    char = _FakeChar(mems=[{"id": "m1", "content": "Event A on Jan 1"}])
    llm = _RecordingLLM(response="Event A - Jan 1\nEvent B - Jan 5\nAnswer: 2 events")
    q = _question("How many events did I attend?")

    result = answer_question(_conversation(), char, q, _cfg(agg_stage=True), llm)

    assert len(llm.calls) == 1
    messages = llm.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": AGG_QA_SYSTEM}
    assert llm.calls[0]["kwargs"]["max_tokens"] == AGG_QA_MAX_TOKENS
    assert result["raw_response"] == "Event A - Jan 1\nEvent B - Jan 5\nAnswer: 2 events"
    assert result["response"] == "2 events"
    # Retrieval ran with query_expansion forced to 3 for this call...
    assert char.retriever.calls[0]["query_expansion"] == 3
    # ...and restored afterward.
    assert get_config().memory.query_expansion == 0


def test_agg_stage_true_non_aggregation_question_untouched():
    """A non-aggregation question, even with agg_stage=True, takes the plain path unchanged:
    QA_SYSTEM, QA_MAX_TOKENS, no raw_response field, retrieval untouched."""
    char = _FakeChar(mems=[])
    llm = _RecordingLLM(response="blue")
    q = _question("What color is my cat?")

    result = answer_question(_conversation(), char, q, _cfg(agg_stage=True), llm)

    messages = llm.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": QA_SYSTEM}
    assert llm.calls[0]["kwargs"]["max_tokens"] == QA_MAX_TOKENS
    assert "raw_response" not in result
    assert result["response"] == "blue"
    assert char.retriever.calls[0]["query_expansion"] == 0


def test_agg_stage_false_is_byte_identical_even_for_aggregation_question():
    """agg_stage=False must never touch the AGG system prompt, even for a question that would
    otherwise match the aggregation regex — the off path is exactly today's behavior."""
    char = _FakeChar(mems=[{"id": "m1", "content": "Event A on Jan 1"}])
    llm = _RecordingLLM(response="Not mentioned")
    q = _question("How many events did I attend?")

    result = answer_question(_conversation(), char, q, _cfg(agg_stage=False), llm)

    messages = llm.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": QA_SYSTEM}
    assert AGG_QA_SYSTEM not in messages[0]["content"]
    assert llm.calls[0]["kwargs"]["max_tokens"] == QA_MAX_TOKENS
    assert "raw_response" not in result
    assert set(result) == {"qid", "response", "prompt_tokens_est", "memories_used", "seconds"}
    assert char.retriever.calls[0]["query_expansion"] == 0


def test_agg_stage_false_never_uses_agg_system_across_many_questions():
    """Sweep both aggregation-shaped and plain questions with agg_stage=False (RunConfig's
    own default) and assert no call ever carries the AGG system text."""
    for question in POSITIVE_QUESTIONS + NEGATIVE_QUESTIONS:
        char = _FakeChar(mems=[])
        llm = _RecordingLLM()
        cfg = RunConfig(bench="longmemeval_s", mode="memory", run_id="t3n-sweep")
        assert cfg.agg_stage is False  # default

        answer_question(_conversation(), char, _question(question), cfg, llm)

        content = llm.calls[0]["messages"][0]["content"]
        assert content == QA_SYSTEM
        assert AGG_QA_SYSTEM not in content


def test_agg_retrieve_exception_restores_query_expansion():
    """If the wrapped retrieve() call raises, `memory.query_expansion` must still be restored
    to whatever it was before — the exception propagates, but the config mutation must not
    leak past it."""
    mem_cfg = get_config().memory
    mem_cfg.query_expansion = 0
    char = _FakeChar(raises=RuntimeError("boom"))
    llm = _RecordingLLM()
    q = _question("How many events did I attend?")

    with pytest.raises(RuntimeError, match="boom"):
        answer_question(_conversation(), char, q, _cfg(agg_stage=True), llm)

    assert mem_cfg.query_expansion == 0
    # The retrieve call itself did see the forced value before raising.
    assert char.retriever.calls[0]["query_expansion"] == 3
    # No LLM call was made — the exception happened before the answer call.
    assert llm.calls == []
