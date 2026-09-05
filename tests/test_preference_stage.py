"""Tier 3q: preference answer stage (grounded advice-request answering).

Spec: docs/superpowers/specs/2026-09-05-tier3q-preference-answer.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from eval.external.aggregation import AGG_QA_SYSTEM
from eval.external.common import Conversation, Question
from eval.external.preference import (
    PREF_QA_SYSTEM,
    is_preference_question,
    pref_qa_messages,
)
from eval.external.prompts import QA_SYSTEM
from eval.external.runner import (
    AGG_QA_MAX_TOKENS,
    PREF_QA_MAX_TOKENS,
    QA_MAX_TOKENS,
    RunConfig,
    answer_question,
)
from woven_imprint import clock

FIXTURES = Path(__file__).resolve().parent.parent / "eval" / "external" / "fixtures"


@pytest.fixture(autouse=True)
def _reset_clock():
    """`answer_question` drives `clock.override()` as a plain (non-restoring) call, by design
    (see tests/test_external_runner.py's own `_reset_clock`) — reset the global clock after
    each test here so it doesn't leak a stale (2023) override into unrelated tests."""
    yield
    clock.override(None)


# ── is_preference_question: regex positives/negatives ───────────────────
#
# All 17 `single-session-preference` question texts, copied verbatim from the committed
# `eval/results/external_lme-s-100-v5.json` (that file is git-tracked but 2MB — not read here;
# hardcoded instead, per the spec). One real text ("... Could there be a reason for this?") is a
# causal-explanation ask, not an advice request, and is deliberately NOT expected to match — the
# spec's bar is >= 15/17, not 17/17.
ALL_V5_PREFERENCE_QUESTIONS = [
    "Can you suggest some useful accessories for my phone?",
    "Can you recommend a show or movie for me to watch tonight?",
    "I noticed my bike seems to be performing even better during my Sunday group rides. "
    "Could there be a reason for this?",
    "I am planning another theme park weekend; do you have any suggestions?",
    "I'm trying to decide whether to buy a NAS device now or wait. What do you think?",
    "I've been feeling nostalgic lately. Do you think it would be a good idea to attend my "
    "high school reunion?",
    "I'm planning my meal prep next week, any suggestions for new recipes?",
    "I’m a bit anxious about getting around Tokyo. Do you have any helpful tips?",
    "I've been thinking about ways to stay connected with my colleagues. Any suggestions?",
    "What should I serve for dinner this weekend with my homegrown ingredients?",
    "I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?",
    "I've been feeling like my chocolate chip cookies need something extra. Any advice?",
    "I've been having trouble with the battery life on my phone lately. Any tips?",
    "Can you suggest some accessories that would complement my current photography setup?",
    "My kitchen's becoming a bit of a mess again. Any tips for keeping it clean?",
    "I was thinking about rearranging the furniture in my bedroom this weekend. Any tips?",
    "I've got some free time tonight, any documentary recommendations?",
]

# The one real text expected NOT to match (see docstring above) — everything else in
# ALL_V5_PREFERENCE_QUESTIONS is a confirmed positive.
_KNOWN_NON_ADVICE_TEXT = (
    "I noticed my bike seems to be performing even better during my Sunday group rides. "
    "Could there be a reason for this?"
)

POSITIVE_QUESTIONS = [q for q in ALL_V5_PREFERENCE_QUESTIONS if q != _KNOWN_NON_ADVICE_TEXT]

# The 7 `_abs` (abstain) question texts, copied verbatim from the same fixture, plus a spread of
# factual questions from other categories/examples — none of these are advice requests.
NEGATIVE_QUESTIONS = [
    "What time did I reach the clinic on Monday?",
    "How often do I play table tennis with my friends at the local park?",
    "What is the name of my hamster?",
    "How much time do I dedicate to practicing violin every day?",
    "How many autographed football have I added to my collection in the first three months "
    "of collection?",
    "How many plants did I initially plant for tomatoes and chili peppers?",
    "How long have I been living in my current apartment in Shinjuku?",
    "How often do I see Dr. Johnson?",
    "What time do I stop checking work emails and messages?",
    "What degree did I graduate with?",
    "What is the name of the music streaming service have I been using lately?",
    "What is the order of the three trips I took in the past three months, from earliest to "
    "latest?",
    "Can you remind me of the Instagram handle of the UK-based designer who works with "
    "unusual gemstones?",
    "How many free night's stays can I redeem at any Hilton property with my accumulated points?",
    "Where did John get the idea for the new logo?",
    "What did Melanie see at the museum?",
]


@pytest.mark.parametrize("question", POSITIVE_QUESTIONS)
def test_is_preference_question_positives(question):
    assert is_preference_question(question) is True


@pytest.mark.parametrize("question", NEGATIVE_QUESTIONS)
def test_is_preference_question_negatives(question):
    assert is_preference_question(question) is False


def test_is_preference_question_v5_fixture_hits_spec_bar():
    """The spec's pre-registered bar: >= 15 of the 17 real v5 preference texts must match."""
    hits = sum(1 for q in ALL_V5_PREFERENCE_QUESTIONS if is_preference_question(q))
    assert hits >= 15, f"only {hits}/17 matched"


def test_is_preference_question_known_non_advice_text_does_not_match():
    """A causal-explanation ask ("Could there be a reason for this?") is not an advice request
    — it correctly falls outside the closed regex set even though it's in the same LME category."""
    assert is_preference_question(_KNOWN_NON_ADVICE_TEXT) is False


def test_is_preference_question_case_insensitive():
    assert is_preference_question("CAN YOU RECOMMEND a good book?") is True
    assert is_preference_question("i need ANY ADVICE on this") is True


def test_is_preference_question_word_boundaried():
    """'recommend' must match as a whole word before the (a|some|any) alternation, not as a
    substring of a larger word — e.g. 'irrecommendable' contains the literal substring
    'recommend' but isn't a recommend-a/some/any advice-request shape."""
    assert is_preference_question("That approach is irrecommendable and a bad idea.") is False


def test_is_preference_question_should_i_matches_mid_sentence():
    """'should i' is word-boundaried on both sides — it must match inside a longer sentence,
    not just at the start of the question (an advice request either way)."""
    assert is_preference_question("What should I say to my boss tomorrow?") is True


def test_is_preference_question_fixture_sweep():
    """Smoke test: every question in the committed LME fixture runs through the classifier
    without error. No assertion on the outcome — this only proves the regex set never raises
    on real question text."""
    data = json.loads((FIXTURES / "longmemeval_mini.json").read_text())
    for item in data:
        result = is_preference_question(item["question"])
        assert isinstance(result, bool)


# ── PREF_QA_SYSTEM / pref_qa_messages ────────────────────────────────────


def test_pref_qa_system_pins_grounded_advice_contract():
    assert "helpfully" in PREF_QA_SYSTEM
    assert "preferences" in PREF_QA_SYSTEM
    assert PREF_QA_SYSTEM != QA_SYSTEM
    assert PREF_QA_SYSTEM != AGG_QA_SYSTEM


def test_pref_qa_messages_mirrors_qa_messages_shape():
    messages = pref_qa_messages("Relevant memories:\nfoo", "Any tips?", "2023-06-10")
    assert messages[0] == {"role": "system", "content": PREF_QA_SYSTEM}
    assert messages[1]["role"] == "user"
    assert "Today is 2023-06-10." in messages[1]["content"]
    assert "Relevant memories:\nfoo" in messages[1]["content"]
    assert "Question: Any tips?" in messages[1]["content"]


def test_pref_qa_messages_omits_empty_memory_block():
    messages = pref_qa_messages("", "Any tips?", "2023-06-10")
    assert messages[1]["content"] == "Today is 2023-06-10.\n\nQuestion: Any tips?"


# ── runner.answer_question wiring ───────────────────────────────────────


class _FakeRetriever:
    def __init__(self, mems=None, raises: Exception | None = None):
        self.mems = mems if mems is not None else []
        self.raises = raises
        self.calls: list[dict] = []

    def retrieve(self, question, limit, relationship_target):
        self.calls.append(
            {"question": question, "limit": limit, "relationship_target": relationship_target}
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


def _cfg(agg_stage: bool = False, pref_stage: bool = False) -> RunConfig:
    return RunConfig(
        bench="longmemeval_s",
        mode="memory",
        run_id="t3q-test",
        agg_stage=agg_stage,
        pref_stage=pref_stage,
    )


def test_pref_stage_true_preference_question_uses_pref_prompt_and_records_plain_response():
    char = _FakeChar(mems=[{"id": "m1", "content": "Likes documentaries about space."}])
    llm = _RecordingLLM(response="You mentioned enjoying space documentaries — try one of those.")
    q = _question("I've got some free time tonight, any documentary recommendations?")

    result = answer_question(_conversation(), char, q, _cfg(pref_stage=True), llm)

    assert len(llm.calls) == 1
    messages = llm.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": PREF_QA_SYSTEM}
    assert llm.calls[0]["kwargs"]["max_tokens"] == PREF_QA_MAX_TOKENS
    assert result["response"] == "You mentioned enjoying space documentaries — try one of those."
    assert "raw_response" not in result
    assert set(result) == {"qid", "response", "prompt_tokens_est", "memories_used", "seconds"}


def test_pref_stage_true_non_preference_question_untouched():
    """A non-preference question, even with pref_stage=True, takes the plain path unchanged:
    QA_SYSTEM, QA_MAX_TOKENS."""
    char = _FakeChar(mems=[])
    llm = _RecordingLLM(response="blue")
    q = _question("What color is my cat?")

    result = answer_question(_conversation(), char, q, _cfg(pref_stage=True), llm)

    messages = llm.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": QA_SYSTEM}
    assert llm.calls[0]["kwargs"]["max_tokens"] == QA_MAX_TOKENS
    assert result["response"] == "blue"


def test_pref_stage_false_is_byte_identical_even_for_preference_question():
    """pref_stage=False must never touch the PREF system prompt, even for a question that would
    otherwise match the preference regex — the off path is exactly today's behavior. This is the
    mutation-binding assertion: bypassing the pref branch (e.g. always taking the default path,
    or never checking `cfg.pref_stage`) collapses this test with
    `test_pref_stage_true_preference_question_uses_pref_prompt_and_records_plain_response`
    into indistinguishable behavior only if BOTH still pass — a bypass that always takes the
    default path fails the pref_stage=True test above; a bypass that always takes the pref path
    fails this one.
    """
    char = _FakeChar(mems=[{"id": "m1", "content": "Likes documentaries about space."}])
    llm = _RecordingLLM(response="Not mentioned")
    q = _question("I've got some free time tonight, any documentary recommendations?")

    result = answer_question(_conversation(), char, q, _cfg(pref_stage=False), llm)

    messages = llm.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": QA_SYSTEM}
    assert PREF_QA_SYSTEM not in messages[0]["content"]
    assert llm.calls[0]["kwargs"]["max_tokens"] == QA_MAX_TOKENS
    assert "raw_response" not in result
    assert set(result) == {"qid", "response", "prompt_tokens_est", "memories_used", "seconds"}


def test_pref_stage_false_never_uses_pref_system_across_many_questions():
    """Sweep both preference-shaped and plain questions with pref_stage=False (RunConfig's own
    default) and assert no call ever carries the PREF system text."""
    for question in POSITIVE_QUESTIONS + NEGATIVE_QUESTIONS:
        char = _FakeChar(mems=[])
        llm = _RecordingLLM()
        cfg = RunConfig(bench="longmemeval_s", mode="memory", run_id="t3q-sweep")
        assert cfg.pref_stage is False  # default

        answer_question(_conversation(), char, _question(question), cfg, llm)

        content = llm.calls[0]["messages"][0]["content"]
        assert content == QA_SYSTEM
        assert PREF_QA_SYSTEM not in content


def test_routing_precedence_aggregation_wins_over_preference():
    """A question matching BOTH the aggregation and preference regexes takes the aggregation
    path when both stages are enabled — aggregation was there first (Tier 3n)."""
    char = _FakeChar(mems=[])
    llm = _RecordingLLM(response="Event A\nEvent B\nAnswer: 2")
    # "in total"/"how many" (aggregation) AND "should i" (preference).
    q = _question("In total, how many tips should I use to fix this?")
    assert is_preference_question(q.question) is True

    result = answer_question(_conversation(), char, q, _cfg(agg_stage=True, pref_stage=True), llm)

    messages = llm.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": AGG_QA_SYSTEM}
    assert llm.calls[0]["kwargs"]["max_tokens"] == AGG_QA_MAX_TOKENS
    assert "raw_response" in result


def test_pref_stage_retrieve_exception_propagates():
    """If retrieve() raises, the exception propagates and no LLM call is made — same contract
    as the plain and aggregation paths."""
    char = _FakeChar(raises=RuntimeError("boom"))
    llm = _RecordingLLM()
    q = _question("Can you suggest a good book?")

    with pytest.raises(RuntimeError, match="boom"):
        answer_question(_conversation(), char, q, _cfg(pref_stage=True), llm)

    assert llm.calls == []
