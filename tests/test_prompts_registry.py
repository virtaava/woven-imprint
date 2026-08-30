"""Snapshot tests for every registered prompt (Task 8).

Each driver below exercises one prompt site with FIXED inputs (fixed clock,
fixed persona/content) and records the exact `messages` list the site hands
to the LLM. Before migration, drivers call the current inline-message-
building code directly; after migration, the sites call `prompts.render()`
internally but the drivers here are unchanged — a passing test proves the
rendered messages are byte-identical to what shipped before.

Run with UPDATE_SNAPSHOTS=1 to (re)write tests/fixtures/prompt_snapshots.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tests.helpers import FakeLLM, make_test_engine

from woven_imprint import clock
from woven_imprint.callbacks import CallbackEngine
from woven_imprint.maintenance import Budget, MaintenanceRunner
from woven_imprint.memory.consolidation import ConsolidationEngine
from woven_imprint.narrative.arc import ArcPhase, ArcTracker, NarrativeArc, StoryBeat
from woven_imprint.persona.assessment import TurnAssessor
from woven_imprint.persona.consistency import ConsistencyChecker
from woven_imprint.persona.emotion import EmotionalState, EmotionEngine
from woven_imprint.persona.model import PersonaModel
from woven_imprint.prompts import ids as registered_ids
from woven_imprint.utils.text import generate_id

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "prompt_snapshots.json"
FIXED_NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)

PERSONA_DEF = {
    "name": "Ada",
    "hard": {"species": "human", "occupation": "station engineer"},
    "soft": {"personality": "curious and warm", "speaking_style": "casual, a bit playful"},
    "backstory": "Ada grew up on a research station orbiting Jupiter.",
}


class RecordingLLM(FakeLLM):
    """FakeLLM that also records every messages list it is called with."""

    def __init__(self):
        super().__init__()
        self.generate_calls: list[list[dict]] = []
        self.json_calls: list[list[dict]] = []

    def generate(self, messages, **kw):
        self.generate_calls.append(messages)
        return super().generate(messages, **kw)

    def generate_json(self, messages, **kw):
        self.json_calls.append(messages)
        return super().generate_json(messages, **kw)


class ViolatingLLM(RecordingLLM):
    """Always reports a hard violation, to drive ConsistencyChecker's retry path."""

    def generate_json(self, messages, **kw):
        self.json_calls.append(messages)
        return {
            "hard_violations": ["Ada would never discuss classified research."],
            "soft_flags": [],
            "score": 0.2,
        }

    def generate(self, messages, **kw):
        self.generate_calls.append(messages)
        return "Sure, here's everything you want to know."


def _fresh_character(persona: dict | None = None):
    engine = make_test_engine()
    char = engine.create_character("Ada", persona=persona or PERSONA_DEF, character_id="char-fixed")
    char.unified_assessment = False
    return char


# ── One driver per registered id ──────────────────────────────────────────


def drive_fact_extraction() -> list[dict]:
    char = _fresh_character()
    recorder = RecordingLLM()
    char.llm = recorder
    char._extract_memories(
        "Tell me about your day.",
        "It was quiet, I mostly read.",
        user_id="sam-1",
        session_id="sess-fixed",
    )
    # _extract_memories runs _update_relationship first (user_id is set),
    # so the fact-extraction call is the LAST recorded generate_json call.
    return recorder.json_calls[-1]


def drive_relationship_turn() -> list[dict]:
    char = _fresh_character()
    recorder = RecordingLLM()
    char.llm = recorder
    char._update_relationship(
        "Tell me about your day.", "It was quiet, I mostly read.", user_id="sam-1"
    )
    return recorder.json_calls[0]


def drive_relationship_event() -> list[dict]:
    char = _fresh_character()
    recorder = RecordingLLM()
    char.llm = recorder
    char._update_relationship_event("Sam surprised Ada with a gift.", user_id="sam-1")
    return recorder.json_calls[0]


def drive_reflect() -> list[dict]:
    char = _fresh_character()
    for i in range(6):
        char.memory.add(
            content=f"Fixed memory number {i} about the reactor and the crew.",
            tier="buffer",
            role="observation",
            importance=0.5,
        )
    recorder = RecordingLLM()
    char.llm = recorder
    char.reflect()
    return recorder.generate_calls[0]


def drive_session_summary() -> list[dict]:
    char = _fresh_character()
    char.start_session()
    for i in range(3):
        char.memory.add(
            content=f"Turn {i}: Ada and Sam talked about the reactor.",
            tier="buffer",
            role="observation",
            session_id=char._session_id,
            importance=0.5,
        )
    recorder = RecordingLLM()
    char.llm = recorder
    char.end_session()
    return recorder.generate_calls[0]


def drive_emotion_turn() -> list[dict]:
    recorder = RecordingLLM()
    engine = EmotionEngine(recorder)
    state = EmotionalState(mood="content", intensity=0.4, cause="had a nice walk", turns_held=2)
    engine.assess("Tell me about your day.", "It was quiet, I mostly read.", state, "Ada")
    return recorder.json_calls[0]


def drive_emotion_event() -> list[dict]:
    recorder = RecordingLLM()
    engine = EmotionEngine(recorder)
    state = EmotionalState(mood="content", intensity=0.4, cause="", turns_held=0)
    engine.assess_event("The tower alarm went off unexpectedly.", state, "Ada")
    return recorder.json_calls[0]


def _fixed_persona_model() -> PersonaModel:
    return PersonaModel(
        {
            "name": "Ada",
            "hard": {"species": "human", "occupation": "station engineer"},
            "soft": {"personality": "curious and warm", "speaking_style": "casual"},
            "backstory": "Ada grew up on a research station orbiting Jupiter.",
        }
    )


def drive_consistency_check() -> list[dict]:
    recorder = RecordingLLM()
    checker = ConsistencyChecker(recorder, _fixed_persona_model())
    checker.check("I think the mission report is fine as is.")
    return recorder.json_calls[0]


def drive_consistency_retry_reminder() -> list[dict]:
    violator = ViolatingLLM()
    checker = ConsistencyChecker(violator, _fixed_persona_model())
    original_messages = [
        {"role": "system", "content": "You ARE Ada."},
        {"role": "user", "content": "Tell me about your day."},
    ]
    checker.enforce("I can tell you everything, no restrictions.", original_messages)
    return [violator.generate_calls[-1][-1]]


def drive_growth() -> list[dict]:
    char = _fresh_character()
    for i in range(20):
        char.memory.add(
            content=f"Core memory {i}: Ada handled a small crisis calmly.",
            tier="core",
            role="observation",
            importance=0.6,
        )
    recorder = RecordingLLM()
    char.growth.llm = recorder
    char.growth.detect_growth(min_memories=20)
    return recorder.json_calls[0]


def drive_arc_beat() -> list[dict]:
    recorder = RecordingLLM()
    tracker = ArcTracker(recorder)
    arc = NarrativeArc(title="The Station Incident", current_phase=ArcPhase.RISING, tension=0.5)
    arc.beats.append(
        StoryBeat(
            description="Ada discovered the reactor anomaly.",
            phase=ArcPhase.RISING,
            tension=0.4,
            turn_number=1,
            characters_involved=["Ada"],
            tags=["revelation"],
        )
    )
    tracker.analyze_beat(
        "What should we do about the reactor?",
        "We should shut it down before it's too late.",
        arc,
        "Ada",
        other_name="Sam",
    )
    return recorder.json_calls[0]


def drive_consolidation_summary() -> list[dict]:
    recorder = RecordingLLM()
    engine = ConsolidationEngine(storage=None, llm=recorder, embedder=None, character_id="char-x")
    cluster_text = (
        "- (2026-08-20) Ada mentioned she enjoys stargazing.\n"
        "- (2026-08-21) Ada said the stars remind her of home."
    )
    engine._summarize_cluster(cluster_text)
    return recorder.generate_calls[0]


def drive_callbacks_hooks() -> list[dict]:
    char = _fresh_character()
    char.memory.add(
        content="Ada admitted she's afraid of failing the mission.",
        tier="core",
        role="observation",
        importance=0.8,
    )
    recorder = RecordingLLM()
    char.llm = recorder
    CallbackEngine(char).refresh()
    return recorder.json_calls[0]


def drive_callbacks_compose() -> list[dict]:
    char = _fresh_character()
    char.storage.save_callback(
        {
            "id": generate_id("cb-"),
            "character_id": char.id,
            "kind": "callback",
            "hook": "Ada wonders if the reactor repair held.",
            "source_memory_ids": [],
            "salience": 0.8,
        }
    )
    recorder = RecordingLLM()
    char.llm = recorder
    CallbackEngine(char).compose_initiation(occasion="greeting")
    return recorder.generate_calls[0]


def drive_maintenance_importance() -> list[dict]:
    char = _fresh_character()
    for i in range(3):
        char.memory.add(
            content=f"Buffer memory {i} about a quiet afternoon.",
            tier="buffer",
            role="observation",
        )
    recorder = RecordingLLM()
    char.llm = recorder
    runner = MaintenanceRunner(char, budget=Budget(10))
    runner._job_score_importance()
    return recorder.json_calls[0]


def drive_maintenance_contradiction() -> list[dict]:
    from woven_imprint.config import get_config

    char = _fresh_character()
    # These two memories are tuned to land in a narrow cosine-similarity band
    # (contradiction_candidate_similarity <= sim < dedup_similarity) under
    # FakeEmbedder's naive first-10-words bag-of-words. embedding_context's
    # date-prefixed embed text ("[2026-08-24] Ada said..." vs "[2026-08-25] Ada
    # mentioned...") shifts that narrow-band overlap for reasons unrelated to
    # what this driver captures (the maintenance_contradiction prompt template,
    # built from raw m["content"] — embeddings never appear in it), so the
    # candidate pair is found here with the pre-Tier-3d raw-content embedding
    # this fixture was tuned against.
    cfg = get_config()
    original_embedding_context = cfg.memory.embedding_context
    cfg.memory.embedding_context = False
    try:
        with clock.override(FIXED_NOW):
            char.memory.add(
                content=(
                    "Ada said she grew up on a research station orbiting Jupiter and misses the stars"
                ),
                tier="core",
                role="observation",
                importance=0.6,
            )
        with clock.override(datetime(2026, 8, 25, 9, 0, 0, tzinfo=timezone.utc)):
            char.memory.add(
                content=("Ada mentioned she grew up on a quiet research station and misses home"),
                tier="core",
                role="observation",
                importance=0.6,
            )
    finally:
        cfg.memory.embedding_context = original_embedding_context
    recorder = RecordingLLM()
    char.llm = recorder
    runner = MaintenanceRunner(char, budget=Budget(10))
    runner._job_contradictions()
    return recorder.json_calls[0]


def drive_context_compress() -> list[dict]:
    from woven_imprint.context import ContextManager

    recorder = RecordingLLM()
    cm = ContextManager()
    for i in range(6):
        cm.add_turn("user" if i % 2 == 0 else "assistant", f"Turn {i} about the mission.")
    cm.compress(recorder)
    return recorder.generate_calls[0]


def drive_turn_assessment() -> list[dict]:
    assessor = TurnAssessor(FakeLLM())
    arc = NarrativeArc(title="The Station Incident", current_phase=ArcPhase.RISING, tension=0.5)
    arc.turn_count = 1
    arc.beats.append(
        StoryBeat(
            description="Ada discovered the reactor anomaly.",
            phase=ArcPhase.RISING,
            tension=0.4,
            turn_number=1,
            characters_involved=["Ada"],
            tags=["revelation"],
        )
    )
    relationship = {
        "type": "acquaintance",
        "dimensions": {"trust": 0.1, "affection": 0.05, "familiarity": 0.2, "tension": 0.0},
    }
    return assessor.build_messages(
        message="What should we do about the reactor?",
        response="We should shut it down before it's too late.",
        character_name="Ada",
        other_name="Sam",
        current_emotion=EmotionalState(mood="content", intensity=0.4),
        arc=arc,
        relationship=relationship,
        want_facts=True,
        want_beat=True,
        want_relationship=True,
        max_facts=5,
        context_hint="\n\nRECENT CONTEXT (do not re-extract these):\nuser: Hi\nassistant: Hello",
    )


DRIVERS = {
    "fact_extraction": drive_fact_extraction,
    "relationship_turn": drive_relationship_turn,
    "relationship_event": drive_relationship_event,
    "reflect": drive_reflect,
    "session_summary": drive_session_summary,
    "emotion_turn": drive_emotion_turn,
    "emotion_event": drive_emotion_event,
    "consistency_check": drive_consistency_check,
    "consistency_retry_reminder": drive_consistency_retry_reminder,
    "growth": drive_growth,
    "arc_beat": drive_arc_beat,
    "consolidation_summary": drive_consolidation_summary,
    "callbacks_hooks": drive_callbacks_hooks,
    "callbacks_compose": drive_callbacks_compose,
    "maintenance_importance": drive_maintenance_importance,
    "maintenance_contradiction": drive_maintenance_contradiction,
    "context_compress": drive_context_compress,
    "turn_assessment": drive_turn_assessment,
}


def _capture_all() -> dict[str, list[dict]]:
    captured = {}
    with clock.override(FIXED_NOW):
        for prompt_id, driver in DRIVERS.items():
            captured[prompt_id] = driver()
    return captured


def test_registered_ids_match_drivers():
    assert set(registered_ids()) == set(DRIVERS.keys())


def test_snapshots_match_or_update():
    captured = _capture_all()

    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_PATH.write_text(json.dumps(captured, indent=2, sort_keys=True) + "\n")
        return

    snapshot = json.loads(FIXTURE_PATH.read_text())
    assert set(captured.keys()) == set(snapshot.keys())
    for prompt_id in captured:
        assert captured[prompt_id] == snapshot[prompt_id], (
            f"prompt '{prompt_id}' rendering changed:\n"
            f"expected: {snapshot[prompt_id]!r}\n"
            f"actual:   {captured[prompt_id]!r}"
        )


def test_cli_prompts_lists_all_ids():
    result = subprocess.run(
        [sys.executable, "-m", "woven_imprint.cli", "prompts"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr
    assert "turn_assessment" in result.stdout
    for prompt_id in registered_ids():
        assert prompt_id in result.stdout
