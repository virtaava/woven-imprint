"""Tier 3e: `_build_context` decouples render order from shedding priority.

The Tier 3d fix wave (68519fd) reordered `optional_parts` to put facts and
memories first. That single list served two purposes at once — it was both
the render order (what position each section ends up at in the volatile
block) and the shedding priority (which sections survive a tight budget).
Measurement (2026-09-01, plus-t3eA/A2/B/C) showed the render-order half of
that change cost ~0.10 cognitive cue-linkage on the chat path: memories
landing away from the end of the prompt (no longer nearest the user
message) hurt the model's ability to link retrieved facts to the current
turn. The shedding-priority half (facts/memories survive over the softer
emotion/arc/relationship framing) was not implicated and is kept.

These tests pin both halves independently so a future change can't
re-couple them by accident.
"""

from tests.helpers import make_test_engine


def _char():
    engine = make_test_engine()
    char = engine.create_character("Ada")
    char.background = False
    char.parallel = False
    char.enforce_consistency = False
    return engine, char


def _make_all_sections_present(char, user_id="toni"):
    """Force every optional section (emotion, arc, relationship, facts,
    memories) to render non-empty text, and return (memories, rel_context)
    for passing into `_build_context`."""
    # Emotion: default state is "neutral" -> describe() is "" (dropped).
    char.emotion.mood = "happy"
    char.emotion.intensity = 0.9
    # Arc: default SETUP phase already renders non-empty text.
    assert char.arc.describe()

    # Facts: _format_facts_block reads from char.facts, keyed by user_id.
    char.facts.add(
        subject="user",
        predicate="likes",
        object="sailing",
        statement="The visitor likes sailing.",
        user_id=user_id,
    )

    # Memories: passed straight into _build_context as the `memories` arg.
    m = char.memory.add("The visitor mentioned a trip to the harbor.", tier="core")
    memories = [m]

    rel_context = "You trust the visitor quite a bit."
    return memories, rel_context


def test_render_order_is_pre_tier3d_with_memories_last():
    """With a generous budget, everything is included, and the assembled
    volatile block orders sections emotion < arc < relationship < facts <
    memories — memories immediately before the closing/user message, per
    the measured-good order (scratch/t3eC, plus-t3eC 0.392 vs 0.289)."""
    engine, char = _char()
    memories, rel_context = _make_all_sections_present(char)

    messages = char._build_context("hi", memories, rel_context, user_id="toni")
    volatile = next(m["content"] for m in messages[1:] if m["role"] == "system")

    emotion_pos = volatile.index("You are currently feeling")
    arc_pos = volatile.index("The story is in its early stages")
    relationship_pos = volatile.index("You trust the visitor quite a bit.")
    facts_pos = volatile.index("What you currently know about toni")
    memories_pos = volatile.index("Your relevant memories:")

    assert emotion_pos < arc_pos < relationship_pos < facts_pos < memories_pos


def test_shedding_priority_keeps_facts_and_memories_under_pressure():
    """With a tight budget, facts and memories (real retrieved content) must
    survive while emotion/arc/relationship (softer framing) are dropped —
    the Tier 3d shedding priority, unchanged by the render-order fix."""
    engine, char = _char()
    memories, rel_context = _make_all_sections_present(char)

    # Derive the budget from actual part sizes so unrelated boilerplate
    # changes can't flip this test (review 2026-09-01). We want: base +
    # facts + memories fit, and the leftover slack is smaller than the
    # smallest soft section (emotion/arc/relationship), so none of them fits.
    facts_part = "\n\n" + char._format_facts_block("toni", set())
    mems_part = "\n\nYour relevant memories:\n" + char._format_memories(memories)
    emotion_part = "\n\n" + char.emotion.describe()
    arc_part = "\n\n" + char.arc.describe()
    rel_part = "\n\n" + rel_context
    # Base = everything the builder always keeps: probe with an over-generous
    # budget and subtract the optional parts we know are present.
    char._context.budget.total = 100_000
    probe = char._build_context("hi", memories, rel_context, user_id="toni")
    probe_volatile = next((m["content"] for m in probe[1:] if m["role"] == "system"), "")
    base_len = len(probe_volatile) - sum(
        len(x) for x in (facts_part, mems_part, emotion_part, arc_part, rel_part)
    )
    min_soft = min(len(emotion_part), len(arc_part), len(rel_part))
    slack = max(4, min_soft - 8)
    # _build_context also counts the system prompt and user message toward
    # base size; measure that overhead from the probe's budget math instead of
    # guessing: give exactly enough for volatile-side content + slack, scaled.
    sys_len = len(probe[0]["content"]) if probe and probe[0]["role"] == "system" else 0
    user_len = len("hi")
    budget_chars = sys_len + user_len + base_len + len(facts_part) + len(mems_part) + slack
    char._context.budget.total = budget_chars // 4
    messages = char._build_context("hi", memories, rel_context, user_id="toni")
    volatile = next((m["content"] for m in messages[1:] if m["role"] == "system"), "")

    assert "What you currently know about toni" in volatile
    assert "Your relevant memories:" in volatile
    assert "You are currently feeling" not in volatile
    assert "The story is in its early stages" not in volatile
    assert "You trust the visitor quite a bit." not in volatile


def test_memories_progressive_halving_still_works():
    """When memories alone don't fit `remaining`, the block is halved
    (not dropped outright) until it fits, or only one memory remains."""
    engine, char = _char()
    for i in range(20):
        char.memory.add(
            f"Filler memory number {i} about errands and the weather on the coast.",
            tier="core",
        )
    memories = char.memory.get_all(tier="core")
    assert len(memories) == 20

    # Budget: enough for base + a few memories, not all 20.
    char._context.budget.total = 300  # -> 1200 chars

    messages = char._build_context("hi", memories, "", user_id="toni")
    volatile = next((m["content"] for m in messages[1:] if m["role"] == "system"), "")

    assert "Your relevant memories:" in volatile
    rendered_count = volatile.count("Filler memory number")
    assert 0 < rendered_count < len(memories)
