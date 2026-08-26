"""Guard test: all four Tier 2 switches off must reproduce pre-Tier-2 (legacy) shapes.

Switches under guard:
- relationship.dynamics       -> RelationshipModel._update_legacy path
- memory.relevance_gate       -> retrieval scores every active memory unconditionally
- context.facts_block         -> no "What you currently know" block in the prompt
- Character.unified_assessment -> sequential per-subsystem bookkeeping (no bookkeeping LLM call)
"""

from tests.helpers import make_test_engine
from woven_imprint.config import get_config


def test_all_tier2_switches_off_matches_legacy_shapes():
    cfg = get_config()
    cfg.relationship.dynamics = False
    cfg.memory.relevance_gate = False
    cfg.context.facts_block = False
    try:
        engine = make_test_engine()
        char = engine.create_character("Ada")
        char.background = False
        char.parallel = False
        char.enforce_consistency = False
        char.unified_assessment = False
        char.chat("hello there friend", user_id="toni")

        rel = char.relationships.get("toni")
        assert (
            rel["type"] == "stranger"
            and rel["key_moments"] == []
            and not (rel.get("state") or {}).get("updates")
        )
        assert "What you currently know" not in char.last_chat_messages[1]["content"]
        assert char.facts.count() == 0
    finally:
        cfg.relationship.dynamics = True
        cfg.memory.relevance_gate = True
        cfg.context.facts_block = True
