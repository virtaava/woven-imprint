# Woven Imprint

**Persistent AI Character Engine**

Characters that remember. Personalities that stay consistent. Relationships that evolve.

Woven Imprint is a local-first engine for creating AI characters with persistent memory,
consistent personality, and evolving relationships. Characters accumulate experiences
across interactions, maintain emotional states, and develop genuine bonds with users
and other characters.

Every interaction leaves an imprint. Every memory is woven into who the character becomes.

## Status: Phase 1 (Core Engine)

Architecture decisions informed by 471 academic papers across 6 domains.
Core memory engine, retrieval, persona model, and relationship tracking implemented.

## Use Cases

- **AI Companions** — persistent personality, remembers your conversations, evolves
- **Game NPCs** — RPG characters with real memory across sessions
- **Interactive Fiction** — characters that develop relationships over branching stories
- **Training Simulations** — consistent role-playing partners
- **Virtual Personalities** — AI characters with maintained identity

## Architecture Principles

- **Local-first** — runs on local hardware, no cloud dependency
- **Model-agnostic** — works with any LLM (Ollama, vLLM, API)
- **Engine, not app** — provides character infrastructure via API/SDK
- **Memory-first** — memory architecture is the core product
- **Zero compromise** — built to be acquired

## Quick Start

```python
from woven_imprint import Engine

engine = Engine("characters.db")

alice = engine.create_character(
    name="Alice",
    birthdate="1998-03-15",
    persona={
        "backstory": "A sharp-witted detective who left the force after her partner's death.",
        "personality": "witty, skeptical, secretly lonely",
        "speaking_style": "clipped sentences, dark humor, avoids emotional topics",
        "occupation": "private investigator",
    },
)

response = alice.chat("Hey Alice, how's the case going?")
print(response)

# Memory persists across sessions
alice.end_session()
response = alice.chat("Remember the harbor case we discussed?")

# Character reflects on experiences
alice.reflect()

# Export full character state
alice.export("alice_backup.json")
```

## License

MIT
