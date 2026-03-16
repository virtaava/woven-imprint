# Chronicle

**Persistent AI Character Engine**

Characters that remember. Personalities that stay consistent. Relationships that evolve.

Chronicle is a local-first engine for creating AI characters with persistent memory,
consistent personality, and evolving relationships. Characters accumulate experiences
across interactions, maintain emotional states, and develop genuine relationships
with users and other characters.

## Status: Research Phase

Currently in deep research phase — no code yet. Architecture decisions will be
informed by academic research across 6 domains:

1. Character Science — what makes AI characters feel real
2. Memory Architecture — hierarchical memory systems (episodic, semantic, procedural)
3. Relationships & Emotions — theory of mind, trust, emotional bonds
4. Game Industry Needs — what studios actually pay for
5. Technical Approaches — vector DBs, knowledge graphs, persona dialogue
6. Competitors — what Character.AI, Inworld, Replika got right and wrong

## Use Cases

- **AI Companions** — persistent personality, remembers conversations, evolves over time
- **Game NPCs** — RPG characters with real memory that remember the player across sessions
- **Interactive Fiction** — characters that develop relationships over a branching story
- **Training Simulations** — consistent role-playing partners
- **Virtual Personalities** — AI characters with maintained identity across content

## Architecture Principles

- **Local-first** — runs entirely on local hardware (DGX Spark, consumer GPUs)
- **Model-agnostic** — works with any LLM via Ollama, vLLM, or API
- **Engine, not app** — provides the character infrastructure, not the UI
- **Memory-first** — memory architecture is the core, not the LLM
- **Zero compromise** — built to be acquired, not to ship fast

## Hardware Target

Primary: NVIDIA DGX Spark (128GB unified, GB10, ARM64)
Secondary: Any system with 16GB+ RAM and Ollama

## License

TBD — will be determined based on business model research.

## Directory Structure

```
chronicle/
  src/              — Core engine source
  tests/            — Test suite
  docs/
    architecture/   — Architecture decisions and design docs
    research/       — Research findings and papers
    api/            — API documentation
  research/         — Raw research data
  config/           — Configuration templates
```
