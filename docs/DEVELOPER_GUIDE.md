# Developer Guide

For Python developers, LLM practitioners, and integrators.

## Install

```bash
pip install woven-imprint           # core (requires only `requests`)
pip install woven-imprint[openai]    # + OpenAI/Azure/vLLM backend
pip install woven-imprint[anthropic] # + Anthropic Claude backend
pip install woven-imprint[mcp]       # + MCP server for IDE integration
pip install woven-imprint[pdf]       # + PDF knowledge file extraction
pip install woven-imprint[demo]      # + React demo UI
pip install woven-imprint[all]       # everything
```

From source:
```bash
git clone https://github.com/virtaava/woven-imprint.git
cd woven-imprint
pip install -e ".[dev]"
```

## LLM Backends

### Config-driven provider selection (recommended)

Since v0.4.0, all entry points (CLI, UI, MCP server, API server) use the configured provider
automatically. Set the provider in `~/.woven_imprint/config.yaml` or via environment variables:

```yaml
# config.yaml
llm:
  llm_provider: openai          # ollama (default), openai, anthropic
  embedding_provider: openai    # ollama (default), openai
  model: gpt-4o-mini
  embedding_model: text-embedding-3-small
  api_key: sk-...
```

```bash
# Or via environment variables
export WOVEN_IMPRINT_LLM_PROVIDER=openai
export WOVEN_IMPRINT_API_KEY_LLM=sk-...
```

The factory functions `create_llm()` and `create_embedding()` dispatch based on config:

```python
from woven_imprint.providers import create_llm, create_embedding

llm = create_llm()        # uses configured provider
embedding = create_embedding()
engine = Engine(llm=llm, embedding=embedding)
```

### Ollama (default)

```bash
ollama pull llama3.2 && ollama pull nomic-embed-text
```

```python
from woven_imprint import Engine
from woven_imprint.llm.ollama import OllamaLLM
from woven_imprint.embedding.ollama import OllamaEmbedding

engine = Engine(
    llm=OllamaLLM(model="llama3.2", num_ctx=8192),
    embedding=OllamaEmbedding(model="nomic-embed-text"),
)
```

### OpenAI

```python
from woven_imprint.llm import OpenAILLM
from woven_imprint.embedding import OpenAIEmbedding

engine = Engine(
    llm=OpenAILLM(model="gpt-4o-mini"),
    embedding=OpenAIEmbedding(model="text-embedding-3-small"),
)
```

Environment variables:
- **Mac/Linux**: `export OPENAI_API_KEY=sk-...`
- **Windows**: `$env:OPENAI_API_KEY = "sk-..."`

### Anthropic Claude

```python
from woven_imprint.llm import AnthropicLLM

engine = Engine(
    llm=AnthropicLLM(model="claude-sonnet-4-6"),
    embedding=OllamaEmbedding(),  # Claude has no embedding API
)
```

### Any OpenAI-compatible endpoint (vLLM, llama.cpp, LiteLLM)

```python
engine = Engine(
    llm=OpenAILLM(model="my-model", base_url="http://localhost:8000/v1", api_key="not-needed"),
    embedding=OllamaEmbedding(),
)

### Gemma edge adapter

Use this when the actual Gemma runtime lives outside Python, for example behind a
Google AI Edge / MediaPipe bridge that exposes local HTTP endpoints.

```python
from woven_imprint import Engine
from woven_imprint.llm import GemmaEdgeLLM

engine = Engine(
    llm=GemmaEdgeLLM(
        model="gemma-3n",
        base_url="http://127.0.0.1:8788",
    ),
)
```

Expected adapter endpoints:

- `POST /generate`
- `POST /generate_json`
```

## Python API

```python
from woven_imprint import Engine

with Engine("characters.db") as engine:
    # Create
    char = engine.create_character(
        name="Marcus",
        birthdate="1995-08-22",
        persona={
            "backstory": "A blacksmith who lost his wife two years ago.",
            "personality": "gruff but kind, dry humor",
            "speaking_style": "short sentences, working-class dialect",
        },
    )

    # Chat (memories and relationships update automatically)
    response = char.chat("I need a sword.", user_id="player_1")

    # Inspect
    print(char.emotion.mood)
    print(char.relationships.describe("player_1"))
    print(char.recall("sword", limit=3))

    # Lifecycle
    char.reflect()                    # generate inner reflection
    char.consolidate()                # compress buffer → core memories
    char.evolve()                     # detect personality growth

    # Session management
    session_id = char.start_session()       # start new session
    char.resume_session(session_id)         # resume a previous session
    summary = char.end_session()            # end session, returns summary

    # Sessions are conversation boundaries — not memory boundaries.
    # The character remembers everything across sessions. Ending a session
    # generates a summary that becomes a core memory.

    char.export("marcus.json")        # full portable export
```

## Background Bookkeeping — `flush()` / `close()` discipline

Since Phase A, `character.background` defaults to `true`: `chat()` and
`chat_stream()` return as soon as generation (and consistency checking) is
done. Emotion assessment, narrative-arc tracking, fact extraction, and
relationship updates run afterward on a per-character daemon thread, so the
caller isn't blocked on 3-4 extra LLM calls it doesn't need the result of
immediately.

This means state that bookkeeping writes — `char.emotion`, extracted core
memories, relationship deltas — may not be visible **immediately** after
`chat()` returns. Two methods manage this:

```python
char.flush(timeout=None)   # block until queued bookkeeping finishes
char.close()               # flush(timeout=10), then stop the worker thread
```

Call `flush()` before reading anything bookkeeping writes to, if you need
it up to date right now — e.g. before `char.recall(...)`, before printing
`char.emotion.mood`, or in tests that assert on post-chat state:

```python
char.chat("I got the job!", user_id="toni")
char.flush()                       # ensure emotion/relationship updates landed
print(char.emotion.mood)           # now reliably reflects this turn
```

`Character.end_session()` calls `flush(timeout=30)` internally before it
reads buffer memories to build the session summary — you don't need to
flush before `end_session()` yourself.

**Always call `close()` before tearing down shared resources** the worker
still writes through — most importantly the storage connection. If you
close the DB (or exit the process) while bookkeeping is still queued, the
in-flight write can fail. `Engine` does not currently call `close()` on
your characters for you when you drop the last reference; call it
explicitly at the end of a character's lifetime (e.g. in a `finally` block,
an MCP-server shutdown hook, or when evicting a cached character):

```python
char = engine.create_character("Marcus", persona={...})
try:
    char.chat("...")
finally:
    char.close()
```

If you don't need this async behavior — e.g. deterministic tests, or a
batch job that shouldn't overlap LLM calls across turns — set
`character.background: false` in config, or `WOVEN_IMPRINT_BACKGROUND=false`.
`chat()` then blocks until bookkeeping finishes, same as pre-Phase-A.

## Streaming — `chat_stream()`

```python
for chunk in char.chat_stream("Tell me about the boathouse.", user_id="toni"):
    print(chunk, end="", flush=True)
```

`chat_stream()` yields response text chunks as the LLM generates them
(native streaming where the provider supports it — see `generate_stream()`
below). It's a generator: nothing happens until you request the first
chunk.

**Consistency-check tradeoff:** `chat()` enforces persona consistency
*before* returning — if the LLM produces a hard violation (e.g. contradicts
a hard-constraint fact), it retries with a different sampling seed and only
returns once it has a consistent response (or exhausts retries). Streaming
makes that impossible: by the time a violation could be detected, the
violating text is already in the caller's hands. So `chat_stream()` checks
**post-hoc** — after the full response has streamed, it runs the same
consistency check and records the result rather than blocking or retracting
anything:

```python
for chunk in char.chat_stream("..."):
    ...
violations = char.last_chat_metrics.get("stream_consistency_violations", 0)
if violations:
    # streamed text already shown to the user — log/flag, don't retry silently
    ...
```

Controlled by `character.consistency_stream_mode`:
- `"log"` (default) — run the post-hoc check, count violations, log a
  warning if any are found.
- `"off"` — skip the check entirely for streamed responses (saves one LLM
  call per turn if you don't need the signal).

If you need guaranteed-consistent responses and can tolerate the latency,
use `chat()` instead of `chat_stream()`.

### Provider streaming (`generate_stream`)

All `LLMProvider` implementations expose `generate_stream(messages, ...)`,
yielding text chunks. Providers with native streaming support (OpenAI,
Ollama) override it; providers without native support fall back to the
`LLMProvider` base default (call `generate()`, yield the whole response as
one chunk) — same external interface either way, just without the
incremental latency benefit.

## Metrics Sink

Every `chat()` / `chat_stream()` call populates `char.last_chat_metrics` —
a dict of per-phase timings in milliseconds (`generate_ms`,
`consistency_ms`, `retrieve_memories_ms`, `total_ms`, etc.) plus a few
non-timing fields (`message_count`, `prompt_chars`, ...). Read it directly
after any call:

```python
response = char.chat("Hello!", user_id="toni")
print(char.last_chat_metrics["total_ms"], char.last_chat_metrics["generate_ms"])
```

For continuous monitoring instead of per-call inspection, set
`character.metrics_path` (or `WOVEN_IMPRINT_METRICS_PATH`) to a file path.
Every `chat()`/`chat_stream()` call then appends one JSON line to that file:

```json
{"ts": "2026-07-11T18:40:00+00:00", "character_id": "char-abc123", "metrics": {"generate_ms": 8406.1, "consistency_ms": 4646.2, "total_ms": 12999.0, ...}}
```

It's opt-in and off by default (`metrics_path: null`) — no disk writes
unless you configure a path. Writes are best-effort: a failed write is
logged at debug level and does not raise or interrupt `chat()`.

`scripts/bench_chat.py` uses `last_chat_metrics` directly (not the sink) to
print p50/p95/max latency per phase over a scripted 12-turn conversation —
see [PERFORMANCE.md](PERFORMANCE.md) for how to run it and the current
numbers.

## Nightly Maintenance

All heavy multi-pass LLM work — consolidation, importance scoring, dedup,
belief reinforcement, contradiction sweeps, reflection, growth, callback
generation — lives in one budgeted, idempotent, headless-callable primitive:
`MaintenanceRunner` (`woven_imprint/maintenance.py`). Chat stays fast; the
character "digests" experience offline.

```python
reports = engine.run_maintenance()                     # all characters, all jobs
reports = engine.run_maintenance(character_id="char-abc123",
                                 jobs=["consolidate", "callbacks"],
                                 budget=20)
```

Each report maps job name → `{status, ...details, llm_calls, duration_ms}`,
where `status` is `ok`, `skipped` (with a `reason`), or `failed` (job
failures are caught — one bad job never aborts the run).

### Jobs (`MaintenanceRunner.DEFAULT_JOBS`, in run order)

| Job | What it does |
|---|---|
| `consolidate` | Drains an over-threshold buffer into core memories in bounded passes (clustering + LLM summaries). |
| `buffer_hygiene` | Archives stale buffer memories (older than `buffer_ttl_days`, importance ≤ `buffer_hygiene_max_importance`). No LLM calls. |
| `score_importance` | One LLM call scores a batch of still-default-importance buffer memories on a 1–10 scale. |
| `dedup` | Archives near-duplicate core observations (cosine ≥ `dedup_similarity`), keeping the more important one and reinforcing its belief. No LLM calls. |
| `reinforce` | Buffer memories semantically close to a core observation reinforce that core memory's certainty. No LLM calls. |
| `contradictions` | Same-topic-but-not-identical core pairs (similarity band below dedup) get an LLM contradiction check; the superseded memory is marked `contradicted` with certainty 0. |
| `reflect` | Triggers `char.reflect()` when enough new importance-weighted buffer material has accumulated since the last reflection. |
| `evolve` | Runs personality-growth detection (`char.evolve()`) once enough core memories exist. |
| `callbacks` | Regenerates the ready-callback queue (see next section). |

### Budget

`Budget(limit)` caps LLM calls per run — `budget.take(n)` returns `False`
when the allowance would be exceeded, and jobs skip gracefully with reason
`budget exhausted`. The default comes from `maintenance.max_llm_calls_per_run`
(env: `WOVEN_IMPRINT_MAINTENANCE_BUDGET`). In `Engine.run_maintenance()` one
budget is **shared across all characters** in the run. Non-LLM jobs
(`buffer_hygiene`, `dedup`, `reinforce`) always run to completion.

### CLI

```bash
woven-imprint maintain                            # all characters, all jobs
woven-imprint maintain --character char-abc123
woven-imprint maintain --jobs consolidate,callbacks
woven-imprint maintain --budget 20
woven-imprint maintain --json                     # machine-readable report
```

### Scheduling on mobile

The runner is deliberately WorkManager-shaped: budgeted, resumable,
idempotent (a killed run just leaves less-processed data for the next one).
On Android, schedule chunked runs under charging + idle constraints — many
small budgeted runs while the device charges overnight are equivalent to one
big run, and safe to interrupt.

## Callbacks & Proactive Initiation

Callbacks are the "she remembered my interview" feature: paraphrased,
in-character conversation hooks generated in **batch** (the `callbacks`
maintenance job, or automatically at `end_session()` — config
`maintenance.callbacks_refresh_on_session_end`) and read **instantly** from
the DB at session start. No LLM call on the hot path.

```python
# Batch (nightly / session end): one LLM call, up to callbacks_refresh_limit hooks
created = char.refresh_callbacks()

# Instant (session start): plain DB read, salience-ordered
hooks = char.get_callbacks(limit=3)
# [{"kind": "open_thread", "hook": "How did the interview go?",
#   "salience": 0.8, "freshness": "2d", "source_memory_ids": [...], ...}]
```

Four kinds: `open_thread` (unresolved thing to ask about), `callback` (warm
reference to a shared moment), `milestone` (anniversary/achievement),
`curiosity` (something the character genuinely wonders about).

**Paraphrase-never-verbatim rule**: hooks are natural in-character sentences
that draw on memories — never verbatim quotes of stored memory text.
Verbatim reads as creepy surveillance; paraphrase reads as genuine memory.
The generation prompt enforces this, and `compose_initiation` re-instructs
the model to paraphrase the hook rather than quote it.

**Proactive initiation** — the character speaks first:

```python
msg = char.compose_initiation(occasion="morning greeting")
# {"text": "Morning! I kept thinking about that interview of yours — how did it go?",
#  "callback": {...} or None}
```

`compose_initiation()` takes the top ready callback (highest salience),
weaves it into a short character-initiated message, and marks it
**consumed**. Consumed-on-use is deliberate scarcity: each hook is used at
most once, so the character never repeats "that thing you said about X"
twice. It composes a generic in-character greeting when no callbacks are
ready. Scheduling *when* to send is the app's job (a game tick, a push
notification window); the library's job is having something in-character to
say.

The ready queue is capped at `maintenance.callbacks_ready_cap` — refreshes
expire the lowest-salience overflow, so the queue stays fresh.

## World Events

`observe()` is how a deterministic game or simulation narrates ground truth
into the character's memory — no dialogue pair, no LLM generation:

```python
char.observe("It rained all day and the roof leaked.")            # lightweight
char.observe("Keeper fed you an extra portion.", user_id="keeper") # assessed
```

Two paths:

- **Lightweight** (no `user_id`): stores an `[Event]`-prefixed buffer memory
  (role `event`, importance default 0.6, source/user recorded in metadata).
  Zero LLM calls — safe to call at game-tick frequency.
- **Assessed** (`user_id` given): additionally runs an event-shaped emotion
  assessment and an LLM-assessed relationship update between the character
  and `user_id` (bounded deltas, same ±0.15 discipline as chat). Runs on the
  background worker when `character.background` is true, synchronously
  otherwise.

Contrast with `ingest(role, content)`, which replays **dialogue** that
happened outside woven-imprint (e.g. a SillyTavern conversation where a
different LLM generated the reply): it stores the turn in the conversation
buffer and memory, and runs the same fact extraction as `chat()` — but no
generation. Use `ingest()` for things said, `observe()` for things that
happened.

## Health Surface

Bookkeeping subsystems fail *soft* by design — a fact-extraction failure
logs at debug level and chat carries on. The danger is silent degradation: a
small model that starts failing extraction means the character quietly stops
learning. `health()` makes that visible:

```python
h = char.health()
# {"subsystems": {"extraction": {"success": 41, "failure": 2, "last_error": "ValueError: ..."},
#                 "emotion": {...}, "arc": {...}, "relationship": {...},
#                 "consistency": {...}, "observe": {...}, "callbacks": {...}},
#  "worker": {"alive": True, "pending": 0},   # None when background worker never started
#  "generated_at": "2026-07-13T12:00:00+00:00"}
```

Counters exist per subsystem that has run at least once: `emotion`, `arc`,
`extraction`, `relationship`, `consistency`, `observe`, `callbacks` — each
with `success`/`failure` counts and the truncated `last_error`. Counters are
in-memory per `Character` instance (reset on reload), so poll them from the
process that owns the character. Also exposed as `GET
/characters/{id}/health` on the sidecar, `/api/characters/{id}/health` on
the demo server, and the `get_health` MCP tool.

## Persona Structure

```python
persona = {
    # Shorthand (auto-classified)
    "backstory": "...",              # → hard constraint
    "personality": "...",            # → soft constraint
    "speaking_style": "...",         # → soft constraint

    # Explicit levels
    "hard": {"name": "Marcus", "species": "human"},         # never changes
    "temporal": {"location": "the forge"},                    # changes by event
    "soft": {"opinion_of_strangers": "wary at first"},       # evolves through experience
    # emergent traits form automatically through interaction
}
```

## Multi-Character Interaction

```python
from woven_imprint import Engine, interact, group_interaction

engine = Engine("world.db")
greta = engine.create_character("Greta", persona={...})
cael = engine.create_character("Cael", persona={...})

# Two characters talk
result = interact(greta, cael, situation="A stranger enters the tavern.", rounds=3)
for turn in result.turns:
    print(f"{turn.speaker}: {turn.response[:200]}")

# Group scene
results = group_interaction([greta, cael], situation="Town meeting.", rounds=2)
```

## Migration from Other Systems

```python
from woven_imprint.migrate import CharacterImporter

importer = CharacterImporter(engine)

char = importer.from_file("conversations.json")     # ChatGPT export
char = importer.from_file("card.png")                # SillyTavern card
char = importer.from_file("/path/to/claude/project") # Claude project
char = importer.from_text("You are Marcus...")        # any text

# With Custom GPT knowledge files
char = importer.from_custom_gpt(
    instructions="You are a product expert...",
    knowledge_files=["manual.pdf", "faq.txt"],
)

# Relationship baseline is auto-calculated from conversation history
print(char.relationships.describe("imported_user"))
```

## Integration

### MCP Server (Claude Desktop, Cursor, Hermes, OpenClaw)

See [MCP Setup](../examples/mcp_setup.md) for config. 24 tools available:
`list_characters`, `create_character`, `chat`, `recall`, `get_relationship`,
`reflect`, `evolve`, `new_session`, `end_session`, `consolidate`, `get_facts`,
`edit_memory`, `delete_memory`, `pin_memory`, `list_pinned`, `edit_fact`,
`retract_fact`, `get_stats`, `get_callbacks`, `observe`, `get_health`, `maintain`,
`delete_character`, `migrate_from_text`.

### OpenAI-Compatible API Proxy

```bash
woven-imprint serve --port 8650
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8650/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="marcus",  # character name = model name
    messages=[{"role": "user", "content": "I need a sword."}],
)
```

### React Demo UI

```bash
woven-imprint demo                          # localhost:7860, auto-open browser
woven-imprint demo --port 8080              # custom port
woven-imprint demo --host 0.0.0.0           # expose on all network interfaces
woven-imprint demo --no-browser             # don't open browser
```

The demo server runs on `localhost:7860` by default. Configure in `~/.woven_imprint/config.yaml`:

```yaml
server:
  demo_port: 7860
  demo_browser: true
```

Features:
- **Chat** with any character, with markdown rendering
- **Character management** — create, delete, export, import (JSON/PNG/markdown), migrate from text
- **X-Ray panel** — real-time memory feed, relationship radar, emotion, session history (collapsible, responsive)
- **Session management** — start new sessions, rename them, resume old ones
- **Provider configuration** — Ollama, OpenAI, Anthropic, DeepSeek, NVIDIA NIM, or any OpenAI-compatible API
- **Live model discovery** — queries the provider API for available models (no hardcoded lists)
- **Reflect** — trigger character self-reflection
- **Help** — click the **?** icon for the full [UI Guide](UI_GUIDE.md)

> **Note:** No model is configured by default. On first launch, click Settings to choose a provider and model.

## Configuration

All settings are managed through `~/.woven_imprint/config.yaml`:

```bash
woven-imprint config --init    # create default config with all options documented
woven-imprint config           # view current settings
```

Priority: CLI flags > environment variables > config file > built-in defaults.

See **[Configuration Reference](CONFIGURATION.md)** for all 60+ settings including
LLM providers, memory, context window, relationships, persona, server, storage, and migration options.

## CLI Reference

```bash
# Getting started
woven-imprint demo                        # React demo UI
woven-imprint chat alice                  # Chat with character

# Character management
woven-imprint create "Name"               # Create character
woven-imprint chat <name-or-id>           # Chat
woven-imprint list                        # List characters
woven-imprint stats <name-or-id>          # Character info
woven-imprint export <name-or-id>         # Export to JSON
woven-imprint export-card <name-or-id>    # Export as a SillyTavern V2 character card
woven-imprint import <path>               # Import from JSON
woven-imprint delete <name-or-id>         # Delete

# Migration
woven-imprint migrate <path>              # Auto-detect format
woven-imprint migrate --text "..."        # From text
woven-imprint migrate <path> -k f1 f2    # With knowledge files

# Server
woven-imprint serve --port 8650           # OpenAI-compatible API
woven-imprint demo --port 7860            # React demo UI (default)
woven-imprint demo --host 0.0.0.0         # Network access

# Offline maintenance (nightly batch — see "Nightly Maintenance" above)
woven-imprint maintain                    # All characters, all jobs
woven-imprint maintain --character <id>   # One character
woven-imprint maintain --jobs consolidate,callbacks --budget 20
woven-imprint maintain --json             # Machine-readable report

# Housekeeping
woven-imprint config --init               # Create default config.yaml
woven-imprint prompts                     # List registered prompts (id, version)
woven-imprint reembed <name-or-id>         # Recompute a character's memory embeddings
woven-imprint link-entities <name-or-id>   # Backfill metadata.entities on existing memories
woven-imprint update                      # Update to latest version
woven-imprint --version                   # Show version
```

### Chat Slash Commands

During `woven-imprint chat` or `demo` — everything without `/` goes to the character:

```
/help       — list commands
/stats      — memory, emotion, relationships
/reflect    — character inner reflection
/memories   — search memories (prompts for term)
/recall X   — search memories for "X"
/quit       — end session and exit
```
