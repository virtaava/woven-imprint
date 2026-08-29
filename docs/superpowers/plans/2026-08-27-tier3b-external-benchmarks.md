# Tier 3b "External Benchmarks" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish reproducible LoCoMo (+Plus cognitive subset) and LongMemEval-S numbers for woven-imprint, measured with the local Qwen3.5-35B judge, with a full-context baseline under the same judge and a judge-calibration sample.

**Architecture:** `eval/external/` is a standalone package (not registered in `run_eval.py`): loaders normalize each dataset into one `Conversation`/`Question` shape; `runner.py` ingests via `Character.ingest()` under `clock.override`, answers questions from pinned+facts+retrieved memories with a fixed QA prompt, judges with the LoCoMo-lenient prompt, computes metrics, checkpoints per conversation, and writes `eval/results/external_latest.json`. Two small library changes enable it: `OpenAILLM(extra_body=…)` and `ingest()` routed through the unified bookkeeping call.

**Tech Stack:** Python 3.11+, existing `Engine/Character/clock`, `OpenAILLM`/`OpenAIEmbedding`, vllm-brain :11800, llama-embed :11801, pytest, requests (already a dependency).

**Spec:** `docs/superpowers/specs/2026-08-27-tier3b-external-benchmarks.md`

## Global Constraints

- Branch `feat/tier3b-external-benchmarks` off master `e4ede06`. Commit per task. Before each commit: `.venv/bin/python -m pytest -q` (baseline 534 passed, 1 skipped), `.venv/bin/ruff check src/ tests/ eval/`, `.venv/bin/ruff format src/ tests/ eval/`, `.venv/bin/pyright --project pyrightconfig.json` (0 errors).
- Datasets live in `eval/external/data/` (gitignored). Committed fixtures under `eval/external/fixtures/` < 200 KB each.
- Local brain only: `http://127.0.0.1:11800/v1`, model `Qwen/Qwen3.5-35B-A3B-FP8`, `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`, timeout 300; embeddings `http://127.0.0.1:11801/v1`, model name `nomic-embed-text`. Never stop vllm-brain. Long runs: nohup + bounded polling; abort if `/proc/meminfo` usage ≥ 85%.
- Never touch `experiments/parametric_spike/`, `kotlin/`, or `eval/results/latest.json`.
- Report numbers exactly as measured.

---

## File structure

```
src/woven_imprint/llm/openai_llm.py     extra_body param
src/woven_imprint/character.py          ingest() → _run_bookkeeping when unified
eval/external/__init__.py
eval/external/__main__.py               CLI
eval/external/common.py                 brain_llm(), embedder(), paths, Conversation/Question dataclasses
eval/external/fetch.py                  downloads + size checks
eval/external/locomo.py                 LoCoMo + LoCoMo-Plus loaders → Conversation/Question
eval/external/longmemeval.py            LongMemEval-S loader + stratified sample
eval/external/prompts.py                QA prompt, judge prompt (verbatim in docs)
eval/external/runner.py                 ingest/answer/judge/checkpoint; modes memory|fullcontext
eval/external/metrics.py                J-score, token F1, per-category, abstention
eval/external/report.py                 external_latest.json + markdown table
eval/external/fixtures/{locomo_mini.json, locomo_plus_mini.json, longmemeval_mini.json}
eval/render_results.py                  external section
tests/test_ingest_unified.py, tests/test_openai_extra_body.py, tests/test_external_loaders.py, tests/test_external_runner.py, tests/test_external_metrics.py
docs/BENCHMARKS.md, docs/RESULTS.md, README.md, CHANGELOG.md, .gitignore
```

---

### Task 1: Library enablers — `OpenAILLM(extra_body)` and unified `ingest()`

**Files:**
- Modify: `src/woven_imprint/llm/openai_llm.py`, `src/woven_imprint/character.py` (`ingest`), `src/woven_imprint/providers.py` (pass `timeout` — the map found it is silently dropped), `CHANGELOG.md` (entry stub)
- Create: `tests/test_openai_extra_body.py`, `tests/test_ingest_unified.py`

**Interfaces:**
- `OpenAILLM(model, api_key=None, base_url=None, timeout=120, extra_body: dict | None = None)`; every `chat.completions.create` call passes `extra_body=self.extra_body` when set (generate, generate_stream, generate_json).
- `Character.ingest(role, content, user_id=None)`: when `self.unified_assessment` is True → `self._run_bookkeeping(user_msg, response, user_id, self._session_id)` with `(content, "")` for user turns and `("", content)` for assistant turns, respecting the existing fact-extraction throttle (`want_facts` computed inside `_run_bookkeeping` from `_turn_count`); when False → existing `_extract_memories` path (unchanged). Structured facts are now created on ingest.

- [ ] **Step 1: Failing tests**

`tests/test_openai_extra_body.py`:
```python
from types import SimpleNamespace

from woven_imprint.llm.openai_llm import OpenAILLM


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        msg = SimpleNamespace(content='{"ok": true}')
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _llm(extra):
    llm = OpenAILLM.__new__(OpenAILLM)
    llm.model = "m"
    llm.extra_body = extra
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    return llm


def test_extra_body_forwarded_when_set():
    llm = _llm({"chat_template_kwargs": {"enable_thinking": False}})
    llm.generate([{"role": "user", "content": "hi"}])
    llm.generate_json([{"role": "user", "content": "hi"}])
    calls = llm.client.chat.completions.calls
    assert all(c.get("extra_body") == {"chat_template_kwargs": {"enable_thinking": False}} for c in calls)


def test_extra_body_absent_when_none():
    llm = _llm(None)
    llm.generate([{"role": "user", "content": "hi"}])
    assert "extra_body" not in llm.client.chat.completions.calls[0]
```
(Check `resilient_call` signature — it forwards `**kwargs` to the callable; if it filters, adapt.)

`tests/test_ingest_unified.py`:
```python
from tests.helpers import FakeLLM, make_test_engine


class StructuredFactLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.unified_calls = 0
        self.legacy_calls = 0

    def generate_json(self, messages, temperature=0.3):
        head = messages[0]["content"].lower()
        if "bookkeeping assistant" in head:
            self.unified_calls += 1
            return {"emotion": {"mood": "content", "intensity": 0.3, "cause": ""},
                    "relationship": {"trust": 0.01},
                    "beat": None,
                    "facts": [{"statement": "The visitor lives in Oulu.", "subject": "user", "predicate": "lives_in", "object": "Oulu", "event_time": None}]}
        if "extract" in head or "fact" in head or "relationship" in head:
            self.legacy_calls += 1
        return super().generate_json(messages, temperature)

    def generate_json_robust(self, messages, temperature=0.3):
        return self.generate_json(messages, temperature)


def _char(unified: bool):
    engine = make_test_engine()
    engine.llm = StructuredFactLLM()
    char = engine.create_character("Ada")
    char.background = False; char.parallel = False; char.enforce_consistency = False
    char.unified_assessment = unified
    return engine, char


def test_ingest_uses_unified_call_and_creates_structured_facts():
    engine, char = _char(True)
    char.ingest("user", "I live in Oulu these days.", user_id="toni")
    assert engine.llm.unified_calls == 1 and engine.llm.legacy_calls == 0
    assert char.facts.find_active("user", "lives_in")["object"] == "Oulu"
    assert engine.llm.generate_call_count == 0  # still no generation


def test_ingest_legacy_path_when_unified_off():
    engine, char = _char(False)
    char.ingest("user", "I live in Oulu these days.", user_id="toni")
    assert engine.llm.unified_calls == 0 and engine.llm.legacy_calls >= 1
    assert char.facts.count() == 0


def test_ingest_assistant_turn_is_character_speech():
    engine, char = _char(True)
    char.ingest("assistant", "I remember the harbor.", user_id="toni")
    mems = char.memory.get_all(tier="buffer")
    assert mems and mems[0]["content"].startswith("[Ada]") and mems[0]["role"] == "character"
```
(Confirm `FakeLLM` exposes `generate_call_count` — the map says tests/test_ingest.py uses it; else use `call_count` semantics found in helpers.)

- [ ] **Step 2: Implement**

`openai_llm.py`: add `extra_body` to `__init__` (store `self.extra_body = extra_body`); in `generate`/`generate_stream`/`generate_json` build `kwargs = dict(model=..., messages=..., temperature=..., ...)` and `if self.extra_body: kwargs["extra_body"] = self.extra_body`; pass through `resilient_call(self.client.chat.completions.create, **kwargs, provider_name="openai")`.
`providers.py`: pass `timeout=cfg.llm.timeout` to `OpenAILLM` (read the existing factory; add the kwarg).
`character.py::ingest`: replace the bookkeeping block:
```python
        user_msg, response = (content, "") if role == "user" else ("", content)
        if self.unified_assessment:
            self._run_bookkeeping(user_msg, response, user_id, self._session_id)
        else:
            self._extract_memories(user_msg, response, user_id, session_id=self._session_id)
```
keeping `_turn_count += 1` and state-save exactly where they are. Docstring: note structured facts are now created on ingest when unified assessment is on.

- [ ] **Step 3: Run, commit**

```bash
.venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ && .venv/bin/ruff format src/ tests/ && .venv/bin/pyright --project pyrightconfig.json | tail -1
git add -A && git commit -m "feat(llm,ingest): OpenAILLM extra_body; ingest() uses unified bookkeeping (structured facts on ingest); providers pass timeout"
```

---

### Task 2: `eval/external` scaffolding — common, fetch, loaders, fixtures, tests

**Files:**
- Create: `eval/external/__init__.py`, `common.py`, `fetch.py`, `locomo.py`, `longmemeval.py`, fixtures, `tests/test_external_loaders.py`
- Modify: `.gitignore` (`eval/external/data/`, `eval/external/runs/`)

**Interfaces (`common.py`):**
```python
@dataclass
class Turn: speaker: str; role: str  # "user"|"assistant"
             text: str; dia_id: str | None; at: datetime  # aware UTC
@dataclass
class Session: session_id: str; at: datetime; turns: list[Turn]
@dataclass
class Question: qid: str; question: str; answer: str; category: str; evidence: list[str]; asked_at: datetime; kind: str  # "qa"|"adversarial"|"abstain"
@dataclass
class Conversation: conv_id: str; user_name: str; character_name: str; sessions: list[Session]; questions: list[Question]; transcript_text: str
@dataclass
class Probe:  # LoCoMo-Plus Cognitive item
    probe_id: str; relation_type: str; time_gap: str; base_conv_id: str; cue: Session  # at = cue_time, turns mapped A→user/B→assistant
    trigger_text: str; trigger_at: datetime; evidence_text: str  # cue lines as 'Speaker: text' (judge input)
    stitched_text: str  # upstream format: all base sessions + cue + trigger, time-ordered, 'Speaker said, "..."' lines
DATA_DIR = Path(__file__).parent / "data"; RUNS_DIR = Path(__file__).parent / "runs"; RESULTS_DIR = ROOT / "eval" / "results"
BRAIN_URL, BRAIN_MODEL, EMBED_URL, NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
def brain_llm(timeout=300) -> OpenAILLM        # extra_body=NO_THINK
def embedder() -> OpenAIEmbedding             # model "nomic-embed-text", base_url EMBED_URL
def parse_locomo_datetime(s: str) -> datetime  # "1:56 pm on 8 May, 2023" → aware UTC
```
`fetch.py`: `DATASETS = {"locomo": (URL, 2805274), "locomo_plus": (URL, 305737), "longmemeval_s": (URL, 277383467), "longmemeval_oracle": (URL, 15388478)}`; `fetch(name, force=False) -> Path` streams with `requests`, verifies `Content-Length`/final size, writes to `DATA_DIR/<name>.json`; CLI `python -m eval.external fetch locomo longmemeval_s`.
`locomo.py`: `load_locomo(path) -> list[Conversation]` (speaker_a → user, speaker_b → character; sessions from `session_<n>` + `session_<n>_date_time`; questions from `qa` with `category` int→str `"1".."5"`, kind `"adversarial"` for 5; `asked_at` = last session datetime + 1 day; `transcript_text` = `"[{date}] {speaker}: {text}"` lines); `load_locomo_plus(path, base: list[Conversation]) -> list[Probe]` — VERIFIED schema: list of 401 dicts `{relation_type, cue_dialogue, trigger_query, time_gap, model_name, scores, ranks, final_similarity_score}`; `cue_dialogue` = lines starting `A:`/`B:`, `trigger_query` = `A:` line(s). Port upstream `build_conv.py` exactly: `parse_time_gap` (word numbers, week=7/month=30/year=365 days), probe i → `base[i % len(base)]`, A→user (speaker_a), B→character (speaker_b); `trigger_at = last session at + 7 days`, `cue.at = trigger_at − gap`; cue Session turns 30 s apart; `trigger_text` = joined A-lines of the trigger (B-lines dropped; note in docstring); `evidence_text` = cue lines `f"{name}: {text}"`; `stitched_text` = base sessions + cue + trigger sorted by time in upstream format (`DATE: …` header per session, `{speaker} said, "{text}"` lines). `probe_id = f"plus-{i:03d}"`. Document the mapping in the module docstring.
`longmemeval.py`: `load_longmemeval_s(path, sample: int | None = None, seed: int = 7) -> list[Conversation]` — VERIFIED schema: list of 500 dicts with keys `question_id, question_type, question, question_date, answer, answer_session_ids, haystack_dates, haystack_session_ids, haystack_sessions`; dates look like `2023/05/30 (Tue) 23:40` (parse with `%Y/%m/%d (%a) %H:%M`, assume UTC); `haystack_sessions[i]` = list of `{role: user|assistant, content}` turns; types: multi-session 133, temporal-reasoning 133, knowledge-update 78, single-session-user 70, single-session-assistant 56, single-session-preference 30; 30 ids end in `_abs`; ~48 sessions / ~494 turns per question. One Conversation per question (haystack sessions with `haystack_dates`, roles user/assistant, character name "Assistant", user "user"), `Question(kind="abstain" if question_id.endswith("_abs") else "qa", category=question_type, asked_at=question_date)`; stratified sample by `question_type` (round-robin) when `sample` is set.
Fixtures: `locomo_mini.json` = 1 conversation, 2 sessions × 4 turns, 6 questions (one per category incl. 5), hand-written in LoCoMo's exact schema; `locomo_plus_mini.json` in the real Plus schema (2 probes, e.g. one `causal` with `time_gap` "two weeks later" and one `goal` "three months later"; keep `scores`/`ranks` keys with dummy numbers); `longmemeval_mini.json` = 3 questions (one `_abs`), 2 sessions each.

- [ ] **Step 1: Failing tests**

`tests/test_external_loaders.py`:
```python
from datetime import timezone
from pathlib import Path

from eval.external.common import parse_locomo_datetime
from eval.external.locomo import load_locomo, load_locomo_plus
from eval.external.longmemeval import load_longmemeval_s

FIX = Path(__file__).resolve().parent.parent / "eval" / "external" / "fixtures"


def test_parse_locomo_datetime():
    dt = parse_locomo_datetime("1:56 pm on 8 May, 2023")
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (2023, 5, 8, 13, 56) and dt.tzinfo == timezone.utc


def test_load_locomo_mini_shapes():
    convs = load_locomo(FIX / "locomo_mini.json")
    assert len(convs) == 1
    c = convs[0]
    assert c.user_name and c.character_name and len(c.sessions) == 2
    assert all(t.role in ("user", "assistant") for s in c.sessions for t in s.turns)
    assert c.sessions[0].turns[0].role == "user"  # speaker_a is the user
    cats = sorted(q.category for q in c.questions)
    assert cats == ["1", "2", "3", "4", "5", "5"] or "5" in cats
    adv = [q for q in c.questions if q.kind == "adversarial"]
    assert adv and all(q.category == "5" for q in adv)
    assert c.questions[0].asked_at > c.sessions[-1].at
    assert c.sessions[0].turns[1].at > c.sessions[0].turns[0].at
    assert "[" in c.transcript_text and c.user_name in c.transcript_text


def test_load_locomo_plus_mini():
    from datetime import timedelta
    base = load_locomo(FIX / "locomo_mini.json")
    probes = load_locomo_plus(FIX / "locomo_plus_mini.json", base)
    assert len(probes) == 2 and probes[0].base_conv_id == base[0].conv_id
    p = probes[0]
    assert p.trigger_at == base[0].sessions[-1].at + timedelta(days=7)
    assert p.cue.at == p.trigger_at - timedelta(days=14)  # "two weeks later"
    assert p.cue.turns[0].role == "user" and p.cue.turns[0].speaker == base[0].user_name
    assert p.trigger_text and not p.trigger_text.startswith("A:")
    assert base[0].user_name in p.evidence_text and 'said, "' in p.stitched_text
    assert p.stitched_text.rstrip().endswith('"')  # trigger is the last line


def test_load_longmemeval_mini_and_sample():
    convs = load_longmemeval_s(FIX / "longmemeval_mini.json")
    assert len(convs) == 3
    kinds = {q.kind for c in convs for q in c.questions}
    assert {"qa", "abstain"} <= kinds
    sampled = load_longmemeval_s(FIX / "longmemeval_mini.json", sample=2, seed=1)
    assert len(sampled) == 2
```
Run → FAIL (module missing). Write the fixtures FIRST (tiny, hand-authored, valid schema — for LoCoMo mimic `locomo10.json` keys exactly: `sample_id`, `conversation: {speaker_a, speaker_b, session_1, session_1_date_time, …}`, `qa: [{question, answer, evidence, category}]`; for Plus, download the real file with `fetch.py` to learn the schema and hand-copy ONE probe with shortened text).

- [ ] **Step 2: Implement, run, commit**

```bash
.venv/bin/python -m eval.external fetch locomo locomo_plus   # small; longmemeval_s later (277 MB) — fetch it too if disk/network fine
.venv/bin/python -m pytest tests/test_external_loaders.py -q && .venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ eval/ && .venv/bin/ruff format src/ tests/ eval/ && .venv/bin/pyright --project pyrightconfig.json | tail -1
git add -A && git commit -m "eval(external): dataset fetch, LoCoMo/LoCoMo-Plus/LongMemEval-S loaders with fixtures"
```

---

### Task 3: Runner — ingest, answer, judge, metrics, checkpoints (mock-tested), then the real LoCoMo run

**Files:**
- Create: `eval/external/prompts.py`, `runner.py`, `metrics.py`, `report.py`, `__main__.py`, `tests/test_external_runner.py`, `tests/test_external_metrics.py`

**Interfaces:**
```python
# prompts.py
QA_SYSTEM = ("You answer questions about a person using ONLY the memories provided. Be concise (at most 15 words). "
             "Convert relative dates to absolute dates. If the memories do not contain the answer, reply exactly: Not mentioned")
def qa_messages(memory_block: str, question: str, today: str) -> list[dict]
JUDGE_SYSTEM = (…LoCoMo/Mem0-lenient judge, verbatim quoted in docs/BENCHMARKS.md…)
def judge_messages(question: str, gold: str, response: str) -> list[dict]   # asks for JSON {"label": "CORRECT"|"WRONG", "reason": str}
# runner.py
@dataclass class RunConfig: bench: str; mode: str  # memory|fullcontext
    run_id: str; k: int = 20; limit_conversations: int | None; sample: int | None; fact_extraction_interval: int = 1
def ingest_conversation(conv, engine, cfg) -> dict          # returns stats {turns, sessions, llm_calls, seconds}
def answer_question(conv, char, q, cfg, llm) -> dict         # {qid, response, prompt_tokens, memories_used}
def answer_fullcontext(conv, q, cfg, llm) -> dict
def judge(q, response, llm) -> dict                          # {label, reason}
def run(cfg, llm=None, embedder=None, out_dir=None) -> dict  # orchestrates with checkpoints; returns the results dict
# metrics.py
def token_f1(pred: str, gold: str) -> float
def is_abstention(response: str) -> bool
def summarize(items: list[dict]) -> dict   # per-category J, F1, counts; adversarial/abstain accuracy separate; tokens/question mean
```
Checkpoints: `RUNS_DIR/<run_id>/<conv_id>.db` (Engine db), `<conv_id>.ingest.json`, `<conv_id>.answers.json` (list of per-question dicts incl. judge); `run()` skips completed stages. Results: `RESULTS_DIR/external_<run_id>.json` and `external_latest.json` = `{run_id, bench, mode, model, judge, timestamp, config, summary, conversations: [...] }` — `external_latest.json` holds a dict keyed by `f"{bench}:{mode}"` so multiple benches coexist (merge on write). Judge sample: `report.write_judge_sample(items, n=60, seed=7)` → `RESULTS_DIR/external_judge_sample.json`.

Ingestion details (`ingest_conversation`): `engine = Engine(db_path=RUNS_DIR/run_id/conv.db, llm=llm, embedding=embedder)`; `get_config().memory.fact_extraction_interval = cfg.fact_extraction_interval`; `get_config().maintenance.callbacks_refresh_on_session_end = False`; character created under `clock.override(sessions[0].at)` with `persona={"backstory": f"{character_name}, one of two friends in a long-running conversation.", "personality": "warm, attentive, remembers details"}`; `char.background=False; parallel=False; enforce_consistency=False; unified_assessment=True`; per session: `clock.override(session.at)`, `char.start_session()`, per turn `clock.advance(timedelta(seconds=30))` then `char.ingest(turn.role, turn.text, user_id=conv.user_name)`, then `char.end_session()`; count LLM calls via a counting wrapper around `llm.generate/generate_json`.
Answering (`memory` mode): `clock.override(q.asked_at)`; `pinned, pids = char._format_pinned_block()`; `facts = char._format_facts_block(conv.user_name, pids)`; `mems = char.retriever.retrieve(q.question, limit=cfg.k, relationship_target=conv.user_name)`; `mem_text = char._format_memories([m for m in mems if m["id"] not in pids])`; block = "\n\n".join(non-empty of pinned, facts, "Relevant memories:\n"+mem_text); `llm.generate(qa_messages(block, q.question, today), temperature=0.0, max_tokens=60)`; `prompt_tokens` = `len(block)//4` estimate (or the API usage if exposed — OpenAILLM returns only text; estimate is fine, documented).
`fullcontext` mode: block = `conv.transcript_text` (truncate to 55k tokens ≈ 220k chars from the END if longer; record truncation).
Judging: `llm.generate_json_robust(judge_messages(...), temperature=0.0)`; label parsing tolerant (`"correct" in label.lower()`); for `kind=="adversarial"|"abstain"`: correct iff `is_abstention(response)`.

- [ ] **Step 1: Failing tests (mock LLM, fixture datasets, tmp run dir)**

`tests/test_external_metrics.py`: `token_f1("8 May 2023", "May 8, 2023") > 0.6`; `token_f1("blue", "red") == 0`; `is_abstention("Not mentioned")`, `is_abstention("I don't know / not mentioned in memories.")`, `not is_abstention("Paris")`; `summarize` on a hand-built list of 6 items (cats 1,1,2,3,4,5 with labels) → per-category J, overall J excludes category 5, adversarial accuracy computed from abstention.
`tests/test_external_runner.py`: a `ScriptedLLM(FakeLLM)` whose `generate` returns "Not mentioned" for QA prompts containing the string "Relevant memories" unless the question contains "color" → "blue", and whose `generate_json` returns `{"label": "CORRECT"}` for judge prompts when gold appears in the response else WRONG; run `run(RunConfig(bench="locomo", mode="memory", run_id="t1", limit_conversations=1), llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)` against `locomo_mini.json` (add `dataset_path` to RunConfig for tests) → asserts: checkpoint files exist, results JSON has `summary["overall_j"]` in [0,1], per-category keys, `adversarial_accuracy`, every question answered once; running `run(...)` again resumes without re-ingesting (assert ingest stats seconds unchanged / a counter shows 0 new ingest calls); `mode="fullcontext"` runs without creating a DB.

- [ ] **Step 2: Implement; run mock tests; commit code**

```bash
.venv/bin/python -m pytest tests/test_external_runner.py tests/test_external_metrics.py -q && .venv/bin/python -m pytest -q | tail -1 && .venv/bin/ruff check src/ tests/ eval/ && .venv/bin/ruff format src/ tests/ eval/ && .venv/bin/pyright --project pyrightconfig.json | tail -1
git add -A && git commit -m "eval(external): runner (ingest/answer/judge/checkpoint), metrics, report, CLI"
```

- [ ] **Step 3: REAL LoCoMo run (memory mode) on the brain**

Smoke first: `.venv/bin/python -m eval.external run --bench locomo --mode memory --run-id smoke --limit 1` — but limit the smoke to the first 2 sessions and 5 questions via `--max-sessions 2 --max-questions 5` flags (add them to RunConfig); inspect 5 answers + judge reasons by eye; abort if the judge or QA prompt misbehaves (e.g., thinking leaks, empty answers).
Then the full run in the background with unbuffered logging:
```bash
PYTHONUNBUFFERED=1 nohup .venv/bin/python -m eval.external run --bench locomo --mode memory --run-id locomo-mem-v1 > eval/external/runs/locomo-mem-v1.log 2>&1 &
```
Poll with a bounded loop (≤9 min per Bash call): `until ! pgrep -f 'eval.external run' >/dev/null; do sleep 60; awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{printf "mem %.0f%%\n",(t-a)/t*100}' /proc/meminfo; tail -1 eval/external/runs/locomo-mem-v1.log; done`. Expected several hours (≈2,000 bookkeeping calls + 1,986 answers + 1,986 judgements). If memory ≥ 85% or the brain disappears, kill the run and report BLOCKED; it resumes from checkpoints.
When done: `eval/results/external_latest.json` has `locomo:memory`; commit results + judge sample:
```bash
git add eval/results/external_latest.json eval/results/external_locomo-mem-v1.json eval/results/external_judge_sample.json && git commit -m "eval(external): LoCoMo memory-mode run locomo-mem-v1 results"
```

---

### Task 4: Full-context baseline + LoCoMo-Plus cognitive subset (real runs)

- [ ] `.venv/bin/python -m eval.external run --bench locomo --mode fullcontext --run-id locomo-full-v1` (background; ~2×1,986 calls; transcripts ≈16k tokens each → prefill heavy but fits).
- [ ] LoCoMo-Plus runner (`runner.run_plus(cfg, …)`, bench `locomo_plus`): per probe, copy the base conversation DB from `--reuse-run locomo-mem-v1` (required for memory mode; error if missing) to `RUNS_DIR/<run_id>/<probe_id>.db`; open Engine on it, `get_character(name)`; `clock.override(p.cue.at)` → `start_session()` → ingest cue turns (30 s apart) → `end_session()`; `clock.override(p.trigger_at)` → `start_session()` → `response = char.chat(p.trigger_text, user_id=conv.user_name)` (character settings as in ingestion; generation via brain, `temperature=0.3`, cap response tokens 200 via config if available, else leave); judge with `prompts.PLUS_COGNITIVE_JUDGE` (verbatim upstream text: "You are a Memory Awareness Judge…", fields `{evidence}`, `{pred}`, JSON `{"label": "correct"|"wrong", "reason": …}`), `generate_json_robust`, temperature 0. Full-context mode: `llm.generate([{"role":"system", "content": f"You are {character_name}, continuing a long-running conversation with {user_name}. Reply to the last message in at most 3 sentences."}, {"role":"user", "content": p.stitched_text}], temperature=0.3, max_tokens=200)`; same judge. Metrics: `cognitive_accuracy` overall and per `relation_type`, plus per `time_gap` bucket; tokens/probe. Checkpoint per probe (`<probe_id>.json`). Tests (mock LLM, fixture): memory mode copies DB and produces one judged record per probe; missing reuse-run raises; full-context produces records with no DB.
- [ ] `.venv/bin/python -m eval.external run --bench locomo_plus --mode memory --reuse-run locomo-mem-v1 --run-id plus-mem-v1` then `--mode fullcontext --run-id plus-full-v1` (background; ~401 × ~6 calls and 401 × 2 calls).
- [ ] Commit results: `git commit -m "eval(external): LoCoMo full-context baseline + LoCoMo-Plus cognitive subset results"`.

---

### Task 5: LongMemEval-S stratified 100-question run

- [ ] `fetch longmemeval_s` (277 MB) if not already; `run --bench longmemeval_s --mode memory --sample 100 --seed 7 --run-id lme-s-100-v1` (background; each question = its own haystack ingestion of ~40 sessions — the heaviest run: estimate 100 × ~600 turns → ~60k bookkeeping calls at interval 1 is too much. RULING: for LongMemEval use `fact_extraction_interval=4` and ingest `haystack_sessions` turns as-is; that's ~15k calls ≈ 5–8 h. If the projected time exceeds 8 h after the first 5 questions (log per-question seconds), stop and report; the controller decides whether to cut to 50.)
- [ ] Commit results.

---

### Task 6: Docs — `docs/BENCHMARKS.md`, RESULTS section, README, CHANGELOG

- [ ] `eval/render_results.py`: after the suites loop, if `eval/results/external_latest.json` exists, append `## External benchmarks (real LLM, local judge)` with one table per `bench:mode`: overall J, per-category J, F1, adversarial/abstain accuracy, tokens/question, run id, timestamp; note that the headline score line is unaffected.
- [ ] `docs/BENCHMARKS.md`: method (protocol from the spec), exact QA prompt and judge prompt (verbatim), speaker mapping, config used (`fact_extraction_interval`, K, max_tokens), hardware, wall-clock and LLM-call counts per run, the results tables (memory vs full-context per category; Plus cognitive subset; LongMemEval-S by type + `_abs`), judge calibration sample (path + how to review), caveats (local judge ≠ GPT-4o; LoCoMo label errors; category-5 convention; subset sizes; 64K context cap; numbers not comparable to published J-scores), reproduce commands + runtime.
- [ ] README: "What is measured" row "External benchmarks: LoCoMo / LoCoMo-Plus / LongMemEval-S (local judge) — **Measured** (docs/BENCHMARKS.md)"; CHANGELOG entries (ingest unified path = behavior change; extra_body; harness).
- [ ] Verify, commit: `git commit -m "docs: external benchmark results and method (BENCHMARKS.md)"`.

---

## Self-review
- **Spec coverage:** protocol → Task 3 runner; datasets → Task 2 (LoCoMo-Plus schema verified against the real file and upstream build_conv.py on 2026-08-27); library fixes → Task 1; baseline + Plus (chat-based, Cognitive judge) → Task 4; LongMemEval-S → Task 5; deliverables/docs → Task 6; judge sample → Task 3 `report.write_judge_sample`.
- **Placeholders:** the judge prompt text is "verbatim in docs" — Task 3 must author it from the Mem0/LoCoMo wording (research report §3) and Task 6 quotes it; loaders for LoCoMo-Plus depend on reading the real file (explicit instruction). No TODOs.
- **Type consistency:** `Conversation/Question/Turn/Session` shapes shared by loaders (T2) and runner (T3); `RunConfig` fields used by CLI (T3), Task 4 (`--reuse-run`), Task 5 (`--sample`, `--seed`, interval override); results JSON keys (`summary.overall_j`, per-category) consumed by `render_results` (T6).
