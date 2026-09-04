"""Ingest -> answer -> judge -> checkpoint runner for eval/external.

Two modes:

- ``memory``: ingests the conversation into a real ``Engine``/``Character`` (one SQLite DB per
  conversation, under ``RUNS_DIR/<run_id>/<conv_id>.db``) via ``Character.ingest()`` under
  ``clock.override``, then answers each question from the pinned block + facts block + retrieved
  memories, same as the product's real prompt-assembly path.
- ``fullcontext``: no Engine/DB at all — the question is answered directly from the
  conversation's flat transcript text (truncated from the end if it would blow the context
  window), as a baseline for what memory retrieval adds or loses.

Both modes score with the same LoCoMo/Mem0-lenient judge (``prompts.judge_messages``), except
adversarial/abstain-kind questions, which are scored by abstention detection alone (no judge
call — the "correct" answer to those questions *is* "Not mentioned").

Checkpointing: per conversation, ``<conv_id>.ingest.json`` marks ingestion done (memory mode
only) and ``<conv_id>.answers.json`` is a list of fully-judged per-question records, appended to
and rewritten after every question — a killed run loses at most one in-flight question, and
re-running ``run()`` skips whatever is already on disk.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from woven_imprint import clock
from woven_imprint.config import get_config
from woven_imprint.embedding.base import EmbeddingProvider
from woven_imprint.engine import Engine
from woven_imprint.llm.base import LLMProvider

from . import metrics, report
from .aggregation import agg_qa_messages, is_aggregation_question, parse_agg_answer
from .common import DATA_DIR, EMBED_MODEL, RESULTS_DIR, RUNS_DIR, Conversation, Probe, Question
from .common import brain_llm as _brain_llm
from .common import embedder as _make_embedder
from .locomo import load_locomo, load_locomo_plus
from .longmemeval import load_longmemeval_s
from .prompts import judge_messages, plus_judge_messages, qa_messages

# Full-context transcripts are truncated to this many characters (from the END — the most
# recent, most-likely-relevant material) before being sent as the answering prompt. This never
# actually triggers for LoCoMo (max transcript ~117,075 chars, well under this limit).
# LongMemEval-S transcripts run ~500K chars — well over this limit — but LongMemEval-S is NOT
# run in fullcontext mode, so its transcripts never hit this truncation path either.
FULLCONTEXT_CHAR_LIMIT = 220_000

QA_MAX_TOKENS = 60
# Tier 3n aggregation-stage QA calls (cfg.agg_stage): the enumerate-then-answer contract needs
# room to list every matching item before the final "Answer:" line.
AGG_QA_MAX_TOKENS = 700
QA_TEMPERATURE = 0.0
JUDGE_TEMPERATURE = 0.0
_LOG_EVERY = 20

# LoCoMo-Plus full-context response generation (per the spec's "LoCoMo-Plus answering" protocol).
PLUS_FULLCONTEXT_TEMPERATURE = 0.3
PLUS_FULLCONTEXT_MAX_TOKENS = 200


def _load_locomo(path: Path, cfg: "RunConfig") -> list[Conversation]:
    return load_locomo(path)


def _load_longmemeval(path: Path, cfg: "RunConfig") -> list[Conversation]:
    return load_longmemeval_s(path, sample=cfg.sample, seed=cfg.seed)


_LOADERS = {
    "locomo": _load_locomo,
    "longmemeval_s": _load_longmemeval,
}


@dataclass
class RunConfig:
    bench: str
    mode: str  # "memory" | "fullcontext"
    run_id: str
    k: int = 20
    limit_conversations: int | None = None
    sample: int | None = None
    seed: int = 7
    fact_extraction_interval: int = 1
    max_sessions: int | None = None
    max_questions: int | None = None
    reuse_run: str | None = None  # bench "locomo_plus": required — base LoCoMo memory-mode run id
    dataset_path: str | Path | None = None
    # bench "locomo_plus" only: the base LoCoMo conversations file probes are stitched onto
    # (probe i -> base[i % len(base)]); defaults to DATA_DIR/"locomo.json". `dataset_path` above
    # is the probes file itself (default DATA_DIR/"locomo_plus.json") for that bench.
    base_dataset_path: str | Path | None = None
    shard: tuple[int, int] | None = (
        None  # (index, count): process conversations where i % count == index
    )
    write_results: bool = True  # shard workers pass False; the final unsharded call aggregates
    overrides: tuple[str, ...] = ()  # "section.key=value" applied to get_config() during the run
    timeout: int = 900  # seconds; passed to brain_llm(timeout=...)
    force_aggregate: bool = False  # bypass the unsharded-aggregation safety check (see run())
    # Pair each (user, assistant) turn during ingestion into one `Character.ingest_exchange()`
    # call instead of two `Character.ingest()` calls — halves bookkeeping LLM calls, since
    # ingest_exchange makes one unified-assessment call per exchange rather than one per turn.
    # `None` (the CLI's unset default) resolves in `__post_init__` to True for
    # bench=="longmemeval_s" (where the haystack is turn-by-turn dialogue and the call-count
    # savings matter most) and False otherwise (LoCoMo's speaker order isn't guaranteed
    # strict user/assistant alternation, and existing LoCoMo/LoCoMo-Plus runs should keep their
    # established per-turn ingestion behavior).
    pair_turns: bool | None = None
    # After a conversation's answers are complete and judged, delete its .db (+ -wal/-shm) —
    # keeps only the ingest.json/answers.json checkpoints. LongMemEval-S's per-question DBs
    # (~40 sessions each, one full haystack ingested) add up across 100 questions; nothing
    # downstream needs the DB once every question for that conversation is answered.
    delete_db_after_answer: bool = False
    # `bench="locomo"`/`bench="longmemeval_s"` mode="memory" only: another run under the same
    # root (sibling of this run's own run_id — see `run_root.parent`) whose per-conversation
    # `<conv>.db`/`.ingest.json` this run copies from when its own conversation directory is
    # missing both (see `_maybe_reuse_ingest`). Lets an answer-only re-run (e.g. testing new
    # `--set` config overrides) skip re-ingesting entirely instead of requiring the caller to
    # manually copy files first. Unlike `reuse_run` (locomo_plus's mandatory base-DB source),
    # this is optional and applies to `run()`'s own per-conversation loop, not `run_plus()`.
    reuse_ingest: str | None = None
    # Tier 3n: route aggregation-shaped questions (`aggregation.is_aggregation_question`)
    # through the enumerate-then-answer path in `answer_question` — per-question
    # `memory.query_expansion=3` retrieval + `aggregation.AGG_QA_SYSTEM` + a larger answer
    # token budget. Default False: a run without `--agg-stage` is byte-identical to today.
    agg_stage: bool = False

    def __post_init__(self) -> None:
        if self.pair_turns is None:
            self.pair_turns = self.bench == "longmemeval_s"


class _CountingLLM(LLMProvider):
    """Wraps an LLMProvider, counting every generate/generate_json(_robust) call.

    ``ingest_conversation`` reads the delta across its own ingestion loop so ingestion
    bookkeeping-call counts are accurate even when the same counting wrapper is reused across
    multiple conversations (answering/judging calls made after ingestion don't retroactively
    inflate an already-recorded ingest stat).
    """

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.calls = 0

    def generate(
        self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048
    ) -> str:
        self.calls += 1
        return self._inner.generate(messages, temperature=temperature, max_tokens=max_tokens)

    def generate_json(self, messages: list[dict], temperature: float = 0.3) -> dict | list:
        self.calls += 1
        return self._inner.generate_json(messages, temperature=temperature)

    def generate_json_robust(self, messages: list[dict], temperature: float = 0.3) -> dict | list:
        self.calls += 1
        return self._inner.generate_json_robust(messages, temperature=temperature)

    def generate_stream(
        self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048
    ):
        self.calls += 1
        yield from self._inner.generate_stream(
            messages, temperature=temperature, max_tokens=max_tokens
        )


def _load_json(path: Path, default: Any) -> Any:
    """Load JSON from ``path``, or ``default`` if it's missing or unreadable.

    A checkpoint file that fails to parse (truncated by a kill mid-write, or — before
    :func:`_save_json` was made atomic — a torn write racing a reader) is treated the same as a
    missing checkpoint rather than crashing the run: a warning is printed and ``default`` is
    returned, so the caller just redoes that unit of work.
    """
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"warning: {path} is not valid JSON ({exc}); treating as missing", flush=True)
        return default


def _save_json(path: Path, data: Any) -> None:
    """Write ``data`` to ``path`` via a same-directory tmp file + ``Path.replace``.

    ``Path.replace`` is atomic on POSIX (same filesystem), so a reader (this process resuming
    after a kill, or a sibling shard worker) never observes a partially-written checkpoint —
    it's either the old complete content or the new complete content, never a torn mix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def _conv_paths(run_root: Path, conv_id: str) -> dict[str, Path]:
    return {
        "db": run_root / f"{conv_id}.db",
        "ingest": run_root / f"{conv_id}.ingest.json",
        "answers": run_root / f"{conv_id}.answers.json",
    }


def _character_persona(conv: Conversation) -> dict:
    return {
        "backstory": f"{conv.character_name}, one of two friends in a long-running conversation.",
        "personality": "warm, attentive, remembers details",
    }


def _ingest_session_turns(char, turns: list, user_id: str | None, pair_turns: bool) -> int:
    """Ingest one session's turns into ``char``, advancing the clock 30s per original turn
    consumed (before ingesting, so ``created_at`` stays monotonic exactly as the un-paired
    per-turn loop did).

    When ``pair_turns``, a ``user`` turn immediately followed by an ``assistant`` turn is
    ingested as a single :meth:`Character.ingest_exchange` call (one bookkeeping LLM call for
    both sides, clock advanced 60s — 30s per turn consumed). An unpaired turn — two ``user``
    turns in a row, an ``assistant`` turn with no preceding ``user`` turn, or a trailing
    ``user`` turn at the end of the session — falls back to :meth:`Character.ingest` (clock
    advanced 30s). Returns the number of turns consumed (``== len(turns)``).
    """
    i = 0
    n = len(turns)
    while i < n:
        turn = turns[i]
        if pair_turns and turn.role == "user" and i + 1 < n and turns[i + 1].role == "assistant":
            nxt = turns[i + 1]
            clock.advance(timedelta(seconds=30 * 2))
            char.ingest_exchange(turn.text, nxt.text, user_id=user_id)
            i += 2
        else:
            clock.advance(timedelta(seconds=30))
            char.ingest(turn.role, turn.text, user_id=user_id)
            i += 1
    return n


def ingest_conversation(conv: Conversation, engine: Engine, cfg: "RunConfig") -> dict:
    """Ingest ``conv``'s sessions into ``engine``, message by message, under clock control.

    Creates the character (id = ``conv.conv_id``) and ingests every session (capped at
    ``cfg.max_sessions`` for smoke runs), advancing the clock 30s per turn so ``created_at`` is
    monotonic. When ``cfg.pair_turns`` is set, consecutive (user, assistant) turns within a
    session are ingested as a single ``Character.ingest_exchange()`` call each — see
    :func:`_ingest_session_turns`. Returns ``{turns, sessions, llm_calls, seconds, pair_turns}``.
    Idempotency (skip if already done) is the caller's responsibility via the
    ``<conv_id>.ingest.json`` checkpoint.
    """
    started = time.perf_counter()
    llm = engine.llm
    calls_before = llm.calls if isinstance(llm, _CountingLLM) else 0

    sessions = conv.sessions[: cfg.max_sessions] if cfg.max_sessions else conv.sessions
    if not sessions:
        return {
            "turns": 0,
            "sessions": 0,
            "llm_calls": 0,
            "seconds": 0.0,
            "pair_turns": bool(cfg.pair_turns),
        }

    clock.override(sessions[0].at)
    char = engine.create_character(
        conv.character_name, persona=_character_persona(conv), character_id=conv.conv_id
    )
    char.background = False
    char.parallel = False
    char.enforce_consistency = False
    char.unified_assessment = True

    turns = 0
    for session in sessions:
        clock.override(session.at)
        char.start_session()
        turns += _ingest_session_turns(char, session.turns, conv.user_name, bool(cfg.pair_turns))
        char.end_session()

    elapsed = time.perf_counter() - started
    calls_after = llm.calls if isinstance(llm, _CountingLLM) else 0
    stats = {
        "turns": turns,
        "sessions": len(sessions),
        "llm_calls": calls_after - calls_before,
        "seconds": elapsed,
        "pair_turns": bool(cfg.pair_turns),
    }
    # Tier 3d fact dedup: record how many fact-derived core inserts this
    # ingest deduped, if the counter exists (optional — older Character
    # builds won't have it; keep this harness change cheap/non-breaking).
    dedup_skipped = getattr(char, "_dedup_skipped", None)
    if dedup_skipped is not None:
        stats["dedup_skipped"] = dedup_skipped
    return stats


def answer_question(conv: Conversation, char, q: Question, cfg: "RunConfig", llm) -> dict:
    """Answer ``q`` from ``char``'s pinned block + facts block + retrieved memories.

    Tier 3n (``cfg.agg_stage``, opt-in): when ``q.question`` matches
    ``aggregation.is_aggregation_question``, retrieval runs with ``memory.query_expansion``
    temporarily forced to 3 (restored in a ``finally``, even if retrieval raises) and the QA
    call uses the enumerate-then-answer ``aggregation.AGG_QA_SYSTEM`` prompt with a larger token
    budget — the list needs room. The raw response is kept in ``raw_response``; ``response`` is
    the parsed ``Answer:`` line (``aggregation.parse_agg_answer``). Every other question, and
    every question when ``cfg.agg_stage`` is False, takes the path below unchanged.
    """
    started = time.perf_counter()
    clock.override(q.asked_at)

    pinned, pinned_ids = char._format_pinned_block()
    facts = char._format_facts_block(conv.user_name, pinned_ids)

    agg = bool(cfg.agg_stage) and is_aggregation_question(q.question)
    if agg:
        mem_cfg = get_config().memory
        prior_expansion = mem_cfg.query_expansion
        mem_cfg.query_expansion = 3
        try:
            mems = char.retriever.retrieve(
                q.question, limit=cfg.k, relationship_target=conv.user_name
            )
        finally:
            mem_cfg.query_expansion = prior_expansion
    else:
        mems = char.retriever.retrieve(q.question, limit=cfg.k, relationship_target=conv.user_name)

    filtered = [m for m in mems if m["id"] not in pinned_ids]
    mem_text = char._format_memories(filtered)

    blocks = [
        b for b in (pinned, facts, f"Relevant memories:\n{mem_text}" if mem_text else "") if b
    ]
    block = "\n\n".join(blocks)
    today = q.asked_at.date().isoformat()

    if agg:
        raw_response = llm.generate(
            agg_qa_messages(block, q.question, today),
            temperature=QA_TEMPERATURE,
            max_tokens=AGG_QA_MAX_TOKENS,
        )
        raw_response = raw_response or ""
        return {
            "qid": q.qid,
            "response": parse_agg_answer(raw_response),
            "raw_response": raw_response.strip(),
            "prompt_tokens_est": len(block) // 4,
            "memories_used": len(filtered),
            "seconds": time.perf_counter() - started,
        }

    response = llm.generate(
        qa_messages(block, q.question, today), temperature=QA_TEMPERATURE, max_tokens=QA_MAX_TOKENS
    )
    return {
        "qid": q.qid,
        "response": (response or "").strip(),
        "prompt_tokens_est": len(block) // 4,
        "memories_used": len(filtered),
        "seconds": time.perf_counter() - started,
    }


def answer_fullcontext(conv: Conversation, q: Question, cfg: "RunConfig", llm) -> dict:
    """Answer ``q`` from the conversation's flat transcript, truncated from the end if too long."""
    started = time.perf_counter()
    clock.override(q.asked_at)

    text = conv.transcript_text
    truncated = len(text) > FULLCONTEXT_CHAR_LIMIT
    block = text[-FULLCONTEXT_CHAR_LIMIT:] if truncated else text
    today = q.asked_at.date().isoformat()

    response = llm.generate(
        qa_messages(block, q.question, today), temperature=QA_TEMPERATURE, max_tokens=QA_MAX_TOKENS
    )
    return {
        "qid": q.qid,
        "response": (response or "").strip(),
        "prompt_tokens_est": len(block) // 4,
        "memories_used": 0,
        "truncated": truncated,
        "seconds": time.perf_counter() - started,
    }


def _judge_call(llm, messages: list[dict], temperature: float) -> dict:
    """``generate_json_robust`` that never raises on unparseable judge output.

    A truncated/garbled JSON reply (e.g. an over-long ``reason`` cut at ``max_tokens``) raises
    ``ValueError`` from the provider after its one retry; a benchmark shard must not die on that —
    it is recorded as an unparsed verdict (``judge_parsed`` False, counted in ``n_unparsed``).

    Only ``ValueError`` (a parse/format failure) is caught here. A transport error (connection
    drop, timeout, etc. — typically ``RuntimeError`` or an ``openai``/``requests`` exception) is
    a different failure mode: recording it as an unparsed verdict would silently mark a real
    connectivity outage as a wrong-but-scored answer instead of stopping the run so it can be
    resumed from checkpoint once the provider is back. Such errors propagate.
    """
    try:
        result = llm.generate_json_robust(messages, temperature=temperature)
    except ValueError as exc:
        return {"label": "", "reason": f"judge error: {type(exc).__name__}: {str(exc)[:200]}"}
    return (
        result if isinstance(result, dict) else {"label": "", "reason": "judge returned non-dict"}
    )


def judge(q: Question, response: str, llm) -> dict:
    """Judge ``response`` against ``q.answer``.

    Adversarial (LoCoMo category 5) and abstain (LongMemEval ``_abs``) questions are scored by
    abstention detection alone — the gold behavior *is* declining to answer, so there is nothing
    for an LLM judge to usefully compare against, and skipping the call saves one generation per
    such question. A label is still recorded for reporting symmetry with judged questions.

    ``judge_parsed`` is ``True`` for the rule-scored (no-LLM-call) path — there is nothing to
    fail to parse — and, for the LLM-judged path, ``False`` when the LLM's JSON response wasn't
    a dict or carried no non-empty ``label`` (a parse/format failure, not a WRONG verdict).
    """
    started = time.perf_counter()
    if q.kind in ("adversarial", "abstain"):
        correct = metrics.is_abstention(response)
        return {
            "label": "CORRECT" if correct else "WRONG",
            "reason": "scored by abstention rule (no judge call)",
            "correct": correct,
            "judge_parsed": True,
            "seconds": time.perf_counter() - started,
        }

    asked_on = q.asked_at.date().isoformat()
    result = _judge_call(
        llm, judge_messages(q.question, q.answer, response, asked_on), temperature=JUDGE_TEMPERATURE
    )
    label = ""
    reason = ""
    parsed = isinstance(result, dict) and bool(str(result.get("label", "")).strip())
    if isinstance(result, dict):
        label = str(result.get("label", ""))
        reason = str(result.get("reason", ""))
    # Exact match, not substring: "INCORRECT" contains "CORRECT" as a substring and must grade
    # WRONG, not CORRECT.
    correct = label.strip().upper() == "CORRECT"
    return {
        "label": "CORRECT" if correct else "WRONG",
        "reason": reason,
        "correct": correct,
        "judge_parsed": parsed,
        "seconds": time.perf_counter() - started,
    }


def _score(cfg: "RunConfig", q: Question, ans: dict, llm) -> dict:
    jr = judge(q, ans["response"], llm)
    return {
        **ans,
        "bench": cfg.bench,
        "mode": cfg.mode,
        "category": q.category,
        "kind": q.kind,
        "question": q.question,
        "gold": q.answer,
        "label": jr["label"],
        "judge_reason": jr["reason"],
        "correct": jr["correct"],
        "judge_parsed": jr["judge_parsed"],
        "judge_seconds": jr["seconds"],
        "judge_version": 2,
        "f1": metrics.token_f1(ans["response"], q.answer),
        "asked_on": q.asked_at.date().isoformat(),
    }


def _answer_questions(
    conv: Conversation,
    cfg: "RunConfig",
    llm,
    paths: dict[str, Path],
    answer_fn,
) -> list[dict]:
    """Shared answer/judge/checkpoint loop for both memory and fullcontext modes."""
    questions = conv.questions[: cfg.max_questions] if cfg.max_questions else conv.questions
    answered: dict[str, dict] = {a["qid"]: a for a in _load_json(paths["answers"], [])}

    for i, q in enumerate(questions):
        if q.qid in answered:
            continue
        ans = answer_fn(q)
        record = {"conv_id": conv.conv_id, **_score(cfg, q, ans, llm)}
        answered[q.qid] = record
        _save_json(paths["answers"], list(answered.values()))
        if (i + 1) % _LOG_EVERY == 0:
            print(f"  [{conv.conv_id}] {i + 1}/{len(questions)} questions answered", flush=True)

    return [answered[q.qid] for q in questions]


def _delete_db_with_sidecars(db_path: Path) -> None:
    """Delete ``db_path`` and its ``-wal``/``-shm`` sidecars (SQLite's write-ahead-log journal
    files) — missing files are fine. Shared by every call site that removes a benchmark SQLite
    DB, so none of them can forget a sidecar and leave a stale ``-wal`` next to a deleted/absent
    ``.db`` (which would replay stray pages into whatever gets created at that path next).
    """
    db_path.unlink(missing_ok=True)
    db_path.with_name(db_path.name + "-wal").unlink(missing_ok=True)
    db_path.with_name(db_path.name + "-shm").unlink(missing_ok=True)


def _cleanup_mid_ingest_db(paths: dict[str, Path]) -> None:
    """If ``paths["db"]`` exists without its ``paths["ingest"]`` checkpoint, a prior run was
    killed mid-ingestion — delete the DB (and its WAL/SHM sidecar files) so the conversation
    restarts ingestion from scratch rather than risk double-ingesting on top of it.
    """
    if not paths["ingest"].exists() and paths["db"].exists():
        _delete_db_with_sidecars(paths["db"])


def _delete_conversation_db(paths: dict[str, Path]) -> None:
    """Delete ``paths["db"]`` and its ``-wal``/``-shm`` sidecars (checkpoints untouched)."""
    _delete_db_with_sidecars(paths["db"])


def _maybe_reuse_ingest(
    cfg: "RunConfig", conv_id: str, paths: dict[str, Path], run_root: Path
) -> None:
    """If ``cfg.reuse_ingest`` names a sibling run and this conversation hasn't been ingested
    into *this* run yet, copy the sibling's already-ingested ``<conv_id>.db``/``.ingest.json``
    into this run's directory so the existing skip-ingestion path (``paths["ingest"].exists()``
    in :func:`_process_conversation_memory`) takes over — no re-ingestion, no bookkeeping LLM
    calls.

    A no-op unless ``reuse_ingest`` is set, this run's own conversation directory has neither
    file yet, and the reuse run actually has both (missing either on the source side is treated
    as "nothing to reuse", not an error — the normal ingest path just runs instead). The reuse
    run's sibling directory is ``run_root.parent / cfg.reuse_ingest`` — the same base ``run_root``
    was built from (whether that's the real ``RUNS_DIR`` or a test's ``out_dir``), so this works
    identically under both.
    """
    if not cfg.reuse_ingest or paths["db"].exists() or paths["ingest"].exists():
        return
    reuse_root = run_root.parent / cfg.reuse_ingest
    reuse_paths = _conv_paths(reuse_root, conv_id)
    if not (reuse_paths["db"].exists() and reuse_paths["ingest"].exists()):
        return
    _checkpoint_and_verify_wal(reuse_paths["db"])
    paths["db"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reuse_paths["db"], paths["db"])
    shutil.copy2(reuse_paths["ingest"], paths["ingest"])


def _process_conversation_memory(
    conv: Conversation,
    cfg: "RunConfig",
    base_llm: LLMProvider,
    base_embedder: EmbeddingProvider,
    run_root: Path,
) -> tuple[dict, list[dict]]:
    paths = _conv_paths(run_root, conv.conv_id)
    expected = conv.questions[: cfg.max_questions] if cfg.max_questions else conv.questions

    # cfg.delete_db_after_answer may have removed the .db on a prior (fully completed) pass —
    # if every expected question is already answered, there is nothing left that needs the DB,
    # so don't recreate an empty one (Engine() would silently make a fresh, character-less
    # file) and don't re-ingest; just replay the checkpoints.
    if expected and not paths["db"].exists() and paths["answers"].exists():
        answered = {a["qid"]: a for a in _load_json(paths["answers"], [])}
        if all(q.qid in answered for q in expected):
            ingest_stats = _load_json(
                paths["ingest"],
                {
                    "turns": 0,
                    "sessions": 0,
                    "llm_calls": 0,
                    "seconds": 0.0,
                    "pair_turns": bool(cfg.pair_turns),
                },
            )
            return ingest_stats, [answered[q.qid] for q in expected]

    _maybe_reuse_ingest(cfg, conv.conv_id, paths, run_root)
    _cleanup_mid_ingest_db(paths)

    counting_llm = _CountingLLM(base_llm)
    engine = Engine(db_path=paths["db"], llm=counting_llm, embedding=base_embedder)
    completed = False
    try:
        if paths["ingest"].exists():
            ingest_stats = _load_json(paths["ingest"], {})
            char = engine.get_character(conv.conv_id)
        else:
            ingest_stats = ingest_conversation(conv, engine, cfg)
            _save_json(paths["ingest"], ingest_stats)
            char = engine.get_character(conv.conv_id)

        char.background = False
        char.parallel = False
        char.enforce_consistency = False

        def _answer(q: Question) -> dict:
            return answer_question(conv, char, q, cfg, counting_llm)

        records = _answer_questions(conv, cfg, counting_llm, paths, _answer)
        print(
            f"[{conv.conv_id}] ingested {ingest_stats.get('sessions', 0)} sessions, "
            f"{ingest_stats.get('turns', 0)} turns, {ingest_stats.get('llm_calls', 0)} bookkeeping "
            f"calls ({ingest_stats.get('seconds', 0.0):.1f}s); {len(records)} questions answered",
            flush=True,
        )
        completed = True
        return ingest_stats, records
    finally:
        engine.close()
        # Only delete once every expected question has been answered and judged (`completed`
        # is set just before returning, above) — never on an exception mid-way, which would
        # strand the conversation with no way to resume ingestion.
        if completed and cfg.delete_db_after_answer:
            _delete_conversation_db(paths)


def _process_conversation_fullcontext(
    conv: Conversation, cfg: "RunConfig", base_llm: LLMProvider, run_root: Path
) -> tuple[dict, list[dict]]:
    paths = _conv_paths(run_root, conv.conv_id)
    counting_llm = _CountingLLM(base_llm)

    def _answer(q: Question) -> dict:
        return answer_fullcontext(conv, q, cfg, counting_llm)

    records = _answer_questions(conv, cfg, counting_llm, paths, _answer)
    print(f"[{conv.conv_id}] fullcontext: {len(records)} questions answered", flush=True)
    ingest_stats = {"turns": 0, "sessions": 0, "llm_calls": 0, "seconds": 0.0}
    return ingest_stats, records


def judge_plus(evidence: str, response: str, llm) -> dict:
    """Judge ``response`` against ``evidence`` with the LoCoMo-Plus Cognitive judge.

    Unlike :func:`judge`, there is no gold answer — the probe's cue is the only evidence, and
    the label is lowercase (``"correct"``/``"wrong"``) per the upstream JSON schema.
    ``judge_parsed`` is ``False`` when the LLM's JSON response wasn't a dict or carried no
    non-empty ``label`` — see :func:`judge`.
    """
    started = time.perf_counter()
    result = _judge_call(
        llm, plus_judge_messages(evidence, response), temperature=JUDGE_TEMPERATURE
    )
    label = ""
    reason = ""
    parsed = isinstance(result, dict) and bool(str(result.get("label", "")).strip())
    if isinstance(result, dict):
        label = str(result.get("label", ""))
        reason = str(result.get("reason", ""))
    # Exact match, not substring — same reasoning as judge()'s CORRECT/INCORRECT fix.
    correct = label.strip().lower() == "correct"
    return {
        "label": "correct" if correct else "wrong",
        "reason": reason,
        "correct": correct,
        "judge_parsed": parsed,
        "seconds": time.perf_counter() - started,
    }


def _probe_checkpoint_path(run_root: Path, probe_id: str) -> Path:
    return run_root / f"{probe_id}.json"


def _checkpoint_and_verify_wal(db_path: Path) -> None:
    """Force a WAL checkpoint on ``db_path`` and verify it actually cleared.

    The storage layer opens SQLite in WAL mode and only checkpoints on the *last* connection's
    close — so a base DB whose ``-wal`` sidecar is non-empty (another connection still has it
    open, or the base run's Engine hasn't been closed yet) would be copied with recent writes
    still sitting in the WAL, nearly empty on disk, silently producing garbage probes if copied
    as-is. ``PRAGMA wal_checkpoint(TRUNCATE)`` forces those pages back into the main DB file and
    truncates the WAL; if the WAL still has content afterward, some other connection is holding
    it open and it isn't safe to copy.
    """
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
    finally:
        conn.close()

    wal_path = db_path.with_name(db_path.name + "-wal")
    if wal_path.exists() and wal_path.stat().st_size > 0:
        raise RuntimeError(
            f"base DB {db_path} has a live WAL ({wal_path} is non-empty after checkpoint) — "
            "base run still in progress? Refusing to copy a possibly-incomplete DB."
        )


def _process_probe_memory(
    probe: Probe,
    conv: Conversation,
    cfg: "RunConfig",
    base_llm: LLMProvider,
    base_embedder: EmbeddingProvider,
    reuse_root: Path,
    run_root: Path,
) -> dict:
    """LoCoMo-Plus memory mode for one probe: WAL-checkpoint and copy the base conversation's
    ingested DB, ingest the cue as its own session, then answer the trigger via
    ``Character.chat()`` — the real product path, per the spec's "LoCoMo-Plus answering"
    protocol.

    The DB copy is deleted after judging (each is small, but 401 of them add up on disk, and
    nothing after this function needs it — only the checkpoint JSON is kept). The source DB is
    checkpointed (see :func:`_checkpoint_and_verify_wal`) before every copy, not just the first,
    since the same base DB is reused across many probes.
    """
    started = time.perf_counter()
    source_db = reuse_root / f"{probe.base_conv_id}.db"
    source_ingest = reuse_root / f"{probe.base_conv_id}.ingest.json"
    if not source_db.exists() or not source_ingest.exists():
        raise ValueError(
            f"--reuse-run {cfg.reuse_run!r} is missing {probe.base_conv_id}.db/.ingest.json "
            f"under {reuse_root} — run the LoCoMo memory-mode bench with --run-id {cfg.reuse_run!r} "
            "first (locomo_plus memory mode reuses its ingested DBs)."
        )

    _checkpoint_and_verify_wal(source_db)

    db_path = run_root / f"{probe.probe_id}.db"
    shutil.copy2(source_db, db_path)

    counting_llm = _CountingLLM(base_llm)
    engine = Engine(db_path=db_path, llm=counting_llm, embedding=base_embedder)
    try:
        char = engine.get_character(probe.base_conv_id)
        char.background = False
        char.parallel = False
        char.enforce_consistency = False
        char.unified_assessment = True

        clock.override(probe.cue.at)
        char.start_session()
        for turn in probe.cue.turns:
            clock.advance(timedelta(seconds=30))
            char.ingest(turn.role, turn.text, user_id=conv.user_name)
        char.end_session()

        clock.override(probe.trigger_at)
        char.start_session()
        response = (char.chat(probe.trigger_text, user_id=conv.user_name) or "").strip()
        prompt_chars = sum(len(m.get("content", "")) for m in char.last_chat_messages)
    finally:
        engine.close()

    jr = judge_plus(probe.evidence_text, response, counting_llm)
    record = {
        "probe_id": probe.probe_id,
        "relation_type": probe.relation_type,
        "time_gap": probe.time_gap,
        "base_conv_id": probe.base_conv_id,
        "response": response,
        "label": jr["label"],
        "reason": jr["reason"],
        "correct": jr["correct"],
        "judge_parsed": jr["judge_parsed"],
        "judge_version": 2,
        "prompt_tokens_est": prompt_chars // 4,
        "llm_calls": counting_llm.calls,
        "seconds": time.perf_counter() - started,
    }
    _delete_db_with_sidecars(db_path)
    return record


def _process_probe_fullcontext(
    probe: Probe, conv: Conversation, cfg: "RunConfig", base_llm: LLMProvider
) -> dict:
    """LoCoMo-Plus full-context mode for one probe: stitched transcript + trigger -> reply, no
    Engine/DB at all, same judge as memory mode."""
    started = time.perf_counter()
    counting_llm = _CountingLLM(base_llm)

    system = (
        f"You are {conv.character_name}, continuing a long-running conversation with "
        f"{conv.user_name}. Reply to the last message in at most 3 sentences."
    )
    response = (
        counting_llm.generate(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": probe.stitched_text},
            ],
            temperature=PLUS_FULLCONTEXT_TEMPERATURE,
            max_tokens=PLUS_FULLCONTEXT_MAX_TOKENS,
        )
        or ""
    ).strip()

    jr = judge_plus(probe.evidence_text, response, counting_llm)
    return {
        "probe_id": probe.probe_id,
        "relation_type": probe.relation_type,
        "time_gap": probe.time_gap,
        "base_conv_id": probe.base_conv_id,
        "response": response,
        "label": jr["label"],
        "reason": jr["reason"],
        "correct": jr["correct"],
        "judge_parsed": jr["judge_parsed"],
        "judge_version": 2,
        "prompt_tokens_est": len(probe.stitched_text) // 4,
        "llm_calls": counting_llm.calls,
        "seconds": time.perf_counter() - started,
    }


def run_plus(
    cfg: "RunConfig",
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Orchestrate a LoCoMo-Plus Cognitive-subset run (bench ``"locomo_plus"``).

    Memory mode requires ``cfg.reuse_run`` — a prior ``bench="locomo"``/``mode="memory"`` run
    whose per-conversation ``<conv_id>.db``/``.ingest.json`` this run copies from (one fresh copy
    per probe, deleted after judging). Full-context mode needs no reuse run.

    Checkpointing is per probe (``<probe_id>.json``, written atomically), unlike
    :func:`run`'s per-conversation answer list — probes are independent single-shot units, so
    there is nothing to accumulate within one probe's file.
    """
    started_at = datetime.now(timezone.utc)

    base = Path(out_dir) if out_dir else None
    run_root = (base / cfg.run_id) if base else (RUNS_DIR / cfg.run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    results_dir = base if base else RESULTS_DIR

    if cfg.mode == "memory":
        if not cfg.reuse_run:
            raise ValueError(
                "locomo_plus memory mode requires --reuse-run <id> (a prior "
                "bench=locomo mode=memory run to copy ingested DBs from)"
            )
        reuse_root = (base / cfg.reuse_run) if base else (RUNS_DIR / cfg.reuse_run)
    elif cfg.mode != "fullcontext":
        raise ValueError(f"Unknown mode {cfg.mode!r}; expected 'memory' or 'fullcontext'")
    else:
        reuse_root = None

    base_dataset_path = (
        Path(cfg.base_dataset_path) if cfg.base_dataset_path else DATA_DIR / "locomo.json"
    )
    probes_dataset_path = (
        Path(cfg.dataset_path) if cfg.dataset_path else DATA_DIR / "locomo_plus.json"
    )
    base_convs = load_locomo(base_dataset_path)
    probes = load_locomo_plus(probes_dataset_path, base_convs)
    conv_by_id = {c.conv_id: c for c in base_convs}

    if cfg.limit_conversations:
        probes = probes[: cfg.limit_conversations]
    if cfg.shard is not None:
        index, count = cfg.shard
        if not (0 <= index < count):
            raise ValueError(f"bad shard {cfg.shard!r}")
        probes = [p for i, p in enumerate(probes) if i % count == index]

    base_llm = llm if llm is not None else _brain_llm(timeout=cfg.timeout)
    base_embedder = embedder if embedder is not None else _make_embedder()

    # Memory mode probes drive char.chat() (see _process_probe_memory) — apply the same
    # benchmark config run() applies to memory-mode ingestion/answering, so a Plus probe's
    # end_session() doesn't fire the callbacks-hooks LLM call and fact_extraction_interval is
    # consistent across benches. Full-context mode never touches an Engine/Character, so it
    # needs no config mutation.
    benchmark_cm = _benchmark_config(cfg) if cfg.mode == "memory" else nullcontext()

    with benchmark_cm:
        records: list[dict] = []
        for i, probe in enumerate(probes):
            checkpoint_path = _probe_checkpoint_path(run_root, probe.probe_id)
            if checkpoint_path.exists():
                records.append(_load_json(checkpoint_path, {}))
                continue

            conv = conv_by_id[probe.base_conv_id]
            if cfg.mode == "memory":
                assert reuse_root is not None
                record = _process_probe_memory(
                    probe, conv, cfg, base_llm, base_embedder, reuse_root, run_root
                )
            else:
                record = _process_probe_fullcontext(probe, conv, cfg, base_llm)

            _save_json(checkpoint_path, record)
            records.append(record)
            if (i + 1) % _LOG_EVERY == 0:
                print(f"  {i + 1}/{len(probes)} probes judged", flush=True)

    summary = metrics.summarize_plus(records)
    results = {
        "run_id": cfg.run_id,
        "bench": cfg.bench,
        "mode": cfg.mode,
        "model": getattr(base_llm, "model", None),
        "judge": getattr(base_llm, "model", None),
        "embedding": getattr(base_embedder, "model", None) if cfg.mode == "memory" else None,
        "timestamp": started_at.isoformat(),
        "config": {
            "limit_conversations": cfg.limit_conversations,
            "shard": list(cfg.shard) if cfg.shard else None,
            "overrides": list(cfg.overrides),
            "reuse_run": cfg.reuse_run,
            "temperature": {
                "judge": JUDGE_TEMPERATURE,
                "fullcontext_response": PLUS_FULLCONTEXT_TEMPERATURE,
            },
        },
        "summary": summary,
        "probes": records,
    }

    # Judge-sample calibration (report.write_judge_sample) assumes LoCoMo/LongMemEval's
    # category/kind/gold schema, which Plus records don't carry (relation_type/time_gap, no
    # gold answer) — rather than force-fit them into that stratification, Plus judge decisions
    # are simply not sampled for human review here (documented choice; the full judged
    # `probes` list is already in the results file for spot-checking).
    if cfg.write_results and not cfg.run_id.startswith("smoke"):
        report.write_results(results, results_dir=results_dir)

    return results


def _assert_safe_to_aggregate(
    run_root: Path, conversations: list[Conversation], cfg: "RunConfig"
) -> None:
    """Refuse to run the final unsharded aggregation call while shard workers may still be
    writing to ``run_root``.

    A ``run(shard=None, write_results=True)`` call is meant to be the *last* step, run once all
    parallel ``--shard i/n`` workers have finished — it reads/aggregates every conversation, so
    if it races a live shard worker it can e.g. see a ``.db`` that worker just created but
    hasn't finished ingesting into yet, or a partial/missing ``.answers.json``. Three signs of
    "still in progress" are checked per conversation:

    - a ``<conv>.db`` exists without its ``<conv>.ingest.json`` (mid-ingestion, or a kill this
      call's own restart-cleanup would normally handle — but that cleanup only runs once this
      call actually starts processing the conversation, which is exactly what we're trying to
      gate here).
    - a ``<conv>.ingest.json`` exists but ``<conv>.answers.json`` does not yet (ingestion
      finished, answering hasn't started or hasn't checkpointed its first question yet —
      ``_answer_questions`` only creates ``.answers.json`` after the first question is scored).
    - a ``<conv>.answers.json`` exists with fewer records than the conversation has questions
      (mid-answering, at least one question already checkpointed).

    Pass ``cfg.force_aggregate=True`` (CLI ``--force-aggregate``) to skip this check, e.g. when
    intentionally aggregating a deliberately-incomplete run.
    """
    if cfg.force_aggregate:
        return

    problems: list[str] = []
    for conv in conversations:
        paths = _conv_paths(run_root, conv.conv_id)
        if paths["db"].exists() and not paths["ingest"].exists():
            problems.append(f"{conv.conv_id}: .db exists without .ingest.json (mid-ingest)")
            continue

        n_questions = (
            len(conv.questions[: cfg.max_questions]) if cfg.max_questions else len(conv.questions)
        )

        if paths["ingest"].exists() and not paths["answers"].exists() and n_questions:
            problems.append(
                f"{conv.conv_id}: .ingest.json exists without .answers.json "
                f"(mid-answering, 0/{n_questions} questions answered)"
            )
            continue

        if paths["answers"].exists():
            n_answered = len(_load_json(paths["answers"], []))
            if n_answered < n_questions:
                problems.append(
                    f"{conv.conv_id}: {n_answered}/{n_questions} questions answered (mid-answering)"
                )

    if problems:
        raise RuntimeError(
            "Refusing unsharded aggregation run — this run looks like it's still in progress "
            f"under {run_root}: " + "; ".join(problems) + ". If shard workers are still running, "
            "wait for them to finish. If this is intentional (e.g. a deliberately partial run), "
            "pass --force-aggregate."
        )


@contextmanager
def _benchmark_config(cfg: "RunConfig"):
    """Mutate the process-global woven-imprint config for the duration of a memory-mode
    benchmark run/probe, restoring the previous values in ``finally``.

    Sets ``memory.fact_extraction_interval = cfg.fact_extraction_interval`` and
    ``maintenance.callbacks_refresh_on_session_end = False`` — the latter so a benchmark
    conversation's `end_session()` never fires the callbacks-hooks LLM call, which is product
    behavior irrelevant to (and an uncounted cost in) a benchmark run. Used by both :func:`run`
    and :func:`run_plus` so memory mode applies the identical benchmark config in either path;
    ``get_config()`` is process-global state, so a caller running multiple benches/modes in one
    process (or a test suite sharing a process) must not leak one run's settings into the next.

    ``apply_overrides`` runs *inside* the ``try`` (not before it): if it raises partway through
    ``cfg.overrides`` (an unknown field, a bad value on the 2nd/3rd override, ...),
    ``apply_overrides`` itself rolls back whatever it already applied before re-raising (see its
    docstring), and this function's ``finally`` still restores ``fact_extraction_interval``/
    ``callbacks_refresh_on_session_end`` either way — a failing override no longer leaks any
    config mutation past this context manager.
    """
    woven_cfg = get_config()
    prev_interval = woven_cfg.memory.fact_extraction_interval
    prev_refresh = woven_cfg.maintenance.callbacks_refresh_on_session_end
    restores: list[tuple[str, str, object]] = []
    try:
        woven_cfg.memory.fact_extraction_interval = cfg.fact_extraction_interval
        woven_cfg.maintenance.callbacks_refresh_on_session_end = False
        restores = apply_overrides(woven_cfg, cfg.overrides)
        yield woven_cfg
    finally:
        woven_cfg.memory.fact_extraction_interval = prev_interval
        woven_cfg.maintenance.callbacks_refresh_on_session_end = prev_refresh
        for section, key, prev in reversed(restores):
            setattr(getattr(woven_cfg, section), key, prev)


_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"0", "false", "no", "off"}


def apply_overrides(woven_cfg, overrides) -> list[tuple[str, str, object]]:
    """Apply ``"section.key=value"`` overrides to the config object; return (section, key, previous)
    triples for restoration. Values are coerced to the type of the current attribute.

    Bool coercion is strict: the value (case-insensitive, surrounding whitespace stripped) must
    be exactly one of ``1``/``0``/``true``/``false``/``yes``/``no``/``on``/``off``, else
    ``ValueError`` — a typo like ``--set memory.relevance_gate=fasle`` must fail loudly rather
    than silently coercing to ``False`` (every non-``"true"``-ish string would otherwise be
    falsy). Unknown section/key raises ``ValueError`` too, so a typo there can't silently run
    the default configuration either.

    If applying override N fails (unknown field, bad value), overrides 1..N-1 that were already
    applied are rolled back before the exception propagates — a caller never has to distinguish
    "no overrides applied" from "some overrides applied, then it failed".
    """
    restores: list[tuple[str, str, object]] = []
    try:
        for item in overrides:
            if "=" not in item or "." not in item.split("=", 1)[0]:
                raise ValueError(f"override must look like section.key=value, got {item!r}")
            path, raw = item.split("=", 1)
            section, key = path.split(".", 1)
            if not hasattr(woven_cfg, section) or not hasattr(getattr(woven_cfg, section), key):
                raise ValueError(f"unknown config field {path!r}")
            target = getattr(woven_cfg, section)
            prev = getattr(target, key)
            if isinstance(prev, bool):
                normalized = raw.strip().lower()
                if normalized in _TRUE_STRINGS:
                    value: object = True
                elif normalized in _FALSE_STRINGS:
                    value = False
                else:
                    raise ValueError(
                        f"override {path}={raw!r} is not a valid bool "
                        f"(expected one of {sorted(_TRUE_STRINGS | _FALSE_STRINGS)})"
                    )
            elif isinstance(prev, int):
                value = int(raw)
            elif isinstance(prev, float):
                value = float(raw)
            else:
                value = raw
            setattr(target, key, value)
            restores.append((section, key, prev))
    except Exception:
        for section, key, prev in reversed(restores):
            setattr(getattr(woven_cfg, section), key, prev)
        raise
    return restores


def run(
    cfg: "RunConfig",
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Orchestrate a full benchmark run: load dataset, ingest/answer/judge per conversation,
    checkpoint as it goes, summarize, and (unless ``run_id`` starts with "smoke") write results.

    ``out_dir``, when given, replaces both the checkpoint root (``RUNS_DIR``) and the results
    directory (``RESULTS_DIR``) — tests pass a tmp_path here so nothing touches the real
    ``eval/external/runs/`` or ``eval/results/``.
    """
    started_at = datetime.now(timezone.utc)

    base = Path(out_dir) if out_dir else None
    run_root = (base / cfg.run_id) if base else (RUNS_DIR / cfg.run_id)
    results_dir = base if base else RESULTS_DIR
    run_root.mkdir(parents=True, exist_ok=True)

    loader = _LOADERS.get(cfg.bench)
    if loader is None:
        raise ValueError(f"Unknown bench {cfg.bench!r}; choices: {sorted(_LOADERS)}")

    dataset_path = Path(cfg.dataset_path) if cfg.dataset_path else DATA_DIR / f"{cfg.bench}.json"
    conversations = loader(dataset_path, cfg)
    if cfg.limit_conversations:
        conversations = conversations[: cfg.limit_conversations]
    if cfg.shard is not None:
        index, count = cfg.shard
        if not (0 <= index < count):
            raise ValueError(f"bad shard {cfg.shard!r}")
        conversations = [c for i, c in enumerate(conversations) if i % count == index]

    if cfg.shard is None and cfg.write_results:
        _assert_safe_to_aggregate(run_root, conversations, cfg)

    base_llm = llm if llm is not None else _brain_llm(timeout=cfg.timeout)
    base_embedder = embedder if embedder is not None else _make_embedder()

    if cfg.mode == "memory":
        benchmark_cm = _benchmark_config(cfg)
    elif cfg.mode != "fullcontext":
        raise ValueError(f"Unknown mode {cfg.mode!r}; expected 'memory' or 'fullcontext'")
    else:
        benchmark_cm = nullcontext()

    with benchmark_cm:
        all_records: list[dict] = []
        ingest_totals = {"turns": 0, "sessions": 0, "llm_calls": 0, "seconds": 0.0}

        for conv in conversations:
            if cfg.mode == "memory":
                ingest_stats, records = _process_conversation_memory(
                    conv, cfg, base_llm, base_embedder, run_root
                )
            else:
                ingest_stats, records = _process_conversation_fullcontext(
                    conv, cfg, base_llm, run_root
                )

            for key in ingest_totals:
                ingest_totals[key] += ingest_stats.get(key, 0)
            all_records.extend(records)

        summary = metrics.summarize(all_records)

        results = {
            "run_id": cfg.run_id,
            "bench": cfg.bench,
            "mode": cfg.mode,
            "model": getattr(base_llm, "model", None),
            "judge": getattr(base_llm, "model", None),
            "embedding": getattr(base_embedder, "model", None),
            "timestamp": started_at.isoformat(),
            "config": {
                "k": cfg.k,
                "fact_extraction_interval": cfg.fact_extraction_interval,
                "pair_turns": bool(cfg.pair_turns),
                "agg_stage": bool(cfg.agg_stage),
                "max_tokens": {"qa": QA_MAX_TOKENS, "agg_qa": AGG_QA_MAX_TOKENS},
                "temperature": {"qa": QA_TEMPERATURE, "judge": JUDGE_TEMPERATURE},
                "limit_conversations": cfg.limit_conversations,
                "sample": cfg.sample,
                "seed": cfg.seed,
                "max_sessions": cfg.max_sessions,
                "max_questions": cfg.max_questions,
                "shard": list(cfg.shard) if cfg.shard else None,
                "overrides": list(cfg.overrides),
            },
            "summary": summary,
            "ingest_totals": ingest_totals,
            "conversations": all_records,
        }

        if cfg.write_results and not cfg.run_id.startswith("smoke"):
            report.write_results(results, results_dir=results_dir)
            report.write_judge_sample(all_records, n=60, seed=7, results_dir=results_dir)

        return results


def rejudge(
    cfg: "RunConfig",
    llm: LLMProvider | None = None,
    out_dir: str | Path | None = None,
    only_unjudged: bool = False,
) -> dict:
    """Re-judge every answered question in an existing run, in place — no re-ingesting or
    re-answering.

    For every ``<conv_id>.answers.json`` under the run root: reloads the dataset with the same
    loader :func:`run` uses (to recover each question's ``asked_at``, needed for the judge's
    reference-date line — see ``prompts.judge_messages``) and maps ``qid -> Question``. The
    reload uses ``cfg.sample``/``cfg.seed`` (LongMemEval-S) same as :func:`run` — pass
    ``--sample``/``--seed`` matching the original run so the reload reproduces the same subset;
    omitting them (the CLI default) reloads the *full* dataset instead, which is still safe here
    since this function only touches conversations that already have an ``.answers.json`` on
    disk — the extra (never-run) conversations a superset reload produces are just skipped.
    ``cfg.max_questions`` is never round-tripped from the original run either way.

    Per record: ``kind == "qa"`` records are re-judged with :func:`judge` (``response``/``gold``
    untouched; ``label``/``judge_reason``/``correct``/``judge_parsed``/``judge_seconds``
    overwritten). ``adversarial``/``abstain`` records are re-scored with the current
    ``metrics.is_abstention`` (no judge call, same as :func:`judge`'s own short-circuit;
    ``judge_parsed`` set ``True``, matching :func:`judge`'s rule-scored path). Every
    touched record gets ``token_f1`` recomputed, an ``asked_on`` date recorded (going forward,
    :func:`run` records this itself — see ``_score``), and ``judge_version: 2`` set. With
    ``only_unjudged=True``, a record already carrying ``judge_version == 2`` is skipped —
    lets a killed rejudge pass resume cheaply. Each conversation's checkpoint is rewritten
    atomically (via :func:`_save_json`) once all its records are processed.

    Finally recomputes ``summary`` and calls ``report.write_results``/``write_judge_sample``
    exactly like :func:`run` — same results dict shape. ``ingest_totals`` is always derived
    fresh by summing every ``<conv_id>.ingest.json`` checkpoint actually on disk under the run
    root (not read from any stored results file — those checkpoints are ground truth for what
    ingestion actually did), and ``embedding`` is always the embedder model constant
    (``common.EMBED_MODEL``) when ``cfg.mode == "memory"`` (``None`` for fullcontext, which never
    embeds). Only ``config`` is copied from the run's existing ``external_<run_id>.json`` (its
    persisted run metadata) when one exists under the results dir, so a rewritten LoCoMo/
    LongMemEval-S results file still records the original run's ``k``/``sample``/``seed``/
    ``fact_extraction_interval``/etc.; when no such file exists yet, ``config`` is built fresh
    from this call's own ``cfg`` (the CLI's ``--sample``/``--seed``/``--interval``/``--k`` flags
    are forwarded into it — see ``__main__.py``'s ``rejudge`` subcommand).
    """
    started_at = datetime.now(timezone.utc)

    base = Path(out_dir) if out_dir else None
    run_root = (base / cfg.run_id) if base else (RUNS_DIR / cfg.run_id)
    results_dir = base if base else RESULTS_DIR
    if not run_root.exists():
        raise ValueError(f"run root {run_root} does not exist — nothing to rejudge")

    loader = _LOADERS.get(cfg.bench)
    if loader is None:
        raise ValueError(f"Unknown bench {cfg.bench!r}; choices: {sorted(_LOADERS)}")

    dataset_path = Path(cfg.dataset_path) if cfg.dataset_path else DATA_DIR / f"{cfg.bench}.json"
    conversations = loader(dataset_path, cfg)
    questions_by_qid: dict[str, Question] = {
        q.qid: q for conv in conversations for q in conv.questions
    }

    base_llm = llm if llm is not None else _brain_llm(timeout=cfg.timeout)

    all_records: list[dict] = []
    for conv in conversations:
        answers_path = run_root / f"{conv.conv_id}.answers.json"
        if not answers_path.exists():
            continue
        records = _load_json(answers_path, [])
        if not records:
            continue

        for rec in records:
            if only_unjudged and rec.get("judge_version") == 2:
                continue
            kind = rec.get("kind")
            response = rec.get("response", "")
            q = questions_by_qid.get(rec.get("qid"))

            if kind == "qa":
                if q is None:
                    continue  # unknown qid (dataset changed underneath us) — leave as-is
                jr = judge(q, response, base_llm)
                rec["label"] = jr["label"]
                rec["judge_reason"] = jr["reason"]
                rec["correct"] = jr["correct"]
                rec["judge_parsed"] = jr["judge_parsed"]
                rec["judge_seconds"] = jr["seconds"]
            elif kind in ("adversarial", "abstain"):
                correct = metrics.is_abstention(response)
                rec["label"] = "CORRECT" if correct else "WRONG"
                rec["judge_reason"] = "scored by abstention rule (no judge call)"
                rec["correct"] = correct
                rec["judge_parsed"] = True
            else:
                continue

            if q is not None:
                rec["asked_on"] = q.asked_at.date().isoformat()
            rec["f1"] = metrics.token_f1(response, rec.get("gold", ""))
            rec["judge_version"] = 2

        _save_json(answers_path, records)
        print(
            f"[rejudge] {conv.conv_id}: {len(records)} records rewritten (judge_version 2)",
            flush=True,
        )
        all_records.extend(records)

    summary = metrics.summarize(all_records)

    model = getattr(base_llm, "model", None)
    judge_model = model
    embedding = EMBED_MODEL if cfg.mode == "memory" else None

    stored = _load_json(results_dir / f"external_{cfg.run_id}.json", None)
    if stored:
        run_config = stored.get("config", {})
    else:
        run_config = {
            "k": cfg.k,
            "fact_extraction_interval": cfg.fact_extraction_interval,
            "pair_turns": bool(cfg.pair_turns),
            "agg_stage": bool(cfg.agg_stage),
            "max_tokens": {"qa": QA_MAX_TOKENS, "agg_qa": AGG_QA_MAX_TOKENS},
            "temperature": {"qa": QA_TEMPERATURE, "judge": JUDGE_TEMPERATURE},
            "limit_conversations": cfg.limit_conversations,
            "sample": cfg.sample,
            "seed": cfg.seed,
            "max_sessions": cfg.max_sessions,
            "max_questions": cfg.max_questions,
            "shard": None,
        }

    # Ground truth for what ingestion actually did — always summed fresh from every
    # <conv_id>.ingest.json checkpoint on disk, never trusted from a stored results file
    # (which may predate a later re-ingest, or simply not exist yet).
    ingest_totals: dict[str, Any] = {"turns": 0, "sessions": 0, "llm_calls": 0, "seconds": 0.0}
    if cfg.mode == "memory":
        for ingest_path in sorted(run_root.glob("*.ingest.json")):
            stats = _load_json(ingest_path, {})
            for key in ("turns", "sessions", "llm_calls", "seconds"):
                ingest_totals[key] += stats.get(key, 0)
            if "pair_turns" in stats:
                ingest_totals["pair_turns"] = stats["pair_turns"]

    results = {
        "run_id": cfg.run_id,
        "bench": cfg.bench,
        "mode": cfg.mode,
        "model": model,
        "judge": judge_model,
        "embedding": embedding,
        "timestamp": started_at.isoformat(),
        "rejudged_at": started_at.isoformat(),
        "config": run_config,
        "summary": summary,
        "ingest_totals": ingest_totals,
        "conversations": all_records,
    }

    if not cfg.run_id.startswith("smoke"):
        report.write_results(results, results_dir=results_dir)
        report.write_judge_sample(all_records, n=60, seed=7, results_dir=results_dir)

    return results
