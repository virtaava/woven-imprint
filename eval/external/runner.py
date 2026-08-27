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
import time
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
from .common import DATA_DIR, RESULTS_DIR, RUNS_DIR, Conversation, Question
from .common import brain_llm as _brain_llm
from .common import embedder as _make_embedder
from .locomo import load_locomo
from .longmemeval import load_longmemeval_s
from .prompts import judge_messages, qa_messages

# Full-context transcripts are truncated to this many characters (from the END — the most
# recent, most-likely-relevant material) before being sent as the answering prompt.
FULLCONTEXT_CHAR_LIMIT = 220_000

QA_MAX_TOKENS = 60
QA_TEMPERATURE = 0.0
JUDGE_TEMPERATURE = 0.0
_LOG_EVERY = 20


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
    reuse_run: str | None = None  # Task 4 stub — not implemented here
    dataset_path: str | Path | None = None


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
    if path.exists():
        return json.loads(path.read_text())
    return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


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


def ingest_conversation(conv: Conversation, engine: Engine, cfg: "RunConfig") -> dict:
    """Ingest ``conv``'s sessions into ``engine``, message by message, under clock control.

    Creates the character (id = ``conv.conv_id``) and ingests every session (capped at
    ``cfg.max_sessions`` for smoke runs), advancing the clock 30s per turn so ``created_at`` is
    monotonic. Returns ``{turns, sessions, llm_calls, seconds}``. Idempotency (skip if already
    done) is the caller's responsibility via the ``<conv_id>.ingest.json`` checkpoint.
    """
    started = time.perf_counter()
    llm = engine.llm
    calls_before = llm.calls if isinstance(llm, _CountingLLM) else 0

    sessions = conv.sessions[: cfg.max_sessions] if cfg.max_sessions else conv.sessions
    if not sessions:
        return {"turns": 0, "sessions": 0, "llm_calls": 0, "seconds": 0.0}

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
        for turn in session.turns:
            clock.advance(timedelta(seconds=30))
            char.ingest(turn.role, turn.text, user_id=conv.user_name)
            turns += 1
        char.end_session()

    elapsed = time.perf_counter() - started
    calls_after = llm.calls if isinstance(llm, _CountingLLM) else 0
    return {
        "turns": turns,
        "sessions": len(sessions),
        "llm_calls": calls_after - calls_before,
        "seconds": elapsed,
    }


def answer_question(conv: Conversation, char, q: Question, cfg: "RunConfig", llm) -> dict:
    """Answer ``q`` from ``char``'s pinned block + facts block + retrieved memories."""
    started = time.perf_counter()
    clock.override(q.asked_at)

    pinned, pinned_ids = char._format_pinned_block()
    facts = char._format_facts_block(conv.user_name, pinned_ids)
    mems = char.retriever.retrieve(q.question, limit=cfg.k, relationship_target=conv.user_name)
    filtered = [m for m in mems if m["id"] not in pinned_ids]
    mem_text = char._format_memories(filtered)

    blocks = [
        b for b in (pinned, facts, f"Relevant memories:\n{mem_text}" if mem_text else "") if b
    ]
    block = "\n\n".join(blocks)
    today = q.asked_at.date().isoformat()

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


def judge(q: Question, response: str, llm) -> dict:
    """Judge ``response`` against ``q.answer``.

    Adversarial (LoCoMo category 5) and abstain (LongMemEval ``_abs``) questions are scored by
    abstention detection alone — the gold behavior *is* declining to answer, so there is nothing
    for an LLM judge to usefully compare against, and skipping the call saves one generation per
    such question. A label is still recorded for reporting symmetry with judged questions.
    """
    started = time.perf_counter()
    if q.kind in ("adversarial", "abstain"):
        correct = metrics.is_abstention(response)
        return {
            "label": "CORRECT" if correct else "WRONG",
            "reason": "scored by abstention rule (no judge call)",
            "correct": correct,
            "seconds": time.perf_counter() - started,
        }

    result = llm.generate_json_robust(
        judge_messages(q.question, q.answer, response), temperature=JUDGE_TEMPERATURE
    )
    label = ""
    reason = ""
    if isinstance(result, dict):
        label = str(result.get("label", ""))
        reason = str(result.get("reason", ""))
    correct = "correct" in label.lower()
    return {
        "label": "CORRECT" if correct else "WRONG",
        "reason": reason,
        "correct": correct,
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
        "judge_seconds": jr["seconds"],
        "f1": metrics.token_f1(ans["response"], q.answer),
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


def _process_conversation_memory(
    conv: Conversation,
    cfg: "RunConfig",
    base_llm: LLMProvider,
    base_embedder: EmbeddingProvider,
    run_root: Path,
) -> tuple[dict, list[dict]]:
    paths = _conv_paths(run_root, conv.conv_id)

    # A DB with no ingest checkpoint means a prior run was killed mid-ingestion — restart that
    # conversation's ingestion from scratch rather than risk double-ingesting on top of it.
    if not paths["ingest"].exists() and paths["db"].exists():
        paths["db"].unlink()

    counting_llm = _CountingLLM(base_llm)
    engine = Engine(db_path=paths["db"], llm=counting_llm, embedding=base_embedder)
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
        return ingest_stats, records
    finally:
        engine.close()


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

    base_llm = llm if llm is not None else _brain_llm()
    base_embedder = embedder if embedder is not None else _make_embedder()

    if cfg.mode == "memory":
        woven_cfg = get_config()
        woven_cfg.memory.fact_extraction_interval = cfg.fact_extraction_interval
        woven_cfg.maintenance.callbacks_refresh_on_session_end = False
    elif cfg.mode != "fullcontext":
        raise ValueError(f"Unknown mode {cfg.mode!r}; expected 'memory' or 'fullcontext'")

    all_records: list[dict] = []
    ingest_totals = {"turns": 0, "sessions": 0, "llm_calls": 0, "seconds": 0.0}

    for conv in conversations:
        if cfg.mode == "memory":
            ingest_stats, records = _process_conversation_memory(
                conv, cfg, base_llm, base_embedder, run_root
            )
        else:
            ingest_stats, records = _process_conversation_fullcontext(conv, cfg, base_llm, run_root)

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
            "max_tokens": {"qa": QA_MAX_TOKENS},
            "temperature": {"qa": QA_TEMPERATURE, "judge": JUDGE_TEMPERATURE},
            "limit_conversations": cfg.limit_conversations,
            "sample": cfg.sample,
            "seed": cfg.seed,
            "max_sessions": cfg.max_sessions,
            "max_questions": cfg.max_questions,
        },
        "summary": summary,
        "ingest_totals": ingest_totals,
        "conversations": all_records,
    }

    if not cfg.run_id.startswith("smoke"):
        report.write_results(results, results_dir=results_dir)
        report.write_judge_sample(all_records, n=60, seed=7, results_dir=results_dir)

    return results
