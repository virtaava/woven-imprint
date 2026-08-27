"""LongMemEval-S loader (xiaowu0162/longmemeval-cleaned, ``longmemeval_s_cleaned.json``).

Each of the 500 items is one question with its own haystack (~48 sessions,
~494 turns) — one ``Conversation`` per question, user "user" / character
"Assistant". ``haystack_dates[i]`` is a single timestamp for
``haystack_sessions[i]`` (format ``%Y/%m/%d (%a) %H:%M``, assumed UTC); turns
within a session are spaced 1s apart so ``Turn.at`` is monotonic.
``question_id`` values ending in ``_abs`` are the abstention subset (30 of
500) -> ``Question.kind = "abstain"``; the rest -> ``"qa"``.
``Question.category`` is the raw ``question_type`` (multi-session,
temporal-reasoning, knowledge-update, single-session-user,
single-session-assistant, single-session-preference).
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .common import Conversation, Question, Session, Turn


def parse_longmemeval_datetime(s: str) -> datetime:
    """Parse LongMemEval's timestamp format, e.g. "2023/05/30 (Tue) 23:40"."""
    return datetime.strptime(s.strip(), "%Y/%m/%d (%a) %H:%M").replace(tzinfo=timezone.utc)


def _stratified_sample(items: list[dict], k: int, seed: int) -> list[dict]:
    """Round-robin sample across ``question_type`` groups, seeded per-group shuffle."""
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(it["question_type"], []).append(it)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    types = sorted(groups)
    out: list[dict] = []
    i = 0
    while len(out) < k and any(groups[t] for t in types):
        t = types[i % len(types)]
        if groups[t]:
            out.append(groups[t].pop(0))
        i += 1
    return out


def load_longmemeval_s(
    path: str | Path, sample: int | None = None, seed: int = 7
) -> list[Conversation]:
    data = json.loads(Path(path).read_text())
    if sample is not None:
        data = _stratified_sample(data, sample, seed)

    conversations: list[Conversation] = []
    for item in data:
        qid = item["question_id"]
        kind = "abstain" if qid.endswith("_abs") else "qa"

        sessions: list[Session] = []
        session_ids = item.get("haystack_session_ids") or []
        # haystack_dates are NOT chronological in the source file for 211/500
        # questions; ingest in time order so recency/temporal reasoning is sane.
        indexed = sorted(
            enumerate(zip(item["haystack_sessions"], item["haystack_dates"])),
            key=lambda e: parse_longmemeval_datetime(e[1][1]),
        )
        for i, (raw_turns, date_str) in indexed:
            at = parse_longmemeval_datetime(date_str)
            turns = [
                Turn(
                    speaker="user" if t["role"] == "user" else "Assistant",
                    role=t["role"],
                    text=t["content"],
                    dia_id=None,
                    at=at + timedelta(seconds=j),
                )
                for j, t in enumerate(raw_turns)
            ]
            session_id = session_ids[i] if i < len(session_ids) else f"session_{i}"
            sessions.append(Session(session_id=session_id, at=at, turns=turns))

        question = Question(
            qid=qid,
            question=item["question"],
            answer=str(item.get("answer", "")),
            category=item["question_type"],
            evidence=list(item.get("answer_session_ids") or []),
            asked_at=parse_longmemeval_datetime(item["question_date"]),
            kind=kind,
        )

        transcript_lines = [
            f"[{t.at.strftime('%Y-%m-%d %H:%M')}] {t.speaker}: {t.text}"
            for s in sessions
            for t in s.turns
        ]

        conversations.append(
            Conversation(
                conv_id=qid,
                user_name="user",
                character_name="Assistant",
                sessions=sessions,
                questions=[question],
                transcript_text="\n".join(transcript_lines),
            )
        )

    return conversations
