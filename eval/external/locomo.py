"""LoCoMo and LoCoMo-Plus (Cognitive) loaders.

``load_locomo`` normalizes ``locomo10.json`` (snap-research/locomo) into
``Conversation``/``Question``. Category mapping: LoCoMo's ``category`` is an
int 1-4 (multi-hop/temporal/common-sense/single-hop, all scored against
``answer``) or 5 (adversarial: no in-conversation answer exists; the gold
label is abstention, and the dataset carries the plausible-but-wrong
``adversarial_answer`` instead of ``answer`` for almost all category-5 items
— a couple carry both keys; category 5 always uses ``adversarial_answer``
here for consistency). ``speaker_a`` is always the user, ``speaker_b`` the
character (matches the spec's ingestion-unit convention).

``load_locomo_plus`` ports ``xjtuleeyf/Locomo-Plus``'s ``data/build_conv.py``
(verified against the live repo 2026-08-27) for probe->base-conversation
mapping, time-gap parsing, and cue/trigger placement:

- probe ``i`` stitches onto ``base[i % len(base)]``.
- ``trigger_at`` = last base session's time + 7 days (upstream ``query_time``).
- ``cue.at`` = ``trigger_at`` - ``parse_time_gap(time_gap)`` days (upstream
  ``cue_time``); the cue's turns are synthesized 30s apart starting at
  ``cue.at`` (upstream keeps the whole cue as one time-ordering event; we
  need per-turn timestamps for ``Turn.at``, so we space them out
  deterministically).
- ``parse_time_gap`` is ported verbatim (word numbers one..twelve, digits,
  "a"/"an" = 1; week=7/month=30/year=365 days) *including* its upstream
  quirk: it only matches a number token immediately followed by whitespace
  and the unit, so phrases like "several months later", "a couple of months
  later", or "several years later" match nothing and parse to 0 days. This
  is upstream behavior, not a bug we introduced — see the real
  ``locomo_plus.json`` for how often it occurs.
- ``trigger_text`` keeps only the trigger's A-line(s) (joined with a space);
  B-lines, if any, are dropped since the product path drives the trigger as
  a single user message via ``Character.chat()``. The full ``stitched_text``
  (full-context baseline) includes every trigger turn, A and B alike.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path

from .common import Conversation, Probe, Question, Session, Turn, parse_locomo_datetime

_NUM_WORD = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_TIME_GAP_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a|an)\b"
    r"\s*(week|weeks|month|months|year|years)\b"
)


def parse_time_gap(time_gap: str) -> int:
    """Port of upstream ``build_conv.parse_time_gap`` — returns a day count."""
    s = time_gap.lower().strip()
    m = _TIME_GAP_RE.search(s)
    if not m:
        return 0
    num, unit = m.groups()
    if num.isdigit():
        count = int(num)
    elif num in {"a", "an"}:
        count = 1
    else:
        count = _NUM_WORD.get(num, 0)
    if unit.startswith("week"):
        return count * 7
    if unit.startswith("month"):
        return count * 30
    if unit.startswith("year"):
        return count * 365
    return 0


def parse_ab_dialogue(text: str) -> list[dict[str, str]]:
    """Port of upstream ``build_conv.parse_ab_dialogue``: split "A:"/"B:" lines."""
    turns = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("A:"):
            turns.append({"speaker": "A", "text": line[2:].strip()})
        elif line.startswith("B:"):
            turns.append({"speaker": "B", "text": line[2:].strip()})
    return turns


def _turn_text(t: dict) -> str:
    """Turn text with the image caption folded in.

    LoCoMo image turns carry ``blip_caption`` (and ``img_url``); upstream renders
    them as ``{speaker} said, "{text}" and shared {caption}.`` and 857/1986
    questions have such a turn as evidence, so the caption is part of the content.
    """
    text = (t.get("text") or "").strip()
    caption = (t.get("blip_caption") or "").strip()
    if caption:
        return f"{text} (shared a photo: {caption})" if text else f"(shared a photo: {caption})"
    return text


def load_locomo(path: str | Path) -> list[Conversation]:
    data = json.loads(Path(path).read_text())
    conversations: list[Conversation] = []

    for item in data:
        conv = item["conversation"]
        user_name = conv["speaker_a"]
        character_name = conv["speaker_b"]

        sessions: list[Session] = []
        idx = 1
        while f"session_{idx}" in conv:
            raw_turns = conv[f"session_{idx}"]
            at = parse_locomo_datetime(conv[f"session_{idx}_date_time"])
            turns = [
                Turn(
                    speaker=t["speaker"],
                    role="user" if t["speaker"] == user_name else "assistant",
                    text=_turn_text(t),
                    dia_id=t.get("dia_id"),
                    at=at + timedelta(seconds=j),
                )
                for j, t in enumerate(raw_turns)
            ]
            sessions.append(Session(session_id=f"session_{idx}", at=at, turns=turns))
            idx += 1

        asked_at = sessions[-1].at + timedelta(days=1) if sessions else None

        questions: list[Question] = []
        for k, q in enumerate(item.get("qa", [])):
            category = str(q["category"])
            kind = "adversarial" if category == "5" else "qa"
            raw_answer = q.get("adversarial_answer") if category == "5" else q.get("answer")
            questions.append(
                Question(
                    qid=f"{item['sample_id']}-q{k}",
                    question=q["question"],
                    answer="" if raw_answer is None else str(raw_answer),
                    category=category,
                    evidence=list(q.get("evidence") or []),
                    asked_at=asked_at,  # type: ignore[arg-type]
                    kind=kind,
                )
            )

        transcript_lines = [
            f"[{t.at.strftime('%Y-%m-%d %H:%M')}] {t.speaker}: {t.text}"
            for s in sessions
            for t in s.turns
        ]

        conversations.append(
            Conversation(
                conv_id=item["sample_id"],
                user_name=user_name,
                character_name=character_name,
                sessions=sessions,
                questions=questions,
                transcript_text="\n".join(transcript_lines),
            )
        )

    return conversations


def _speaker_name(letter: str, conv: Conversation) -> str:
    return conv.user_name if letter == "A" else conv.character_name


def load_locomo_plus(path: str | Path, base: list[Conversation]) -> list[Probe]:
    items = json.loads(Path(path).read_text())
    probes: list[Probe] = []

    for i, item in enumerate(items):
        conv = base[i % len(base)]
        gap_days = parse_time_gap(item["time_gap"])
        trigger_at = conv.sessions[-1].at + timedelta(days=7)
        cue_at = trigger_at - timedelta(days=gap_days)

        cue_ab = parse_ab_dialogue(item["cue_dialogue"])
        cue_turns = [
            Turn(
                speaker=_speaker_name(t["speaker"], conv),
                role="user" if t["speaker"] == "A" else "assistant",
                text=t["text"],
                dia_id=None,
                at=cue_at + timedelta(seconds=30 * j),
            )
            for j, t in enumerate(cue_ab)
        ]
        cue = Session(session_id=f"{conv.conv_id}-cue-{i:03d}", at=cue_at, turns=cue_turns)

        trigger_ab = parse_ab_dialogue(item["trigger_query"])
        trigger_text = " ".join(t["text"] for t in trigger_ab if t["speaker"] == "A")
        evidence_text = "\n".join(
            f"{_speaker_name(t['speaker'], conv)}: {t['text']}" for t in cue_ab
        )

        blocks: list[tuple] = []
        for s in conv.sessions:
            blocks.append((s.at, [(t.speaker, t.text) for t in s.turns]))
        blocks.append((cue_at, [(t.speaker, t.text) for t in cue_turns]))
        blocks.append(
            (trigger_at, [(_speaker_name(t["speaker"], conv), t["text"]) for t in trigger_ab])
        )
        blocks.sort(key=lambda b: b[0])

        stitched_lines = []
        for at, lines in blocks:
            stitched_lines.append(f"DATE: {at.strftime('%Y-%m-%d %H:%M')}")
            for speaker, text in lines:
                stitched_lines.append(f'{speaker} said, "{text}"')
        stitched_text = "\n".join(stitched_lines)

        probes.append(
            Probe(
                probe_id=f"plus-{i:03d}",
                relation_type=item["relation_type"],
                time_gap=item["time_gap"],
                base_conv_id=conv.conv_id,
                cue=cue,
                trigger_text=trigger_text,
                trigger_at=trigger_at,
                evidence_text=evidence_text,
                stitched_text=stitched_text,
            )
        )

    return probes
