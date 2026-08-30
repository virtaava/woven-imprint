"""Why does the answer model fail when the evidence memory is already in the top-20 block?

For a judged LoCoMo memory-mode run (default ``locomo-mem-v2d``), this script isolates the
``WRONG`` questions whose gold evidence memory made the top-K retrieved block (i.e.
``evidence_class == "retrieved"`` in :mod:`eval.external.diagnose_recall`'s terms) and asks: if
the evidence was right there, why did the model abstain ("Not mentioned") or answer wrong
anyway?

It reuses :func:`eval.external.diagnose_recall.build_diagnostics` verbatim for the
evidence-location/retrieval-recall computation (same DB-copy discipline: the run's own
``eval/external/runs/<run-id>/`` directory is opened read-only, never written to; every DB is
first copied into this script's own diagnostics directory and only the copy is opened — see
``build_diagnostics``/``_copy_db``), then, for exactly the WRONG+retrieved rows it selects, goes
one step further than that script does: it reconstructs the EXACT prompt block text the model
saw, using the current code, and pulls out the exact rendered line(s) that carried the answer
evidence.

Block reconstruction mirrors :func:`eval.external.runner.answer_question` line for line —
``char._format_pinned_block()`` + ``char._format_facts_block()`` + ``"Relevant memories:\n" +
char._format_memories(top-K)`` — against the run's own copied-and-reopened per-conversation DB,
under ``clock.override(q.asked_at)`` so retrieval and relative-date rendering match the original
run. No LLM call is made anywhere in this script (a ``FakeLLM`` stands in for ``Engine``'s
required LLM provider — retrieval/formatting never call it); the embedding server at
``127.0.0.1:11801`` is called for query embeddings, exactly as the original run did.

For every question analyzed, ``len(filtered)`` (the reconstructed top-K memory count actually
rendered) is compared against the run's own recorded ``memories_used`` — a mismatch is reported,
not silently ignored, since it would mean the reconstruction is drifting from what the model
really saw (see ``--verify-all`` / the printed verification summary).

Classification rubric (see ``CATEGORY_DESCRIPTIONS``) was hand-derived by reading ~60
reconstructed rows before this script's classifier logic was written. That reading surfaced a
category the brief's suggested rubric didn't anticipate — ``h_truncated_evidence`` —
``Character._format_memories`` hard-slices every memory's content to 200 characters before
rendering it, and over half of these cases have at least one evidence memory long enough to be
cut by that slice; it turned out to be the single largest root cause, ahead of the abstention
instruction. The classifier applies simple, auditable string/structural rules per category — see
``classify_row`` — falling back to "other" when none apply. A row can match more than one
category (checked in priority order; the FIRST matching category is recorded as primary, but
`categories` on each row lists every category that matched, for cross-tab sanity checks).

Run (repo root, product venv):
    ./.venv/bin/python -m eval.external.abstain_analysis --run-id locomo-mem-v2d

Outputs under ``eval/external/runs/diagnostics/abstain/<run-id>/`` (gitignored):
    cases.jsonl              -- one reconstructed+classified record per WRONG+retrieved question
    category_report.md       -- counts per category x response_class x locomo category
    format_quotes.md         -- one quoted example of each rendered memory-row shape
    worked_examples.md       -- N (default 12) fully worked examples
    recommendation.md        -- ranked PRODUCT/HARNESS recommendation list
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

from woven_imprint import clock
from woven_imprint.engine import Engine
from woven_imprint.llm.base import LLMProvider

from tests.helpers import FakeLLM

from . import diagnose_recall, metrics
from .common import DATA_DIR, RESULTS_DIR, RUNS_DIR, embedder
from .locomo import load_locomo
from .runner import apply_overrides

RUN_ID = "locomo-mem-v2d"
ABSTAIN_ROOT = RUNS_DIR / "diagnostics" / "abstain"
LOCOMO_PATH = DATA_DIR / "locomo.json"

_WORD_RE = re.compile(r"[a-z0-9]+")

# Relative-time phrases that force the model to do date arithmetic against the memory's OWN
# formed-date (shown in the rendered "(YYYY-MM-DD, N ... ago)" prefix), not against "today" (the
# question's asked_at, given in the QA prompt's "Today is ..." line) -- these are two different
# reference points, and the QA_SYSTEM instruction ("Convert relative dates to absolute dates")
# doesn't say which one to use.
_RELATIVE_TIME_RE = re.compile(
    r"\b(next|last|this coming|coming up|a few (?:days|weeks|months|years)|"
    r"in a (?:few|couple of)|(?:yesterday|tomorrow|tonight)|"
    r"(?:mon|tues|wednes|thurs|fri|satur|sun)day|"
    r"next week|next month|next year|last week|last month|last year|"
    r"in \d+ (?:days?|weeks?|months?|years?)|\d+ (?:days?|weeks?|months?|years?) ago|"
    r"the other day|recently|soon|upcoming)\b",
    re.IGNORECASE,
)

_DATE_QUESTION_RE = re.compile(
    r"\bwhen\b|\bwhat (?:date|day|month|year)\b|\bhow (?:long|many (?:days|weeks|months|years))\b",
    re.IGNORECASE,
)

_PHOTO_RE = re.compile(r"\(shared a photo:", re.IGNORECASE)

CATEGORY_DESCRIPTIONS = {
    "c_multi_hop": (
        "2+ evidence dia_ids are needed for the gold answer, and not all of them resolved into "
        "the top-20 block -- only a partial evidence set was actually visible."
    ),
    "h_truncated_evidence": (
        "`Character._format_memories` hard-slices every memory's content to `[:200]` chars "
        "before rendering it -- the evidence memory's full stored content is > 200 chars AND "
        "the gold answer's words are present in the full content but fall after the cut, so the "
        "line the model actually saw ends mid-sentence, before the answer. Found by reading real "
        "examples, not in the brief's suggested rubric -- turned out to be the largest single "
        "root cause."
    ),
    "d_consolidated_or_dropped_detail": (
        "The evidence line is a [Consolidated] summary, or (independent of the 200-char cap) a "
        "paraphrase whose text no longer contains the gold answer's key detail even though the "
        "original dataset turn did -- a lossy rewrite, not a rendering cutoff. CAVEAT: found 0/251 "
        "here -- structurally near-invisible to this 251-case set by construction, since "
        "diagnose_recall's evidence-location only counts a memory as 'stored' at all when its "
        "content is an exact or >=80%-word-overlap fuzzy match to the original dataset turn; a "
        "heavily-paraphrased consolidated summary usually falls below that threshold and the "
        "question lands in 'not_stored'/'stored_archived_only' instead of 'retrieved' -- i.e. "
        "outside this script's target set entirely, not evidence that consolidation is lossless."
    ),
    "b_relative_time": (
        "The evidence line embeds a relative-time phrase ('next month', 'last Saturday', ...) "
        "and the question asks for a date/time -- the model must combine the phrase with the "
        "line's own '(YYYY-MM-DD, ...)' formed-date, not with the prompt's 'Today is' line."
    ),
    "a_speaker_ambiguous": (
        "The rendered evidence line is a user turn tagged '[User]' (never the person's actual "
        "name, even though facts/questions refer to them by name) -- pronouns ('I'/'my') in the "
        "line have no name to resolve against inside the block itself."
    ),
    "f_photo_caption": (
        "The evidence line visibly contains an image-caption fragment ('(shared a photo: ...)') "
        "in the block the model actually saw -- the answer is folded into a caption aside rather "
        "than stated directly. (A caption present in the dataset turn but cut off by the 200-char "
        "truncation before it renders is counted under h_truncated_evidence instead, not here.)"
    ),
    "e_explicit_evidence_abstained": (
        "Single-hop question, the RENDERED evidence line (post-truncation) verbatim contains the "
        "gold answer text, yet the model still abstained or answered wrong -- no structural or "
        "rendering excuse found; points at the QA prompt/instruction itself."
    ),
    "g_other": "None of the above applied by the auditable string/structural rules.",
}


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _render_single_line(char, mem: dict, ref) -> str:
    """Render exactly one memory the way it appears embedded in the real block: run
    ``Character._format_memories`` (the actual product formatter) on a singleton list and strip
    its shared header line, so the text returned is a byte-identical substring of the real
    multi-memory block (the header is the only part that isn't per-item)."""
    text = char._format_memories([mem], now=ref)
    lines = text.split("\n")
    return lines[1] if len(lines) > 1 else ""


def _lines_for_ids(mem_text: str, ordered_ids: list[str]) -> dict[str, str]:
    """Map memory id -> its rendered line, using positional correspondence between
    ``ordered_ids`` (the exact list ``_format_memories`` was called with) and ``mem_text``'s
    lines-after-header (one line per input memory, same order, same content — see
    ``Character._format_memories``: it emits exactly one ``"- ..."`` line per list item, in
    order, with no filtering)."""
    lines = mem_text.split("\n")[1:] if mem_text else []
    return dict(zip(ordered_ids, lines))


def reconstruct_block(
    conv, q, char, user_name: str, k: int
) -> tuple[str, str, str, str, list[dict], dict[str, str]]:
    """Rebuild the exact QA prompt block (pinned + facts + "Relevant memories:\\n" + top-K) the
    same way :func:`eval.external.runner.answer_question` does. Returns
    ``(block, pinned, facts, mem_text, filtered_memories, line_by_id)``."""
    clock.override(q.asked_at)
    pinned, pinned_ids = char._format_pinned_block()
    facts = char._format_facts_block(user_name, pinned_ids)
    mems = char.retriever.retrieve(q.question, limit=k, relationship_target=user_name)
    filtered = [m for m in mems if m["id"] not in pinned_ids]
    mem_text = char._format_memories(filtered)
    blocks = [
        b for b in (pinned, facts, f"Relevant memories:\n{mem_text}" if mem_text else "") if b
    ]
    block = "\n\n".join(blocks)
    line_by_id = _lines_for_ids(mem_text, [m["id"] for m in filtered])
    return block, pinned, facts, mem_text, filtered, line_by_id


def gold_words_present(gold: str, line: str) -> bool:
    """Loose containment check: every non-trivial gold word/number appears in the line
    (case-insensitive) -- used only for the "explicit evidence" heuristic (category e), which is
    deliberately permissive (paraphrase-tolerant) since we want to flag cases where the model had
    every ingredient it needed, not just an exact substring match."""
    gw = [w for w in _words(gold) if len(w) > 2]
    if not gw:
        return False
    lw = set(_words(line))
    hits = sum(1 for w in gw if w in lw)
    return hits / len(gw) >= 0.8


def classify_row(row: dict) -> list[str]:
    """Return every category (from ``CATEGORY_DESCRIPTIONS``) that plausibly applies to ``row``,
    in the fixed priority order the module docstring/report use as "primary" (structural gaps
    first, then the rendering bug that turned out to dominate, then paraphrase loss, then the
    softer content-shape/prompt categories). Pure string/structural rules only, checked against
    the ACTUAL RENDERED evidence line (post-200-char-truncation) wherever the distinction
    matters -- no LLM call, fully auditable."""
    cats: list[str] = []
    ev_lines = row["evidence_lines_in_block"]
    stored_list = row["evidence_turn_texts_as_stored"]  # raw `content` field, pre-truncation
    question = row["question"]
    gold = row["gold"] or ""
    n_evidence = len(row["evidence_dia_ids"])
    n_resolved_in_block = len(ev_lines)

    if n_evidence >= 2 and n_resolved_in_block < n_evidence:
        cats.append("c_multi_hop")

    # h: the 200-char slice in `_format_memories` specifically cut off the gold answer's words --
    # present in the full stored content, absent from the first 200 chars that were rendered.
    truncated_cut = any(
        len(stored) > 200
        and gold
        and gold_words_present(gold, stored)
        and not gold_words_present(gold, stored[:200])
        for stored in stored_list
    )
    if truncated_cut:
        cats.append("h_truncated_evidence")

    any_consolidated = any(stored.startswith("[Consolidated]") for stored in stored_list)
    # A dropped-detail paraphrase independent of truncation: the un-truncated stored content
    # (<=200 chars, so truncation isn't the explanation) doesn't contain the gold answer's key
    # words even loosely, but the raw dataset turn_text does -- something was lost in the
    # ingest-time content itself (e.g. consolidation), not by the display-time slice.
    dropped_detail = False
    for turn_text, stored in zip(row["evidence_turn_texts"], stored_list):
        if (
            turn_text
            and gold
            and len(stored) <= 200
            and not gold_words_present(gold, stored)
            and gold_words_present(gold, turn_text)
        ):
            dropped_detail = True
    if any_consolidated or dropped_detail:
        cats.append("d_consolidated_or_dropped_detail")

    any_relative = any(_RELATIVE_TIME_RE.search(t) for t in row["evidence_turn_texts"])
    if any_relative and _DATE_QUESTION_RE.search(question):
        cats.append("b_relative_time")

    any_user_tag = any("[User]" in line for line in ev_lines)
    # Category (a) needs the question to plausibly be asking about the NAMED user (not "you"),
    # so a first-person pronoun in the line is genuinely ambiguous without the name.
    pronoun_evidence = any(re.search(r"\b(I|my|me|mine)\b", t) for t in row["evidence_turn_texts"])
    if any_user_tag and pronoun_evidence:
        cats.append("a_speaker_ambiguous")

    # f: caption VISIBLE in what the model actually saw -- a caption present in the dataset turn
    # but sliced off by truncation before it renders is `h_truncated_evidence` instead, not this.
    any_photo_visible = any(_PHOTO_RE.search(line) for line in ev_lines)
    if any_photo_visible:
        cats.append("f_photo_caption")

    single_hop = n_evidence <= 1 and n_resolved_in_block == n_evidence
    # Checked against the RENDERED line, so a truncation-cut line can never satisfy this --
    # "explicit" means the model's own input, as truncated, verbatim contained the answer.
    explicit = single_hop and any(gold and gold_words_present(gold, line) for line in ev_lines)
    if explicit:
        cats.append("e_explicit_evidence_abstained")

    if not cats:
        cats.append("g_other")
    return cats


def build_cases(run_id: str, out_dir: Path, k_override: int | None) -> dict:
    results_file = RESULTS_DIR / f"external_{run_id}.json"
    run_dir = RUNS_DIR / run_id
    results = json.loads(results_file.read_text())
    k = k_override if k_override is not None else int(results["config"]["k"])
    overrides = tuple(results["config"].get("overrides") or ())

    # Step 1: reuse diagnose_recall's evidence-mapping/retrieval-recall pipeline verbatim to
    # find the target set (WRONG + evidence_class == "retrieved") -- this also does the
    # required DB-copy-before-open into `out_dir` (never opens run_dir's DBs directly).
    diag = diagnose_recall.build_diagnostics(
        results_file=results_file, run_dir=run_dir, diag_dir=out_dir
    )
    target_rows = [
        r for r in diag["rows"] if r["label"] == "WRONG" and r["evidence_class"] == "retrieved"
    ]

    dataset_convs = load_locomo(LOCOMO_PATH)
    dataset_by_id = {c.conv_id: c for c in dataset_convs}
    question_by_qid = {q.qid: q for c in dataset_convs for q in c.questions}

    by_conv: dict[str, list[dict]] = defaultdict(list)
    for r in target_rows:
        by_conv[r["conv_id"]].append(r)

    restores = apply_overrides(_get_config(), overrides) if overrides else []
    try:
        cases: list[dict] = []
        verify_mismatches: list[dict] = []
        for conv_id in sorted(by_conv):
            dataset_conv = dataset_by_id[conv_id]
            db_path = out_dir / f"{conv_id}.db"  # already copied by build_diagnostics above
            engine = Engine(db_path=str(db_path), llm=cast(LLMProvider, FakeLLM()), embedding=embedder())
            char = engine.get_character(conv_id)
            char.background = False
            char.parallel = False
            char.enforce_consistency = False

            for diag_row in by_conv[conv_id]:
                q = question_by_qid[diag_row["qid"]]
                block, pinned, facts, mem_text, filtered, line_by_id = reconstruct_block(
                    dataset_conv, q, char, dataset_conv.user_name, k
                )
                cases.append(_build_case(diag_row, block, pinned, facts, filtered, line_by_id))
        clock.override(None)
    finally:
        for section, key, prev in reversed(restores):
            setattr(getattr(_get_config(), section), key, prev)

    # Cross-check reconstructed memory count against the run's own recorded `memories_used`,
    # for every case (cheap once already computed) -- not just a sample, since it's free here.
    qa_records = {
        r["qid"]: r for r in results["conversations"] if r.get("kind") == "qa"
    }
    for c in cases:
        rec = qa_records.get(c["qid"], {})
        c["memories_used_recorded"] = rec.get("memories_used")
        c["memories_used_reconstructed"] = len(c["top20_ids"])
        c["memories_used_match"] = c["memories_used_recorded"] == c["memories_used_reconstructed"]
        if not c["memories_used_match"]:
            verify_mismatches.append(
                {"qid": c["qid"], "recorded": c["memories_used_recorded"], "reconstructed": c["memories_used_reconstructed"]}
            )

    for c in cases:
        c["categories"] = classify_row(c)
        c["primary_category"] = c["categories"][0]

    return {
        "run_id": run_id,
        "k": k,
        "n_target": len(cases),
        "cases": cases,
        "verify_mismatches": verify_mismatches,
    }


def _get_config():
    from woven_imprint.config import get_config

    return get_config()


def _build_case(diag_row, block, pinned, facts, filtered, line_by_id) -> dict:
    top20_ids = [m["id"] for m in filtered]
    evidence_lines: list[str] = []
    evidence_turn_texts: list[str] = []
    evidence_turn_texts_as_stored: list[str] = []
    for d in diag_row["evidence_details"]:
        matched = (set(d["exact_ids"]) | set(d["fuzzy_ids"])) & set(top20_ids)
        for mid in matched:
            line = line_by_id.get(mid, "")
            if line:
                evidence_lines.append(line)
                evidence_turn_texts.append(d["turn_text"])
                mem = next((m for m in filtered if m["id"] == mid), None)
                evidence_turn_texts_as_stored.append((mem or {}).get("content", ""))

    response = diag_row["response"]
    any_truncated = any(len(s) > 200 for s in evidence_turn_texts_as_stored)
    return {
        "qid": diag_row["qid"],
        "conv_id": diag_row["conv_id"],
        "category_locomo": diag_row["category"],
        "question": diag_row["question"],
        "gold": diag_row["gold"],
        "response": response,
        "response_class": diag_row["response_class"],
        "abstained_recomputed": metrics.is_abstention(response),
        "evidence_dia_ids": diag_row["evidence_dia_ids"],
        "evidence_details": diag_row["evidence_details"],
        "evidence_lines_in_block": evidence_lines,
        "evidence_turn_texts": evidence_turn_texts,
        "evidence_turn_texts_as_stored": evidence_turn_texts_as_stored,
        # Broader signal than the strict `h_truncated_evidence` category: True whenever ANY
        # evidence memory's full stored content exceeds the 200-char render cap, whether or not
        # the specific gold words happen to survive into the visible prefix. Reported as an
        # aggregate (not a classification bucket on its own) because a half-sentence fragment
        # plausibly reads as "incomplete" to the model even when the literal answer words survive.
        "any_evidence_truncated": any_truncated,
        "top20_ids": top20_ids,
        "pinned_block": pinned,
        "facts_block": facts,
        "full_block": block,
    }


# ── Reporting ────────────────────────────────────────────────────────────


def render_category_report(data: dict) -> str:
    cases = data["cases"]
    lines = [
        f"# Abstain-with-evidence category report — {data['run_id']}",
        "",
        f"Target set: WRONG questions with evidence in the top-{data['k']} block — "
        f"n={data['n_target']}.",
        "",
        "## Category descriptions",
        "",
    ]
    for cat, desc in CATEGORY_DESCRIPTIONS.items():
        lines.append(f"- **{cat}**: {desc}")
    lines.append("")

    lines.append("## Primary-category counts (abstained vs answered_wrong)")
    lines.append("")
    lines.append("| category | abstained | answered_wrong | total |")
    lines.append("|---|---:|---:|---:|")
    by_cat: dict[str, Counter] = defaultdict(Counter)
    for c in cases:
        by_cat[c["primary_category"]][c["response_class"]] += 1
    for cat in CATEGORY_DESCRIPTIONS:
        ab = by_cat[cat]["abstained"]
        aw = by_cat[cat]["answered_wrong"]
        if ab + aw == 0 and cat not in by_cat:
            continue
        lines.append(f"| {cat} | {ab} | {aw} | {ab + aw} |")
    lines.append("")

    lines.append("## Any-match counts (a row may match multiple categories)")
    lines.append("")
    lines.append("| category | n_matching |")
    lines.append("|---|---:|")
    any_counts: Counter = Counter()
    for c in cases:
        for cat in c["categories"]:
            any_counts[cat] += 1
    for cat in CATEGORY_DESCRIPTIONS:
        lines.append(f"| {cat} | {any_counts[cat]} |")
    lines.append("")

    lines.append("## Primary category x LoCoMo category")
    lines.append("")
    locomo_cats = sorted({c["category_locomo"] for c in cases})
    header = "| category | " + " | ".join(f"loco-{lc}" for lc in locomo_cats) + " | total |"
    lines.append(header)
    lines.append("|---|" + "---:|" * (len(locomo_cats) + 1))
    for cat in CATEGORY_DESCRIPTIONS:
        row_cases = [c for c in cases if c["primary_category"] == cat]
        if not row_cases:
            continue
        counts = Counter(c["category_locomo"] for c in row_cases)
        cells = " | ".join(str(counts.get(lc, 0)) for lc in locomo_cats)
        lines.append(f"| {cat} | {cells} | {len(row_cases)} |")
    lines.append("")

    n_any_trunc = sum(1 for c in cases if c["any_evidence_truncated"])
    lines.append("## Broader truncation signal (not a classification bucket)")
    lines.append("")
    lines.append(
        f"{n_any_trunc}/{len(cases)} cases have at least one evidence memory whose full stored "
        "content exceeds `_format_memories`'s 200-char render cap (so the model never saw the "
        "un-cut text at all, regardless of whether the specific gold words happen to fall before "
        "or after char 200). Only the subset where this demonstrably ate the gold words is "
        "counted under the strict `h_truncated_evidence` category above; this wider number is "
        "the real exposure -- any of these cases could plausibly read to the model as an "
        "incomplete, cut-off sentence."
    )
    lines.append("")

    mism = data["verify_mismatches"]
    lines.append("## memories_used verification")
    lines.append("")
    lines.append(
        f"{data['n_target'] - len(mism)}/{data['n_target']} cases: reconstructed top-{data['k']} "
        f"count matches the run's own recorded `memories_used`."
    )
    if mism:
        lines.append("")
        lines.append("Mismatches:")
        for m in mism[:20]:
            lines.append(f"- {m['qid']}: recorded={m['recorded']}, reconstructed={m['reconstructed']}")
    lines.append("")

    return "\n".join(lines)


def _first_example_of_shape(out_dir: Path, cases: list[dict]) -> dict[str, str]:
    """Scan the already-copied per-conversation DBs for one representative line each of: a user
    turn, a character turn, a core fact (from the facts block, not a memory row), and a
    [Consolidated] summary. Returns raw text quotes (not full case objects)."""
    import sqlite3

    quotes: dict[str, str] = {}
    conv_ids = sorted({c["conv_id"] for c in cases})
    for conv_id in conv_ids:
        db_path = out_dir / f"{conv_id}.db"
        if not db_path.exists():
            continue
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, tier, status, role, content, created_at FROM memories "
                "WHERE character_id = ? AND status='active' ORDER BY rowid",
                (conv_id,),
            ).fetchall()
        finally:
            conn.close()
        for r in rows:
            content = r["content"] or ""
            if "user_turn" not in quotes and r["tier"] == "buffer" and content.startswith("[User]"):
                quotes["user_turn"] = f"({conv_id}, {r['created_at'][:10] if r['created_at'] else ''}) {content[:220]}"
            elif (
                "character_turn" not in quotes
                and r["tier"] == "buffer"
                and content.startswith("[")
                and not content.startswith("[User]")
            ):
                quotes["character_turn"] = f"({conv_id}, {r['created_at'][:10] if r['created_at'] else ''}) {content[:220]}"
            elif "consolidated" not in quotes and content.startswith("[Consolidated]"):
                quotes["consolidated"] = f"({conv_id}, {r['created_at'][:10] if r['created_at'] else ''}) {content[:400]}"
        if len(quotes) >= 3:
            break
    return quotes


def render_format_quotes(out_dir: Path, cases: list[dict]) -> str:
    quotes = _first_example_of_shape(out_dir, cases)
    # A core fact quote: pull one straight from a reconstructed case's facts_block (real
    # rendered text, byte-identical to what the model saw).
    fact_line = ""
    for c in cases:
        fb = c["facts_block"]
        if fb:
            for line in fb.split("\n"):
                if line.startswith("- (since"):
                    fact_line = line
                    break
        if fact_line:
            break

    lines = [
        "# Rendered memory-row shapes — one quote of each",
        "",
        "## User turn (buffer tier, `_format_memories`/`ingest` prefix)",
        "",
        f"    {quotes.get('user_turn', '(none found)')}",
        "",
        "Note the literal `[User]` tag — never the person's actual name (`ingest()` hardcodes "
        '`prefix = "[User]" if role == "user" else f"[{self.name}]"`), even though the facts '
        "block below addresses them by name.",
        "",
        "## Character turn (buffer tier)",
        "",
        f"    {quotes.get('character_turn', '(none found)')}",
        "",
        "## Core fact (from `_format_facts_block`, not a memory-row line)",
        "",
        f"    {fact_line or '(none found)'}",
        "",
        "## Consolidated summary (core tier, `[Consolidated]` prefix)",
        "",
        f"    {quotes.get('consolidated', '(none found)')}",
        "",
        "## What metadata exists but is NOT rendered",
        "",
        "- **Speaker identity for user turns**: the memory row's `role` column is `'user'` and "
        "the conversation's actual speaker name (`conv.user_name`, e.g. `Caroline`) is known to "
        "the harness — neither reaches the rendered line, which only ever says `[User]`.",
        "- **`session_id`**: every memory row carries the session it was formed in "
        "(`self.memory.add(..., session_id=self._session_id, ...)`); `_format_memories` never "
        "prints it, so the model can't tell two memories came from the same conversation turn "
        "or session vs. weeks apart other than via the printed date.",
        "- **Absolute vs. relative date**: the row's real `created_at` timestamp (full "
        "date+time) is reduced to a `(YYYY-MM-DD, N days/months ago)` prefix — the *time of day* "
        "is dropped, and the relative phrase is computed against the *question's* asked_at, not "
        "against any relative-time phrase embedded in the turn's own text (e.g. \"next month\") "
        "— the model has to combine the two itself.",
        "- **`certainty`**: only shown when < 0.5 (`\" (uncertain)\"` tag); a certainty of "
        "e.g. 0.6 renders identically to 1.0.",
        "- **`metadata.consolidated_into` / provenance links**: a consolidated summary's source "
        "turns (kept retrievable per `memory.consolidation_keep_sources`) carry no visible link "
        "back to the summary or to each other in the rendered text.",
        "",
    ]
    return "\n".join(lines)


def _stratified_sample(cases: list[dict], n: int, seed: int) -> list[dict]:
    """Pick ``n`` cases covering every category present at least once (round-robin over
    categories in ``CATEGORY_DESCRIPTIONS`` order, one per pass) before filling any remaining
    slots randomly — a pure random sample of ``n`` would likely miss the smallest categories
    (e.g. `e_explicit_evidence_abstained`, n=13) entirely."""
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_cat[c["primary_category"]].append(c)
    for pool in by_cat.values():
        rng.shuffle(pool)

    chosen: list[dict] = []
    cat_order = [cat for cat in CATEGORY_DESCRIPTIONS if by_cat.get(cat)]
    idx = 0
    while len(chosen) < n and any(by_cat[cat] for cat in cat_order):
        cat = cat_order[idx % len(cat_order)]
        if by_cat[cat]:
            chosen.append(by_cat[cat].pop())
        idx += 1
    return chosen


def render_worked_examples(cases: list[dict], n: int, seed: int) -> str:
    chosen = _stratified_sample(cases, n, seed)

    lines = [f"# {n} worked examples", ""]
    for i, c in enumerate(chosen, 1):
        lines.append(f"## {i}. {c['qid']} (loco-cat {c['category_locomo']}, {c['response_class']}, "
                     f"primary={c['primary_category']})")
        lines.append("")
        lines.append(f"- **Question**: {c['question']!r}")
        lines.append(f"- **Gold**: {c['gold']!r}")
        lines.append(f"- **Response**: {c['response']!r}")
        if c["evidence_lines_in_block"]:
            lines.append("- **Evidence line(s) verbatim, as rendered in the block**:")
            for line in c["evidence_lines_in_block"]:
                lines.append(f"  - `{line.strip()}`")
        else:
            lines.append(
                "- **Evidence line(s)**: none of the evidence dia_ids resolved into the "
                "reconstructed top-K set (multi-hop gap — see category c)."
            )
        lines.append(f"- **All matched categories**: {', '.join(c['categories'])}")
        lines.append(f"- **Plausible fix**: {_plausible_fix(c)}")
        lines.append("")
    return "\n".join(lines)


def _plausible_fix(c: dict) -> str:
    fixes = {
        "c_multi_hop": (
            "Raise K or improve multi-hop retrieval so all needed evidence lines co-occur in one "
            "block — PRODUCT (retrieval), out of scope for a rendering/prompt fix."
        ),
        "h_truncated_evidence": (
            "Raise (or remove) `_format_memories`'s hard `[:200]` character slice per memory — "
            "PRODUCT. The cheapest, highest-confidence fix found in this analysis."
        ),
        "d_consolidated_or_dropped_detail": (
            "Tighten the consolidation summarization prompt to preserve concrete details (dates, "
            "names, numbers) rather than gisting them away — PRODUCT."
        ),
        "b_relative_time": (
            "Have `_format_memories` pre-resolve relative-time phrases in the stored content at "
            "ingest/format time (or explicitly instruct the QA prompt that the memory's own "
            "'(YYYY-MM-DD, ...)' prefix, not 'Today is', is the anchor for phrases inside the "
            "line) — PRODUCT (rendering) + HARNESS (instruction wording)."
        ),
        "a_speaker_ambiguous": (
            "Render the user's actual name instead of the generic '[User]' tag "
            "(`Character.ingest`/`ingest_exchange` prefix) — PRODUCT."
        ),
        "f_photo_caption": (
            "Render photo-caption turns with an explicit 'In a shared photo, X ...' framing "
            "instead of a parenthetical aside so the caption detail reads as a first-class fact "
            "— PRODUCT."
        ),
        "e_explicit_evidence_abstained": (
            "Loosen or reword the abstention instruction ('Not mentioned') / lift the 15-word cap "
            "for a case with explicit evidence present — HARNESS."
        ),
        "g_other": "No single rendering/prompt change stands out from the rubric; needs its own read.",
    }
    return fixes.get(c["primary_category"], "")


def render_recommendation(data: dict) -> str:
    cases = data["cases"]
    n = data["n_target"]
    counts = Counter(c["primary_category"] for c in cases)
    e_count = counts["e_explicit_evidence_abstained"]
    a_count = counts["a_speaker_ambiguous"]
    b_count = counts["b_relative_time"]
    d_count = counts["d_consolidated_or_dropped_detail"]
    c_count = counts["c_multi_hop"]
    f_count = counts["f_photo_caption"]
    h_count = counts["h_truncated_evidence"]
    n_any_trunc = sum(1 for c in cases if c["any_evidence_truncated"])
    abstain_total = sum(1 for c in cases if c["response_class"] == "abstained")
    any_counts = Counter(cat for c in cases for cat in c["categories"])
    e_any = any_counts["e_explicit_evidence_abstained"]

    lines = [
        f"# Recommendation — {data['run_id']} abstain-with-evidence ({n} cases)",
        "",
        "## Headline finding: a rendering bug, not (mainly) the abstention instruction",
        "",
        f"`Character._format_memories` hard-slices every memory's `content` to `[:200]` chars "
        "before it ever reaches the model. "
        f"**{n_any_trunc}/{n} ({n_any_trunc / n:.0%})** of these WRONG-but-evidence-in-block "
        f"cases have at least one evidence memory whose full content exceeds that cap — the "
        "model was shown a sentence fragment that stops mid-thought, not the full evidence line "
        f"the run's own recall accounting credited it with. In **{h_count}/{n} ({h_count / n:.0%})** "
        "of cases this is demonstrable, not just plausible: the gold answer's own words are "
        "present in the full stored content and specifically fall after the 200-char cut, so "
        "they were never rendered at all. This was not in the brief's suggested rubric — it only "
        "surfaced from reading real reconstructed blocks — and it is the single largest, cheapest, "
        "highest-confidence fix found in this analysis.",
        "",
        "## Is the abstention instruction the main cause?",
        "",
        f"{abstain_total}/{n} ({abstain_total / n:.0%}) of these cases abstained outright "
        '(\"Not mentioned\") despite evidence sitting in the block. Of those, '
        f"{e_count} are classified `e_explicit_evidence_abstained` — single-hop, the RENDERED "
        "(post-truncation) evidence line verbatim contains the gold answer, model still "
        "declined. That is real signal that the QA_SYSTEM rule (\"If the memories do not contain "
        "the answer, reply exactly: Not mentioned\") over-fires — but at "
        f"{e_count}/{n} ({e_count / n:.0%}) it is smaller than the truncation exposure above. "
        "**Flag: the abstention instruction is a real, secondary lever, not the main cause** — "
        f"it plausibly compounds every other category (a model handed a cut-off fragment, an "
        "unnamed '[User]' pronoun, or a bare relative-time phrase has an easy escape hatch in "
        "'Not mentioned' rather than committing to an inference), but the truncation numbers "
        "above show the model is often failing on incomplete input, not on complete input it "
        f"chose to decline. Supporting that reading: {e_any}/{n} cases have the gold words "
        f"present verbatim in a single-hop rendered line at all (the broader, any-match count), "
        f"but only {e_count} of those have NO other confound (no `[User]` ambiguity, no "
        "multi-hop gap, no truncation) — most 'evidence was explicit' cases are ALSO ambiguous "
        "in some other way, so a clean 'the instruction alone is at fault' story only fits a "
        f"minority ({e_count}/{n}) of cases.",
        "",
        "## Ranked changes",
        "",
        "| # | change | side | cases plausibly addressed |",
        "|---|---|---|---:|",
        f"| 1 | Raise (or remove) `_format_memories`'s per-memory `[:200]` character cap | "
        f"PRODUCT | ~{h_count} demonstrable, up to {n_any_trunc} exposed (`h_truncated_evidence` "
        f"+ broader truncation signal) |",
        f"| 2 | Improve multi-hop coverage (raise K conditionally, or a second retrieval pass "
        f"keyed on partial matches) so co-dependent evidence lines land in the same block | "
        f"PRODUCT (retrieval, not rendering) | ~{c_count} (`c_multi_hop`) |",
        f"| 3 | Audit consolidation fidelity directly (this 251-case set structurally can't see "
        f"it -- 0/251 evidence memories here are `[Consolidated]`, because the evidence-location "
        f"method only counts near-verbatim matches as 'stored'; a lossy consolidated paraphrase "
        f"would show up as `not_stored`/`stored_archived_only` WRONG rows instead -- follow-up: "
        f"rerun this same reconstruction against THAT bucket) | PRODUCT | {d_count} found here "
        f"(structurally undercounted, see caveat) |",
        f"| 4 | Teach `_format_memories` (or the ingested content) to resolve relative-time "
        f"phrases against the memory's own formed-date, not leave the model to combine the "
        f"printed date with an untouched 'next month'/'last Saturday' in the text | PRODUCT | "
        f"~{b_count} (`b_relative_time`) |",
        f"| 5 | Render the real user name instead of the generic `[User]` tag in "
        f"`Character.ingest`/`ingest_exchange` | PRODUCT | ~{a_count} (`a_speaker_ambiguous`) |",
        f"| 6 | Render photo-caption turns as a first-class fact line rather than a parenthetical "
        f"aside | PRODUCT | ~{f_count} (`f_photo_caption`) |",
        f"| 7 | Reword/relax the abstention instruction (require the model to attempt an answer "
        f"from any evidence-bearing line before it may reply 'Not mentioned'; or move the "
        f"15-word cap off explanatory reasoning) | HARNESS | ~{e_count} direct (`e_*`), plausibly "
        f"a fraction of the other {n - e_count} too since abstention is available as an escape "
        f"hatch in any ambiguous or incomplete case |",
        "",
        "PRODUCT changes are justified independent of this benchmark: the 200-char cap silently "
        "truncates ANY character's memories in ANY deployment, not just this eval (it is plain "
        "information loss, unconditional on LoCoMo); `[User]` vs. a real name is a "
        "personalization gap wherever the user's name is known (Telegram, SillyTavern character "
        "card, etc.); relative-time resolution, consolidation fidelity, and multi-hop retrieval "
        "quality all affect any long-running character's recall quality; photo-caption framing "
        "affects any multimodal ingestion. HARNESS changes (item 7, and the QA prompt's word "
        "cap/'Not mentioned' wording generally) must be applied identically to both memory mode "
        "and full-context mode to keep the two comparable — changing only the memory-mode prompt "
        "would confound 'we fixed retrieval/rendering' with 'we fixed the answer prompt'.",
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--run-id", default=RUN_ID, help=f"Run id to analyze (default: {RUN_ID!r}).")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write outputs into (default: eval/external/runs/diagnostics/abstain/<run-id>/).",
    )
    parser.add_argument("--k", type=int, default=None, help="Override top-K (default: run's own config.k).")
    parser.add_argument("--n-examples", type=int, default=12, help="Number of worked examples (default 12).")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for worked-example sampling.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else ABSTAIN_ROOT / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_cases(args.run_id, out_dir, args.k)

    (out_dir / "cases.jsonl").write_text(
        "\n".join(json.dumps(c, default=str) for c in data["cases"]) + "\n"
    )
    (out_dir / "category_report.md").write_text(render_category_report(data))
    (out_dir / "format_quotes.md").write_text(render_format_quotes(out_dir, data["cases"]))
    (out_dir / "worked_examples.md").write_text(
        render_worked_examples(data["cases"], args.n_examples, args.seed)
    )
    (out_dir / "recommendation.md").write_text(render_recommendation(data))

    print(f"n_target={data['n_target']} k={data['k']}")
    print(f"mismatches={len(data['verify_mismatches'])}")
    for cat in CATEGORY_DESCRIPTIONS:
        n = sum(1 for c in data["cases"] if c["primary_category"] == cat)
        print(f"  {cat}: {n}")
    print(f"wrote outputs under {out_dir}")


if __name__ == "__main__":
    main()
