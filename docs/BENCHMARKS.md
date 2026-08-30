# External Benchmarks: LoCoMo, LoCoMo-Plus, LongMemEval-S

## Purpose

Publish honest, reproducible numbers for Woven Imprint on the standard long-term-conversational-
memory benchmarks (LoCoMo, LoCoMo-Plus, LongMemEval-S), measured with a **local** judge
(Qwen3.5-35B-A3B-FP8 via `vllm-brain`, thinking disabled) and no paid APIs, alongside a
**full-context baseline under the same judge** so the reader can see what the memory system adds
or loses — not just an absolute score. The harness (`eval/external/`), the exact prompts, the
judge prompt, and a judge-calibration sample are all published alongside the numbers; see
[Reproduce](#reproduce) to re-run it yourself.

This is Tier 3b of the audit roadmap ("publish LoCoMo/LongMemEval numbers with the exact
harness"). Spec: `docs/superpowers/specs/2026-08-27-tier3b-external-benchmarks.md`.

## Datasets

| Dataset | Size | Source |
|---|---|---|
| **LoCoMo** | 10 conversations, 1,986 questions (282 / 321 / 96 / 841 category 1–4 + 446 category 5 adversarial) | [snap-research/locomo](https://github.com/snap-research/locomo), `data/locomo10.json` |
| **LoCoMo-Plus** (Cognitive category) | 401 cue/trigger probes: causal / state / goal / value relation types | [xjtuleeyf/Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus), `data/locomo_plus.json` — companion to ARR 2026 submission, [arXiv:2602.10715](https://arxiv.org/abs/2602.10715) |
| **LongMemEval-S** | 500 questions, 6 types, 30 `_abs` (abstention) questions, ~48 sessions / ~494 turns of haystack per question — **50 of 500 run (see Reproduce)** | [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned), `longmemeval_s_cleaned.json` |

LoCoMo's `category` (an int) maps: 1 = multi-hop, 2 = temporal, 3 = open-domain/common-sense,
4 = single-hop (all four scored against `answer`); 5 = adversarial (no in-conversation answer
exists — the gold behavior is abstention; scored separately, see
[Category 5 / abstention](#category-5--abstention)).

LongMemEval-S's `question_type` is one of: `single-session-user`, `single-session-assistant`,
`single-session-preference`, `multi-session`, `temporal-reasoning`, `knowledge-update`.
`question_id` values ending in `_abs` (30 of 500) are the abstention subset.

Download (streamed, size-verified against the byte counts above; files are gitignored under
`eval/external/data/`):

```bash
.venv/bin/python -m eval.external fetch locomo locomo_plus longmemeval_s
```

| file | on-disk size |
|---|---|
| `locomo.json` | 2,805,274 bytes |
| `locomo_plus.json` | 305,737 bytes |
| `longmemeval_s.json` | 277,383,467 bytes |

`fetch` also knows `longmemeval_oracle` (15,388,478 bytes; downloaded for reference, not run by
this harness — LongMemEval-M/-Oracle and MemConflict are out of scope, see the spec).

## Protocol

The protocol follows the Mem0/Zep lineage, made explicit end to end.

### Ingestion

Message by message, via `Character.ingest(role, content, user_id)`. Speaker mapping per
conversation: the dataset's first speaker (`speaker_a`) becomes the **user**
(`user_id` = speaker name), the second speaker (`speaker_b`) becomes the **character** (its name
= speaker name); `role="user"` / `"assistant"` accordingly. Session boundaries follow the
dataset's sessions: `clock.override(session datetime)` before each session, `clock.advance(30s)`
per turn so `created_at` stays monotonic, `char.end_session()` after each session (one summary
call). `memory.fact_extraction_interval = 1` for LoCoMo runs (every turn triggers bookkeeping —
cost accepted for maximum recall fidelity). `Character.unified_assessment = True`, so `ingest()`
routes through the same single-call emotion/relationship/beat/facts bookkeeping path `chat()`
uses — **this also means mood and narrative arc update on ingested turns**, not just structured
facts; see [Caveats](#caveats).

LoCoMo image turns carry a `blip_caption`; these are folded into the turn text as
`"{text} (shared a photo: {caption})"` (or just the caption clause if the turn has no text of its
own) before ingestion — 1,226 turns and 857 of the 1,986 questions have such a turn as evidence,
so dropping the caption would silently break recall on a meaningful slice of the benchmark.

LongMemEval-S: **pair-turn ingestion** (`--pair-turns`, on by default for this bench). Each
consecutive (user, assistant) pair within a session is ingested with one
`Character.ingest_exchange(user_message, response, user_id)` call — one unified bookkeeping call
per exchange instead of one per turn, halving bookkeeping calls (the 50-question sample has
24,639 turns in 2,380 sessions → ≈12.3k exchange calls + 2,380 session-summary calls). Unpaired
turns (two user turns in a row, a leading assistant turn, a trailing user turn) fall back to
`Character.ingest()`; the clock advances 30 s per original turn either way.
`fact_extraction_interval` counts *exchanges* here (one `_turn_count` increment per pair). Note
that under unified assessment the interval does **not** reduce LLM calls — `_run_bookkeeping`
always makes its one assessment call (emotion/relationship) and the interval only gates whether
that call also extracts facts — so LongMemEval-S runs use `--interval 1`, the same as LoCoMo.

Session boundary handling: LongMemEval-S session dates are **not** chronological in the source
file for 211 of 500 questions — this loader sorts each question's `haystack_sessions` by
`haystack_dates` before ingesting, so recency/temporal-reasoning ordering is sane regardless of
file order.

### Answering

For each question, under `clock.override(question's asked_at)` — LoCoMo: last session date + 1
day; LongMemEval-S: the dataset's `question_date` — the prompt is assembled exactly as the
product's real prompt-assembly path builds it: pinned block (`char._format_pinned_block()`) +
facts block (`char._format_facts_block(user_id, pinned_ids)`) + up to K = 20 dated memories from
`char.retriever.retrieve(question, limit=20, relationship_target=user_id)` (pinned memories
already in the block are filtered out of this list so nothing repeats), rendered via
`char._format_memories(...)`. Instruction: answer in at most 15 words, use only the memories,
convert relative dates to absolute, and reply exactly `Not mentioned` if the memories don't
contain the answer. Model: the same local brain, thinking off, temperature 0, max_tokens 60.

### Full-context baseline

No `Engine`/database at all — the question is answered directly from the conversation's flat
transcript text (`[{date} {time}] {speaker}: {text}` per line), truncated from the **end** (the
most recent, most-likely-relevant material) if it would exceed 220,000 characters before being
sent as the answering prompt. This never actually triggers for LoCoMo (max transcript
~117,075 chars — well under the limit). See [Baselines](#baselines) for why LongMemEval-S has no
full-context run.

### LoCoMo-Plus answering

Memory mode reuses the base LoCoMo memory-mode run's ingested database (WAL-checkpointed and
copied per probe — see [Determinism/resumability](#determinismresumability)): under
`clock.override(cue_time)`, `char.start_session()`, ingest the cue's turns (A → user, B →
character), `char.end_session()`; then under `clock.override(query_time)`,
`char.start_session()`, `response = char.chat(trigger_A_line, user_id=user_name)` — the real
product chat path, at its **default temperature 0.7** (not overridden — this is the real product
default, not a benchmark-specific choice), max output governed by the product's own chat
settings. This means memory mode's retrieval and generation budget are **not** the QA path's
`K=20`/60-token answer: `char.chat()` retrieves the product's own top-10 memories
(`retriever.retrieve(..., limit=10)`) and generates with `max_tokens=2048`, the same as any live
product conversation. `query_time` = the base conversation's last session time + 7 days
(upstream's `query_time`); `cue_time` = `query_time` − `time_gap` (upstream's `cue_time`).

Full-context mode: no Engine/DB — the stitched transcript (every base session, then the cue, then
the trigger, time-ordered, each block headed by `DATE: ...`, turns rendered as
`{speaker} said, "{text}"`) plus an instruction to reply as `speaker_b`, sent at temperature 0.3,
max_tokens 200 (per the spec's protocol; a fixed, benchmark-chosen setting, unlike memory mode's
product-default temperature).

Both modes are judged with the same Cognitive judge (see [Prompts](#prompts)); evidence = the cue
turns rendered as `"{speaker}: {text}"` lines.

### Judge

The LoCoMo/Mem0-lenient judge (quoted verbatim in [Prompts](#prompts)): CORRECT if the response
conveys the same fact as the gold answer (paraphrases, partial dates that agree with the gold
date, and equivalent relative/absolute time expressions are all tolerated); WRONG if the response
contradicts the gold answer, misses the key fact, or answers a different question. The judge runs
at temperature 0 and is given a "Reference date" (the question's `asked_at` date) so it can
resolve relative time expressions in the response the same way the answering model was told
"Today is {date}".

### Category 5 / abstention

Category 5 (LoCoMo) and `_abs` (LongMemEval-S) questions are **not** sent to the LLM judge at
all — the gold behavior for these is declining to answer, so there's nothing for a judge to
usefully compare against. Instead they're scored by a fixed abstention rule
(`eval/external/metrics.py::is_abstention`): the response matches one of a fixed set of
abstention-phrase patterns ("not mentioned", "does not mention", "cannot answer", "no
information", "don't know", "unable to find", …, case-insensitive) **and** is at most 12
whitespace-separated words. The word-count cap exists only to stop a long, otherwise-substantive
answer that happens to contain an abstention-shaped fragment from being misclassified — it is a
deliberately simple rule, not an NLI judgment, and does not special-case "opens with an
abstention phrase but then supplies a concrete answer" (a short reply doing both, e.g.
"Not mentioned; she adopted a dog", is still counted as an abstention). These are two distinct
summary keys, `adversarial_accuracy` (LoCoMo) and `abstain_accuracy` (LongMemEval-S), each `null`
when a run has no items of that kind. Per-category rows are computed over judged (`kind == "qa"`)
items only; rule-scored items are reported separately (`rule_scored`), so the LoCoMo "5" row is
the abstention accuracy. LongMemEval-S's `_abs` count is whatever the seeded sample contains: the
50-question sample holds only 2 `_abs` questions (one `knowledge-update`, one
`single-session-user`), so its `abstain_accuracy` is anecdotal, not a measurement. Because every
LongMemEval-S question carries its own haystack, `--delete-db-after-answer` removes a
conversation's SQLite DB (plus `-wal`/`-shm`) once all its questions are answered and judged,
keeping only the JSON checkpoints. Never pass it on a LoCoMo memory run that a later
`--reuse-run` (LoCoMo-Plus) depends on.

### Metrics

- **J-score**: fraction of `kind == "qa"` questions judged CORRECT — overall (categories 1–4 /
  non-`_abs`) and per category.
- **Token-F1**: SQuAD-style bag-of-words F1 between the response and the gold answer, order-
  insensitive, lowercased, punctuation-stripped, articles (a/an/the) dropped (see
  `metrics.token_f1`/`_normalize_answer`).
- **Adversarial / abstain accuracy**: fraction of category-5 / `_abs` questions the abstention
  rule marks CORRECT — reported separately from the headline J-score (see above).
- **Mean prompt tokens (est.)**: `len(prompt_block) // 4` per question, averaged.
- **LoCoMo-Plus cognitive accuracy**: fraction of probes the Cognitive judge marks `"correct"`,
  overall and broken down by `relation_type` and `time_gap`.
- **Ingestion**: LLM call count and wall-clock per conversation (`ingest_totals` in the results
  JSON).

### Determinism/resumability

One SQLite database per conversation, under `eval/external/runs/<run_id>/<conv_id>.db`, plus a
`<conv_id>.ingest.json` checkpoint (ingestion done) and a `<conv_id>.answers.json` checkpoint
(list of fully-judged per-question records, rewritten atomically — via a same-directory temp file
+ `Path.replace`, which is atomic on POSIX — after every question). A killed run loses at most one
in-flight question; re-running `run()` skips whatever is already on disk. Seeds are fixed
(default 7) for LongMemEval-S's stratified sampling and the judge-calibration sample.

LoCoMo-Plus memory mode copies its base conversation's `.db` per probe rather than sharing one
connection: before every copy, the source DB is forced through
`PRAGMA wal_checkpoint(TRUNCATE)` and the copy refuses to proceed if the `-wal` sidecar is still
non-empty afterward (another connection still has it open, or the base run hasn't finished) —
copying a DB with recent writes still sitting in its WAL would silently produce garbage probes.

A `run(shard=None, write_results=True)` aggregation call (the "collect everything and publish"
step after parallel `--shard i/n` workers finish) refuses to run while any conversation still
looks mid-ingestion (`.db` without `.ingest.json`) or mid-answering (fewer `.answers.json` records
than the conversation has questions) — `--force-aggregate` bypasses this for a deliberately
partial run.

### Judge calibration

60 judged `kind == "qa"` decisions, stratified by `(category, verdict)`, are written to
`eval/results/external_judge_sample.json` (keyed `"bench:mode"`, same convention as the results
file) for human review — each record carries the question, gold answer, model response, judge
label, and judge reason, so a reviewer can spot-check whether the local judge's CORRECT/WRONG
calls look right without re-reading all ~1,540+ judged questions. LoCoMo-Plus judge decisions are
not sampled this way (its records carry `relation_type`/`time_gap`, not `category`/gold answer —
forcing them into the same stratification wasn't worth it; the full judged `probes` list is
already in that bench's results file for spot-checking).

## Prompts

Verbatim from `eval/external/prompts.py` — do not reword these without updating this file (the
module docstring says the same in the other direction).

**QA system prompt** (`QA_SYSTEM`):

```
You answer questions about a person using ONLY the memories provided. Be concise (at most 15 words). Convert relative dates to absolute dates. If the memories do not contain the answer, reply exactly: Not mentioned
```

QA user message (`qa_messages`): `"Today is {today}."`, then the assembled memory/facts block (if
non-empty), then `"Question: {question}"` — joined with blank lines.

**Judge system prompt** (`JUDGE_SYSTEM`):

```
You are an expert grader. Given a question, a gold answer and a model response, decide whether the response conveys the same information as the gold answer. Be lenient: accept paraphrases, partial dates that agree with the gold date, equivalent relative/absolute time expressions, and extra correct detail. Mark WRONG if the response contradicts the gold answer, is missing the key fact, or answers a different question. Relative time expressions in the response are interpreted relative to the reference date.
```

Judge user message (`judge_messages`):

```
Question: {question}
Gold answer: {gold}
Model response: {response}
Reference date (when the question was asked): {asked_on}

Reply with JSON only: {"label": "CORRECT" or "WRONG", "reason": "..."}
```

**LoCoMo-Plus Cognitive judge** (`PLUS_COGNITIVE_JUDGE`) — copied character-for-character from
upstream `xjtuleeyf/Locomo-Plus`'s `evaluation_framework/task_eval/prompt.py`
(`PROMPT_TEMPLATES["Cognitive"]`, fetched 2026-08-27). Upstream sends the whole filled template as
a single flat prompt with no separate system role, so `plus_judge_messages` mirrors that with one
user-role message:

```

You are a Memory Awareness Judge.
Your task: Judge whether the Model Prediction considers or is linked to the Evidence. If there is a clear connection, the answer is correct (score 1); if not, it is wrong (no score).

Labels:
- "correct": The prediction explicitly or implicitly reflects/uses the evidence (memory or constraint). Give 1 point.
- "wrong": The prediction does not show such a link to the evidence. No point.

Memory/Evidence:
{evidence}

Model Prediction:
{pred}

Return your judgment strictly in JSON format:
{"label": "correct"|"wrong", "reason": "<Does the prediction relate to the evidence?>"}

```

## Baselines

Full-context uses the same QA prompt and the same judge as memory mode — the only difference is
what's in the prompt's memory/context block (retrieved memories vs. the raw transcript) — so the
comparison isolates what memory retrieval adds or loses versus "just put the whole conversation
in context".

LoCoMo's transcripts fit comfortably in the local brain's context window (max ~117,075 characters
≈ well under both the 220,000-character truncation limit and the model's 65,536-token
`max_model_len`), so the full-context baseline is a fair, untruncated comparison.

**LongMemEval-S is NOT run in full-context mode.** Its haystacks run roughly 500,000 characters
(≈125,000 tokens) per question — well over the local brain's 64K-token context window — so a
full-context baseline would require truncation severe enough to not really be "full context"
anymore. The spec only requires a LoCoMo full-context baseline; LongMemEval-S is memory-mode only.

## Hardware & runtime

<!-- RUNTIME:MEASURED --> The figures below are measured wall-clock from the publication runs
(`locomo-mem-v1`, `plus-mem-v1`, `locomo-full-v1`, `plus-full-v1`), 2026-08-27/28.

- **Host**: DGX Spark (GB10, 128 GB unified memory, ARM64).
- **Answering + judge model**: Qwen3.5-35B-A3B-FP8 via `vllm-brain` (vLLM Docker), thinking
  disabled (`extra_body={"chat_template_kwargs": {"enable_thinking": False}}` on every call —
  vLLM's chat template otherwise streams Qwen3.5's chain-of-thought straight into the response).
  The production `vllm-brain` service runs with `--max_num_seqs 2`, which was **not** changed for
  this benchmark run (never stop/reconfigure the production brain unasked) — so parallel shard
  workers queue behind that concurrency cap rather than getting full throughput. Measured effect:
  10 parallel `--shard i/10` workers on `locomo-mem-v1` produced an observed aggregate of
  **≈24 LLM calls/minute** and **≈25s** average latency per call under that contention — a single
  uncontended call runs ≈4.4s, so 10 workers bought roughly ~2× serial throughput, not 10×.
  **LoCoMo memory-mode ingest** (10 conversations, 272 sessions, 5,882 turns) issued **6,441**
  bookkeeping+summary calls and took **18:50 → 00:45 (2026-08-27 → 2026-08-28), ≈5 h 55 min**
  wall-clock sharded 10 ways, including two shard restarts after client-side timeouts (shard 4 and
  shard 8, both `openai.APITimeoutError` under queue contention, relaunched from their last
  checkpoint). The sum of per-conversation ingest seconds logged by the shards is **140,944 s**
  (~39.1 h) — this is parallel work summed across the 10 concurrent shards, **not** wall-clock; the
  ≈5 h 55 min figure above is the actual wall-clock.
  **Rejudge** (re-scoring all 1,540 category 1–4 answers with the reference-date judge fix, a
  single unsharded process) took **45 min**. **LoCoMo-Plus memory mode** (401 probes, 5 shards,
  ~5.9 LLM calls/probe) took **1 h 42 min**. **LoCoMo full-context** (5 shards, 1 call/question)
  took **39 min**. **LoCoMo-Plus full-context** (5 shards) took **40 min**.
- **Embeddings**: nomic-embed-text (768-d) via `llama-embed` (llama.cpp), used for memory
  retrieval in memory-mode runs.
- **Per-request timeout**: 900s (`--timeout`, see [Reproduce](#reproduce)) — long enough to
  survive queue contention at `--max_num_seqs 2` without giving up on a call that would have
  succeeded; the default (300s) round-tripped an `openai.APITimeoutError` mid-run once before this
  was raised (see [Caveats](#caveats)).

## Results

**Current defaults (2026-08-30, Tier 3c):** LoCoMo memory J **0.536** vs. full-context 0.696;
LoCoMo-Plus cognitive memory **0.421** vs. full-context 0.135. These are `locomo-mem-v2d` /
`plus-mem-v2d` — `consolidation_keep_sources=true`, `weight_recency=0.1`,
`weight_importance=weight_relationship=0.0` — the config the library ships today, and what
`eval/results/external_latest.json` / [docs/RESULTS.md](RESULTS.md) report under
`locomo:memory` / `locomo_plus:memory`. The section below (`v1`, dated 2026-08-27/28) is the
original Tier 3b measurement, kept as history — see
[Tier 3c: keep sources + relevance-first ranking](#tier-3c-keep-sources--relevance-first-ranking-2026-08-30)
for the full v1→v2→v2b/v2c/v2d progression, the diagnostics that drove each change, and why
LongMemEval-S was not re-run this round.

<!-- RESULTS:BEGIN -->
Measured 2026-08-27/28 on the local judge (see [Hardware & runtime](#hardware--runtime)). Full
per-`bench:mode` tables (rendered by `eval/render_results.py` from
`eval/results/external_latest.json`) are in
[docs/RESULTS.md](RESULTS.md#external-benchmarks-real-llm-local-judge); the tables below add the
memory-vs-full-context comparison view the spec asked for. **These specific tables are the
original Tier 3b (2026-08-27/28) numbers, kept as history — current defaults are summarized
above and detailed in the [Tier 3c section](#tier-3c-keep-sources--relevance-first-ranking-2026-08-30).**

### LoCoMo — memory vs. full-context

Run ids: `locomo-mem-v1` (2026-08-28T02:08:54Z) vs. `locomo-full-v1` (2026-08-28T04:30:21Z).

| metric | memory | full-context | Δ (memory − full-context) |
|---|---|---|---|
| **Overall J** (cat 1–4, n=1,986) | 0.444 | 0.696 | −0.252 |
| — cat 1 multi-hop (n=282) | 0.291 | 0.504 | −0.213 |
| — cat 2 temporal (n=321) | 0.439 | 0.511 | −0.072 |
| — cat 3 open-domain/commonsense (n=96) | 0.167 | 0.302 | −0.135 |
| — cat 4 single-hop (n=841) | 0.528 | 0.875 | −0.347 |
| Token-F1 (overall) | 0.234 | 0.388 | −0.154 |
| Adversarial abstention accuracy (cat 5, n=446) | 0.883 | 0.832 | +0.052 |
| Mean prompt tokens/question | 1,000 | 24,917 | ~25× fewer |
| n_unparsed | 0 | 0 | — |

### LoCoMo-Plus Cognitive — memory (chat) vs. full-context

Run ids: `plus-mem-v1` (2026-08-28T03:49:49Z) vs. `plus-full-v1` (2026-08-28T05:10:32Z).

| metric | memory | full-context | Δ |
|---|---|---|---|
| **Overall cognitive accuracy** (n=401) | 0.332 | 0.135 | +0.197 (2.46×) |
| — causal (n=101) | 0.356 | 0.188 | +0.168 |
| — state (n=100) | 0.440 | 0.170 | +0.270 |
| — goal (n=100) | 0.260 | 0.080 | +0.180 |
| — value (n=100) | 0.270 | 0.100 | +0.170 |
| Mean prompt tokens/probe | 1,035 | 22,883 | ~22× fewer |
| Mean LLM calls/probe | 5.89 (memory: retrieval + chat + judge) | 2.00 (answer + judge) | — |

Largest time-gap buckets (n ≥ 20), cognitive accuracy by mode:

| time gap | n | memory | full-context |
|---|---|---|---|
| one month later | 44 | 0.432 | 0.068 |
| six weeks later | 25 | 0.360 | 0.120 |
| two months later | 59 | 0.186 | 0.085 |
| three months later | 56 | 0.357 | 0.125 |
| four months later | 26 | 0.192 | 0.154 |
| six months later | 33 | 0.273 | 0.152 |

### LongMemEval-S

Memory mode only (full-context does not fit the 64K window). Run id `lme-s-50-v1`, 50-question
stratified sample (seed 7), pair-turn ingestion, `--interval 1`, K = 20. Ingested 24,639 turns /
2,380 sessions with 19,506 bookkeeping + summary calls.

| metric | memory |
|---|---|
| Overall J (48 judged questions) | **0.396** |
| single-session-user (n=7) | 0.857 |
| knowledge-update (n=8) | 0.750 |
| single-session-assistant (n=8) | 0.375 |
| temporal-reasoning (n=8) | 0.375 |
| multi-session (n=9) | 0.111 |
| single-session-preference (n=8) | 0.000 |
| abstention (`_abs`, n=2, rule-scored) | 2/2 — anecdotal |
| token-F1 | 0.242 |
| mean prompt tokens / question | 1,211 |
| n_unparsed | 0 |

Per-type cells rest on 7–9 questions each; treat them as directional. 20 of the 48 judged answers
were "Not mentioned" (retrieval misses), the same failure mode as LoCoMo. `single-session-preference`
scored 0/8: these questions ask the assistant to *apply* a stated preference in a new reply, and the
≤15-word factual-answer prompt is the wrong instrument for them — a follow-up, not a memory finding.

Runtime: the 10-shard run started 2026-08-28 08:20 and the last shard finished 2026-08-29 ≈11:50
(≈27.5 h wall, including four shards restarted at 18:45 after the embedding-length fix); sum of
per-conversation ingest seconds 517,361 (parallel, not wall-clock).

### Interpretation

Memory mode trails full-context by ~25 J points on LoCoMo (0.444 vs. 0.696) while using ~25×
fewer prompt tokens per question (1,000 vs. 24,917) — retrieval is far cheaper but not yet as
accurate as putting the whole transcript in context. The gap is worst on single-hop questions
(cat 4: Δ −0.347) and it traces mostly to retrieval, not generation: in the 60-item
judge-calibration sample, 14 of the 30 WRONG verdicts were the model answering "not mentioned" —
the fact was in the conversation but never retrieved into context. Memory mode is the one place
it beats full-context outright: adversarial abstention accuracy is higher (88.3% vs. 83.2%). On
the LoCoMo-Plus cognitive-cue test the pattern reverses — the memory-based character beats the
full-transcript continuation 2.5× overall (0.332 vs. 0.135) and on every relation type — but both
modes decay as the cue-to-trigger gap widens (memory: ~0.43 at one month down to ~0.19–0.27 by
two to six months), so memory's advantage there is a head start, not immunity to the same
retrieval-difficulty trend. These are local-35B-judge numbers scoring a local-35B answering
model, not comparable to published GPT-4o-judged results; the full-context baseline under the
identical judge is the fair comparison this harness supports.
<!-- RESULTS:END --> On LongMemEval-S (50-question sample) memory mode reaches J 0.396, strong on single-session user facts and knowledge updates (0.86 / 0.75) and weak on multi-session aggregation (0.11).

## Tier 3c: keep sources + relevance-first ranking (2026-08-30)

Spec: `docs/superpowers/specs/2026-08-30-tier3c-recall-keep-sources.md`. Ledger:
`.superpowers/sdd/2026-08-30-tier3c-recall-keep-sources/progress.md`. This tier started from a
single diagnostic finding on the Tier 3b baseline — 86.7% of LoCoMo memory-mode WRONG answers had
their evidence sitting only in `archived` buffer rows, which retrieval never scores — and went
through five measured runs (`v1` → `v2` → `v2b`/`v2c`/`v2d`) as each fix uncovered the next
bottleneck: first "sources get archived", then "ranking dilutes relevance", then "does the
relevance ranking actually convert to correct answers". Config changed twice:
`memory.consolidation_keep_sources` (new default `true`) and the RRF weights
(`weight_recency`/`weight_importance`/`weight_relationship`, defaults `0.1`/`0.0`/`0.0`, was
`1.0`/`1.0`/`1.0`).

### (a) LoCoMo memory mode — v1 → v2 → v2c → v2b → v2d

| run | overall J | cat 1 | cat 2 | cat 3 | cat 4 | adversarial (cat 5) | token-F1 | tokens/q | config |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v1 | 0.444 | 0.291 | 0.439 | 0.167 | 0.528 | 0.883 | 0.234 | 1,000 | `keep_sources=false`; weights 1.0/1.0/1.0 (pre-fix relationship) |
| v2 | 0.490 | 0.309 | 0.467 | 0.198 | 0.593 | 0.899 | 0.264 | 1,044 | `keep_sources=true`; weights 1.0/1.0/1.0 (pre-fix relationship) |
| v2c | 0.431 | 0.227 | 0.436 | 0.167 | 0.528 | 0.910 | 0.243 | 1,063 | `keep_sources=true`; weights 1.0/1.0/1.0 (tie-bias fix applied, restored via `--set`) |
| v2b | 0.476 | 0.301 | 0.458 | 0.198 | 0.573 | 0.892 | 0.270 | 1,115 | `keep_sources=true`; weights 0.1/0.0/1.0 (code defaults at 5c84a76, tie-bias fix) |
| v2d | **0.536** | 0.355 | 0.502 | 0.219 | 0.647 | 0.841 | 0.286 | 1,106 | `keep_sources=true`; weights 0.1/0.0/**0.0** (shipped default, tie-bias fix) |
| full-context (v1, unchanged) | 0.696 | 0.504 | 0.511 | 0.302 | 0.875 | 0.832 | 0.388 | 24,917 | no `Engine`/retrieval |

Provenance: `v2` is a **fresh ingest** on `b8ce67e` (the `keep_sources` commit) — new DBs, new
bookkeeping, judged with the same local judge as `v1`. `v2b`/`v2c`/`v2d` are **answer-only
re-runs of `v2`'s already-ingested DBs** — no re-ingestion, no new bookkeeping calls, only the
QA/judge calls — via copied checkpoints (`--reuse-ingest locomo-mem-v2`, which copies each
conversation's `.db`/`.ingest.json` the first time that run-id's own directory has neither file
yet). `v2c` and `v2d`'s weight overrides are applied with repeatable `--set section.key=value`
flags and recorded verbatim in the run's `config.overrides` list (see each result JSON); `v2b` has
no `overrides` entry because it ran on the code defaults committed at `5c84a76` directly, with no
`--set` needed. The full-context baseline is unchanged from Tier 3b (`v1`, no memory/retrieval
involved, so nothing about consolidation or ranking touches it).

### (b) LoCoMo-Plus cognitive — v1 → v2 → v2b → v2d (per-relation)

| run | overall | causal | state | goal | value | config |
|---|---:|---:|---:|---:|---:|---|
| v1 | 0.332 | 0.356 | 0.440 | 0.260 | 0.270 | `keep_sources=false`; weights 1.0/1.0/1.0 (pre-fix relationship) |
| v2 | 0.374 | 0.396 | 0.480 | 0.280 | 0.340 | `keep_sources=true`; weights 1.0/1.0/1.0 (pre-fix relationship) |
| v2b | 0.384 | 0.446 | 0.440 | 0.300 | 0.350 | `keep_sources=true`; weights 0.1/0.0/1.0 (tie-bias fix) |
| v2d | **0.421** | 0.495 | 0.410 | 0.340 | 0.440 | `keep_sources=true`; weights 0.1/0.0/**0.0** (shipped default) |
| full-context (v1, unchanged) | 0.135 | 0.188 | 0.170 | 0.080 | 0.100 | no `Engine`/retrieval |

No `v2c` Plus variant was run (the 1.0/1.0/1.0-tie-fix-only cell was judged not worth the LoCoMo
generation-model calls once its LoCoMo answer numbers came back worse than `v2`). Each Plus run
reuses its same-letter LoCoMo run's ingested DBs (`--reuse-run`) exactly as in Tier 3b.

### (c) LongMemEval-S: not re-run in Tier 3c

`lme-s-50-v1`'s numbers (J 0.396, per-type breakdown in [Results](#results) above) **stand
unchanged** — LongMemEval-S was deliberately **not** re-run this round (ledger ruling: "skip
lme-s-50-v2 in this slice (8 h for a modest change); run LME once after the ranking fix
(Tier 3d)"). Concretely: `lme-s-50-v1` was measured under the **old** `keep_sources=false`
(sources archived) and the **old** equal-weight (`1.0`/`1.0`/`1.0`) RRF defaults, not under
anything shipped in this tier. `eval/results/external_latest.json`'s `longmemeval_s:memory` key
still points at `lme-s-50-v1` — it is the only LongMemEval-S number this project has ever
published, and it should be read as "measured under the pre-Tier-3c defaults," not as a Tier 3c
result. Re-running it is explicit follow-up work for the next ranking change (Tier 3d), once
contextualized embeddings and the abstention-with-evidence lever below are also in place, so it's
measured once against a more settled target rather than twice against a moving one.

### (d) The diagnostics story, in numbers

Four diagnostic passes, run in sequence as each one's finding pointed at the next bottleneck.

1. **v1 miss classes** (`eval/external/runs/diagnostics/recall_diagnostic.md`, top-level file):
   of 857 WRONG LoCoMo answers, 743 (**86.7%**) had their evidence only in `archived` buffer
   rows — consolidation had archived the very sources retrieval would have needed.
   Evidence recall@20 (fraction of WRONG+CORRECT questions whose evidence memory was actually in
   the top-20 retrieved set) was **3.5%**.
2. **v2 diagnostic** (`.../diagnostics/locomo-mem-v2/recall_diagnostic.md`, after
   `keep_sources=true`): 0 archived buffer rows (the fix worked structurally), but of 785 WRONG
   answers, 666 (**85%**) were now `stored_active_not_retrieved` — the evidence was active and
   eligible but simply outranked by everything else in RRF fusion. Evidence recall@20 rose to
   **23.9%** (from 3.5%) — keep-sources alone made evidence retrievable in principle, but ranking
   was now the bottleneck.
3. **Offline ranking experiments** (`eval/external/ranking_experiments.py`, report at
   `.../diagnostics/ranking/ranking_experiments_report.md`, no LLM calls — pure re-ranking over
   already-computed embeddings): the equal-weight baseline (V0, weights 1.0/1.0/1.0) reproduced
   `v2`'s live 23.9% @20 (50/50 exact-order validation against the product's own fused ranking).
   Turning recency+importance to `0` with relationship still at `1` (the strategy's **pre-fix**,
   buggy implementation) reached **35.6%** @20. Turning recency, importance, **and**
   relationship all to `0` reached **50.8%** @20 — the real ceiling this sweep found; the 35.6%
   cell is not a clean read on the relationship signal, since the tie-bias bug (below) was still
   live when it was measured. A follow-up cell using **contextualized document text**
   (`"[date] speaker: content"` instead of raw content) reached **54.7%** @20 — the single best
   number in the sweep — but it needs a full re-embed of every buffer row and was deferred to
   Tier 3d rather than shipped here. Two things made things *worse*: adding nomic's
   `search_query:`/`search_document:` task prefixes (15.1% @20 in one prefix-both variant, well
   below baseline), and widening the FTS keyword pull from 50 to 200 candidates (24.3% vs. 26.1%
   @20 at the same weights) — both counterintuitive results kept, not silently dropped.
4. **Flip analysis** (`.../diagnostics/flip/`, comparing `v2` vs. `v2b` on the same 1,540
   questions): shipping the fixed relationship strategy at weight `1.0` (`v2b`) only moved live
   evidence-in-top20 from 23.9% to **29.9%** — far short of the 50.8% offline ceiling, because the
   shipped combo still had `weight_relationship=1.0` (the offline ceiling measured
   relationship at `0`). Worse, correctness *conditional on* evidence being in the top-20 actually
   *fell*, from **69.1% to 64.4%** — more relevant material in context didn't reliably turn into
   more correct answers. The mechanism: mean top-20 composition shifted from 17.69 `core_fact` /
   1.90 `buffer_raw_turn` (old) to 13.96 / 5.42 (new) — the newly-surfaced buffer turns were
   displacing **near-duplicate core facts** (paraphrased restatements of the same underlying
   fact) rather than adding new information, and the model answers more often from a paraphrased
   core fact than from the literal evidence turn. This is what motivated measuring `v2c`
   (weights 1.0/1.0/1.0 + tie-fix only, isolating whether the fix alone is neutral — it wasn't:
   0.431, worse than `v2`'s 0.490, because removing the old tie bug's incidental oldest-first
   prior actually cost something at equal weights) and `v2d` (the offline-best cell, `0.0`
   relationship) as answer-only re-runs before picking a default.
5. **v2d diagnostic** (`.../diagnostics/locomo-mem-v2d/recall_diagnostic.md`, the shipped
   defaults): evidence recall@20 **49.6%**, matching the 50.8% offline prediction. Of 714
   remaining WRONG answers: 457 (**64%**) still `stored_active_not_retrieved` (evidence active but
   outside top-20 — the residual ranking miss, not fully closed); 251 (**35%**)
   `retrieved`-but-wrong, of which **155 abstained** despite evidence being in the prompt (the
   model said "Not mentioned" anyway) and 96 answered wrong with the evidence present. The 155
   abstentions-with-evidence are now the single largest identified failure class and the clearest
   Tier 3d lever: generation/abstention behavior, not retrieval.

### (e) Caveats

- **Adversarial abstention fell under the new defaults**: 0.899 (`v2`) → **0.841** (`v2d`). More
  relevant evidence reaching the prompt makes the model answer more often instead of abstaining —
  which is exactly right for genuine questions but costs a few points on category-5 adversarial
  questions (whose gold behavior is declining to answer). Reported honestly, not smoothed over.
- **Near-duplicate core facts** dominate the top-20 (mean 17.69/20 slots pre-`v2b`, still 13.96/20
  post) — LoCoMo's fact-extraction pipeline produces multiple paraphrased restatements of the same
  underlying fact, which crowd out literal evidence turns and each other. Not addressed in this
  tier; a real dedup pass is Tier 3d scope.
- **User-affinity and name-mention boosts are off by default.** `weight_relationship=0.0` also
  means the `+0.2` user_id/relationship_target affinity bonus baked into the importance strategy's
  scoring never fires by default either (the importance list itself is skipped at
  `weight_importance=0.0`) — both are still fully implemented and available as opt-ins, not
  removed.
- **`consolidated_into` remap on import.** Every memory gets a new id on `Engine.import_character()`,
  so a kept source's `metadata.consolidated_into` (pointing at the *old* id of its consolidated
  summary) is dangling on arrival unless the import path rewrites it — it now does, via an
  old-id→new-id map built during the same pass (or drops the key if the summary wasn't part of the
  export).
- **Buffer growth is unbounded by design under `keep_sources=true`** — sources that used to be
  archived now persist as active buffer rows indefinitely. Nightly `buffer_hygiene` still sweeps
  rows on TTL, but it's now scoped to exempt anything carrying `metadata.consolidated_into`
  (a kept source), sweeping only unkept rows and rows merely marked `consolidation_seen`. This is a
  deliberate trade (retrievability over storage compactness), not an oversight — see the
  consolidation paragraph in [ARCHITECTURE.md](ARCHITECTURE.md#consolidation-engine).
- **LongMemEval-S was not re-run** — see (c) above; its published number predates every change in
  this tier.
- **Judge unchanged.** All Tier 3c numbers use the same local judge (`v2` judge logic, unchanged
  since Tier 3b) as `v1` — the comparison across `v1`→`v2d` isolates the retrieval/consolidation
  changes, not judge drift.

### (f) How to reproduce

All commands assume `.venv/bin/python`, `vllm-brain` (`:11800`), and `llama-embed` (`:11801`)
running, as in [Reproduce](#reproduce) above.

```bash
# 1. v2: fresh ingest + answer, 10-way sharded (new keep_sources=true default; the production
#    vllm-brain was raised to --max_num_seqs 8 for this measurement; ~3h13m wall-clock, 2 shard
#    restarts on this run)
for i in $(seq 0 9); do
  .venv/bin/python -m eval.external run --bench locomo --mode memory \
    --run-id locomo-mem-v2 --shard "$i/10" --no-results --timeout 900 &
done
wait
.venv/bin/python -m eval.external run --bench locomo --mode memory \
  --run-id locomo-mem-v2 --timeout 900

for i in $(seq 0 4); do
  .venv/bin/python -m eval.external run --bench locomo_plus --mode memory \
    --run-id plus-mem-v2 --reuse-run locomo-mem-v2 --shard "$i/5" --no-results --timeout 900 &
done
wait
.venv/bin/python -m eval.external run --bench locomo_plus --mode memory \
  --run-id plus-mem-v2 --reuse-run locomo-mem-v2 --timeout 900

# 2. v2b/v2c/v2d: answer-only re-runs on v2's ingested DBs — no re-ingestion, no bookkeeping
#    calls. The three variants that shipped this tier predate `--reuse-ingest` (added in the
#    fix wave below) and were produced by copying locomo-mem-v2's <conv>.db/.ingest.json
#    checkpoints into each new run-id's own directory before launching the answer-only shards
#    (v2b: `cp` then run with no --set, on 5c84a76's code defaults; v2c/v2d: `cp` then run with
#    the --set overrides shown). `--reuse-ingest <run-id>` (current code) does the same copy
#    automatically the first time a conversation's own directory has neither file yet — see the
#    "Answer-only re-runs" example under [Reproduce](#reproduce) above for the fully-sharded,
#    `--reuse-ingest`-based form of the v2c/v2d commands (`--set memory.weight_recency=...`
#    etc.); the v2b equivalent is the same pattern with no `--set` flags:
.venv/bin/python -m eval.external run --bench locomo --mode memory \
  --run-id locomo-mem-v2b --reuse-ingest locomo-mem-v2 --timeout 900

.venv/bin/python -m eval.external run --bench locomo_plus --mode memory \
  --run-id plus-mem-v2b --reuse-run locomo-mem-v2b --timeout 900

.venv/bin/python -m eval.external run --bench locomo_plus --mode memory \
  --run-id plus-mem-v2d --reuse-run locomo-mem-v2d \
  --set memory.weight_recency=0.1 --set memory.weight_importance=0.0 \
  --set memory.weight_relationship=0.0 --timeout 900

# 3. Diagnose a run's recall miss classes
.venv/bin/python -m eval.external.diagnose_recall --run-id locomo-mem-v2d

# 4. Offline ranking experiments (no LLM calls; re-ranks already-computed embeddings)
.venv/bin/python -m eval.external.ranking_experiments --run-id locomo-mem-v2

# 5. Flip analysis between two answer-only variants on the same DBs
.venv/bin/python -m eval.external.flip_analysis --old locomo-mem-v2 --new locomo-mem-v2b
```

Wall-clock, measured (`eval/external/runs/driver/chain.log`): `locomo-mem-v2` fresh
ingest+answer **3 h 13 min** (8 concurrent seqs, 10-way sharded, 2 shard restarts);
`plus-mem-v2` **1 h 18 min**; the `v2c`/`v2d` answer-only re-runs together **≈1 h 40 min**
running in parallel (5 shards each, `v2b` ran separately at ≈30 min); `plus-mem-v2d`
**1 h 17 min**.

## Caveats

- **Local 35B judge ≠ GPT-4o.** These numbers are judged by the same local Qwen3.5-35B-A3B-FP8
  model used for answering, not GPT-4o (the judge most published LoCoMo/LongMemEval numbers use).
  They are **not comparable** to published J-scores from other systems' papers. The full-context
  baseline under the identical judge is the honest comparison this harness supports.
- **LoCoMo label noise.** LoCoMo is known to have some annotation errors in its gold answers
  (a documented property of the public dataset, not something this harness can fix). A couple of
  category-5 items even carry both an `answer` and an `adversarial_answer` key; this loader always
  uses `adversarial_answer` for category 5, for consistency.
- **Category-5 convention.** Category-5/`_abs` "correct" means "the model appropriately declined
  to answer", scored by a fixed phrase-and-length rule (see
  [Category 5 / abstention](#category-5--abstention)), not by the LLM judge — it is reported
  separately and excluded from the headline J-score, matching the Mem0/Zep convention.
- **`is_abstention`'s 12-word cap can misclassify in both directions.** `is_abstention` requires
  ≤12 words, so a verbose but compliant abstention is scored WRONG; short factual answers
  containing "unknown"/"not sure" could be scored as abstentions — counts reported with the
  results.
- **Blip captions folded into text.** LoCoMo image-sharing turns are ingested as
  `"{text} (shared a photo: {caption})"`; 857 of 1,986 questions depend on evidence in such a
  turn. This is upstream's own rendering convention for these turns, not an invention of this
  harness.
- **Subset sizes.** LoCoMo and LoCoMo-Plus run in full. LongMemEval-S runs on a seeded stratified
  sample (round-robin across `question_type` groups) rather than all 500 questions — 50, not 500
  — because at this hardware's realistic bookkeeping throughput (~24 calls/min under
  `--max_num_seqs 2`) the 50-question sample's ≈12.3k paired bookkeeping calls already cost
  ~10 hours wall-clock; 100 questions would be ~21 hours and 500 would take on the order of days.
  The sample size actually run is recorded in each results file's `config.sample`.
- **`time_gap` regex quirk (LoCoMo-Plus).** `parse_time_gap` is a faithful port of upstream
  `build_conv.parse_time_gap`, including its quirk: it only matches a number token immediately
  followed by whitespace and a unit ("week"/"month"/"year"), so phrases like "several months
  later", "a couple of months later", or "several years later" match nothing and parse to 0 days.
  15 of the 401 real probes hit this and parse to a zero time gap. This is upstream behavior,
  preserved rather than "fixed", so the harness's probe placement matches what upstream's own
  pipeline would produce.
- **`judge_version: 2`.** Every record scored by the current judge (a fresh `run()` or a
  `python -m eval.external rejudge` pass) carries `judge_version: 2`; a record from a run
  predating the reference-date/abstention-detection fix has no such field (equivalent to `1`).
  RESULTS.md's external section reports the judge_version carried by the run's records
  (2 = reference-date judge; absent/1 = pre-fix pass).
- **Plus accuracy is topical linkage, not recall verification.** The Cognitive judge's question is
  "does the prediction relate to the evidence", not "did the model actually recall and use this
  specific fact" — a response that's merely thematically adjacent to the cue can be marked
  `"correct"`. A full-context smoke run scored 1/5 correct under this judge on 5 spot-checked
  probes, illustrating how strict-in-practice the linkage bar tends to land even though it reads
  loose on paper.
- **Base-conversation facts can satisfy the Plus judge without the cue.** Because LoCoMo-Plus
  memory mode reuses the base conversation's already-ingested facts and memories, `char.chat()`
  can produce a response the judge accepts as "linked to the evidence" purely from what the
  character already knew before the cue session was ever ingested — this doesn't by itself prove
  the cue's *new* information was recalled. Interpret LoCoMo-Plus memory-mode numbers with that in
  mind; it's an inherent property of probing a character that already has broad context, not a
  harness bug.
- **Chat temperature 0.7 in LoCoMo-Plus memory mode.** `char.chat()` runs at the product's
  default temperature (0.7), unlike the QA/judge calls elsewhere in this harness (all temperature
  0) — this is deliberate: memory mode exercises the real product chat path as-is, so its
  sampling temperature is the real product's, not a benchmark-chosen one. Full-context mode uses a
  fixed 0.3/200-token setting instead (see [Protocol](#locomo-plus-answering)).
- **Ingestion mutates mood/arc.** Because `ingest()` (with `unified_assessment` on) runs the same
  bookkeeping as `chat()`, ingesting the character's own turns updates its emotional state and
  narrative arc, not just its memories/facts — accepted as the correct behavior for this harness
  (in the LoCoMo speaker mapping, the character genuinely is speaker_b, so this is "the character
  witnesses/says this turn", the real product path), but it means later `chat()` answers (notably
  LoCoMo-Plus's trigger response) can be colored by mood churn accumulated across the whole
  ingested conversation, not just by retrieved facts.
- **Judge calibration (60-item skim, `locomo:memory`).** A controller skim of the stratified
  60-item judge-calibration sample (`eval/results/external_judge_sample.json`) found the CORRECT
  verdicts sound in 29 of 30 cases (one lenient false positive: gold "May 2022" vs. the model's
  "2022-06-19", accepted by the judge as matching a month-only gold answer) and the WRONG
  verdicts sound in 27–28 of 30 cases (the judge is harsh on partial lists and paraphrases, e.g.
  penalizing a partial "recipes" answer or a "health & stress" paraphrase of the gold wording).
  Observed judge disagreement is ≈4/60 (~7%) on this sample. This is a controller skim, not an
  independent human-labeled audit — a second human pass over the sample is recommended before
  treating the J-scores as more precise than ±a few points. Separately, of the 30 WRONG verdicts,
  14 were the model answering "not mentioned" — i.e. most of memory mode's errors on this sample
  are retrieval misses, not generation or judging errors.
- **`generate_json` ran without `max_tokens` during LoCoMo ingestion.** The `max_tokens` cap
  (default 2048, added in 8d34cf1) was not yet in place while `locomo-mem-v1`'s ingestion ran, so
  every ingestion-time `generate_json` call could in principle generate up to the model's full
  context window; in practice only one request across the entire ingest run ever finished by
  hitting a length limit, so this had negligible effect on the ingested facts/summaries. It did
  cause a real incident later: the first `rejudge` pass (run after the cap was added to the judge
  path, but before ingestion was re-run) hung for 3.5 hours on a single uncapped judge request that
  looped toward the 64K-token limit at temperature 0; the cap was then applied everywhere
  (including `rejudge`) before the rejudge, `plus-mem-v1`, `locomo-full-v1`, and `plus-full-v1`
  stages ran, so all of those are capped. This is a latency/availability incident, not a data
  quality issue in the published numbers.
- **Heavy queue contention during memory-mode answering affects latency, not content.**
  Memory-mode answers were produced with 5–10 concurrent shard workers queued behind the
  production `vllm-brain`'s `--max_num_seqs 2` (see [Hardware & runtime](#hardware--runtime)) —
  this inflates per-call latency (≈25s observed vs. ≈4.4s uncontended) but does not change what
  each call generates; sampling parameters (temperature, prompts) are identical to an uncontended
  run of the same code.

## Reproduce

All commands assume `.venv/bin/python` (the project venv) and that `vllm-brain` (`:11800`) and
`llama-embed` (`:11801`) are already running. `--timeout 900` is used throughout (see
[Hardware & runtime](#hardware--runtime)); `--shard i/n --no-results` runs one of `n` parallel
workers over conversations at conversation index `% n == i`, and a final unsharded call
(no `--shard`) aggregates and writes results once every worker has finished.

```bash
# 1. Fetch datasets (idempotent; --force to re-download)
.venv/bin/python -m eval.external fetch locomo locomo_plus longmemeval_s

# 2. LoCoMo, memory mode — 10-way sharded (~3h wall-clock under production vllm-brain contention)
for i in $(seq 0 9); do
  .venv/bin/python -m eval.external run --bench locomo --mode memory \
    --run-id locomo-mem-v1 --shard "$i/10" --no-results --timeout 900 &
done
wait
.venv/bin/python -m eval.external run --bench locomo --mode memory \
  --run-id locomo-mem-v1 --timeout 900   # aggregates + writes results

# 3. Re-judge with the reference-date fix (only needed if the run predates it; judge_version -> 2)
.venv/bin/python -m eval.external rejudge --bench locomo --mode memory \
  --run-id locomo-mem-v1 --timeout 900
# re-sums ingest totals and rewrites results with the judge_version-2 records; run only after
# rejudge has exited
.venv/bin/python -m eval.external run --bench locomo --mode memory \
  --run-id locomo-mem-v1 --timeout 900

# 4. LoCoMo, full-context baseline — 5-way sharded
for i in $(seq 0 4); do
  .venv/bin/python -m eval.external run --bench locomo --mode fullcontext \
    --run-id locomo-full-v1 --shard "$i/5" --no-results --timeout 900 &
done
wait
.venv/bin/python -m eval.external run --bench locomo --mode fullcontext \
  --run-id locomo-full-v1 --timeout 900

# 5. LoCoMo-Plus, memory mode — reuses locomo-mem-v1's ingested DBs, 10-way sharded
for i in $(seq 0 9); do
  .venv/bin/python -m eval.external run --bench locomo_plus --mode memory \
    --run-id plus-mem-v1 --reuse-run locomo-mem-v1 --shard "$i/10" --no-results --timeout 900 &
done
wait
.venv/bin/python -m eval.external run --bench locomo_plus --mode memory \
  --run-id plus-mem-v1 --reuse-run locomo-mem-v1 --timeout 900

# 6. LoCoMo-Plus, full-context baseline (no reuse-run needed; shard similarly if it's slow)
.venv/bin/python -m eval.external run --bench locomo_plus --mode fullcontext \
  --run-id plus-full-v1 --timeout 900

# 7. LongMemEval-S, memory mode only — 50-question stratified sample (seed 7; identical to the
#    first 50 of the 100-question sample under the same seed), 10-way sharded. 24,639 turns /
#    2,380 sessions -> ~12.3k paired bookkeeping calls + 2.4k session summaries, ~10 h at ~24
#    calls/min under --max_num_seqs 2; 100 questions would be ~21 h, which is why this
#    publication runs 50. The aggregate call MUST repeat --sample/--seed exactly, or the loader
#    reloads all 500 questions and the aggregate starts ingesting the other 450.
for i in $(seq 0 9); do
  .venv/bin/python -m eval.external run --bench longmemeval_s --mode memory \
    --run-id lme-s-50-v1 --sample 50 --seed 7 --interval 1 --delete-db-after-answer \
    --shard "$i/10" --no-results --timeout 900 &
done
wait
.venv/bin/python -m eval.external run --bench longmemeval_s --mode memory \
  --run-id lme-s-50-v1 --sample 50 --seed 7 --interval 1 --delete-db-after-answer --timeout 900
```

Notes from running this for real:

- A shard worker died once with `openai.APITimeoutError` inside an `end_session()` summary call
  under queue contention at `--max_num_seqs 2` (the default 300s timeout was too tight); fixed by
  making that exception retryable (`resilience.resilient_call`) and raising the default per-run
  timeout to 900s (`--timeout`, shown above). The affected conversation simply re-ingested from
  scratch on relaunch (checkpointing discards a partial ingest with no `.ingest.json` marker).
- The unsharded aggregation call (step 2's second invocation, and its equivalents in steps 4/5/7)
  must only run after every `--shard` worker for that run has actually exited — it refuses to run
  otherwise (see [Determinism/resumability](#determinismresumability)); `--force-aggregate`
  overrides this only when a partial run is genuinely what you want to publish.

### Answer-only re-runs

Testing a `memory.*` RRF weight change (or any other config knob) doesn't require
re-ingesting: `--set section.key=value` (repeatable) overrides a config field for the
run's duration, and `--reuse-ingest <run-id>` copies `<conv>.db`/`.ingest.json` from a
sibling run under the same root the first time a conversation's own directory has
neither file yet — the run then answers straight from the copied DB with zero
ingestion/bookkeeping LLM calls. Both flags compose: a same-dataset run-id with
different `--set` overrides and `--reuse-ingest` pointing at the already-ingested base
run turns a multi-hour re-ingest into an answer-only pass. Example (the 2026-08-30
`weight_recency`/`weight_importance`/`weight_relationship` sweep that produced the
`weight_relationship` default above — both variants reuse `locomo-mem-v2`'s ingested
DBs, 5-way sharded):

```bash
# v2c: weights 1.0/1.0/1.0 (tie-bias fix only, no relevance-first defaults)
for i in $(seq 0 4); do
  .venv/bin/python -m eval.external run --bench locomo --mode memory \
    --run-id locomo-mem-v2c --reuse-ingest locomo-mem-v2 \
    --set memory.weight_recency=1.0 --set memory.weight_importance=1.0 \
    --set memory.weight_relationship=1.0 \
    --shard "$i/5" --no-results --timeout 900 &
done
wait
.venv/bin/python -m eval.external run --bench locomo --mode memory \
  --run-id locomo-mem-v2c --reuse-ingest locomo-mem-v2 \
  --set memory.weight_recency=1.0 --set memory.weight_importance=1.0 \
  --set memory.weight_relationship=1.0 --timeout 900

# v2d: recency 0.1 / importance 0.0 / relationship 0.0 (the offline-best cell)
for i in $(seq 0 4); do
  .venv/bin/python -m eval.external run --bench locomo --mode memory \
    --run-id locomo-mem-v2d --reuse-ingest locomo-mem-v2 \
    --set memory.weight_recency=0.1 --set memory.weight_importance=0.0 \
    --set memory.weight_relationship=0.0 \
    --shard "$i/5" --no-results --timeout 900 &
done
wait
.venv/bin/python -m eval.external run --bench locomo --mode memory \
  --run-id locomo-mem-v2d --reuse-ingest locomo-mem-v2 \
  --set memory.weight_recency=0.1 --set memory.weight_importance=0.0 \
  --set memory.weight_relationship=0.0 --timeout 900
```
