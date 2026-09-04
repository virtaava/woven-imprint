# Tier 3n — Aggregation answer stage (enumerate-then-answer) — Spec

**Date:** 2026-09-04
**Origin:** Tier 3m — both dials (K=150, arithmetic clause) failed their bars; conclusion
on record: enumeration questions need machinery. Tier 3h/3l failure shape: "how many /
how much in total / what order" answers are scattered across instance memories; the model
either abstains or counts only what it notices in a 100-line list under a 60-token cap.
**Status:** proceeding under the standing flow.

## Mechanism (benchmark-harness protocol machinery; no library change)
1. `eval/external/aggregation.py` (new):
   - `is_aggregation_question(q: str) -> bool` — closed, case-insensitive regex set:
     `how many`, `how much`, `in total`, `total number`, `total amount`, `altogether`,
     `list all`, `what ... order` / `order of` / `sequence of`, `first ... then` style
     ordering asks. Word-boundaried; unit-tested against every multi-session question
     text in the committed LME fixture plus negative examples.
   - `AGG_QA_SYSTEM` — variant of the hardened QA_SYSTEM for aggregation questions only:
     same never-guess contract, plus: "First list each matching item the memories state,
     one per line, with its date/amount. Then give the final line exactly as
     'Answer: <concise answer>'." (≤ 15-word rule applies to the Answer line.)
   - `parse_agg_answer(text) -> str` — returns the text after the last `Answer:` marker
     (stripped); falls back to the whole response if no marker (judged as-is, honest).
2. `answer_question` in `runner.py`: when `cfg.agg_stage` is true AND
   `is_aggregation_question(q.question)`:
   - retrieval: temporarily set `memory.query_expansion = 3` around the retrieve call
     (try/finally restore; the retriever already has the LLM handle via Character —
     shipped opt-in in Tier 3i, designed for exactly these questions),
   - answer: `AGG_QA_SYSTEM` messages, `max_tokens = 700` (the list needs room),
     response recorded raw in a new `raw_response` field, judged on `parse_agg_answer`.
   - every other question: byte-identical to today.
3. `RunConfig.agg_stage: bool = False` + `--agg-stage` CLI flag; recorded in results
   config. Default OFF — a run without the flag is byte-identical to today.

## Honesty rules
- Results from agg-stage runs are labeled as such; BENCHMARKS reports the mode
  explicitly. If adopted, the protocol section documents the two-path answering and the
  regex; the single-path number stays in the lineage tables.
- The parse fallback never invents an answer; unparseable → judged raw.

## Measurement (pre-registered bars, declared before any run)
Answer-only on kept `lme-s-100-v5` DBs: `lme-t3nAgg` (`--agg-stage --k 100`).
- **Adopt** iff: overall J ≥ 0.569 (v5 0.559 + 0.01 — machinery must pay, not just tie)
  AND multi-session ≥ 7/16 AND `_abs` ≥ 6/7 AND no other category loses ≥ 3 questions.
- If adopted: confirm on LoCoMo with `locomo-t3nAgg` answer-only on v3b DBs — adversarial
  ≥ 0.87 AND overall ≥ 0.665 required to keep it in the cross-benchmark protocol;
  LoCoMo failure → agg stage ships as an LME-documented mode only (split protocol,
  documented) — controller decision deferred to the numbers.
- Not adopted → negative result published like Tier 3m; slice closes; entity-linking
  at ingest becomes the next candidate.

## Deliverables & acceptance
aggregation.py + tests (regex positives/negatives incl. fixture sweep, parse_agg_answer
edge cases, AGG system text pins the Answer: contract); runner wiring + tests (agg path
selected only when flag+match, expansion toggle restored on exception, off path
byte-identical — ScriptedLLM test); suite/ruff/pyright green; measurement + BENCHMARKS
section + results committed either way.

## Global constraints
Branch `feat/tier3n-aggregation-answer` off master `95249e8`. Never touch
`experiments/parametric_spike/`, `kotlin/`, or `src/woven_imprint/` (harness-only slice).
Local brain only; never stop vllm-brain; memory < 85%.
