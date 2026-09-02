# Tier 3i — LLM-guided query expansion for aggregation & multi-hop retrieval — Spec

**Date:** 2026-09-02
**Owner:** Toni
**Origin:** follow-up queue ("semantic multi-hop bridging: LLM-guided second query / entity
linking", recorded at Tier 3f); Tier 3h diagnostics (LME-S-100 multi-session 0.31,
temporal 0.375; 12 of 21 wrongs in those categories are "Not mentioned" retrieval misses
on aggregation-shaped questions — "how many X in total", "order of the three trips").
**Status:** proposed under the standing "merge then proceed" flow; assumptions are the
controller's, Toni can override.

## Problem
Single-query retrieval ranks candidates by similarity to the *question*. Aggregation
questions ("How much money did I raise for charity **in total**?") need every instance
memory ("donated $500 at the bake sale"), each of which is individually dissimilar to the
aggregate phrasing; multi-hop questions need evidence reachable only through an
intermediate entity. Tier 3f's lexical seed expansion (zero LLM calls) failed its
pre-registered bar and shipped opt-in — terms from already-found seeds cannot reach
instances that share no vocabulary with the question or the seeds. The recorded next step
is semantic expansion: let the LLM rewrite the question into instance-level search queries.

## Mechanism
One new opt-in retrieval stage inside `MemoryRetriever.retrieve`, gated by config:

- `memory.query_expansion: int = 0` — maximum number of LLM-generated expansion queries
  (0 = off, byte-identical ranking to today). Benchmark runs measure 3.
- `memory.query_expansion_weight: float = 0.5` — RRF weight applied to each expansion
  query's ranked lists (original query's lists keep their existing weights, so the
  original ranking stays dominant).

Flow when `query_expansion > 0` and the query is non-empty:
1. **One** LLM call (temperature 0, `max_tokens` ≤ 300, JSON list contract): given the
   user's question, return up to N alternative *search queries* — instance-level
   rephrasings and decompositions (e.g. "art event attended" → "went to a gallery /
   exhibit / museum / concert"), entity-bridge phrasings for multi-hop. Prompt lives next
   to the code; documented in BENCHMARKS.
2. Parse → strip empties/duplicates (case-folded) and any expansion equal to the original
   query. LLM error, timeout, or unparseable output → log-free silent skip; retrieval
   NEVER fails or degrades below today's behavior.
3. For each surviving expansion: one embed call (batched if the provider supports it) +
   one `fts_search(limit=50)`. New rows join the candidate pool exactly like Tier 3f's
   widening (vectors ride on the FTS rows).
4. Each expansion contributes two ranked lists to the final weighted RRF (semantic
   cosine vs. the expansion's embedding, computed over the widened pool; its FTS keyword
   ranking), both at `query_expansion_weight`. The original query's lists (and, when
   `retrieval_second_pass` ran, its recomputed lists) are reused as-is — strategies 3-5
   are NOT re-run and the relevance gate is not re-derived: rows surfaced only by an
   expansion earn rank credit purely through the expansion lists (the same additive-credit
   rationale as Tier 3f's strategy-2 extension), which keeps the block small and the
   off-path byte-identical.
5. Composability: runs after/independently of `retrieval_second_pass`; both off by
   default; enabling both must not error (test).

**LLM plumbing:** `MemoryRetriever.__init__` gains optional `llm: LLMProvider | None = None`
(default None → expansion silently off even if config asks for it). `Character` passes its
LLM when constructing the retriever; the eval runner passes the answer-model LLM. No other
public-API change.

## Cost
When on: +1 LLM call, +1 batch embed call (≤ N vectors), +≤ N FTS queries per retrieve.
Benchmark records mean seconds/question and mean prompt tokens; docs state the cost
plainly.

## Measurement (pre-registered bar — declared BEFORE any run)
Answer-only LoCoMo run `locomo-t3iA` on the existing `locomo-mem-v3b` DBs
(`--reuse-ingest`, `--set memory.query_expansion=3`), judged identically to baseline
`locomo-t3gA` (J 0.571; cat-1 multi-hop 0.287 n=282; cat-2 0.573; cat-3 0.240;
cat-4 0.704; adversarial 0.895).

- **Ship-on bar:** overall J ≥ 0.565 (no meaningful overall regression) **AND**
  cat-1 ≥ 0.317 (+0.03 on the targeted failure mode).
- Bar met → defaults flip to `query_expansion=3` and docs/headline update.
- Bar missed → ships opt-in (config present, default 0), honest BENCHMARKS section with
  the numbers either way — Tier 3f precedent.
- Adversarial abstention must not drop below 0.87 (expansion must not manufacture false
  evidence confidence); breach = fail regardless of J.

LME-S re-measurement deferred to the next full LME run (`lme-s-100-v4` DBs were deleted
by `--delete-db-after-answer`; a fresh ingest is ~17 h and not this slice).

## Deliverables
- Config fields + validation (`query_expansion ≥ 0`, weight > 0) with docstrings citing
  this spec; config.yaml.example comments.
- Retriever implementation as above; expansion prompt as a module-level constant.
- Tests (FakeLLM/FakeEmbedder): off = byte-identical ranking; on = expansion lists fused
  at the configured weight; LLM failure/garbage-JSON/empty-list → identical to off;
  dedup of expansions; composition with `retrieval_second_pass>0`; retriever without
  `llm` ignores the config; Character wires its LLM through.
- Runner support: nothing new needed (`--set` already covers the flags) — verify only.
- `locomo-t3iA` measured; BENCHMARKS "Tier 3i" section with elimination-table-style
  before/after per category; RESULTS/README/CHANGELOG per the bar outcome.

## Acceptance
pytest/ruff (src tests eval demo)/pyright green; deterministic suites unaffected;
`bench_longhorizon` 12/12; numbers published exactly as measured.

## Global constraints
Branch `feat/tier3i-query-expansion` off master `0ea7298`. Never touch
`experiments/parametric_spike/` or `kotlin/`. Local brain/embedder only; never stop
vllm-brain; memory < 85%. Data files gitignored; results JSON committed with `git add -f`.
