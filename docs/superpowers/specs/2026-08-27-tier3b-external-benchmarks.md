# Tier 3b — External benchmarks on the local brain — Spec

**Date:** 2026-08-27  
**Owner:** Toni  
**Origin:** audit roadmap Tier 3 ("publish LoCoMo/LongMemEval numbers with the exact harness"); research report 2026-08-27 (datasets, protocols, pitfalls).  
**Status:** approved ("merge then proceed"). Assumptions are the controller's; Toni can override.

## Goal
Publish honest, reproducible numbers for woven-imprint on the standard long-term-conversational-memory benchmarks, measured with a **local** judge (Qwen3.5-35B-A3B-FP8 via vllm-brain, thinking disabled) and no paid APIs, alongside a **full-context baseline under the same judge** so the reader can see what the memory system adds or loses. Publish the exact harness, prompts, judge prompt, and a judge-calibration sample.

## Scope
1. **LoCoMo** (snap-research, `data/locomo10.json`, 10 conversations, 1,540 scored Qs in categories 1–4 + 446 adversarial category 5) — full run.
2. **LoCoMo-Plus** Cognitive category (`data/locomo_plus.json`: 401 cue/trigger probes, relation types causal/state/goal/value, no gold answers). Verified schema 2026-08-27: each item = `cue_dialogue` (A:/B: lines), `trigger_query` (A: line), `time_gap`, `relation_type`. Upstream protocol (`data/build_conv.py`, `unified_input.py`): probe i is stitched into LoCoMo conversation `i % 10`; A→speaker_a, B→speaker_b; `query_time = last session + 7 days`, `cue_time = query_time − time_gap`; the cue is inserted as its own session at `cue_time`, the trigger at `query_time`; the model produces a *reply* to the trigger; the **Cognitive judge** (verbatim from `evaluation_framework/task_eval/prompt.py`) labels correct/wrong by whether the reply reflects the cue. Score = mean correct. In our harness memory mode = `Character.chat(trigger)` on the LoCoMo-ingested character + cue session (the real product path); full-context mode = stitched transcript + trigger → reply, same judge.
3. **Full-context baseline** on LoCoMo: same QA prompt + judge, transcript in context (fits the 64K window).
4. **LongMemEval-S** (`xiaowu0162/longmemeval-cleaned`, `longmemeval_s_cleaned.json`, 277 MB): stratified sample of 100 questions (seeded), per-session ingestion with dates; abstention (`_abs`) scored separately.
5. Library fix that the harness exposed: `Character.ingest()` bypasses unified assessment and never creates structured facts.

Out of scope: LongMemEval-M, MemConflict (no public data), MemSyco-Bench and PersonaMem-v2 (different protocols — next slice), Kotlin.

## Protocol (Mem0/Zep lineage, made explicit)
- **Ingestion unit:** message by message via `Character.ingest(role, content, user_id)`. Speaker mapping per conversation: the first speaker becomes the **user** (`user_id` = speaker name), the second speaker becomes the **character** (its name = speaker name); `role="user"` / `"assistant"` accordingly. Session boundaries follow the dataset's sessions; `clock.override(session datetime)` before each session, `clock.advance(seconds=…)` per turn so `created_at` is monotonic; `end_session()` after each session (1 summary call; `maintenance.callbacks_refresh_on_session_end=false`). `memory.fact_extraction_interval = 1` for benchmark runs (documented; cost ↑).
- **Answering:** for each question, under `clock.override(question_time)` (LoCoMo: last session date + 1 day; LongMemEval: `question_date`), build the prompt: pinned block + facts block (`_format_facts_block(user_id)`) + dated memories from `retriever.retrieve(question, limit=K, relationship_target=user_id)` (K = 20; each line as `_format_memories` renders) + the question; instruction: answer concisely (≤ 15 words), use only the memories, convert relative dates to absolute, and reply exactly `Not mentioned` if the memories do not contain the answer. Model: the same local brain (thinking off).
- **LoCoMo-Plus answering:** copy the conversation's ingested DB from the LoCoMo memory run; `clock.override(cue_time)` → `start_session` → ingest cue turns (A=user, B=character) → `end_session`; `clock.override(query_time)` → `start_session` → `response = char.chat(trigger A-line, user_id=user_name)` (brain, thinking off, temperature 0.3 as upstream, max_tokens 200); judge with the Cognitive prompt; evidence = cue with mapped speaker names. Full-context: stitched dialogue in upstream `Speaker said, "..."` format ending with the trigger, instruction to reply as speaker_b.
- **Judge:** the LoCoMo/Mem0 lenient judge prompt (quoted verbatim in `docs/BENCHMARKS.md`): CORRECT if the response conveys the same fact as the gold answer (date-format/relative-time variants tolerated), else WRONG. Judge = local brain at temperature 0. Category 5: CORRECT iff the response is `Not mentioned` (or an equivalent explicit abstention) — reported separately, excluded from the headline J-score, as in Mem0/Zep.
- **Metrics:** J-score (% CORRECT) overall (cats 1–4) and per category; token-F1 vs gold; category-5 abstention accuracy; tokens per question (prompt tokens sent to the answer model); ingestion LLM calls and wall-clock. LongMemEval-S: accuracy per question type, `_abs` accuracy separately.
- **Judge calibration:** 60 judge decisions sampled (stratified by category and verdict) written to `eval/results/external_judge_sample.json` for human review; the docs state the sample size and that scores are not comparable to GPT-4o-judged numbers.
- **Determinism/resumability:** one SQLite DB per conversation under `eval/external/runs/<run_id>/`, checkpoint JSON per conversation (ingested? answered?), rerun resumes; seeds fixed.

## Deliverables
- `src/woven_imprint/llm/openai_llm.py`: `OpenAILLM(..., extra_body: dict | None = None)` threaded into all calls (replaces the spike's local subclass).
- `Character.ingest()` uses `_run_bookkeeping` (one unified call) when `unified_assessment` is on; legacy path when off. Documented as a behavior change (structured facts now created on ingest).
- `eval/external/` package: `fetch.py` (downloads to `eval/external/data/`, gitignored, size/sha check), `locomo.py` (loader + adapter), `longmemeval.py`, `runner.py` (ingest → QA → judge → metrics, checkpointed), `judge.py`, `metrics.py`, `report.py` (writes `eval/results/external_latest.json` and `eval/results/external_<run_id>.json`), `__main__` CLI: `python -m eval.external run --bench locomo|locomo_plus|longmemeval_s --mode memory|fullcontext [--limit N] [--sample 100] [--run-id X]`.
- Tests with FakeLLM/FakeEmbedder on tiny committed fixtures (`eval/external/fixtures/locomo_mini.json`, `longmemeval_mini.json`, `locomo_plus_mini.json`): loaders, adapter, prompt assembly, judge parsing, metrics, checkpoint resume.
- `docs/BENCHMARKS.md`: method, prompts, judge prompt, numbers table (memory vs full-context, per category), tokens/question, caveats (local judge, LoCoMo label errors, category-5 convention, LongMemEval subset), how to reproduce (commands, runtime). `docs/RESULTS.md` gains an "External benchmarks (real LLM)" section from `external_latest.json` (deterministic score line untouched). README claims table row + CHANGELOG.

## Acceptance
- pytest/ruff/pyright green; deterministic suites unaffected.
- LoCoMo full run (memory + full-context) and LoCoMo-Plus cognitive subset completed and published; LongMemEval-S 100-question run completed and published; judge sample file present.
- Numbers reported exactly as measured, including any category where memory loses to full context.

## Global constraints
- Branch `feat/tier3b-external-benchmarks` off master `e4ede06`. Commit per task. Never touch `experiments/parametric_spike/`, `kotlin/`, or the deterministic `eval/results/latest.json`.
- Local brain only (`http://127.0.0.1:11800/v1`, thinking off via `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`); embeddings `http://127.0.0.1:11801/v1` (768-d). Never stop vllm-brain; watch memory (<85%).
- Data files are gitignored; fixtures < 200 KB committed.
