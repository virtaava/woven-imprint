# Tier 3q — Preference answer stage — Spec

**Date:** 2026-09-05
**Origin:** LME `single-session-preference` has been the worst category all campaign
(0/8 → 1/17 → 4/17), documented since Tier 3b as an instrument mismatch: the questions
are advice/recommendation REQUESTS ("Can you recommend a show for me tonight?"), the
golds are rubrics ("The user would prefer responses that reference their stated ..."),
and the hardened ≤15-word factual prompt makes the model ABSTAIN — 4+ of the current 13
wrongs are "Not mentioned" on questions that have no fact to abstain from.

## Mechanism (harness protocol machinery, mirroring Tier 3n's aggregation stage)
1. `eval/external/preference.py` (new, mirroring `aggregation.py`):
   - `is_preference_question(q) -> bool` — closed, case-insensitive, word-boundaried
     regex set for advice-request shapes: `can you (suggest|recommend)`,
     `any suggestions`, `do you have any (suggestions|recommendations|ideas)`,
     `(what|how) do you think`, `do you think it would`, `should i \b`,
     `recommend (a|some|any)`, `help me (choose|decide|plan|pick)`,
     `what would you (recommend|suggest)`. Unit-tested against all 17 v5 preference
     question texts (must match ≥ 15) and the 7 `_abs` questions + 10 factual examples
     (must match 0).
   - `PREF_QA_SYSTEM`: "You are this person's assistant. Answer their request helpfully
     in 2-3 sentences, grounded in the specific preferences, plans, and past experiences
     the memories state about them — name the relevant details you are using. Never
     invent facts the memories do not support. If the memories contain nothing relevant,
     say you don't have their preferences on record and give one brief general answer."
   - `pref_qa_messages(memory_block, question, today)` mirroring `qa_messages`.
2. Runner: `RunConfig.pref_stage: bool = False` + `--pref-stage` flag (recorded in
   results config). In `answer_question`: routing precedence — aggregation stage first
   (if `agg_stage` and matches), then preference stage (if `pref_stage` and matches):
   PREF messages, `max_tokens=200`, response recorded as-is (no marker parsing; the
   whole reply is judged). Everything else byte-identical.
3. Tests mirror `tests/test_aggregation_stage.py`: regex positives (real v5 preference
   texts) / negatives (the `_abs` texts verbatim + factual samples), routing precedence
   (a question matching BOTH regexes takes the aggregation path), off-path
   byte-identity, pref path uses PREF system + max_tokens 200.

## Measurement (pre-registered bars — declared BEFORE any run)
`lme-t3qPref`: answer-only on kept `lme-s-100-v5` DBs, shipping protocol
(`--k 100 --agg-stage --pref-stage`). Baseline `lme-t3nAgg` (0.656; preference 3/17
under agg-stage protocol; `_abs` 7/7).
- **Adopt iff:** preference ≥ 8/17 AND overall ≥ 0.651 AND `_abs` ≥ 6/7 AND no other
  category loses ≥ 3 questions.
- If adopted: LoCoMo confirmation `locomo-t3qPref` on v3b-ent DBs (same flags):
  keep iff overall ≥ 0.669 AND adversarial ≥ 0.87 (misrouted factual/adversarial
  questions would show here). LoCoMo failure → LME-only documented mode (controller
  rules on the split with the numbers in hand).
- Missed → opt-in + honest negative docs, Tier 3f/3i/3p precedent.

## Acceptance
Suite/ruff (src tests eval demo)/pyright green; BENCHMARKS Tier 3q section; results
committed; latest.json pointer per outcome.

## Global constraints
Branch `feat/tier3q-preference-answer` off master `16a694a`. Harness-only — never touch
`src/woven_imprint/`, `experiments/parametric_spike/`, `kotlin/`.
