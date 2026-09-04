# Tier 3m — Aggregation probes on the kept LME DBs — Spec

**Date:** 2026-09-04
**Origin:** Tier 3l — multi-session (LME's aggregation category) went 5/16 → 4/16 under the
new protocol while everything else gained. Failure split (v5 wrongs, n=12): 7 abstain
("Not mentioned" — plausibly the hardened instruction discouraging summing/counting since
no single memory *states* the total), 5 enumerate-but-miss-instances (coverage at K=100 in
~500-session stores).
**Status:** exploratory diagnostics under the standing flow; both arms answer-only on the
KEPT `lme-s-100-v5` DBs (~1-2 h each), sequential so each runs on clean code.

## Arms
1. **`lme-t3mK150`** — coverage arm: current prompt, `--k 150`, `--reuse-ingest lme-s-100-v5`.
2. **`lme-t3mArith`** — instruction arm: K=100 plus one clause appended to the hardened
   sentence: "Counting, adding up, or ordering things the memories state is combining,
   not guessing." (edit committed to this branch before the arm runs; BENCHMARKS quote
   updated only if adopted).

## Pre-registered bars (declared before any run)
- An arm is ADOPTED into the protocol only if: overall J ≥ 0.559 (no regression, n=93)
  AND multi-session ≥ 7/16 (from 4/16; +3 questions on the targeted category) AND
  abstention (`_abs`) ≥ 6/7.
- Both adopted → combine and confirm with one more run at K=150+clause.
- Neither adopted → publish findings as diagnostics; they scope the real slice
  (entity-linking at ingest vs an aggregation answer stage).
- n=16 category noise is acknowledged: +3 questions is the minimum readable signal.

## Constraints
Branch `feat/tier3m-aggregation-probe` off master `2c385c6`. Answer-only; never touch
library code; never stop vllm-brain.
