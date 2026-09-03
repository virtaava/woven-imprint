# Tier 3k — Abstention-at-depth hardening — Spec

**Date:** 2026-09-02
**Owner:** Toni
**Origin:** Tier 3j K-sweep — adversarial abstention decays monotonically with depth
(0.913 at K=40 → 0.848 at K=100) while every answer category gains; K=80/100 J
(0.664/0.674) is already measured but unusable under the pre-registered abstention floor
(0.87). Diagnosis: the 33 K=20→K=100 flipped adversarial answers are plausible near-miss
completions — deeper pools surface related-but-not-answering memories and the model
commits.
**Status:** proceeding under the standing "merge then proceed" flow.

## Change
One prompt edit in the benchmark harness (`eval/external/prompts.py::QA_SYSTEM` — protocol,
not library code). The instruction must forbid speculation beyond the memories while still
permitting multi-memory combination (the cat-1 gains depend on it). New wording:

> "You answer questions about a person using ONLY the memories provided. Be concise (at
> most 15 words). Convert relative dates to absolute dates. You may combine multiple
> memories, but never guess or infer beyond what they state. If the memories do not
> actually state the answer, reply exactly: Not mentioned"

`docs/BENCHMARKS.md` quotes QA_SYSTEM verbatim — the quote updates in the same docs pass.
No library code changes. Controller ruling: diff is two strings + docs; executed directly
without SDD (recorded).

## Measurement (pre-registered bar — declared BEFORE any run)
Answer-only on `locomo-mem-v3b` DBs, hardened prompt, three depths:
`locomo-t3kH60`, `locomo-t3kH80`, `locomo-t3kH100` (10 shards each), judged identically.
Unhardened baselines: t3jK60 (0.632/adv 0.877), t3jK80 (0.664/0.863), t3jK100 (0.674/0.848).

- **Adopt K\* = the largest K ∈ {60, 80, 100} whose hardened run has adversarial ≥ 0.87
  AND overall J ≥ (unhardened J at that K) − 0.005.** That run becomes the LoCoMo
  memory-mode headline and the hardened prompt becomes the protocol prompt at all K.
- If even K=60 hardened fails its leg (J < 0.627 or adversarial < 0.87), hardening FAILS:
  revert the prompt, keep t3jK60/0.632 headline, publish the negative result.
- Cost note: prompt-token deltas are negligible (one sentence); no latency change.

## Acceptance
Suite/ruff/pyright green (prompt string only — expect no test churn); BENCHMARKS Tier 3k
section with the 3×(J, per-cat, adversarial) table vs unhardened; RESULTS/README/CHANGELOG
per outcome; results JSONs committed; latest.json pointer per outcome.

## Global constraints
Branch `feat/tier3k-abstention-hardening` off master `b2e6aa5`. Never touch
`experiments/parametric_spike/` or `kotlin/`. Local brain only; never stop vllm-brain.
