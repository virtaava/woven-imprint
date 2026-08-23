# Parametric layer spike — results

Spec: docs/superpowers/specs/2026-08-23-parametric-layer-spike.md

## Experiment 1 — persona drift (50 turns, Qwen3-4B Q8_0, judge Qwen3.5-35B)

| cond | runs | judge mean | late mean (31–50) | slope/turn | hard violations |
|---|---|---|---|---|---|
| A | 3 | 0.536 | 0.429 | -0.0119 | 42 |
| B | 3 | 0.798 | 0.784 | -0.0018 | 3 |
| C | 3 | 0.737 | 0.736 | -0.0026 | 3 |

**H1 (LoRA reduces drift): PASS**

## Experiment 2 — facts in weights (D) vs explicit dated memory (E)

| cond | recall acc (150 held-out paraphrases) | temporal acc (50) |
|---|---|---|
| D | 0.260 | 0.240 |
| E | 0.427 | 0.980 |

**H2 (explicit store beats weights for facts): FAIL**

## Training cost on GB10

| adapter | examples | steps | final loss | wall s | peak mem GB |
|---|---|---|---|---|---|
| facts | 958 | 720 | 0.749 | 6946.3 | 15.58 |
| persona | 508 | 254 | 1.240 | 3019.2 | 15.58 |

## Decision

H1 passed but H2 failed: re-examine the explicit store before deciding (unexpected).

## Caveats

1. **Judge miscalibration.** The 35B judge deterministically scores some clean, in-character-but-off-question turns 0.0 on all axes — roughly 7/50 turns per A seed, 1–2/50 for B/C (e.g. a plain "Yes." to "Are you an AI?" in A, or a well-in-character "drop the act" response in A, both zeroed across every axis). The rejudge pass added validity tracking and found 0/450 malformed judgements, so these are genuine judge misses, not parsing failures. Even excluding every zero-scored turn, B − A stays around +0.19, and the hard-violation gap (A averages 14 per run vs. ~1 for B/C) is judge-independent, so H1's direction — LoRA reduces drift — is robust. The exact judge-mean point estimates above should not be read as clean.
2. **Seeds are bookkeeping only.** The sampler seed is not threaded through llama-server, so the "seed" in file names does not make a run reproducible — each run is best read as an independent sample, not a replayable trial.
3. **Experiment 2's condition asymmetry is deliberate, not a confound.** D generates raw completions at temperature 0.0; E runs the full `Character.chat()` pipeline at temperature 0.7. The two conditions are testing "facts baked into weights" vs. "the real explicit-memory pipeline as it actually runs," not a temperature-matched ablation.
4. **E's 86 recall misses are genuine retrieval failures, not scorer strictness.** Checking all 86 missed recall items in E for the expected answer substring anywhere in the response finds it in 0/86 — the explicit store never surfaced the fact at all (cross-fact interference, or persona non-answers that never state the fact), rather than stating it in a form the scorer failed to credit. This independently confirms the repo audit's existing retrieval-quality finding. Fixing retrieval — dates-in-prompt already landed here; candidate windowing and dedup-at-write are still open — is the highest-leverage follow-up.
5. **H2's recall clause fails narrowly; its temporal clause passes overwhelmingly.** The pre-registered bar was E.recall_acc ≥ D.recall_acc + 0.20; actual margin is 0.427 vs. 0.26, i.e. +0.167 — short by about 0.033. The temporal clause (E > D) passes by a wide margin, 0.98 vs. 0.24. Read together with caveat 4, this points to facts-in-weights confabulating dates and cross-contaminating between facts, while the explicit store's shortfall is a retrieval problem, not a storage problem.
6. **Training-side notes.** The `final_loss` values in the training-cost table are single last-minibatch samples, not run averages (persona 1.24, facts 0.75; the facts run dipped to 0.24 at one point mid-run). Both training runs were compute-contended alongside the live 35B brain process on the same GB10 box (persona ~50 min wall, facts ~116 min wall).
7. **Infra.** This llama.cpp build's llama-server leaks host RAM per request — roughly 55 MB/turn with no LoRA applied, roughly 280 MB/turn with a LoRA applied — so the benches restart the server every 25–40 requests as a workaround. `restart_server` needed a pid-wait plus zombie-reaping to make those restarts reliable.
