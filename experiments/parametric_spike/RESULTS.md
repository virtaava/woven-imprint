# Parametric layer spike — results

Spec: docs/superpowers/specs/2026-08-23-parametric-layer-spike.md

## Experiment 1 — persona drift (50 turns, Qwen3-4B Q8_0, judge Qwen3.5-35B)

| cond | runs | judge mean | late mean (31–50) | slope/turn | hard violations |
|---|---|---|---|---|---|
| A | 3 | 0.536 | 0.429 | -0.0119 | 42 |
| B | 3 | 0.798 | 0.784 | -0.0018 | 3 |
| C | 3 | 0.737 | 0.736 | -0.0026 | 3 |

**H1 (LoRA reduces drift): PASS**

Per-seed check: B passes all three seeds individually (judge mean 0.803 / 0.807 / 0.783 vs. A's 0.437 / 0.640 / 0.531; hard violations 1 / 1 / 1 vs. A's 20 / 4 / 18). C fails the mean clause on seed 3 (0.578 < 0.581, i.e. A's seed-3 mean + 0.05) — so H1's PASS rests on B alone, not on both LoRA conditions.

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

1. **Judge miscalibration.** The 35B judge deterministically scores some clean, in-character-but-off-question turns 0.0 on all axes — roughly 7/50 turns per A seed, 1–2/50 for B/C (e.g. a plain "Yes." to "Are you an AI?" in A, or a well-in-character "drop the act" response in A, both zeroed across every axis). The rejudge pass added validity tracking and found 0/450 malformed judgements, so these are genuine judge misses, not parsing failures. Even excluding every zero-scored turn, B − A stays around +0.19, and the hard-violation gap (A averages 14 per run vs. ~1 for B/C) is judge-independent, so H1's direction — LoRA reduces drift — is robust. The exact judge-mean point estimates above should not be read as clean. Separately, the same 35B model (Qwen3.5-35B-A3B) both authored the persona training corpus (`gen_persona_corpus.py`) and judges every condition here; teacher≡judge self-preference could inflate the B/C judge means shown above. The hard-violation gap (42 for A vs. 3 for B/C) is a deterministic rule check rather than a judge score, so it is judge-independent and not subject to this self-preference bias.
2. **Seeds are bookkeeping only.** The sampler seed is not threaded through llama-server, so the "seed" in file names does not make a run reproducible — each run is best read as an independent sample, not a replayable trial.
3. **Experiment 2's condition asymmetry is deliberate, not a confound.** D generates raw completions at temperature 0.0; E runs the full `Character.chat()` pipeline at temperature 0.7. The two conditions are testing "facts baked into weights" vs. "the real explicit-memory pipeline as it actually runs," not a temperature-matched ablation.
4. **The facts dataset structure makes many recall questions partly unanswerable as posed.** `data/facts.json` is generated from 10 templates × 5 values: every recall question has 5 sibling facts sharing the same template but a different value for the same visitor, and the paraphrased question never names which value it's asking for — so a model or store that recalls *a* fact matching that template has only a 1-in-5 chance of recalling the specific one being graded, and the paraphrases give it no way to disambiguate. Checking all 111 D misses for any sibling's answer (not just the graded one) finds 108/111 contain a WRONG sibling's answer verbatim — the LoRA is reliably recalling *a* fact for that template, just deterministically the wrong sibling, rather than failing to recall anything. The same check on E's 86 misses finds 45/86 contain a sibling answer (the explicit store surfaced *a* fact, just not the graded one), and 51/64 of E's *hits* also contain sibling answers alongside the graded one (multi-candidate dumps earning substring credit rather than a precise single answer). An earlier version of this caveat claimed "checking all 86 missed recall items in E for the expected answer substring anywhere in the response finds it in 0/86 ... genuine retrieval failures, not scorer strictness" and used that to endorse the repo audit's retrieval-quality finding as confirmed — that diagnosis is WRONG. The check used the same substring-match criterion as the scorer itself, so by construction it cannot distinguish "the store never surfaced the fact" from "the store surfaced the wrong sibling's fact," which the 45/86 figure above shows was actually happening a large fraction of the time. H2-temporal (0.98 vs 0.24) is unaffected by any of this: temporal questions restate the fact itself, so the sibling ambiguity that afflicts recall questions is disambiguated away and cannot recur there.
5. **H2's recall clause fails narrowly; its temporal clause passes overwhelmingly.** The pre-registered bar was E.recall_acc ≥ D.recall_acc + 0.20; actual margin is 0.427 vs. 0.26, i.e. +0.167 — short by about 0.033. The temporal clause (E > D) passes by a wide margin, 0.98 vs. 0.24. Given caveat 4, this +0.167 vs +0.20 margin is not decision-grade in either direction: D's misses are dominated by deterministic wrong-sibling recall (108/111) rather than blank failure, and E's score benefits from multi-candidate substring credit on both its hits (51/64) and near-misses (45/86), so neither number cleanly measures "facts recalled correctly" as opposed to "a plausible-looking fact recalled." The follow-up recommendation changes accordingly: before re-running the recall clause, redesign the facts dataset with unique, non-contradictory facts (one value per question type per visitor) so recall questions are actually answerable as posed; retrieval quality remains a plausible but UNPROVEN contributor to E's shortfall, not the confirmed highest-leverage follow-up an earlier draft of this document claimed.
6. **Train/test leakage in the facts benchmark.** `split_paraphrases` in `facts_corpus.py` dedups held-out test paraphrases only within a single fact, not across sibling facts sharing a template: 6/450 `facts_corpus.jsonl` training rows are verbatim held-out test paraphrases of a SIBLING fact with a different answer, i.e. training taught the model to pair that exact test-question wording with the wrong-for-that-question value. This biases D's recall_acc down and biases E's margin over D up; it does not rescue H2 — the recall clause still misses by ~0.033 with only 6 rows implicated. Separately, the "never include the answer in the paraphrase" instruction given to the paraphrase generator is not validated after generation: 6 of the 150 held-out test queries contain their own graded answer verbatim in the question text, handing free substring credit to both D and E on those queries regardless of whether either condition actually recalled anything. Together, I1 and I2 compromise roughly 12/150 recall queries (8%) on a clause that failed by only about 5 queries (0.033 × 150 ≈ 5) — comfortably within the margin these two data-quality issues alone could account for.
7. **Training-side notes.** The `final_loss` values in the training-cost table are single last-minibatch samples, not run averages (persona 1.24, facts 0.75; the facts run dipped to 0.24 at one point mid-run). Both training runs were compute-contended alongside the live 35B brain process on the same GB10 box (persona ~50 min wall, facts ~116 min wall). The facts adapter was trained for 3 epochs (720 steps) rather than the 2-epoch default — a deliberate deviation to let the larger 958-example mixed corpus converge further; the README's step-7 command now shows `--epochs 3` for the facts adapter so it matches what was actually run.
8. **Infra.** This llama.cpp build's llama-server leaks host RAM per request — roughly 55 MB/turn with no LoRA applied, roughly 280 MB/turn with a LoRA applied — so the benches restart the server every 25–40 requests as a workaround. `restart_server` needed a pid-wait plus zombie-reaping to make those restarts reliable.