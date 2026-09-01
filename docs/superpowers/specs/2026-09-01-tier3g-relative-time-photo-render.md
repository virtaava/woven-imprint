# Tier 3g — Relative-time resolution + photo-caption rendering — Spec

**Date:** 2026-09-01  **Owner:** Toni  **Status:** approved ("merge and proceed" after Tier 3f merge f30ce64).
**Origin:** v3 abstain-with-evidence analysis: `b_relative_time` 55/317 (evidence says "last Saturday"/"next month"; question asks for a date; the model must combine the phrase with the line's formed date and often fails) and `f_photo_caption` 54/317 (the answer sits inside the "(shared a photo: …)" parenthetical and gets skipped). Both are pure rendering changes in `_format_memories` — measurable answer-only on the v3b DBs.

## Design
**T1 (product, both in `_format_memories`):**
1. **Relative-time hints** (`context.resolve_relative_dates: bool = True`): when a memory's content contains one of a CLOSED set of relative phrases (case-insensitive): yesterday, tomorrow, tonight, "last night", "this morning", "last week(end)", "next week(end)", "last month", "next month", "last year", "next year", "last <weekday>", "next <weekday>", "on <weekday>" — append resolved absolute dates to the date prefix, computed from the memory's OWN `created_at`: e.g. `(2023-05-08 Mon, 3 months ago; "tomorrow"→2023-05-09, "last Saturday"→2023-05-06)`. Rules: `last <weekday>` = the most recent strictly-earlier such weekday; `next <weekday>` = the next strictly-later; week starts Monday; `last weekend` = the Sat–Sun before the created date's week (render the Saturday); month/year hints render as `2023-04` / `2024`. At most 3 hints per line, first occurrences win. Pure `datetime`, no LLM. Off → today's format byte-identical.
2. **Photo captions** (`context.photo_caption_line: bool = True`): when content contains `(shared a photo: X)` (the loader/ingest convention), render the caption as its own indented continuation line so it reads as content, not an aside:
   `- (date…) [User: name] Went hiking with the kids! \n    [photo] a mountain trail at sunrise`
   (strip the parenthetical from the main line; multiple photos → multiple `[photo]` lines). Off → byte-identical.
**T2 (measurement, controller):** locomo answer-only on v3b DBs, both flags default-on (they are the new defaults IF the bar is met): run `locomo-t3gA`. Pre-registered bar: overall J not below 0.557 (v3b −0.005) AND cat-2 ≥ +0.02 or cat-4 ≥ +0.015. If met → defaults stay True and ship; if not → flip defaults to False and ship opt-in with numbers (same discipline as Tier 3f).
**T3:** docs (BENCHMARKS Tier 3g), results, merge package. LME 50→100 extension is the NEXT slice (pure measurement), not this one.

## Constraints
Branch `feat/tier3g-relative-time-photo-render` off master f30ce64. Green suite/bench; regex-only, no LLM; never touch experiments/, kotlin/.
