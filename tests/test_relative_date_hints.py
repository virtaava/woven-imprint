"""Tests for `woven_imprint.character._relative_date_hints`.

Tier 3g (docs/superpowers/specs/2026-09-01-tier3g-relative-time-photo-render.md):
the v3 abstain-with-evidence analysis found `b_relative_time` questions (55/317)
failing because the evidence line says "last Saturday"/"next month" and the
model must combine the phrase with the memory line's own formed date — a step
it often skips or gets wrong. This module resolves the closed set of relative
phrases to absolute dates, pure `datetime`, no LLM.
"""

from datetime import datetime, timezone

from woven_imprint.character import _relative_date_hints

# 2023-05-08 is a Monday (matches the spec's own worked example).
CREATED = datetime(2023, 5, 8, 12, 0, tzinfo=timezone.utc)


def test_empty_when_no_relative_phrase():
    assert _relative_date_hints("Nothing relative here.", CREATED) == ""


def test_yesterday():
    assert _relative_date_hints("I saw her yesterday.", CREATED) == '; "yesterday"→2023-05-07'


def test_tomorrow_matches_spec_example():
    assert _relative_date_hints("See you tomorrow!", CREATED) == '; "tomorrow"→2023-05-09'


def test_tonight_is_same_date():
    assert _relative_date_hints("Big party tonight.", CREATED) == '; "tonight"→2023-05-08'


def test_last_night_is_previous_date():
    assert _relative_date_hints("Rough sleep last night.", CREATED) == '; "last night"→2023-05-07'


def test_this_morning_is_same_date():
    assert _relative_date_hints("Went for a run this morning.", CREATED) == (
        '; "this morning"→2023-05-08'
    )


def test_last_week():
    assert _relative_date_hints("We spoke last week.", CREATED) == '; "last week"→2023-05-01'


def test_next_week():
    assert _relative_date_hints("Let's meet next week.", CREATED) == '; "next week"→2023-05-15'


def test_last_month_renders_year_month():
    assert _relative_date_hints("It happened last month.", CREATED) == '; "last month"→2023-04'


def test_next_month_renders_year_month():
    assert _relative_date_hints("Due next month.", CREATED) == '; "next month"→2023-06'


def test_last_month_crosses_year_boundary():
    created = datetime(2023, 1, 15, tzinfo=timezone.utc)
    assert _relative_date_hints("last month", created) == '; "last month"→2022-12'


def test_next_month_crosses_year_boundary():
    created = datetime(2023, 12, 15, tzinfo=timezone.utc)
    assert _relative_date_hints("next month", created) == '; "next month"→2024-01'


def test_last_year_renders_year_only():
    assert _relative_date_hints("last year", CREATED) == '; "last year"→2022'


def test_next_year_renders_year_only():
    assert _relative_date_hints("next year", CREATED) == '; "next year"→2024'


def test_spec_example_last_saturday():
    """Spec worked example: created 2023-05-08 Mon, "last Saturday"→2023-05-06."""
    assert _relative_date_hints("last Saturday", CREATED) == '; "last Saturday"→2023-05-06'


def test_last_friday_across_month_boundary():
    """Brief example: created 2023-01-02 Mon, "last Friday"→2022-12-30."""
    created = datetime(2023, 1, 2, tzinfo=timezone.utc)
    assert _relative_date_hints("last Friday", created) == '; "last Friday"→2022-12-30'


def test_next_sunday_across_month_boundary():
    """Brief example: created 2023-01-02 Mon, "next Sunday"→2023-01-08."""
    created = datetime(2023, 1, 2, tzinfo=timezone.utc)
    assert _relative_date_hints("next Sunday", created) == '; "next Sunday"→2023-01-08'


def test_last_same_weekday_as_created_goes_back_a_full_week():
    """ "last <weekday>" is strictly earlier: on a Monday, "last Monday" is 7 days back,
    never "today"."""
    created = datetime(2023, 1, 2, tzinfo=timezone.utc)  # Monday
    assert _relative_date_hints("last Monday", created) == '; "last Monday"→2022-12-26'


def test_next_same_weekday_as_created_goes_forward_a_full_week():
    created = datetime(2023, 1, 2, tzinfo=timezone.utc)  # Monday
    assert _relative_date_hints("next Monday", created) == '; "next Monday"→2023-01-09'


def test_weekday_matching_is_case_insensitive():
    created = datetime(2023, 1, 2, tzinfo=timezone.utc)
    assert _relative_date_hints("last friday", created) == '; "last friday"→2022-12-30'


def test_last_weekend_from_a_monday_is_previous_saturday():
    """Brief: "last weekend from a Monday → previous Saturday"."""
    created = datetime(2023, 1, 2, tzinfo=timezone.utc)  # Monday
    assert _relative_date_hints("last weekend", created) == '; "last weekend"→2022-12-31'


def test_next_weekend_renders_a_saturday():
    created = datetime(2023, 1, 2, tzinfo=timezone.utc)  # Monday
    text = _relative_date_hints("next weekend", created)
    assert text == '; "next weekend"→2023-01-14'
    resolved = datetime.fromisoformat(text.split("→")[1]).weekday()
    assert resolved == 5  # Saturday


# --- word-boundary safety --------------------------------------------------


def test_lastly_does_not_match_last():
    assert _relative_date_hints("She lastly mentioned the trip.", CREATED) == ""


def test_nextdoor_does_not_match_next():
    assert _relative_date_hints("The nextdoor neighbor waved.", CREATED) == ""


def test_last_week_does_not_match_inside_last_weekend():
    # CREATED is Mon 2023-05-08; the most recent Sat-Sun pair before that week is
    # 2023-05-06/07, same Saturday as the "last Saturday" weekday example above.
    text = _relative_date_hints("Party last weekend was great.", CREATED)
    assert text == '; "last weekend"→2023-05-06'


def test_next_week_does_not_match_inside_next_weekend():
    # The Sat-Sun pair after CREATED's week (2023-05-08..14) is 2023-05-20/21.
    text = _relative_date_hints("Party next weekend, come along.", CREATED)
    assert text == '; "next weekend"→2023-05-20'


# --- cap at 3, first occurrences, order of appearance ----------------------


def test_caps_at_three_hints_in_order_of_appearance():
    content = (
        "yesterday we talked, tomorrow we'll meet, tonight is quiet, and next week is busy too."
    )
    text = _relative_date_hints(content, CREATED)
    hints = text[2:].split(", ")
    assert len(hints) == 3
    assert hints[0].startswith('"yesterday"')
    assert hints[1].startswith('"tomorrow"')
    assert hints[2].startswith('"tonight"')
    assert "next week" not in text


def test_duplicate_phrase_counted_once_first_occurrence_wins():
    content = "tomorrow is the day. tomorrow, really, tomorrow."
    text = _relative_date_hints(content, CREATED)
    assert text == '; "tomorrow"→2023-05-09'


def test_multiple_distinct_phrases_appear_in_content_order():
    content = "next week we leave, but yesterday we packed."
    text = _relative_date_hints(content, CREATED)
    assert text == '; "next week"→2023-05-15, "yesterday"→2023-05-07'
