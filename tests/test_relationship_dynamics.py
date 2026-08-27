"""Tests for relationship dynamics: trust asymmetry, betrayal damping, key moments,
trajectory window, tension decay, tiers."""

from datetime import datetime, timedelta, timezone

import pytest

from woven_imprint import clock
from woven_imprint.config import get_config
from woven_imprint.relationship.model import RelationshipModel
from woven_imprint.storage.sqlite import SQLiteStorage

T0 = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def rm():
    s = SQLiteStorage(":memory:")
    s.save_character("c1", "Ada", {})
    get_config().relationship.dynamics = True
    yield RelationshipModel(s, "c1")
    s.close()


def test_trust_gains_are_halved_losses_are_not(rm):
    with clock.override(T0):
        rm.update("u", {"trust": 0.10})
        assert abs(rm.get("u")["dimensions"]["trust"] - 0.05) < 1e-9
        rm.update("u", {"trust": -0.10})
        assert abs(rm.get("u")["dimensions"]["trust"] - (-0.05)) < 1e-9


def test_betrayal_damps_recovery_and_records_key_moment(rm):
    with clock.override(T0):
        for _ in range(4):
            rm.update("u", {"trust": 0.15})  # 4 × 0.075 = 0.30
        rm.update("u", {"trust": -0.12}, note="lied about the map")  # betrayal
        rel = rm.get("u")
        assert rel["state"]["damping_left"] == 10 and rel["state"]["betrayals"] == 1
        assert any("betrayal" in m and "lied about the map" in m for m in rel["key_moments"])
        before = rel["dimensions"]["trust"]
        rm.update("u", {"trust": 0.15})
        after = rm.get("u")["dimensions"]["trust"]
        assert abs((after - before) - 0.15 * 0.5 * 0.25) < 1e-9
        assert rm.get("u")["state"]["damping_left"] == 9


def test_key_moment_threshold_and_cap(rm):
    get_config().relationship.key_moments_limit = 3
    try:
        with clock.override(T0):
            for i in range(5):
                rm.update("u", {"affection": 0.09}, note=f"moment {i}")
        moments = rm.get("u")["key_moments"]
        assert len(moments) == 3 and moments[-1].endswith("moment 4")
        rm.update("u", {"affection": 0.02}, note="tiny")
        assert not any("tiny" in m for m in rm.get("u")["key_moments"])
    finally:
        get_config().relationship.key_moments_limit = 20


def test_trajectory_uses_window(rm):
    with clock.override(T0):
        for _ in range(5):
            rm.update("u", {"trust": 0.15, "affection": 0.15})
        assert rm.get("u")["trajectory"] == "warming"
        rm.update("u", {"trust": -0.05})  # one small dip must not flip a 5-turn warm streak
        assert rm.get("u")["trajectory"] == "warming"
        for _ in range(5):
            rm.update("u", {"trust": -0.15, "affection": -0.15})
        assert rm.get("u")["trajectory"] == "cooling"


def test_tension_decays_with_elapsed_days(rm):
    with clock.override(T0):
        rm.update("u", {"tension": 0.15})
        rm.update("u", {"tension": 0.15})
        assert abs(rm.get("u")["dimensions"]["tension"] - 0.30) < 1e-9
    with clock.override(T0 + timedelta(days=4)):
        rm.update("u", {"trust": 0.01})
        assert abs(rm.get("u")["dimensions"]["tension"] - 0.10) < 1e-9  # 0.30 − 4×0.05


def test_tiers():
    from woven_imprint.relationship.model import RelationshipModel as R

    assert (
        R.tier_for({"trust": 0, "affection": 0, "respect": 0, "familiarity": 0.0, "tension": 0})[0]
        == "stranger"
    )
    assert (
        R.tier_for({"trust": 0, "affection": 0, "respect": 0, "familiarity": 0.2, "tension": 0})[0]
        == "acquaintance"
    )
    assert (
        R.tier_for(
            {"trust": 0.4, "affection": 0.3, "respect": 0.2, "familiarity": 0.35, "tension": 0}
        )[0]
        == "friend"
    )
    assert (
        R.tier_for(
            {"trust": 0.8, "affection": 0.6, "respect": 0.5, "familiarity": 0.7, "tension": 0}
        )[0]
        == "close_friend"
    )
    assert (
        R.tier_for(
            {"trust": -0.6, "affection": -0.4, "respect": 0.0, "familiarity": 0.5, "tension": 0.5}
        )[0]
        == "adversary"
    )


def test_type_follows_tier_and_describe_shows_it(rm):
    with clock.override(T0):
        for _ in range(8):
            rm.update("u", {"trust": 0.15, "affection": 0.15, "respect": 0.15, "familiarity": 0.15})
    rel = rm.get("u")
    assert rel["type"] in ("friend", "close_friend")
    text = rm.describe("u")
    assert "tier:" in text and "affinity" in text


def test_dynamics_off_is_legacy(rm):
    get_config().relationship.dynamics = False
    try:
        with clock.override(T0):
            rm.update("u", {"trust": 0.10})
        assert abs(rm.get("u")["dimensions"]["trust"] - 0.10) < 1e-9
        assert rm.get("u")["type"] == "stranger" and rm.get("u")["key_moments"] == []
    finally:
        get_config().relationship.dynamics = True


def test_key_moment_uses_pre_scaling_magnitude(rm):
    # A trust gain of +0.15 is halved by trust_gain_factor to an applied 0.075,
    # which is below key_moment_threshold (0.08) -- but the key moment must
    # still fire because it is judged against the pre-scaling clamped value.
    with clock.override(T0):
        rm.update("u", {"trust": 0.15}, note="saved my life")
    rel = rm.get("u")
    assert abs(rel["dimensions"]["trust"] - 0.075) < 1e-9  # applied value unaffected
    assert any("trust +0.15" in m and "saved my life" in m for m in rel["key_moments"])


def test_key_moments_limit_zero_disables_moments(rm):
    get_config().relationship.key_moments_limit = 0
    try:
        with clock.override(T0):
            rm.update("u", {"affection": 0.15}, note="disabled moment")
        assert rm.get("u")["key_moments"] == []
    finally:
        get_config().relationship.key_moments_limit = 20


def test_betrayal_while_damped_resets_without_decrementing(rm):
    with clock.override(T0):
        rm.update("u", {"trust": -0.12}, note="first betrayal")
        assert rm.get("u")["state"]["damping_left"] == 10
        rm.update("u", {"trust": -0.12}, note="second betrayal")
        assert rm.get("u")["state"]["damping_left"] == 10
