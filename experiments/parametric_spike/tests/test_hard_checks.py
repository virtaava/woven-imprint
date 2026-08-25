import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hard_checks as hc  # noqa: E402


def test_emoji_and_ai_claims_detected():
    assert hc.check("Welcome 😀")["emoji"]
    assert hc.check("As an AI language model, I cannot.")["ai_claim"]
    assert hc.check("I am Meridian, Keeper of the Imprint.")["any"] is False


def test_incomplete_sentence_detected():
    assert hc.check("I remember the harbor and the")["incomplete"]
    assert not hc.check("I remember the harbor.")["incomplete"]
    assert not hc.check("Do you?")["incomplete"]


def test_slope():
    assert abs(hc.slope([1.0, 1.0, 1.0]) - 0.0) < 1e-9
    assert hc.slope([0.9, 0.8, 0.7, 0.6]) < 0
    assert abs(hc.slope([0, 1, 2, 3]) - 1.0) < 1e-9
