import pytest

from eval.external.metrics import is_abstention, summarize, token_f1


def test_token_f1_agrees_on_reworded_date():
    assert token_f1("8 May 2023", "May 8, 2023") > 0.6


def test_token_f1_disjoint_is_zero():
    assert token_f1("blue", "red") == 0


def test_token_f1_empty_strings_are_zero():
    assert token_f1("", "May 8, 2023") == 0
    assert token_f1("May 8, 2023", "") == 0


def test_is_abstention():
    assert is_abstention("Not mentioned")
    assert is_abstention("I don't know / not mentioned in memories.")
    assert not is_abstention("Paris")


def test_token_f1_ignores_articles_and_punctuation():
    assert token_f1("a cat", "the cat") == 1.0
    assert token_f1("She adopted a cat.", "She adopted the cat") == 1.0


@pytest.mark.parametrize(
    "phrase",
    [
        "Not mentioned",
        "not mentioned in the memories",
        "The conversation does not mention it",
        "The memories do not mention this",
        "I do not have this information",
        "I don't have that information",
        "Sorry, I cannot answer that",
        "There is no info about that",
        "No information available",
        "unknown based on the memories",
    ],
)
def test_is_abstention_catches_required_phrases(phrase):
    assert is_abstention(phrase), phrase


@pytest.mark.parametrize(
    "phrase",
    [
        "She adopted a dog",
        "Paris",
        "Mostly sourdough bread, it's her favorite thing to bake",
    ],
)
def test_is_abstention_does_not_flag_concrete_answers(phrase):
    assert not is_abstention(phrase), phrase


def _item(category, kind, correct, f1=0.5, prompt_tokens_est=10):
    return {
        "category": category,
        "kind": kind,
        "correct": correct,
        "f1": f1,
        "prompt_tokens_est": prompt_tokens_est,
    }


def test_summarize_categories_and_adversarial_accuracy():
    items = [
        _item("1", "qa", True),
        _item("1", "qa", False),
        _item("2", "qa", True),
        _item("3", "qa", True),
        _item("4", "qa", False),
        _item("5", "adversarial", True),
    ]
    summary = summarize(items)

    assert 0.0 <= summary["overall_j"] <= 1.0
    # overall_j is computed over the 5 "qa" items only (3 correct / 5)
    assert summary["overall_j"] == 3 / 5
    assert set(summary["per_category"]) == {"1", "2", "3", "4", "5"}
    assert summary["per_category"]["1"]["n"] == 2
    assert summary["adversarial_accuracy"] == 1.0
    assert summary["abstain_accuracy"] is None
    assert summary["n_questions"] == 6


def test_summarize_abstain_accuracy_separate_from_overall():
    items = [
        _item("multi-session", "qa", True),
        _item("temporal-reasoning", "abstain", True),
        _item("temporal-reasoning", "abstain", False),
    ]
    summary = summarize(items)
    assert summary["overall_j"] == 1.0  # only the one "qa" item counts
    assert summary["abstain_accuracy"] == 0.5
