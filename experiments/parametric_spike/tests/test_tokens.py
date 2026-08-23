import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HF = Path.home() / "models" / "qwen3-4b-hf"
pytestmark = pytest.mark.skipif(not (HF / "tokenizer.json").exists(), reason="base tokenizer not downloaded")


def test_assistant_only_labels():
    from transformers import AutoTokenizer
    import train_lora as t

    tok = AutoTokenizer.from_pretrained(str(HF))
    msgs = [{"role": "system", "content": "You are Meridian."},
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am Meridian, Keeper of the Imprint."},
            {"role": "user", "content": "And?"},
            {"role": "assistant", "content": "And I remember."}]
    ex = t.build_example_tokens(tok, msgs, max_len=256)
    assert ex is not None and len(ex["input_ids"]) == len(ex["labels"])
    kept = [i for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    text = tok.decode(kept)
    assert "Keeper of the Imprint" in text and "I remember" in text
    assert "Who are you?" not in text and "You are Meridian." not in text
    assert ex["labels"][0] == -100


def test_truncation_returns_none_when_no_assistant_tokens():
    from transformers import AutoTokenizer
    import train_lora as t

    tok = AutoTokenizer.from_pretrained(str(HF))
    msgs = [{"role": "system", "content": "x " * 300}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    assert t.build_example_tokens(tok, msgs, max_len=64) is None
