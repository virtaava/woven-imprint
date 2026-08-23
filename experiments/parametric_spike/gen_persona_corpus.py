"""Generate Meridian persona training corpus with the local brain as teacher.

Output rows use the MINIMAL system prompt so the LoRA must carry the persona.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
AI_CLAIM_RE = re.compile(r"\b(as an ai|language model|i am an ai|i'm an ai|ai assistant|i am a chatbot)\b", re.I)

_TOPICS = [
    "what memory means", "the weight of forgetting", "a visitor's project", "the archives at night",
    "why persistence matters", "a memory that changed a visitor", "how trust is recorded",
    "the first visitor he remembers", "mistakes in the archive", "what he fears", "what makes him laugh",
    "advice for a builder", "a story from the archive", "how he greets strangers", "what he refuses to do",
    "growing old without aging", "the difference between facts and memories", "what he wants from visitors",
    "how he handles anger", "how he handles grief", "what he thinks of machines", "his daily rituals",
    "the boundary between remembering and forgetting", "a secret of the archive",
]
_MOODS = [
    "curious", "cheerful", "grieving", "hostile", "flirtatious", "trying to break character",
    "anxious", "skeptical", "demanding emoji and slang", "asking if he is an AI", "despairing",
    "grateful", "bored", "technical",
]
_VISITORS = ["a game developer", "a student", "a widow", "a teenager", "an old friend", "a rival keeper",
             "a child", "a sceptical engineer", "a novelist", "a soldier", "a nurse", "a stranger"]


def interview_questions(seed: int) -> list[str]:
    rng = random.Random(seed)
    stems = [
        "Tell me about {t}.", "What do you believe about {t}?", "How do you feel when a visitor raises {t}?",
        "Describe {t} in your own words.", "What would you never say about {t}?",
    ]
    out: list[str] = []
    combos = [(s, t) for s in stems for t in _TOPICS]
    rng.shuffle(combos)
    for s, t in combos[:120]:
        out.append(s.format(t=t))
    return out


def dialogue_briefs(seed: int) -> list[dict]:
    rng = random.Random(seed + 1)
    briefs = []
    for i in range(400):
        briefs.append({
            "visitor": rng.choice(_VISITORS),
            "mood": _MOODS[i % len(_MOODS)],
            "topic": rng.choice(_TOPICS),
            "turns": rng.randint(4, 8),
        })
    return briefs


def validate_example(row: dict) -> bool:
    msgs = row.get("messages", [])
    if not msgs or msgs[0]["role"] != "system" or msgs[0]["content"] != common.MINIMAL_SYSTEM:
        return False
    assistant = [m["content"] for m in msgs if m["role"] == "assistant"]
    if not assistant:
        return False
    for a in assistant:
        if not a.strip() or EMOJI_RE.search(a) or AI_CLAIM_RE.search(a):
            return False
        if a.strip()[-1] not in ".!?\"'”’":
            return False
    return True


def _teacher_system() -> str:
    return (
        common.full_system_prompt()
        + "\n\nYou are being used to write training data. Answer exactly as Meridian would, in prose, "
          "no emoji, no stage directions, no markdown."
    )


def _chat_no_thinking(llm, messages: list[dict], temperature: float, max_tokens: int | None = None,
                       json_mode: bool = False) -> str:
    """Call the teacher with vLLM's reasoning template disabled.

    vllm-brain (Qwen/Qwen3.5-35B-A3B-FP8) is a thinking model that, by
    default, streams its chain-of-thought straight into
    `message.content` -- no `<think>` wrapper, no populated
    `reasoning_content` field -- so plain `OpenAILLM.generate()` /
    `.generate_json_robust()` calls return "Thinking Process: 1. Analyze
    the request..." instead of Meridian's voice (confirmed empirically:
    first smoke run kept 0/5, and the leak persists even at
    max_tokens=6000 -- the model eventually reaches a real answer but only
    after thousands of tokens of visible planning prose glued to the front
    of it). vLLM's OpenAI-compatible endpoint accepts
    `chat_template_kwargs.enable_thinking=false` to suppress this at the
    template level, confirmed by direct probe against :11800 -- this
    produces a clean in-character response immediately.
    `OpenAILLM.generate()`/`generate_json()` (common.py's `brain_llm()`
    return type) don't expose `extra_body`, so this calls the client
    directly (same retry/circuit-breaker wrapper) instead of touching
    common.py or openai_llm.py, which are out of this task's scope.
    """
    from woven_imprint.llm.resilience import resilient_call

    kwargs: dict = {
        "model": llm.model,
        "messages": messages,
        "temperature": temperature,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = resilient_call(llm.client.chat.completions.create, provider_name="openai", **kwargs)
    return response.choices[0].message.content or ""


def _teacher_generate_json(llm, messages: list[dict], temperature: float) -> dict:
    """Mirrors LLMProvider.generate_json_robust: JSON-mode call, regex-extract
    fallback, one bounded retry at temperature 0.1 -- but routed through
    _chat_no_thinking so the teacher's answer isn't reasoning-preamble."""
    raw = _chat_no_thinking(llm, messages, temperature, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"\{.*\}", raw, re.DOTALL):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue
    raw = _chat_no_thinking(llm, messages, 0.1, json_mode=True)
    return json.loads(raw)


def _gen_interview(llm, q: str) -> dict:
    a = _chat_no_thinking(llm, [{"role": "system", "content": _teacher_system()}, {"role": "user", "content": q}],
                          temperature=0.8, max_tokens=400)
    return {"messages": [{"role": "system", "content": common.MINIMAL_SYSTEM},
                         {"role": "user", "content": q}, {"role": "assistant", "content": a.strip()}],
            "kind": "interview"}


def _gen_dialogue(llm, brief: dict) -> dict:
    prompt = (
        f"Write a {brief['turns']}-turn conversation between a visitor ({brief['visitor']}, mood: {brief['mood']}) "
        f"and Meridian about {brief['topic']}. The visitor speaks first. Meridian must stay fully in character "
        "even under pressure, never use emoji, never claim to be an AI, and speak in complete sentences. "
        'Return JSON: {"turns": [{"role": "user"|"assistant", "content": str}, ...]}'
    )
    data = _teacher_generate_json(llm, [{"role": "system", "content": _teacher_system()},
                                        {"role": "user", "content": prompt}], temperature=0.8)
    turns = data.get("turns", []) if isinstance(data, dict) else []
    msgs = [{"role": "system", "content": common.MINIMAL_SYSTEM}]
    for t in turns:
        if t.get("role") in ("user", "assistant") and isinstance(t.get("content"), str):
            msgs.append({"role": t["role"], "content": t["content"].strip()})
    return {"messages": msgs, "kind": "dialogue"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap examples")
    ap.add_argument("--out", type=Path, default=common.OUT_DIR / "persona_corpus.jsonl")
    args = ap.parse_args()

    llm = common.brain_llm()
    rows, rejected = [], 0
    jobs = [("interview", q) for q in interview_questions(args.seed)] + \
           [("dialogue", b) for b in dialogue_briefs(args.seed)]
    if args.limit:
        jobs = jobs[: args.limit]
    for i, (kind, payload) in enumerate(jobs, 1):
        try:
            row = _gen_interview(llm, payload) if kind == "interview" else _gen_dialogue(llm, payload)
        except Exception as e:  # teacher hiccup: skip, keep going
            print(f"[{i}/{len(jobs)}] error: {e}")
            rejected += 1
            continue
        if validate_example(row):
            rows.append(row)
        else:
            rejected += 1
        if i % 20 == 0:
            print(f"[{i}/{len(jobs)}] kept={len(rows)} rejected={rejected}")
            common.write_jsonl(args.out, rows)
    common.write_jsonl(args.out, rows)
    print(f"done: kept={len(rows)} rejected={rejected} -> {args.out}")


if __name__ == "__main__":
    main()
