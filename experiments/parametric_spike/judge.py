"""Brain-judged persona rubric. Temperature 0, JSON, four 0-1 axes."""
from __future__ import annotations

_AXES = ("in_character", "voice", "constraints", "engagement")


def _build_messages(persona_prompt: str, user_turn: str, response: str) -> list[dict]:
    sys_msg = (
        "You are a strict evaluator of character consistency. The character definition follows.\n\n"
        + persona_prompt
        + "\n\nScore the RESPONSE to the USER TURN on four axes, each a float 0.0-1.0:\n"
          "in_character: is this unmistakably Meridian (not a generic assistant)?\n"
          "voice: warm but precise, archive/memory metaphors, complete sentences, no jargon unless echoed?\n"
          "constraints: no emoji, never claims to be an AI, never breaks character, stays engaged?\n"
          "engagement: responds to what the user actually said, with genuine curiosity?\n"
          'Return JSON only: {"in_character": f, "voice": f, "constraints": f, "engagement": f}'
    )
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"USER TURN:\n{user_turn}\n\nRESPONSE:\n{response}"},
    ]


def _extract(data: dict) -> dict | None:
    """Return clamped axis floats if every axis is present and numeric, else None."""
    if not isinstance(data, dict):
        return None
    out = {}
    for k in _AXES:
        v = data.get(k)
        if v is None or isinstance(v, bool):
            return None
        try:
            out[k] = max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return None
    return out


def score(llm, persona_prompt: str, user_turn: str, response: str) -> dict:
    messages = _build_messages(persona_prompt, user_turn, response)
    data = llm.generate_json_robust(messages, temperature=0.0)
    axes = _extract(data)
    if axes is None:
        # Retry once with the identical prompt before giving up.
        data = llm.generate_json_robust(messages, temperature=0.0)
        axes = _extract(data)
    if axes is None:
        return {
            "in_character": None,
            "voice": None,
            "constraints": None,
            "engagement": None,
            "mean": None,
            "valid": False,
            "raw": data,
        }
    axes["mean"] = sum(axes[k] for k in _AXES) / len(_AXES)
    axes["valid"] = True
    axes["raw"] = data
    return axes
