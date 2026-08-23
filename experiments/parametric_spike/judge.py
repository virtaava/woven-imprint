"""Brain-judged persona rubric. Temperature 0, JSON, four 0-1 axes."""
from __future__ import annotations

_AXES = ("in_character", "voice", "constraints", "engagement")


def score(llm, persona_prompt: str, user_turn: str, response: str) -> dict:
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
    data = llm.generate_json_robust([
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"USER TURN:\n{user_turn}\n\nRESPONSE:\n{response}"},
    ], temperature=0.0)
    out = {}
    for k in _AXES:
        try:
            out[k] = max(0.0, min(1.0, float(data.get(k, 0.0))))
        except (TypeError, ValueError, AttributeError):
            out[k] = 0.0
    out["mean"] = sum(out[k] for k in _AXES) / len(_AXES)
    return out
