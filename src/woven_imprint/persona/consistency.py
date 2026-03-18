"""NLI-inspired consistency checker — verifies responses against persona constraints."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import LLMProvider
from .model import PersonaModel


@dataclass
class ConsistencyReport:
    """Result of a consistency check."""

    consistent: bool
    hard_violations: list[str] = field(default_factory=list)
    soft_flags: list[str] = field(default_factory=list)
    score: float = 1.0  # 0.0 = completely inconsistent, 1.0 = fully consistent


class ConsistencyChecker:
    """Check if a character response is consistent with their persona.

    Uses LLM-based NLI (Natural Language Inference) to detect contradictions
    between the generated response and the character's persona constraints.

    Three-step process:
    1. Extract claims from the response
    2. Check claims against hard constraints → reject if contradiction
    3. Check claims against soft constraints → flag but allow with growth context
    """

    def __init__(
        self,
        llm: LLMProvider,
        persona: PersonaModel,
        config=None,
    ):
        self.llm = llm
        self.persona = persona
        # Configurable values from CharacterConfig
        if config is not None:
            self._max_retries = config.consistency_max_retries
            self._retry_temp = config.consistency_temperature
            self._fail_open_score = config.consistency_fail_open_score
        else:
            self._max_retries = 2
            self._retry_temp = 0.5
            self._fail_open_score = 0.8

    def check(self, response: str, context: str = "") -> ConsistencyReport:
        """Check a response for persona consistency.

        Args:
            response: The character's generated response.
            context: Optional conversation context for growth justification.

        Returns:
            ConsistencyReport with violations and score.
        """
        hard_facts = self.persona.get_hard_facts()
        if not hard_facts:
            return ConsistencyReport(consistent=True)

        # Build the NLI prompt
        facts_text = "\n".join(f"- {f}" for f in hard_facts)

        soft_traits = []
        if self.persona.soft.get("personality"):
            soft_traits.append(f"Personality: {self.persona.soft['personality']}")
        if self.persona.soft.get("speaking_style"):
            soft_traits.append(f"Speaking style: {self.persona.soft['speaking_style']}")
        soft_text = "\n".join(f"- {t}" for t in soft_traits) if soft_traits else "None specified"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a consistency verification system. Check if a character's "
                    "response contradicts their established facts or personality.\n\n"
                    "Output JSON with:\n"
                    "- hard_violations: list of strings describing contradictions with "
                    "  immutable facts (name, backstory, species, etc.)\n"
                    "- soft_flags: list of strings describing potential personality "
                    "  inconsistencies (may be acceptable as character growth)\n"
                    "- score: float 0.0-1.0 (1.0 = fully consistent)\n\n"
                    "Be strict about hard facts. Be lenient about personality — "
                    "characters can have complex moments."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CHARACTER HARD FACTS:\n{facts_text}\n\n"
                    f"CHARACTER SOFT TRAITS:\n{soft_text}\n\n"
                    f"RESPONSE TO CHECK:\n{response}\n\n"
                    f"{'CONVERSATION CONTEXT:\n' + context + '\n\n' if context else ''}"
                    f"Check for contradictions. Return JSON."
                ),
            },
        ]

        try:
            result = self.llm.generate_json(messages)
            if not isinstance(result, dict):
                # Retry once at low temperature before falling through
                result = self.llm.generate_json(messages, temperature=0.1)
                if not isinstance(result, dict):
                    return ConsistencyReport(
                        consistent=True, score=self._fail_open_score
                    )
            hard = result.get("hard_violations", [])
            soft = result.get("soft_flags", [])
            score = float(result.get("score", 1.0))

            return ConsistencyReport(
                consistent=len(hard) == 0,
                hard_violations=hard if isinstance(hard, list) else [],
                soft_flags=soft if isinstance(soft, list) else [],
                score=max(0.0, min(1.0, score)),
            )
        except (ValueError, KeyError, TypeError):
            # If consistency check fails, assume consistent (don't block generation)
            return ConsistencyReport(consistent=True, score=self._fail_open_score)

    def enforce(
        self,
        response: str,
        messages: list[dict[str, str]],
        max_retries: int | None = None,
    ) -> tuple[str, ConsistencyReport]:
        """Check response and regenerate if hard violations found.

        Args:
            response: Initial response to check.
            messages: Original message list for regeneration.
            max_retries: Max regeneration attempts (uses config default if None).

        Returns:
            Tuple of (best_response, final_report).
        """
        if max_retries is None:
            max_retries = self._max_retries

        # Build conversation context from the last 3 non-system message pairs
        context_parts = []
        non_system = [m for m in messages if m.get("role") != "system"]
        for m in non_system[-6:]:  # last 3 pairs (user+assistant)
            role = m.get("role", "unknown")
            content = m.get("content", "")[:300]
            context_parts.append(f"{role}: {content}")
        context = "\n".join(context_parts)

        report = self.check(response, context=context)
        if report.consistent:
            return response, report

        best_response = response
        best_score = report.score

        for attempt in range(max_retries):
            # Add constraint reminder to messages
            violation_text = "\n".join(f"- {v}" for v in report.hard_violations)
            retry_messages = messages + [
                {
                    "role": "system",
                    "content": (
                        f"IMPORTANT: Your previous response contradicted these "
                        f"established facts about your character:\n{violation_text}\n\n"
                        f"Regenerate your response while staying consistent with "
                        f"who you are. Do not contradict your backstory or identity."
                    ),
                }
            ]

            try:
                new_response = self.llm.generate(
                    retry_messages, temperature=self._retry_temp
                )
                new_report = self.check(new_response, context=context)

                if new_report.consistent:
                    return new_response, new_report

                if new_report.score > best_score:
                    best_response = new_response
                    best_score = new_report.score
                    report = new_report
            except Exception:
                break

        return best_response, report
