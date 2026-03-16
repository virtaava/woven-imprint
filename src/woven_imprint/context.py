"""Context window management — conversation buffer with overflow handling.

Manages the sliding window of recent conversation turns sent to the LLM,
with automatic compression when the context budget is exceeded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextBudget:
    """How to allocate the context window.

    All values in estimated tokens (1 token ≈ 4 chars).
    """

    total: int = 6000  # total tokens available for our content
    system_prompt: int = 1000  # persona + emotion + arc
    memories: int = 1500  # retrieved memories
    conversation: int = 3000  # recent conversation turns
    reserve: int = 500  # safety margin


@dataclass
class ConversationTurn:
    """A single turn in the conversation buffer."""

    role: str  # "user" or "assistant"
    content: str

    @property
    def estimated_tokens(self) -> int:
        return len(self.content) // 4 + 1


class ContextManager:
    """Manage the conversation buffer with context window awareness.

    Keeps recent turns in a sliding window. When the window exceeds the
    budget, older turns are summarized and compressed.
    """

    def __init__(self, budget: ContextBudget | None = None, max_turns: int = 20):
        self.budget = budget or ContextBudget()
        self.max_turns = max_turns
        self._turns: list[ConversationTurn] = []
        self._summary: str = ""  # compressed older turns

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def add_turn(self, role: str, content: str) -> None:
        """Add a conversation turn to the buffer."""
        self._turns.append(ConversationTurn(role=role, content=content))
        self._enforce_limits()

    def get_messages(self) -> list[dict[str, str]]:
        """Get the conversation buffer as a message list for the LLM.

        Returns messages including the compressed summary of older turns
        (if any) followed by recent turns.
        """
        messages = []

        if self._summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Summary of earlier conversation:\n{self._summary}",
                }
            )

        for turn in self._turns:
            messages.append({"role": turn.role, "content": turn.content})

        return messages

    def get_conversation_tokens(self) -> int:
        """Estimate total tokens in the conversation buffer."""
        total = sum(t.estimated_tokens for t in self._turns)
        if self._summary:
            total += len(self._summary) // 4
        return total

    def clear(self) -> None:
        """Clear the conversation buffer."""
        self._turns.clear()
        self._summary = ""

    def compress(self, llm=None) -> str:
        """Compress older turns into a summary.

        If an LLM is provided, uses it to generate the summary.
        Otherwise, creates a simple concatenation of older turns.

        Returns the summary text.
        """
        if len(self._turns) <= 4:
            return self._summary

        # Keep the last 4 turns, compress the rest
        to_compress = self._turns[:-4]
        self._turns = self._turns[-4:]

        if llm:
            # LLM-powered compression
            turns_text = "\n".join(f"{t.role}: {t.content[:200]}" for t in to_compress)
            try:
                summary = llm.generate(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Summarize this conversation excerpt concisely. "
                                "Preserve key facts, decisions, and emotional moments. "
                                "2-3 sentences maximum."
                            ),
                        },
                        {"role": "user", "content": turns_text},
                    ],
                    temperature=0.3,
                    max_tokens=200,
                )
            except Exception:
                summary = self._simple_compress(to_compress)
        else:
            summary = self._simple_compress(to_compress)

        # Append to existing summary
        if self._summary:
            self._summary = f"{self._summary}\n{summary}"
        else:
            self._summary = summary

        # Trim summary if too long
        max_summary_chars = self.budget.conversation * 2  # chars, not tokens
        if len(self._summary) > max_summary_chars:
            self._summary = self._summary[-max_summary_chars:]

        return self._summary

    def _enforce_limits(self) -> None:
        """Drop oldest turns if we exceed max_turns."""
        while len(self._turns) > self.max_turns:
            dropped = self._turns.pop(0)
            # Add to summary as simple text
            if self._summary:
                self._summary += f"\n{dropped.role}: {dropped.content[:100]}"
            else:
                self._summary = f"{dropped.role}: {dropped.content[:100]}"

    def _simple_compress(self, turns: list[ConversationTurn]) -> str:
        """Create a simple text summary without LLM."""
        parts = []
        for t in turns:
            prefix = "User said" if t.role == "user" else "Character said"
            parts.append(f"{prefix}: {t.content[:100]}")
        return "; ".join(parts)

    def to_dict(self) -> dict:
        return {
            "turns": [{"role": t.role, "content": t.content} for t in self._turns],
            "summary": self._summary,
        }

    @classmethod
    def from_dict(cls, data: dict, budget: ContextBudget | None = None) -> ContextManager:
        cm = cls(budget=budget)
        cm._summary = data.get("summary", "")
        for t in data.get("turns", []):
            cm._turns.append(ConversationTurn(role=t["role"], content=t["content"]))
        return cm
