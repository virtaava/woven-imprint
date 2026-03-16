"""Multi-character interaction — two characters talk to each other."""

from __future__ import annotations

from dataclasses import dataclass, field

from .character import Character
from .utils.text import generate_id


@dataclass
class InteractionTurn:
    """A single turn in a multi-character interaction."""

    speaker: str
    listener: str
    message: str
    response: str


@dataclass
class InteractionResult:
    """Full result of a multi-character interaction."""

    participants: list[str]
    situation: str
    turns: list[InteractionTurn] = field(default_factory=list)
    session_id: str = ""


def interact(
    char_a: Character,
    char_b: Character,
    situation: str,
    rounds: int = 2,
    a_opens: bool = True,
) -> InteractionResult:
    """Run a multi-round conversation between two characters.

    Each round: one character speaks, the other responds. Memories and
    relationships update for both sides automatically.

    Args:
        char_a: First character.
        char_b: Second character.
        situation: Scene description / context for the interaction.
        rounds: Number of back-and-forth exchanges.
        a_opens: If True, char_a speaks first. Otherwise char_b.

    Returns:
        InteractionResult with all turns.
    """
    first, second = (char_a, char_b) if a_opens else (char_b, char_a)
    result = InteractionResult(
        participants=[first.name, second.name],
        situation=situation,
        session_id=generate_id("int-"),
    )

    # Ensure sessions are active
    if not first._session_id:
        first.start_session()
    if not second._session_id:
        second.start_session()

    last_response = ""

    for round_num in range(rounds):
        # First character speaks
        if round_num == 0:
            prompt_a = (
                f"[Scene: {situation}]\nYou encounter {second.name}. React and speak in character."
            )
        else:
            prompt_a = (
                f"[Scene: {situation}]\n"
                f'{second.name} says to you: "{last_response[:500]}"\n'
                f"Respond in character."
            )

        response_a = first.chat(prompt_a, user_id=second.id)

        # Second character responds
        prompt_b = (
            f"[Scene: {situation}]\n"
            f'{first.name} says to you: "{response_a[:500]}"\n'
            f"Respond in character."
        )
        response_b = second.chat(prompt_b, user_id=first.id)

        result.turns.append(
            InteractionTurn(
                speaker=first.name,
                listener=second.name,
                message=prompt_a[:200],
                response=response_a,
            )
        )
        result.turns.append(
            InteractionTurn(
                speaker=second.name,
                listener=first.name,
                message=prompt_b[:200],
                response=response_b,
            )
        )

        last_response = response_b

        # Swap who leads next round
        first, second = second, first

    return result


def group_interaction(
    characters: list[Character],
    situation: str,
    rounds: int = 1,
) -> list[InteractionResult]:
    """Run a group scene where each character reacts to the situation.

    Each character observes the situation and previous responses, then
    contributes. Not a direct dialogue — more like a group scene.

    Args:
        characters: All participating characters.
        situation: Scene description.
        rounds: Number of full rotations.

    Returns:
        List of InteractionResults (one per round).
    """
    results = []
    accumulated_context = ""

    for round_num in range(rounds):
        round_result = InteractionResult(
            participants=[c.name for c in characters],
            situation=situation,
            session_id=generate_id("grp-"),
        )

        for i, char in enumerate(characters):
            if not char._session_id:
                char.start_session()

            other_names = [c.name for c in characters if c.id != char.id]
            others_str = ", ".join(other_names)

            prompt = f"[Scene: {situation}]\nPresent: {others_str}\n"
            if accumulated_context:
                prompt += f"\nWhat has happened so far:\n{accumulated_context}\n"
            prompt += f"\nAs {char.name}, react to this scene. What do you do and say?"

            # Use first other character as relationship target
            target_id = characters[(i + 1) % len(characters)].id
            response = char.chat(prompt, user_id=target_id)

            round_result.turns.append(
                InteractionTurn(
                    speaker=char.name,
                    listener=others_str,
                    message=prompt[:200],
                    response=response,
                )
            )

            accumulated_context += f"\n{char.name}: {response[:200]}"

        # Trim accumulated context to prevent unbounded growth
        if len(accumulated_context) > 5000:
            accumulated_context = accumulated_context[-4000:]

        results.append(round_result)

    return results
