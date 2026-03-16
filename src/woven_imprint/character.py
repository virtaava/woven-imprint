"""Character — the core entity that remembers, stays consistent, and evolves."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .llm.base import LLMProvider
from .embedding.base import EmbeddingProvider
from .memory.store import MemoryStore
from .memory.retrieval import MemoryRetriever
from .memory.belief import BeliefReviser
from .memory.consolidation import ConsolidationEngine
from .persona.model import PersonaModel
from .persona.consistency import ConsistencyChecker
from .relationship.model import RelationshipModel
from .storage.sqlite import SQLiteStorage
from .utils.text import generate_id


class Character:
    """A persistent AI character with memory, persona, and relationships.

    Usage:
        # Usually created via Engine, not directly
        character = engine.create_character("Alice", persona={...})
        response = character.chat("Hello!")
        character.reflect()
    """

    def __init__(
        self,
        char_id: str,
        storage: SQLiteStorage,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        persona: PersonaModel,
    ):
        self.id = char_id
        self.storage = storage
        self.llm = llm
        self.embedder = embedder
        self.persona = persona

        # Sub-systems
        self.memory = MemoryStore(storage, embedder, char_id)
        self.retriever = MemoryRetriever(storage, embedder, char_id)
        self.belief = BeliefReviser(storage, char_id)
        self.relationships = RelationshipModel(storage, char_id)
        self.consolidator = ConsolidationEngine(storage, llm, embedder, char_id)
        self.consistency = ConsistencyChecker(llm, persona)

        # Session tracking
        self._session_id: str | None = None
        self._turn_count: int = 0

        # Config
        self.enforce_consistency: bool = True

    @property
    def name(self) -> str:
        return self.persona.name

    def start_session(self) -> str:
        """Start a new conversation session. Returns session ID."""
        self._session_id = generate_id("sess-")
        self._turn_count = 0
        self.storage.save_session(
            {
                "id": self._session_id,
                "character_id": self.id,
            }
        )
        return self._session_id

    def chat(self, message: str, user_id: str | None = None) -> str:
        """Send a message and get an in-character response.

        Args:
            message: The user's message.
            user_id: Optional user identifier for relationship tracking.

        Returns:
            The character's response.
        """
        if not self._session_id:
            self.start_session()

        # 1. Store user message as buffer memory
        self.memory.add(
            content=f"[User] {message}",
            tier="buffer",
            role="user",
            session_id=self._session_id,
            importance=0.5,
        )

        # 2. Retrieve relevant memories
        memories = self.retriever.retrieve(
            query=message,
            limit=10,
            relationship_target=user_id,
        )

        # 3. Get relationship context
        rel_context = ""
        if user_id:
            self.relationships.get_or_create(user_id)
            rel_context = self.relationships.describe(user_id)

        # 4. Build prompt with persona + memory + relationship context
        memory_text = self._format_memories(memories)

        system_prompt = self.persona.build_system_prompt()
        if memory_text:
            system_prompt += f"\n\nYour relevant memories:\n{memory_text}"
        if rel_context:
            system_prompt += f"\n\n{rel_context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

        # 5. Generate response (with optional consistency enforcement)
        response = self.llm.generate(messages, temperature=0.7)

        if self.enforce_consistency:
            response, report = self.consistency.enforce(response, messages)
            if report.soft_flags:
                # Store soft flags as metadata on the response memory
                pass  # Flags captured in the consistency report, not blocking

        # 6. Store character response as buffer memory
        self.memory.add(
            content=f"[{self.name}] {response}",
            tier="buffer",
            role="character",
            session_id=self._session_id,
            importance=0.5,
        )

        # 7. Extract and store notable facts from the exchange
        self._extract_memories(message, response, user_id)

        self._turn_count += 1
        return response

    def reflect(self) -> str:
        """Generate higher-level reflections from accumulated memories.

        Stanford-style: synthesize recent memories into insights about
        the character's situation, relationships, and goals.

        Returns:
            The reflection text.
        """
        recent = self.memory.get_all(tier="buffer", limit=50)
        if len(recent) < 5:
            return "Not enough recent memories to reflect on."

        recent_text = "\n".join(f"- {m['content'][:200]}" for m in recent[:30])

        messages = [
            {"role": "system", "content": self.persona.build_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Based on your recent experiences, reflect on:\n"
                    f"1. What patterns do you notice?\n"
                    f"2. How do you feel about recent interactions?\n"
                    f"3. Have your opinions or feelings changed about anything?\n"
                    f"4. What do you want to do next?\n\n"
                    f"Recent memories:\n{recent_text}\n\n"
                    f"Write your reflection as inner thoughts, in first person. "
                    f"Be honest with yourself. 3-5 sentences."
                ),
            },
        ]

        reflection = self.llm.generate(messages, temperature=0.6)

        # Store reflection as core memory (higher tier)
        self.memory.add(
            content=f"[Reflection] {reflection}",
            tier="core",
            role="observation",
            session_id=self._session_id,
            importance=0.8,
        )

        return reflection

    def consolidate(self) -> dict:
        """Compress buffer memories into core memories.

        Clusters semantically similar buffer entries and summarizes them.
        Original entries are archived, not deleted.

        Returns:
            Stats dict: {clusters, summarized, created, archived}.
        """
        return self.consolidator.consolidate()

    def recall(self, query: str, limit: int = 10) -> list[dict]:
        """Explicitly recall memories relevant to a query."""
        return self.retriever.retrieve(query, limit=limit)

    def end_session(self) -> str | None:
        """End the current session and generate a summary.

        Returns:
            Session summary text, or None if no session active.
        """
        if not self._session_id:
            return None

        # Get session memories
        session_memories = [
            m
            for m in self.memory.get_all(tier="buffer", limit=200)
            if m.get("session_id") == self._session_id
        ]

        if not session_memories:
            self._session_id = None
            return None

        # Generate session summary
        mem_text = "\n".join(f"- {m['content'][:150]}" for m in session_memories[:30])
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are summarizing a conversation session for {self.name}. "
                    f"Capture: key events, emotional beats, relationship changes, "
                    f"new information learned, commitments made."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Summarize this session:\n{mem_text}\n\n"
                    f"Write a concise summary (3-5 sentences) capturing the most important "
                    f"moments and any changes in relationships or beliefs."
                ),
            },
        ]

        summary = self.llm.generate(messages, temperature=0.3)

        # Store summary as core memory
        self.memory.add(
            content=f"[Session Summary] {summary}",
            tier="core",
            role="observation",
            session_id=self._session_id,
            importance=0.7,
        )

        # Update session record
        self.storage.save_session(
            {
                "id": self._session_id,
                "character_id": self.id,
                "summary": summary,
            }
        )

        self._session_id = None
        self._turn_count = 0
        return summary

    def export(self, path: str | None = None) -> dict:
        """Export full character state as JSON.

        Args:
            path: Optional file path to write JSON to.

        Returns:
            Character state dict.
        """
        data = {
            "id": self.id,
            "persona": self.persona.to_dict(),
            "birthdate": self.persona.birthdate.isoformat() if self.persona.birthdate else None,
            "memories": {
                "buffer": self.memory.get_all(tier="buffer"),
                "core": self.memory.get_all(tier="core"),
                "bedrock": self.memory.get_all(tier="bedrock"),
            },
            "relationships": self.relationships.get_all(),
            "sessions": self.storage.get_sessions(self.id),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        # Strip embeddings from export (too large, recomputable)
        for tier_name in ("buffer", "core", "bedrock"):
            for mem in data["memories"][tier_name]:
                mem.pop("embedding", None)

        if path:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        return data

    def _extract_memories(self, user_msg: str, response: str, user_id: str | None) -> None:
        """Extract notable facts and update relationships from an exchange."""
        # Relationship updates happen every turn
        if user_id:
            self._update_relationship(user_msg, response, user_id)

        # Fact extraction is throttled — every 3rd turn
        if self._turn_count % 3 != 0 and self._turn_count > 0:
            return

        messages = [
            {
                "role": "system",
                "content": (
                    "Extract specific facts, opinions, or commitments from this exchange "
                    "that are worth remembering long-term. Return a JSON array of strings. "
                    "Each string should be a single fact. Return [] if nothing notable."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User said: {user_msg}\n"
                    f"{self.name} responded: {response}\n\n"
                    f"What facts should {self.name} remember?"
                ),
            },
        ]

        try:
            result = self.llm.generate_json(messages)
            facts = result if isinstance(result, list) else result.get("facts", [])
            for fact in facts[:5]:  # Cap at 5 facts per extraction
                if isinstance(fact, str) and len(fact) > 10:
                    self.memory.add(
                        content=fact,
                        tier="core",
                        role="observation",
                        session_id=self._session_id,
                        importance=0.6,
                        metadata={"source": "extraction", "user_id": user_id},
                    )
        except (ValueError, KeyError):
            pass  # Extraction failed — not critical

    def _update_relationship(self, user_msg: str, response: str, user_id: str) -> None:
        """LLM-assess how an interaction shifts relationship dimensions."""
        current = self.relationships.get_or_create(user_id)
        dims = current["dimensions"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You assess how a conversation exchange affects a relationship "
                    "between two people. Return a JSON object with these float fields "
                    "(each between -0.15 and 0.15, use 0.0 for no change):\n"
                    "- trust: did this interaction build or erode trust?\n"
                    "- affection: did warmth increase or decrease?\n"
                    "- respect: did admiration change?\n"
                    "- familiarity: how much did they learn about each other? (0.0 to 0.15 only)\n"
                    "- tension: did unresolved conflict increase or decrease?\n\n"
                    "Be conservative. Most single exchanges cause small changes (0.01-0.05). "
                    "Only dramatic moments warrant larger shifts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current relationship: {current['type']}, "
                    f"trust={dims.get('trust', 0):.2f}, "
                    f"affection={dims.get('affection', 0):.2f}, "
                    f"familiarity={dims.get('familiarity', 0):.2f}\n\n"
                    f"User said: {user_msg[:300]}\n"
                    f"{self.name} responded: {response[:300]}\n\n"
                    f"How does this exchange shift the relationship? Return JSON."
                ),
            },
        ]

        try:
            result = self.llm.generate_json(messages)
            deltas = {}
            for key in ("trust", "affection", "respect", "familiarity", "tension"):
                val = result.get(key, 0.0)
                if isinstance(val, (int, float)):
                    deltas[key] = float(val)
            if deltas:
                self.relationships.update(user_id, deltas)
        except (ValueError, KeyError, TypeError):
            pass  # Relationship update failed — not critical

    def _format_memories(self, memories: list[dict]) -> str:
        """Format retrieved memories for inclusion in prompt."""
        if not memories:
            return ""
        lines = []
        for m in memories:
            tier_tag = f"[{m['tier']}]" if m["tier"] != "buffer" else ""
            certainty = m.get("certainty", 1.0)
            cert_tag = " (uncertain)" if certainty < 0.5 else ""
            lines.append(f"- {tier_tag}{cert_tag} {m['content'][:200]}")
        return "\n".join(lines)
