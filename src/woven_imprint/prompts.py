"""Central registry of every LLM prompt in Woven Imprint.

Each prompt site in the codebase renders its messages through `render()`
instead of building `[{"role": ..., "content": ...}, ...]` inline. This
gives every prompt a stable id + version, a one-line `expects` note for
`woven-imprint prompts`, and a single place to audit wording.

`render()` uses `str.format_map`, so literal `{`/`}` in a template (JSON
examples, mostly) must be escaped as `{{`/`}}`. Sites that need to choose
between two alternative literal phrasings (e.g. "No prior beats." vs.
"Recent beats:\n...") keep that branching in the call site, but the
literal wording for each branch lives here as a small fragment constant
the site fills in and passes as an already-rendered string.

Nothing here changes prompt wording — this module holds the exact strings
that used to live inline at each site, converted to templates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    id: str
    version: int
    system: str
    user: str
    expects: str  # human-readable schema note, shown by `woven-imprint prompts`


def render(prompt_id: str, **kwargs) -> list[dict]:
    """Render a registered prompt's messages with the given substitutions."""
    spec = PROMPTS[prompt_id]
    messages: list[dict] = []
    if spec.system:
        messages.append({"role": "system", "content": spec.system.format_map(kwargs)})
    if spec.user:
        messages.append({"role": "user", "content": spec.user.format_map(kwargs)})
    return messages


def ids() -> list[str]:
    return list(PROMPTS.keys())


# ── Fragment constants ──────────────────────────────────────────────────
# Literal wording for the branch-dependent pieces of a couple of prompts
# whose message construction is too dynamic to express as one flat
# str.format_map template (per-site conditional assembly, wording sourced
# from here). Sites `.format(...)` these directly, then pass the result
# as a plain kwarg to `render()`.

SOMEONE_FALLBACK = "Someone"

ARC_BEAT_RECENT_TMPL = "Recent beats:\n{recent}"
ARC_BEAT_NONE_TMPL = "No prior beats."

TURN_ASSESSMENT_RECENT_TMPL = "Recent beats:\n{recent}\n"
TURN_ASSESSMENT_NONE_TMPL = "No prior beats.\n"

CONSISTENCY_CONTEXT_FRAGMENT = "CONVERSATION CONTEXT:\n{context}\n\n"

CALLBACKS_EMOTION_SUFFIX = "\n\n{emotion_desc}"
CALLBACKS_HOOK_SUFFIX = (
    "\nNaturally work in this thought of yours (paraphrase, don't quote): {hook}"
)

TURN_ASSESSMENT_EMOTION_SECTION = (
    '"emotion": {{"mood": one of {moods}, "intensity": float 0.0-1.0, '
    '"cause": one sentence}}. Be realistic: most exchanges produce mild emotions (0.2-0.5).'
)
TURN_ASSESSMENT_RELATIONSHIP_SECTION = (
    '"relationship": {{"trust", "affection", "respect", "familiarity", "tension"}} — each a float '
    "between -0.15 and 0.15 (familiarity 0.0 to 0.15) describing how THIS exchange shifts the "
    "relationship; 0.0 for no change. Be conservative: most exchanges are 0.01-0.05."
)
TURN_ASSESSMENT_BEAT_SECTION = (
    '"beat": {{"is_beat": bool, "description": one sentence, "phase": setup|rising_action|climax|'
    'falling_action|resolution|epilogue, "tension": float 0.0-1.0, "tags": [strings]}} or null. '
    "Only genuine story beats (revelations, confrontations, betrayals, decisions) are beats."
)
TURN_ASSESSMENT_FACTS_SECTION = (
    '"facts": up to {max_facts} strings — specific NEW facts, opinions, preferences, biographical '
    "details or commitments worth remembering long-term, one per string; [] if nothing notable."
)


# ── Registry ─────────────────────────────────────────────────────────────

PROMPTS: dict[str, PromptSpec] = {
    "fact_extraction": PromptSpec(
        id="fact_extraction",
        version=1,
        system=(
            "Extract specific NEW facts, opinions, preferences, biographical details, "
            "or commitments from this exchange that are worth remembering long-term. "
            "Return a JSON array of strings. Each string should be a single fact. "
            "Focus on NEW information not already present in the recent context. "
            "Return [] if nothing notable."
        ),
        user=(
            "User said: {user_msg}\n"
            "{character_name} responded: {response}\n\n"
            "What facts should {character_name} remember?"
            "{context_hint}"
        ),
        expects="JSON array of fact strings",
    ),
    "relationship_turn": PromptSpec(
        id="relationship_turn",
        version=1,
        system=(
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
        user=(
            "Current relationship: {rel_type}, "
            "trust={trust:.2f}, "
            "affection={affection:.2f}, "
            "familiarity={familiarity:.2f}\n\n"
            "User said: {user_msg}\n"
            "{character_name} responded: {response}\n\n"
            "How does this exchange shift the relationship? Return JSON."
        ),
        expects="JSON: trust/affection/respect/familiarity/tension deltas",
    ),
    "relationship_event": PromptSpec(
        id="relationship_event",
        version=1,
        system=(
            "You assess how an EVENT affects a relationship between a character "
            "and another party. Return a JSON object with float fields between "
            "-0.15 and 0.15 (0.0 = no change): trust, affection, respect, "
            "familiarity (0.0 to 0.15 only), tension. Be conservative."
        ),
        user=(
            "Current: trust={trust:.2f}, "
            "affection={affection:.2f}\n"
            "Event involving {user_id}: {event}\n\n"
            "Return JSON."
        ),
        expects="JSON: trust/affection/respect/familiarity/tension deltas",
    ),
    "reflect": PromptSpec(
        id="reflect",
        version=1,
        system="{persona_system}",
        user=(
            "Based on your recent experiences, reflect on:\n"
            "1. What patterns do you notice?\n"
            "2. How do you feel about recent interactions?\n"
            "3. Have your opinions or feelings changed about anything?\n"
            "4. What do you want to do next?\n\n"
            "Recent memories:\n{recent_text}\n\n"
            "Write your reflection as inner thoughts, in first person. "
            "Be honest with yourself. 3-5 sentences."
        ),
        expects="free text reflection, first person, 3-5 sentences",
    ),
    "session_summary": PromptSpec(
        id="session_summary",
        version=1,
        system=(
            "You are summarizing a conversation session for {character_name}. "
            "Capture: key events, emotional beats, relationship changes, "
            "new information learned, commitments made."
        ),
        user=(
            "Summarize this session:\n{mem_text}\n\n"
            "Write a concise summary (3-5 sentences) capturing the most important "
            "moments and any changes in relationships or beliefs."
        ),
        expects="free text session summary, 3-5 sentences",
    ),
    "emotion_turn": PromptSpec(
        id="emotion_turn",
        version=1,
        system=(
            "You assess how a conversation affects a character's emotional state. "
            "Return JSON with:\n"
            "- mood: one of: joyful, content, excited, anxious, angry, sad, fearful, "
            "  disgusted, surprised, neutral, contemplative, melancholic, determined, "
            "  vulnerable, amused\n"
            "- intensity: float 0.0-1.0 (how strongly they feel this)\n"
            "- cause: brief reason for the mood (1 sentence)\n\n"
            "Be realistic. Most conversations produce mild emotions (0.2-0.5). "
            "Only dramatic events warrant high intensity."
        ),
        user=(
            "Character: {character_name}\n"
            "Current mood: {mood} (intensity {intensity:.1f})\n\n"
            "Someone said: {message}\n"
            "{character_name} responded: {response}\n\n"
            "What is {character_name}'s emotional state now? Return JSON."
        ),
        expects="JSON: mood/intensity/cause",
    ),
    "emotion_event": PromptSpec(
        id="emotion_event",
        version=1,
        system=(
            "You assess how an event affects {character_name}'s emotional state. "
            'Return JSON: {{"mood": <label>, "intensity": 0.0-1.0, "cause": <short reason>}}. '
            "Valid moods: {valid_moods}. "
            "Small events cause small shifts; keep continuity with the current mood."
        ),
        user=(
            "Current mood: {mood} (intensity {intensity:.1f}).\n"
            "Event: {event}\n\n"
            "How does {character_name} feel now? Return JSON."
        ),
        expects="JSON: mood/intensity/cause",
    ),
    "consistency_check": PromptSpec(
        id="consistency_check",
        version=1,
        system=(
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
        user=(
            "CHARACTER HARD FACTS:\n{facts_text}\n\n"
            "CHARACTER SOFT TRAITS:\n{soft_text}\n\n"
            "RESPONSE TO CHECK:\n{response}\n\n"
            "{context_section}"
            "Check for contradictions. Return JSON."
        ),
        expects="JSON: hard_violations/soft_flags/score",
    ),
    "consistency_retry_reminder": PromptSpec(
        id="consistency_retry_reminder",
        version=1,
        system=(
            "IMPORTANT: Your previous response contradicted these "
            "established facts about your character:\n{violation_text}\n\n"
            "Regenerate your response while staying consistent with "
            "who you are. Do not contradict your backstory or identity."
        ),
        user="",
        expects="single system reminder appended before regeneration",
    ),
    "growth": PromptSpec(
        id="growth",
        version=1,
        system=(
            "You analyze a character's accumulated experiences to detect "
            "genuine personality growth or change. Characters evolve slowly "
            "through meaningful experiences.\n\n"
            "Rules:\n"
            "- Only detect changes supported by multiple memories\n"
            "- Changes must be gradual, not sudden reversals\n"
            "- A character becoming 'slightly more trusting' is realistic; "
            "  'completely changed personality' is not\n"
            "- Return JSON array of growth events, or empty array []\n"
            "- Each event: {{trait, old_value, new_value, reason, confidence}}\n"
            "- confidence: 0.0-1.0 (how strongly the evidence supports this change)"
        ),
        user=(
            "CHARACTER'S CURRENT TRAITS:\n{soft_text}\n\n"
            "RECENT EXPERIENCES:\n{memory_text}\n\n"
            "Based on these experiences, has this character grown or changed "
            "in any way? Return JSON array of growth events."
        ),
        expects="JSON array of growth events",
    ),
    "arc_beat": PromptSpec(
        id="arc_beat",
        version=1,
        system=(
            "You analyze conversations for narrative significance. "
            "Not every exchange matters — only flag genuine story beats: "
            "revelations, confrontations, betrayals, reconciliations, "
            "decisions, turning points.\n\n"
            "Return JSON with:\n"
            "- is_beat: boolean (true if this is narratively significant)\n"
            "- description: what happened (1 sentence)\n"
            "- phase: setup|rising_action|climax|falling_action|resolution|epilogue\n"
            "- tension: float 0.0-1.0 (narrative tension level)\n"
            "- tags: list of story tags (e.g. 'revelation', 'confrontation', 'romantic')\n\n"
            "Be conservative. Most exchanges are NOT story beats."
        ),
        user=(
            "Current arc phase: {phase}\n"
            "Current tension: {tension:.1f}\n"
            "{beats_line}\n\n"
            "[{other_display}]: {message}\n"
            "[{character_name}]: {response}\n\n"
            "Is this a story beat?"
        ),
        expects="JSON: is_beat/description/phase/tension/tags",
    ),
    "consolidation_summary": PromptSpec(
        id="consolidation_summary",
        version=1,
        system=(
            "You are a memory consolidation system. Summarize the following "
            "related memories into a single dense entry that preserves all "
            "important facts, emotions, and relationships. Be concise but "
            "complete. Write in third person or as an observation."
        ),
        user=(
            "Consolidate these related memories into one summary:\n\n"
            "{cluster_text}\n\n"
            "Write a single paragraph capturing the key information."
        ),
        expects="free text single-paragraph summary",
    ),
    "callbacks_hooks": PromptSpec(
        id="callbacks_hooks",
        version=1,
        system=(
            "You create conversation hooks for {name} — things the "
            "character would naturally bring up with the person they know. "
            "Return a JSON array (max {limit}) of objects: "
            '{{"hook": <one natural in-character sentence>, '
            '"kind": "open_thread"|"callback"|"milestone"|"curiosity", '
            '"sources": [<numbers of the memories it draws on>]}}. '
            "RULES: paraphrase naturally — NEVER quote the memory text "
            "verbatim. open_thread = unresolved thing to ask about; "
            "callback = warm reference to a shared moment; milestone = "
            "anniversary or achievement; curiosity = something the "
            "character genuinely wonders about."
        ),
        user="Memories and moments:\n{numbered}",
        expects="JSON array of hook objects (hook/kind/sources)",
    ),
    "callbacks_compose": PromptSpec(
        id="callbacks_compose",
        version=1,
        system="{persona_system}{emotion_suffix}",
        user=(
            "Compose a short (1-2 sentence) message where you, {name}, "
            "initiate contact. Occasion: {occasion}.{hook_suffix}"
        ),
        expects="free text 1-2 sentence initiation message",
    ),
    "maintenance_importance": PromptSpec(
        id="maintenance_importance",
        version=1,
        system=(
            "You score the long-term importance of memories on a 1-10 scale "
            "(1 = mundane small talk, 10 = life-changing). "
            "Return a JSON array of integers, one per numbered memory, in order."
        ),
        user="Score these memories:\n{numbered}",
        expects="JSON array of integer scores 1-10",
    ),
    "maintenance_contradiction": PromptSpec(
        id="maintenance_contradiction",
        version=1,
        system=(
            "You check whether two remembered facts contradict each other. "
            'Return JSON: {{"contradictory": true|false, '
            '"current": "first"|"second"|"unclear"}} — "current" is the fact '
            "that reflects the present state if they contradict "
            "(the newer statement usually, unless it is clearly speculative)."
        ),
        user=("First (older, {a_date}): {a_content}\nSecond (newer, {b_date}): {b_content}"),
        expects="JSON: contradictory bool + current first/second/unclear",
    ),
    "context_compress": PromptSpec(
        id="context_compress",
        version=1,
        system=(
            "Summarize this conversation excerpt concisely. "
            "Preserve key facts, decisions, and emotional moments. "
            "2-3 sentences maximum."
        ),
        user="{turns_text}",
        expects="free text 2-3 sentence summary",
    ),
    "turn_assessment": PromptSpec(
        id="turn_assessment",
        version=1,
        system=(
            "You are the bookkeeping assistant for the character {character_name}. Assess the "
            "exchange and return ONE JSON object with exactly these keys:\n- {sections}"
        ),
        user=(
            "Character: {character_name}\n"
            "Current mood: {mood} (intensity {intensity:.1f})\n"
            "{rel_line}{arc_line}\n"
            "[{other_display}]: {message}\n"
            "[{character_name}]: {response}\n"
            "{context_hint}\n\nReturn the JSON object."
        ),
        expects="JSON: emotion + optional relationship/beat/facts",
    ),
}
