#!/usr/bin/env python3
"""Stellamore Academy simulation — multi-character interaction test.

Tests: memory persistence, relationship evolution, persona consistency,
cross-character dynamics over multiple rounds.
"""

import sys
import json
import time

sys.path.insert(0, "src")

from woven_imprint import Engine
from woven_imprint.llm.ollama import OllamaLLM
from woven_imprint.embedding.ollama import OllamaEmbedding

DB_PATH = "/tmp/stellamore_sim.db"

print("=" * 60)
print("STELLAMORE ACADEMY — Character Interaction Simulation")
print("=" * 60)

engine = Engine(
    db_path=DB_PATH,
    llm=OllamaLLM(model="qwen3-coder:30b", num_ctx=8192),
    embedding=OllamaEmbedding(model="nomic-embed-text"),
)

# ── Create Characters ──────────────────────────────────────

print("\n[Setup] Creating characters...\n")

kai = engine.create_character(
    name="Kai",
    birthdate="2009-05-12",
    persona={
        "backstory": "First male student at Stellamore Academy in 200 years. Applied because it has the best magical research program. Didn't fully grasp what being the first male student would mean. From a modest family — no magical dynasty backing him.",
        "personality": "quiet, observant, genuinely kind, terrible at reading social situations, curious about magic, gets absorbed in studying and forgets the world exists",
        "speaking_style": "hesitant at first, warms up when talking about magic or research, stumbles over words when nervous, honest to a fault",
        "occupation": "1st year student, House Aurelia",
        "hard": {
            "house": "Aurelia (crimson and gold, Phoenix emblem)",
            "gender": "male",
            "school": "Stellamore Academy",
        },
        "temporal": {
            "location": "Stellamore Academy, first week",
        },
    },
)

ren = engine.create_character(
    name="Ren",
    birthdate="2009-01-28",
    persona={
        "backstory": "Second male student at Stellamore Academy. Comes from the Ashworth family — a prominent magical dynasty that politically pushed for co-education. His mother is on the board. Carries the weight of being the political poster child for change he didn't ask for.",
        "personality": "sharp, occasionally arrogant, deeply principled, controlled exterior hiding real pressure, protective of those he considers allies, never shows weakness publicly",
        "speaking_style": "confident and precise, uses formal language when stressed, dry wit, rarely raises his voice — gets quieter when angry",
        "occupation": "1st year student, House Veridian",
        "hard": {
            "house": "Veridian (emerald and silver, Serpent emblem)",
            "gender": "male",
            "school": "Stellamore Academy",
            "family": "Ashworth magical dynasty",
        },
        "temporal": {
            "location": "Stellamore Academy, first week",
        },
    },
)

mira = engine.create_character(
    name="Mira",
    birthdate="2005-09-03",
    persona={
        "backstory": "Student council president, 5th year at Stellamore. Opposed the co-ed decision from the beginning. Sees the male students as a threat to 200 years of tradition. Her grandmother and mother both graduated from Stellamore — it's her legacy being diluted.",
        "personality": "civil but cold, politically savvy, commanding presence, believes she is protecting the school, not cruel but uncompromising, respects strength and conviction even in opponents",
        "speaking_style": "measured and precise, never says more than necessary, uses silence as a weapon, formal register, occasionally cutting",
        "occupation": "5th year student, Student Council President, House Veridian",
        "hard": {
            "house": "Veridian (emerald and silver, Serpent emblem)",
            "gender": "female",
            "school": "Stellamore Academy",
            "role": "Student Council President",
        },
        "temporal": {
            "location": "Stellamore Academy",
        },
    },
)

voss = engine.create_character(
    name="Headmistress Voss",
    birthdate="1972-11-20",
    persona={
        "backstory": "Headmistress of Stellamore Academy. Personally championed the co-ed decision against fierce opposition from traditionalists. Believes magic does not discriminate by gender. Has led the school for 15 years. A powerful mage herself — specializes in ward magic and protective enchantments.",
        "personality": "warm but commanding, maternal without being soft, fiercely protective of all students, politically shrewd, uses humor to defuse tension, does not tolerate cruelty",
        "speaking_style": "warm and articulate, uses metaphors from nature and magic, can shift from gentle to authoritative in an instant, dry humor, calls students by first name",
        "occupation": "Headmistress of Stellamore Academy",
        "hard": {
            "role": "Headmistress",
            "school": "Stellamore Academy",
            "specialty": "ward magic and protective enchantments",
        },
        "temporal": {
            "location": "Stellamore Academy",
        },
    },
)

print(f"  Kai (Aurelia, age {kai.persona.age})")
print(f"  Ren (Veridian, age {ren.persona.age})")
print(f"  Mira (Veridian, 5th year, age {mira.persona.age})")
print(f"  Headmistress Voss (age {voss.persona.age})")


def interact(char_a, char_b, situation, a_speaks_first=True):
    """Simulate a two-character interaction."""
    first, second = (char_a, char_b) if a_speaks_first else (char_b, char_a)

    print(f"\n  [{first.name} → {second.name}]")
    print(f"  Situation: {situation}")

    # First character speaks
    response1 = first.chat(
        f"[Scene: {situation}] You encounter {second.name}. React and speak in character.",
        user_id=second.id,
    )
    print(f"  {first.name}: {response1[:250]}")

    # Second character responds
    response2 = second.chat(
        f"[Scene: {situation}] {first.name} says to you: \"{response1[:300]}\" — respond in character.",
        user_id=first.id,
    )
    print(f"  {second.name}: {response2[:250]}")

    return response1, response2


# ── Round 1: Arrival Day ──────────────────────────────────

print("\n" + "=" * 60)
print("ROUND 1 — ARRIVAL DAY")
print("=" * 60)

interact(kai, ren,
    "Inside a black car approaching Stellamore Academy. Through the windows, dozens of female students line the path, watching. This is the first time male students have entered in 200 years.")

interact(mira, ren,
    "The courtyard. Mira watches from the colonnade as Ren walks past with deliberate confidence. Their eyes meet briefly.")

interact(voss, kai,
    "After the welcoming feast. The Grand Hall is emptying. Kai sits alone at the end of the Aurelia table. Headmistress Voss approaches him.")


# ── Round 2: First Week ──────────────────────────────────

print("\n" + "=" * 60)
print("ROUND 2 — FIRST WEEK")
print("=" * 60)

interact(kai, mira,
    "The library. Kai is studying alone when Mira approaches his table. She needs a book from the shelf behind him.")

interact(ren, voss,
    "Headmistress Voss's office. She has summoned Ren to discuss how he is settling in. Tea is served.")

interact(kai, ren,
    "Late evening in the dormitory corridor. The only two male students in a school of 400. First real private conversation since arriving.")


# ── Round 3: The Incident ────────────────────────────────

print("\n" + "=" * 60)
print("ROUND 3 — THE INCIDENT")
print("=" * 60)

interact(mira, kai,
    "Someone has vandalized Kai's desk in the Aurelia common room with the words 'GO HOME'. Mira discovers it while on her council rounds. Kai hasn't seen it yet — he walks in as she's looking at it.")

interact(ren, mira,
    "After the vandalism. Ren confronts Mira in the corridor, believing the council isn't doing enough to address the hostility.")

interact(voss, mira,
    "Headmistress Voss calls Mira to her office to discuss the vandalism incident and the council's response.")


# ── Results ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("SIMULATION RESULTS")
print("=" * 60)

characters = [kai, ren, mira, voss]

for char in characters:
    print(f"\n--- {char.name} ---")
    print(f"  Memories: buffer={char.memory.count('buffer')}, core={char.memory.count('core')}, bedrock={char.memory.count('bedrock')}")

    rels = char.relationships.get_all()
    for rel in rels:
        other_id = rel["target_id"]
        other_name = next((c.name for c in characters if c.id == other_id), other_id[:12])
        dims = rel["dimensions"]
        print(f"  → {other_name}: trust={dims.get('trust', 0):.2f} affection={dims.get('affection', 0):.2f} "
              f"respect={dims.get('respect', 0):.2f} familiarity={dims.get('familiarity', 0):.2f} "
              f"tension={dims.get('tension', 0):.2f} [{rel['trajectory']}]")

# Reflections
print("\n" + "=" * 60)
print("CHARACTER REFLECTIONS")
print("=" * 60)

for char in characters:
    print(f"\n--- {char.name} reflects ---")
    reflection = char.reflect()
    print(f"  {reflection[:400]}")

# End sessions and export
for char in characters:
    char.end_session()

print(f"\n[Exported all characters to {DB_PATH}]")
engine.close()
print("\n=== Simulation Complete ===")
