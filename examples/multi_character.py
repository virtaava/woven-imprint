#!/usr/bin/env python3
"""Multi-character interaction — two characters talk to each other."""

from woven_imprint import Engine, interact

engine = Engine("tavern.db")

# Create two NPCs
innkeeper = engine.create_character(
    name="Greta",
    persona={
        "backstory": "Runs the Golden Stag tavern. Knows everyone's business. Fiercely loyal to regulars.",
        "personality": "warm, shrewd, motherly, gossips strategically",
        "speaking_style": "hearty, uses food metaphors, calls everyone 'love' or 'dear'",
    },
)

stranger = engine.create_character(
    name="Cael",
    persona={
        "backstory": "A wandering sellsword who just arrived in town. Hiding from something. Covered in road dust.",
        "personality": "guarded, observant, quick to distrust, surprisingly gentle with children",
        "speaking_style": "clipped, economical with words, avoids personal questions",
    },
)

# They interact
print("Scene: A tired stranger enters the Golden Stag at dusk.\n")

result = interact(
    innkeeper, stranger,
    situation="A dusty stranger pushes open the tavern door at dusk. The tavern is half-full. Greta is behind the bar.",
    rounds=3,
)

for turn in result.turns:
    print(f"{turn.speaker}: {turn.response[:200]}\n")

# Check their relationship
print(f"\nGreta's view: {innkeeper.relationships.describe(stranger.id)}")
print(f"Cael's view: {stranger.relationships.describe(innkeeper.id)}")

engine.close()
