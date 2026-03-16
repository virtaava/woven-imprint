#!/usr/bin/env python3
"""Basic usage — create a character and chat."""

from woven_imprint import Engine

engine = Engine("my_characters.db")

# Create a character
npc = engine.create_character(
    name="Marcus",
    birthdate="1995-08-22",
    persona={
        "backstory": "A seasoned blacksmith in a small village. Lost his wife to illness two years ago. Throws himself into work to cope.",
        "personality": "gruff but kind, protective of the village, dry humor, uncomfortable with emotions",
        "speaking_style": "short sentences, working-class dialect, occasionally poetic when caught off guard",
    },
)

# Conversation
print(f"Chatting with {npc.name} (age {npc.persona.age})\n")

response = npc.chat("Marcus, I need a sword forged by tomorrow.", user_id="player")
print(f"Marcus: {response}\n")

response = npc.chat("It's urgent. Bandits are coming to the village.", user_id="player")
print(f"Marcus: {response}\n")

response = npc.chat("Your wife... she would have wanted you to protect this place.", user_id="player")
print(f"Marcus: {response}\n")

# Check what happened
print(f"\nEmotion: {npc.emotion.mood} (intensity {npc.emotion.intensity:.1f})")
print(f"Relationship: {npc.relationships.describe('player')}")

# Reflect
reflection = npc.reflect()
print(f"\nMarcus reflects: {reflection}")

engine.close()
