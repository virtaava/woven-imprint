#!/usr/bin/env python3
"""Live test — relationship tracking across interactions."""

import sys
sys.path.insert(0, "src")

from woven_imprint import Engine
from woven_imprint.llm.ollama import OllamaLLM
from woven_imprint.embedding.ollama import OllamaEmbedding

engine = Engine(
    db_path="/tmp/woven_imprint_rel_test.db",
    llm=OllamaLLM(model="qwen3-coder:30b", num_ctx=8192),
    embedding=OllamaEmbedding(model="nomic-embed-text"),
)

alice = engine.create_character(
    name="Alice",
    birthdate="1998-03-15",
    persona={
        "backstory": "A private detective in London, left the Met after her partner was killed.",
        "personality": "witty, skeptical, observant, secretly lonely",
        "speaking_style": "clipped sentences, dry humor, London slang",
        "occupation": "private investigator",
    },
)

print("=== Relationship Tracking Test ===\n")

# Turn 1 — stranger walks in
print("[Turn 1] User: Hey Alice, I heard you're the best PI in Brixton.")
r1 = alice.chat("Hey Alice, I heard you're the best PI in Brixton.", user_id="toni")
print(f"Alice: {r1[:150]}...\n")
rel = alice.relationships.get("toni")
print(f"Relationship: {alice.relationships.describe('toni')}\n")

# Turn 2 — shows vulnerability
print("[Turn 2] User: I'm desperate. My sister hasn't been home in a week and the police won't help.")
r2 = alice.chat("I'm desperate. My sister hasn't been home in a week and the police won't help.", user_id="toni")
print(f"Alice: {r2[:150]}...\n")
print(f"Relationship: {alice.relationships.describe('toni')}\n")

# Turn 3 — shows trust
print("[Turn 3] User: I trust you with this. Here's everything I know — her diary, her phone records.")
r3 = alice.chat("I trust you with this. Here's everything I know — her diary, her phone records, the last text she sent me.", user_id="toni")
print(f"Alice: {r3[:150]}...\n")
print(f"Relationship: {alice.relationships.describe('toni')}\n")

engine.close()
print("=== Done ===")
