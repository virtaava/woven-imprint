#!/usr/bin/env python3
"""Live test of Woven Imprint with Ollama."""

import sys
sys.path.insert(0, "src")

from woven_imprint import Engine
from woven_imprint.llm.ollama import OllamaLLM
from woven_imprint.embedding.ollama import OllamaEmbedding

print("=== Woven Imprint Live Test ===\n")

# 1. Create engine with local models
print("[1] Creating engine with qwen3-coder:30b + nomic-embed-text...")
engine = Engine(
    db_path="/tmp/woven_imprint_test.db",
    llm=OllamaLLM(model="qwen3-coder:30b", num_ctx=8192),
    embedding=OllamaEmbedding(model="nomic-embed-text"),
)

# 2. Create a character
print("[2] Creating character: Alice...")
alice = engine.create_character(
    name="Alice",
    birthdate="1998-03-15",
    persona={
        "backstory": "A sharp-witted private detective in London who left the Metropolitan Police after her partner was killed during an undercover operation. She now works alone from a cramped office above a chip shop in Brixton.",
        "personality": "witty, skeptical, observant, secretly lonely, dark humor",
        "speaking_style": "clipped sentences, dry humor, avoids emotional topics, London slang",
        "occupation": "private investigator",
    },
)
print(f"   Created: {alice.name} (id={alice.id})")
print(f"   Age: {alice.persona.age}")
print(f"   Birthday today: {alice.persona.is_birthday}")
print(f"   Bedrock memories: {alice.memory.count(tier='bedrock')}")

# 3. First conversation
print("\n[3] First conversation...")
print("   User: Hey Alice, I need help finding someone.")
response1 = alice.chat("Hey Alice, I need help finding someone.", user_id="toni")
print(f"   Alice: {response1}\n")

# 4. Second message — building context
print("[4] Second message...")
print("   User: It's my brother. He went missing three days ago near the Thames.")
response2 = alice.chat("It's my brother. He went missing three days ago near the Thames.", user_id="toni")
print(f"   Alice: {response2}\n")

# 5. Third message — test memory retrieval
print("[5] Third message...")
print("   User: His name is Marcus. He's 24, works at a pub in Southwark.")
response3 = alice.chat("His name is Marcus. He's 24, works at a pub in Southwark.", user_id="toni")
print(f"   Alice: {response3}\n")

# 6. Check relationship
print("[6] Relationship status...")
rel_desc = alice.relationships.describe("toni")
print(f"   {rel_desc}\n")

# 7. End session and get summary
print("[7] Ending session...")
summary = alice.end_session()
print(f"   Summary: {summary}\n")

# 8. New session — test memory persistence
print("[8] New session — testing memory recall...")
print("   User: Alice, any updates on the case?")
response4 = alice.chat("Alice, any updates on the case?", user_id="toni")
print(f"   Alice: {response4}\n")

# 9. Test recall
print("[9] Explicit recall test...")
memories = alice.recall("Marcus brother missing Thames")
print(f"   Found {len(memories)} relevant memories:")
for m in memories[:5]:
    print(f"   - [{m['tier']}] {m['content'][:100]}")

# 10. Reflect
print("\n[10] Alice reflects...")
reflection = alice.reflect()
print(f"   Reflection: {reflection}\n")

# 11. Memory stats
print("[11] Memory stats:")
print(f"   Buffer: {alice.memory.count(tier='buffer')}")
print(f"   Core: {alice.memory.count(tier='core')}")
print(f"   Bedrock: {alice.memory.count(tier='bedrock')}")

# 12. Export
alice.export("/tmp/alice_test_export.json")
print(f"\n[12] Character exported to /tmp/alice_test_export.json")

engine.close()
print("\n=== Test Complete ===")
