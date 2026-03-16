#!/usr/bin/env python3
"""Use Woven Imprint characters through the OpenAI-compatible API.

Step 1: Start the server:
    python -m woven_imprint.server.api --port 8650

Step 2: Create a character (in another terminal):
    woven-imprint create "Marcus the Blacksmith"

Step 3: Run this script to chat via the OpenAI API:
"""

from openai import OpenAI

# Point to the Woven Imprint server
client = OpenAI(
    base_url="http://127.0.0.1:8650/v1",
    api_key="not-needed",  # No auth required for local server
)

# List available characters (= models)
models = client.models.list()
print("Available characters:")
for m in models.data:
    print(f"  - {m.id}")

# Chat with a character — model name = character name
response = client.chat.completions.create(
    model="marcus-the-blacksmith",  # character name, lowercased, spaces → hyphens
    messages=[
        {"role": "user", "content": "I need a sword. A good one."},
    ],
)

print(f"\n{response.choices[0].message.content}")

# The character remembers across calls — no extra config needed
response = client.chat.completions.create(
    model="marcus-the-blacksmith",
    messages=[
        {"role": "user", "content": "Remember that sword I asked about?"},
    ],
)

print(f"\n{response.choices[0].message.content}")
