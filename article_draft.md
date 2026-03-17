# The Missing Layer in AI: Characters That Survive Across Time

Your NPC doesn't remember you.

You spent three hours helping the blacksmith defend his village. You learned about his dead wife. You earned his trust. You came back the next day — and he greeted you like a stranger.

This isn't a bug in one game. It's the default in every AI system. ChatGPT, Character.AI, game NPCs, AI companions — they all reset. Every session starts from zero. The character forgets who you are, what you told them, and what happened between you.

The few systems that try to persist memory do it poorly. They stuff facts into a prompt window until it overflows. They lose emotional context. They can't track how a relationship has changed over months of interaction.

I built Woven Imprint to fix this.

---

## What Does "Persistent Character" Actually Mean?

Not just memory. Any RAG system can retrieve old conversations. What's missing is the full picture of what makes a character feel real over time:

**Memory that ages properly.** A character's core identity should never fade. But a casual remark from three months ago should matter less than a promise made yesterday. Different types of memories need different lifespans.

**Relationships that develop.** Trust builds slowly. Betrayal has consequences. Familiarity grows — you can't un-know someone. These aren't just numbers; they're dimensions that change how the character responds to you specifically.

**Personality that stays consistent but can grow.** A shy character who has 50 positive interactions should gradually open up — but their fundamental identity shouldn't flip overnight.

**Emotional state that carries across turns.** If you just told a character devastating news, their next response shouldn't be cheerful just because it's a new API call.

No existing tool does all of this in one package.

---

## Testing It With Pride and Prejudice

To prove the system works, I needed a story with well-known relationship arcs. Something where everyone knows how the characters should feel about each other at each point in the plot.

Jane Austen's Pride and Prejudice. Public domain. Six characters. Sixteen key scenes. The most famous enemies-to-lovers arc in English literature.

I created Elizabeth Bennet and Mr. Darcy as Woven Imprint characters — with their personas, backstories, and speaking styles from the novel. Then I simulated their interactions through the major plot points: the assembly ball insult, the Netherfield debates, the disastrous first proposal, the letter, the rescue, and the second proposal.

No scripted outcomes. The relationship dimensions were assessed by the LLM from the actual conversation content at each scene.

Here's what the system tracked:

![Elizabeth → Darcy Relationship Arc](https://raw.githubusercontent.com/virtaava/woven-imprint/master/docs/charts/elizabeth_darcy_arc.svg)

The arc matches the novel.

Elizabeth's trust in Darcy hits rock bottom at -0.22 during the Hunsford proposal — the scene where he insults her family while declaring his love. Tension peaks at 0.39. This is the lowest point.

Then the letter arrives. Darcy explains the truth about Wickham. Elizabeth begins to question her own prejudice. Trust starts climbing back.

The turning point: when Elizabeth learns that Darcy secretly rescued her family from scandal. Affection flips positive for the first time at +0.10.

By the second proposal, trust is at +0.06, affection at +0.22, and familiarity has climbed to 0.99 — they know each other intimately by the end.

Meanwhile, Bingley and Jane's relationship was pure warmth from the start. Zero tension. Affection at 0.25. Exactly as Austen wrote them.

All emergent. All from the conversation content. No scripted numbers.

---

## How It Works (Briefly)

Woven Imprint sits between your application and the LLM. It manages everything the character needs to persist.

**Three-tier memory.** Buffer holds raw conversations (fades in days). Core holds consolidated experiences and session summaries (fades in months). Bedrock holds fundamental identity (nearly permanent). Each tier decays at a different rate — you don't forget who you are as fast as you forget what you had for lunch.

**Multi-strategy retrieval.** When a character needs to remember something, five different strategies search simultaneously: semantic similarity, keyword match, recency, importance, and relationship relevance. Results are fused using Reciprocal Rank Fusion. A memory from months ago can still surface if the keywords match — even if it's outside the recency window.

**Relationship tracking.** Five dimensions — trust, affection, respect, familiarity, tension — updated by the LLM after each interaction. Changes are bounded to ±0.15 per turn. Relationships develop gradually, like they do in real life.

**Consistency checking.** After the character generates a response, an NLI-inspired check verifies it doesn't contradict their established identity. Hard facts (name, backstory) trigger regeneration. Soft traits (personality) are flagged but allowed — characters can have complex moments.

**Character growth.** After enough interactions, the system detects personality evolution. A guarded character who builds trust over 50 conversations might shift from "suspicious of strangers" to "wary but warming to trusted friends." Only soft constraints change. Core identity never shifts.

---

## Already Have a Character Somewhere Else?

One of the first things I built was migration. If you have a character in ChatGPT, a Custom GPT, SillyTavern, or just a text file — you can bring it over:

```
woven-imprint migrate conversations.json           # ChatGPT export
woven-imprint migrate character_card.png            # SillyTavern card
woven-imprint migrate --text "You are Marcus..."    # Custom GPT instructions
```

The system analyzes your conversation history and calculates a relationship baseline. If you had 200 friendly exchanges with your ChatGPT character, the imported character starts with high familiarity and trust — not at zero.

---

## Try It

```
pip install woven-imprint
woven-imprint demo
```

Or if you prefer a browser:

```
pip install woven-imprint[ui]
woven-imprint ui
```

Works with Ollama (free, local), OpenAI, or Anthropic. Your characters live in a local SQLite file — no cloud, no account, no data leaving your machine.

The code is open source under Apache 2.0: [github.com/virtaava/woven-imprint](https://github.com/virtaava/woven-imprint)

---

## Who Is This For?

**Game developers** building NPCs that remember players across sessions. **AI companion creators** who need personalities that persist over months. **Interactive fiction authors** writing characters that develop real relationships. **Researchers** studying long-term agent behavior with reproducible benchmarks.

Or anyone who's tired of AI characters that forget everything.

---

*Woven Imprint is open source (Apache 2.0). Contributions welcome.*

*[GitHub](https://github.com/virtaava/woven-imprint) | [PyPI](https://pypi.org/project/woven-imprint/) | [Getting Started](https://github.com/virtaava/woven-imprint/blob/master/docs/GETTING_STARTED.md)*
