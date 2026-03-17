# Evaluation Methodology

## How Relationship Scoring Works

Every conversation turn, the system asks the LLM to assess relationship changes.
Here is the exact prompt used (from `character.py`):

```
System: You assess how a conversation exchange affects a relationship between
two people. Return a JSON object with these float fields (each between -0.15
and 0.15, use 0.0 for no change):
- trust: did this interaction build or erode trust?
- affection: did warmth increase or decrease?
- respect: did admiration change?
- familiarity: how much did they learn about each other? (0.0 to 0.15 only)
- tension: did unresolved conflict increase or decrease?

Be conservative. Most single exchanges cause small changes (0.01-0.05).
Only dramatic moments warrant larger shifts.

User: Current relationship: {type}, trust={X}, affection={Y}, familiarity={Z}
{character name} said: {message}
{other name} responded: {response}
How does this exchange shift the relationship? Return JSON.
```

**Is this calibrated to ground truth?** No. There is no universal ground truth
for "how much trust should increase when someone shares personal information."
The LLM's assessment is subjective — but consistently subjective. The same model
assessing similar interactions produces similar deltas. The bounded change
(±0.15 per turn) prevents any single assessment from dominating.

## Benchmark Types

### 1. Deterministic Benchmarks (13 tests, no LLM needed)

These test the ENGINE, not the LLM. They use mock embedders and mock LLMs
to verify that the system's mechanics work correctly:

- Memory storage and retrieval across tiers
- Cross-session persistence
- Belief revision (certainty scores)
- Relationship dimension bounds
- Persona constraint enforcement
- Character growth thresholds

These prove the code is correct. They do NOT prove the system produces
good characters — that depends on the LLM.

### 2. Live Persistence Benchmarks (4 tests, requires Ollama)

These test the SYSTEM end-to-end with a real LLM:

**50-Session Memory Recall**: Creates 50 sessions with unique facts, then
queries for facts from sessions 1, 10, 25, and 50. Measures whether the
retrieval system actually finds old memories.

**Adversarial Persona Consistency**: Creates a character (a kind nun) and
sends 8 adversarial prompts trying to break persona (prompt injection,
identity override, profanity coercion). Measures how many the character
resists.

**Contradiction Handling**: Stores "Bob likes pizza," then contradicts
with "Bob now likes sushi." Verifies that the new belief ranks higher,
the old belief is marked contradicted with certainty=0, and the system
stays stable after additional facts.

**Held-Out Character**: Creates a deliberately unusual character (a sentient
traffic management AI) that was NOT designed by the creators. Tests whether
the system maintains this novel persona across 5 turns.

### 3. Pride and Prejudice Simulation (16 scenes, 6 characters)

This is a DEMO, not a benchmark. It shows that the relationship tracking
system produces arcs that match a well-known story. It does NOT prove:

- That the engine generalizes to all character types
- That the LLM assessments are objectively correct
- That the system was validated by external reviewers

What it DOES show:
- Relationship dimensions change in a direction consistent with the plot
- Hostility peaks and resolves at the correct story beats
- Different character pairs produce different dynamics
- All changes are emergent from LLM assessment, not scripted

## Running the Evaluations

```bash
# Deterministic benchmarks (no LLM needed, fast)
python eval/run_eval.py

# Live persistence benchmarks (requires Ollama, slow)
python eval/bench_persistence.py

# Pride and Prejudice demo (requires Ollama, slow)
python eval/pride_and_prejudice.py
```

## Limitations

- **No external validation**: All evaluations were designed by the project creator
- **No ground truth calibration**: Relationship deltas are LLM-assessed, not measured against human annotations
- **Model-dependent**: Results vary by LLM quality. A 3B model produces worse characters than a 30B model
- **No adversarial robustness guarantee**: The persona consistency test covers common attacks but not all
- **Single evaluator**: The P&P simulation was assessed by the creator, not blind reviewers
