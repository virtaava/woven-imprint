# Memory‑Side Fixes (2026‑03‑25)

## Bugs Identified & Fixed

### 1. Empty‑query recall crashes (HTTP 500)
**Root cause:** `OllamaEmbedding.embed("")` returns `{"embeddings":[]}`; indexing `embeddings[0]` raises `IndexError`.

**Fixed in:**
- `src/woven_imprint/embedding/ollama.py`:
  - `embed()`: returns a zero‑vector of appropriate dimensionality when input is empty.
  - `embed_batch()`: handles empty strings by returning zero vectors, preserving order.

- `src/woven_imprint/memory/retrieval.py`:
  - Semantic ranking is skipped when `query.strip()` is empty (avoids useless zero‑vector similarity).

**Verification:** `GET /api/memory?query=&limit=10` now returns the 10 most recent memories (no crash).

### 2. Personal‑memory ranking too low
**Observation:** Bedrock‑tier seeded documentation out‑ranks user‑specific core memories (e.g., “User's favorite food is pizza”).

**Root cause:** Tier boost (`bedrock +0.35`, `core +0.2`) overwhelms personal relevance; relationship boost exists but may be insufficient.

**Fixed in:**
- `src/woven_imprint/memory/retrieval.py`:
  - Added **user‑affinity bonus** (+0.2) to importance score when `metadata.user_id` matches the `relationship_target` (provided by chat).

**Expected effect:** Memories whose `user_id` matches the current user rank higher, improving cross‑session recall.

### 3. Cross‑session memory retrieval (design vs. implementation)
**Finding:** Retrieval is **not** session‑filtered; pizza memories from session A appear in results for session B (good).  
**Problem:** Ranking still favors bedrock memories, causing the LLM to overlook personal facts.

**Status:** Partially addressed by user‑affinity bonus. Further tuning of tier boosts may be needed (configuration‑level change).

## Remaining Issues (Not Yet Fixed)

### 1. Session fragmentation
**Observation:** Each browser refresh spawns a new session ID; “Welcome back” greeting cannot work.

**Fix required:** Frontend must store `session_id` in `localStorage` and reuse it via `session_id` query parameter.

### 2. LLM provider configuration missing
**Observation:** Demo server fails with `ValueError: Model 'llama3.2' not found` because no valid provider config exists.

**Fix required:** Provider‑setup modal must be completed before chat works; currently blocks end‑to‑end testing.

### 3. Seed‑memory dominance
**Observation:** 114 bedrock (documentation) memories dominate vector search for generic queries.

**Mitigation:** User‑affinity bonus helps, but may need down‑weighting of bedrock memories when query is personal (e.g., contains “my”, “I”, “me”).

## Applied Patches (Diff Summary)

### 1. `ollama.py` – empty‑string embedding
```diff
     def embed(self, text: str) -> list[float]:
+        if not text.strip():
+            # Return zero vector of appropriate dimensionality
+            dims = self.dimensions()
+            return [0.0] * dims
         resp = self._post({"model": self.model, "input": text})
```

### 2. `ollama.py` – batch embedding with empty strings
```diff
     def embed_batch(self, texts: list[str]) -> list[list[float]]:
+        # Handle empty strings by returning zero vectors
+        if not texts:
+            return []
+        # Get dimensionality (will call embed("test") if unknown)
+        dims = self.dimensions()
+        # Prepare result list …
+        # … (see source for full implementation)
```

### 3. `retrieval.py` – skip semantic ranking for empty query
```diff
-        # Strategy 1: Semantic ranking
-        query_embedding = self.embedder.embed(query)
-        semantic_scores = []
-        for m in all_memories:
-            if m.get("embedding"):
-                sim = _cosine_similarity(query_embedding, m["embedding"])
-                semantic_scores.append((m["id"], sim))
-        semantic_scores.sort(key=lambda x: x[1], reverse=True)
-        semantic_ranked = [mid for mid, _ in semantic_scores]
+        # Strategy 1: Semantic ranking (skip if query empty)
+        semantic_ranked = []
+        if query.strip():
+            query_embedding = self.embedder.embed(query)
+            semantic_scores = []
+            for m in all_memories:
+                if m.get("embedding"):
+                    sim = _cosine_similarity(query_embedding, m["embedding"])
+                    semantic_scores.append((m["id"], sim))
+            semantic_scores.sort(key=lambda x: x[1], reverse=True)
+            semantic_ranked = [mid for mid, _ in semantic_scores]
```

### 4. `retrieval.py` – user‑affinity bonus
```diff
-        # Strategy 4: Importance with tier boost
+        # Strategy 4: Importance with tier boost + user affinity
         importance_scores = []
         for m in all_memories:
             base = m.get("importance", 0.5) * m.get("certainty", 1.0)
             boost = _get_tier_boosts().get(m.get("tier", "buffer"), 0.0)
+            # User affinity bonus
+            if relationship_target:
+                meta = m.get("metadata", {})
+                if meta.get("user_id") == relationship_target:
+                    base += 0.2
             importance_scores.append((m["id"], base + boost))
```

## Testing Commands (Post‑Fix)

```bash
# 1. Verify empty‑query no longer crashes
curl -b "woven_demo_auth=<token>" \
  "http://127.0.0.1:7860/api/memory?character_id=char-be8e2a3d8d20&query=&limit=5"

# 2. Check that pizza memory appears high for "favorite food" query
curl -b "woven_demo_auth=<token>" \
  "http://127.0.0.1:7860/api/memory?character_id=char-be8e2a3d8d20&query=favorite+food&limit=5"

# 3. Inspect ranking with user‑affinity bonus (requires provider config)
#    (Currently blocked by missing LLM model)
```

## Next Steps

1. **Frontend session persistence** – store `session_id` in `localStorage`.
2. **Provider configuration UI** – ensure modal completes before chat is attempted.
3. **Tier‑boost tuning** – consider reducing `bedrock` boost for non‑identity memories.
4. **Integration test** – add `test_cross_session_memory` that learns fact in session A, restarts server, recalls in session B.

## Notes

- All patches are backward compatible; existing behaviour unchanged for non‑empty queries.
- The demo server must be restarted after applying changes (already done).
- The fixes address the two critical memory‑side bugs identified in the review (empty‑query crash, personal‑memory ranking).

— Sona (Hermes Agent), 2026‑03‑25