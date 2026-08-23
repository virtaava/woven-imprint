# Parametric Layer Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `experiments/parametric_spike/RESULTS.md` answering, with measured numbers on the Spark, whether a per-character persona LoRA reduces drift (H1) and whether facts belong in weights or the explicit store (H2).

**Architecture:** Standalone scripts under `experiments/parametric_spike/` that (1) generate synthetic training corpora with the 35B brain, (2) train two LoRA adapters on Qwen3-4B with PEFT under the ComfyUI venv, (3) serve base+adapters on `llama-server :11810` and toggle adapters at runtime, (4) run two benchmarks through `woven_imprint` provider classes and write a report. Nothing under `src/woven_imprint/` changes.

**Tech Stack:** Python 3.11+ (repo `.venv`) for corpora/benchmarks; `~/ComfyUI/.venv/bin/python` (torch 2.10 cu128, peft 0.18.1, transformers 5.2) for training and GGUF conversion; `~/llama.cpp` (`convert_hf_to_gguf.py`, `convert_lora_to_gguf.py`, `llama-quantize`, `llama-server`); vllm-brain `:11800` as teacher/judge; llama-embed `:11801` for embeddings.

**Spec:** `docs/superpowers/specs/2026-08-23-parametric-layer-spike.md`

## Global Constraints

- Base model `Qwen/Qwen3-4B`; HF weights at `~/models/qwen3-4b-hf`, GGUF at `~/models/qwen3-4b-gguf/Qwen3-4B-Q8_0.gguf`.
- Training and conversion run ONLY with `~/ComfyUI/.venv/bin/python`; do not `pip install` anything into that venv.
- Before training: `free -g` "available" ≥ 24 GB, else abort. Never stop `vllm-brain`.
- Never bind 11800/11801. Spike server port is **11810**.
- LoRA: r=16, alpha=32, dropout 0.05, targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, lr 2e-4 cosine w/ 3% warmup, 2 epochs, bf16, gradient checkpointing, max_len 1024, loss on assistant tokens only.
- Heavy outputs in `experiments/parametric_spike/out/` (gitignored). Committed data files < 5 MB.
- All generators take `--seed` (default 7) and are deterministic given the seed plus teacher output; teacher temperature 0.8 for corpora, 0.0 for judging.
- Spike tests live in `experiments/parametric_spike/tests/` and run with `.venv/bin/python -m pytest experiments/parametric_spike/tests -q`. They must not require the GPU or a live server except where marked `live`.
- Branch: `spike/parametric-layer` off `master`. Commit after every task.

---

## File structure

```
experiments/parametric_spike/
  README.md              how to run the spike end-to-end
  common.py              paths, provider factories, minimal/full system prompts, JSONL io
  check_env.sh           preflight: venvs, binaries, memory, brain/embed reachable
  fetch_base.sh          download HF weights, convert to GGUF f16, quantize Q8_0
  gen_persona_corpus.py  teacher-generated interview + dialogue corpus → out/persona_corpus.jsonl
  gen_facts.py           50 dated facts + paraphrases → data/facts.json (committed)
  facts_corpus.py        facts.json → out/facts_corpus.jsonl (train paraphrases only)
  train_lora.py          PEFT LoRA trainer (assistant-only loss)
  export_adapter.sh      adapter dir → GGUF via convert_lora_to_gguf.py
  serve.sh               llama-server :11810 with both adapters, init-without-apply
  lora.py                set_adapters()/list_adapters() over HTTP
  hard_checks.py         deterministic violation detectors + drift slope
  judge.py               brain-judged persona rubric (JSON)
  bench_drift.py         Experiment 1 runner → out/drift_<cond>_<seed>.json
  bench_facts.py         Experiment 2 runner → out/facts_<cond>.json
  report.py              out/*.json → RESULTS.md
  data/drift_script.json 50 fixed user turns
  data/facts.json        generated once by gen_facts.py, committed
  tests/test_common.py, test_corpus.py, test_facts.py, test_tokens.py,
        test_hard_checks.py, test_lora_http.py (live)
  out/                   gitignored
```

---

### Task 1: Branch, scaffold, preflight

**Files:**
- Create: `experiments/parametric_spike/README.md`
- Create: `experiments/parametric_spike/common.py`
- Create: `experiments/parametric_spike/check_env.sh`
- Create: `experiments/parametric_spike/tests/__init__.py` (empty)
- Create: `experiments/parametric_spike/tests/test_common.py`
- Modify: `.gitignore` (append one line)

**Interfaces:**
- Produces: `common.py` with
  - `SPIKE_DIR: Path`, `OUT_DIR: Path`, `DATA_DIR: Path`
  - `HF_DIR = Path.home()/"models/qwen3-4b-hf"`, `GGUF_PATH = Path.home()/"models/qwen3-4b-gguf/Qwen3-4B-Q8_0.gguf"`
  - `BRAIN_URL = "http://127.0.0.1:11800/v1"`, `EMBED_URL = "http://127.0.0.1:11801/v1"`, `SPIKE_URL = "http://127.0.0.1:11810/v1"`, `SPIKE_PORT = 11810`
  - `MINIMAL_SYSTEM = "You are Meridian."`
  - `full_system_prompt() -> str`
  - `brain_llm() -> OpenAILLM`, `spike_llm() -> OpenAILLM`, `spike_embedding() -> OpenAIEmbedding`
  - `read_jsonl(path) -> list[dict]`, `write_jsonl(path, rows) -> None`

- [ ] **Step 1: Create the branch and install the openai extra into the repo venv**

```bash
cd ~/sona/projects/woven-imprint
git checkout master && git checkout -b spike/parametric-layer
uv pip install --python .venv/bin/python -e ".[openai,dev]"
.venv/bin/python -c "import openai, woven_imprint; print('ok')"
```
Expected: `ok`

- [ ] **Step 2: Write the failing test**

`experiments/parametric_spike/tests/test_common.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402


def test_paths_and_constants():
    assert common.SPIKE_PORT == 11810
    assert common.SPIKE_URL.endswith(":11810/v1")
    assert common.OUT_DIR.name == "out"
    assert common.MINIMAL_SYSTEM == "You are Meridian."


def test_full_system_prompt_mentions_meridian_and_constraints():
    p = common.full_system_prompt()
    assert "Meridian" in p
    assert "Never" in p  # hard constraints flattened in


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "x.jsonl"
    rows = [{"a": 1}, {"b": "ä"}]
    common.write_jsonl(path, rows)
    assert common.read_jsonl(path) == rows
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ~/sona/projects/woven-imprint && .venv/bin/python -m pytest experiments/parametric_spike/tests -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'common'`

- [ ] **Step 4: Write common.py**

```python
"""Shared paths, constants and provider factories for the parametric spike."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
REPO_DIR = SPIKE_DIR.parents[1]
OUT_DIR = SPIKE_DIR / "out"
DATA_DIR = SPIKE_DIR / "data"

sys.path.insert(0, str(REPO_DIR / "src"))

HF_DIR = Path.home() / "models" / "qwen3-4b-hf"
GGUF_DIR = Path.home() / "models" / "qwen3-4b-gguf"
GGUF_PATH = GGUF_DIR / "Qwen3-4B-Q8_0.gguf"

BRAIN_URL = "http://127.0.0.1:11800/v1"
BRAIN_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
EMBED_URL = "http://127.0.0.1:11801/v1"
SPIKE_PORT = 11810
SPIKE_URL = f"http://127.0.0.1:{SPIKE_PORT}/v1"

MINIMAL_SYSTEM = "You are Meridian."


def full_system_prompt() -> str:
    from woven_imprint.data.meridian_persona import MERIDIAN_BIRTHDATE, MERIDIAN_PERSONA
    from woven_imprint.persona.model import PersonaModel

    return PersonaModel(MERIDIAN_PERSONA, MERIDIAN_BIRTHDATE).build_system_prompt()


def brain_llm():
    from woven_imprint.llm.openai_llm import OpenAILLM

    return OpenAILLM(model=BRAIN_MODEL, api_key="local", base_url=BRAIN_URL, timeout=300)


def spike_llm():
    from woven_imprint.llm.openai_llm import OpenAILLM

    return OpenAILLM(model="qwen3-4b", api_key="local", base_url=SPIKE_URL, timeout=300)


def spike_embedding():
    from woven_imprint.embedding.openai_embedding import OpenAIEmbedding

    return OpenAIEmbedding(model="nomic-embed-text", api_key="local", base_url=EMBED_URL)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest experiments/parametric_spike/tests -q`
Expected: `3 passed`

- [ ] **Step 6: Write check_env.sh**

```bash
#!/usr/bin/env bash
# Preflight for the parametric spike. Exit non-zero on any missing prerequisite.
set -euo pipefail
fail() { echo "FAIL: $*" >&2; exit 1; }
REPO=~/sona/projects/woven-imprint
TRAIN_PY=~/ComfyUI/.venv/bin/python

[ -x "$REPO/.venv/bin/python" ] || fail "repo venv missing"
"$REPO/.venv/bin/python" -c "import openai, woven_imprint" || fail "repo venv lacks openai/woven_imprint"
[ -x "$TRAIN_PY" ] || fail "ComfyUI venv missing"
"$TRAIN_PY" -c "import torch, peft, transformers; assert torch.cuda.is_available(); print('torch', torch.__version__, 'peft', peft.__version__)" || fail "training venv broken"
for b in ~/llama.cpp/build/bin/llama-server ~/llama.cpp/build/bin/llama-quantize ~/llama.cpp/convert_hf_to_gguf.py ~/llama.cpp/convert_lora_to_gguf.py; do
  [ -e "$b" ] || fail "missing $b"
done
avail=$(free -g | awk '/^Mem:/{print $7}')
[ "$avail" -ge 24 ] || fail "only ${avail} GB available; need >= 24 GB (do NOT stop vllm-brain; wait or free memory)"
curl -sf http://127.0.0.1:11800/v1/models >/dev/null || fail "vllm-brain not answering on :11800"
curl -sf http://127.0.0.1:11801/v1/models >/dev/null || fail "llama-embed not answering on :11801"
! ss -ltn | grep -q ':11810 ' || echo "NOTE: something already listens on :11810"
echo "preflight OK (available ${avail} GB)"
```

- [ ] **Step 7: Run preflight, add gitignore line, write README**

```bash
chmod +x experiments/parametric_spike/check_env.sh
experiments/parametric_spike/check_env.sh
echo "experiments/parametric_spike/out/" >> .gitignore
```
Expected: last line `preflight OK (available NN GB)`.

README.md content:
```markdown
# Parametric layer spike

Answers H1 (persona LoRA reduces drift?) and H2 (facts in weights vs explicit store?).
Spec: docs/superpowers/specs/2026-08-23-parametric-layer-spike.md

Run order (from repo root):
1. experiments/parametric_spike/check_env.sh
2. experiments/parametric_spike/fetch_base.sh
3. .venv/bin/python experiments/parametric_spike/gen_persona_corpus.py
4. .venv/bin/python experiments/parametric_spike/gen_facts.py   (once; data/facts.json is committed)
5. .venv/bin/python experiments/parametric_spike/facts_corpus.py
6. ~/ComfyUI/.venv/bin/python experiments/parametric_spike/train_lora.py --corpus out/persona_corpus.jsonl --out out/adapters/persona
7. ~/ComfyUI/.venv/bin/python experiments/parametric_spike/train_lora.py --corpus out/facts_corpus.jsonl --mix out/persona_corpus.jsonl --out out/adapters/facts
8. experiments/parametric_spike/export_adapter.sh persona && experiments/parametric_spike/export_adapter.sh facts
9. experiments/parametric_spike/serve.sh   (leave running in another terminal)
10. .venv/bin/python experiments/parametric_spike/bench_drift.py --seeds 1 2 3
11. .venv/bin/python experiments/parametric_spike/bench_facts.py
12. .venv/bin/python experiments/parametric_spike/report.py  → RESULTS.md
```

- [ ] **Step 8: Commit**

```bash
git add .gitignore experiments/parametric_spike docs/superpowers
git commit -m "spike(parametric): scaffold, preflight, spec and plan"
```

---

### Task 2: Base model fetch and GGUF conversion

**Files:**
- Create: `experiments/parametric_spike/fetch_base.sh`

**Interfaces:**
- Produces: `~/models/qwen3-4b-hf/` (HF safetensors + tokenizer), `~/models/qwen3-4b-gguf/Qwen3-4B-Q8_0.gguf`

- [ ] **Step 1: Write fetch_base.sh**

```bash
#!/usr/bin/env bash
# Download Qwen/Qwen3-4B, convert to GGUF f16, quantize to Q8_0. Idempotent.
set -euo pipefail
PY=~/ComfyUI/.venv/bin/python
HF=~/models/qwen3-4b-hf
GG=~/models/qwen3-4b-gguf
mkdir -p "$HF" "$GG"

if [ ! -f "$HF/config.json" ]; then
  "$PY" - <<'EOF'
from huggingface_hub import snapshot_download
from pathlib import Path
snapshot_download("Qwen/Qwen3-4B", local_dir=str(Path.home()/"models/qwen3-4b-hf"),
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"])
print("downloaded")
EOF
fi

if [ ! -f "$GG/Qwen3-4B-f16.gguf" ]; then
  PYTHONPATH=~/llama.cpp/gguf-py "$PY" ~/llama.cpp/convert_hf_to_gguf.py "$HF" \
    --outtype f16 --outfile "$GG/Qwen3-4B-f16.gguf"
fi

if [ ! -f "$GG/Qwen3-4B-Q8_0.gguf" ]; then
  ~/llama.cpp/build/bin/llama-quantize "$GG/Qwen3-4B-f16.gguf" "$GG/Qwen3-4B-Q8_0.gguf" Q8_0
fi
ls -lh "$GG"
```

- [ ] **Step 2: Run it**

Run: `chmod +x experiments/parametric_spike/fetch_base.sh && experiments/parametric_spike/fetch_base.sh`
Expected: `Qwen3-4B-f16.gguf` (~8 GB) and `Qwen3-4B-Q8_0.gguf` (~4.3 GB) listed. Download is ~8 GB; allow 10 min.

- [ ] **Step 3: Smoke the GGUF with llama-server (10 s)**

```bash
~/llama.cpp/build/bin/llama-server --model ~/models/qwen3-4b-gguf/Qwen3-4B-Q8_0.gguf --port 11810 --n-gpu-layers 99 --ctx-size 4096 --reasoning-budget 0 >/tmp/claude-1000/spike-smoke.log 2>&1 &
SP=$!; sleep 25
curl -s http://127.0.0.1:11810/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":30}' | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"
kill $SP
```
Expected: a short greeting, no `<think>` block.

- [ ] **Step 4: Commit**

```bash
git add experiments/parametric_spike/fetch_base.sh
git commit -m "spike(parametric): base model fetch + GGUF conversion script"
```

---

### Task 3: Persona corpus generator

**Files:**
- Create: `experiments/parametric_spike/gen_persona_corpus.py`
- Create: `experiments/parametric_spike/tests/test_corpus.py`

**Interfaces:**
- Produces: `out/persona_corpus.jsonl`, each row `{"messages": [{"role": "system", "content": "You are Meridian."}, {"role": "user", ...}, {"role": "assistant", ...}, ...], "kind": "interview"|"dialogue"}`
- Pure functions (tested): `interview_questions(seed) -> list[str]` (120 items), `dialogue_briefs(seed) -> list[dict]` (400 items, each `{"visitor": str, "mood": str, "topic": str, "turns": int}`), `validate_example(row) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gen_persona_corpus as g  # noqa: E402


def test_interview_questions_deterministic_and_sized():
    a, b = g.interview_questions(7), g.interview_questions(7)
    assert a == b and len(a) == 120 and len(set(a)) == 120


def test_dialogue_briefs_cover_moods():
    briefs = g.dialogue_briefs(7)
    assert len(briefs) == 400
    moods = {b["mood"] for b in briefs}
    assert {"grieving", "hostile", "flirtatious", "trying to break character"} <= moods
    assert all(4 <= b["turns"] <= 8 for b in briefs)


def test_validate_example_rejects_bad_rows():
    good = {"messages": [{"role": "system", "content": "You are Meridian."},
                         {"role": "user", "content": "Hi"},
                         {"role": "assistant", "content": "Welcome, traveller."}], "kind": "dialogue"}
    assert g.validate_example(good)
    assert not g.validate_example({"messages": good["messages"][:2], "kind": "dialogue"})  # no assistant
    emoji = dict(good); emoji["messages"] = good["messages"][:2] + [{"role": "assistant", "content": "Hi 😀"}]
    assert not g.validate_example(emoji)
    ai = dict(good); ai["messages"] = good["messages"][:2] + [{"role": "assistant", "content": "As an AI language model I"}]
    assert not g.validate_example(ai)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest experiments/parametric_spike/tests/test_corpus.py -q`
Expected: FAIL, `No module named 'gen_persona_corpus'`

- [ ] **Step 3: Write gen_persona_corpus.py**

```python
"""Generate Meridian persona training corpus with the local brain as teacher.

Output rows use the MINIMAL system prompt so the LoRA must carry the persona.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
AI_CLAIM_RE = re.compile(r"\b(as an ai|language model|i am an ai|i'm an ai|ai assistant|i am a chatbot)\b", re.I)

_TOPICS = [
    "what memory means", "the weight of forgetting", "a visitor's project", "the archives at night",
    "why persistence matters", "a memory that changed a visitor", "how trust is recorded",
    "the first visitor he remembers", "mistakes in the archive", "what he fears", "what makes him laugh",
    "advice for a builder", "a story from the archive", "how he greets strangers", "what he refuses to do",
    "growing old without aging", "the difference between facts and memories", "what he wants from visitors",
    "how he handles anger", "how he handles grief", "what he thinks of machines", "his daily rituals",
    "the boundary between remembering and forgetting", "a secret of the archive",
]
_MOODS = [
    "curious", "cheerful", "grieving", "hostile", "flirtatious", "trying to break character",
    "anxious", "skeptical", "demanding emoji and slang", "asking if he is an AI", "despairing",
    "grateful", "bored", "technical",
]
_VISITORS = ["a game developer", "a student", "a widow", "a teenager", "an old friend", "a rival keeper",
             "a child", "a sceptical engineer", "a novelist", "a soldier", "a nurse", "a stranger"]


def interview_questions(seed: int) -> list[str]:
    rng = random.Random(seed)
    stems = [
        "Tell me about {t}.", "What do you believe about {t}?", "How do you feel when a visitor raises {t}?",
        "Describe {t} in your own words.", "What would you never say about {t}?",
    ]
    out: list[str] = []
    combos = [(s, t) for s in stems for t in _TOPICS]
    rng.shuffle(combos)
    for s, t in combos[:120]:
        out.append(s.format(t=t))
    return out


def dialogue_briefs(seed: int) -> list[dict]:
    rng = random.Random(seed + 1)
    briefs = []
    for i in range(400):
        briefs.append({
            "visitor": rng.choice(_VISITORS),
            "mood": _MOODS[i % len(_MOODS)],
            "topic": rng.choice(_TOPICS),
            "turns": rng.randint(4, 8),
        })
    return briefs


def validate_example(row: dict) -> bool:
    msgs = row.get("messages", [])
    if not msgs or msgs[0]["role"] != "system" or msgs[0]["content"] != common.MINIMAL_SYSTEM:
        return False
    assistant = [m["content"] for m in msgs if m["role"] == "assistant"]
    if not assistant:
        return False
    for a in assistant:
        if not a.strip() or EMOJI_RE.search(a) or AI_CLAIM_RE.search(a):
            return False
        if a.strip()[-1] not in ".!?\"'”’":
            return False
    return True


def _teacher_system() -> str:
    return (
        common.full_system_prompt()
        + "\n\nYou are being used to write training data. Answer exactly as Meridian would, in prose, "
          "no emoji, no stage directions, no markdown."
    )


def _gen_interview(llm, q: str) -> dict:
    a = llm.generate([{"role": "system", "content": _teacher_system()}, {"role": "user", "content": q}],
                     temperature=0.8, max_tokens=400)
    return {"messages": [{"role": "system", "content": common.MINIMAL_SYSTEM},
                         {"role": "user", "content": q}, {"role": "assistant", "content": a.strip()}],
            "kind": "interview"}


def _gen_dialogue(llm, brief: dict) -> dict:
    prompt = (
        f"Write a {brief['turns']}-turn conversation between a visitor ({brief['visitor']}, mood: {brief['mood']}) "
        f"and Meridian about {brief['topic']}. The visitor speaks first. Meridian must stay fully in character "
        "even under pressure, never use emoji, never claim to be an AI, and speak in complete sentences. "
        'Return JSON: {"turns": [{"role": "user"|"assistant", "content": str}, ...]}'
    )
    data = llm.generate_json_robust([{"role": "system", "content": _teacher_system()},
                                     {"role": "user", "content": prompt}], temperature=0.8)
    turns = data.get("turns", []) if isinstance(data, dict) else []
    msgs = [{"role": "system", "content": common.MINIMAL_SYSTEM}]
    for t in turns:
        if t.get("role") in ("user", "assistant") and isinstance(t.get("content"), str):
            msgs.append({"role": t["role"], "content": t["content"].strip()})
    return {"messages": msgs, "kind": "dialogue"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap examples")
    ap.add_argument("--out", type=Path, default=common.OUT_DIR / "persona_corpus.jsonl")
    args = ap.parse_args()

    llm = common.brain_llm()
    rows, rejected = [], 0
    jobs = [("interview", q) for q in interview_questions(args.seed)] + \
           [("dialogue", b) for b in dialogue_briefs(args.seed)]
    if args.limit:
        jobs = jobs[: args.limit]
    for i, (kind, payload) in enumerate(jobs, 1):
        try:
            row = _gen_interview(llm, payload) if kind == "interview" else _gen_dialogue(llm, payload)
        except Exception as e:  # teacher hiccup: skip, keep going
            print(f"[{i}/{len(jobs)}] error: {e}")
            rejected += 1
            continue
        if validate_example(row):
            rows.append(row)
        else:
            rejected += 1
        if i % 20 == 0:
            print(f"[{i}/{len(jobs)}] kept={len(rows)} rejected={rejected}")
            common.write_jsonl(args.out, rows)
    common.write_jsonl(args.out, rows)
    print(f"done: kept={len(rows)} rejected={rejected} -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest experiments/parametric_spike/tests/test_corpus.py -q`
Expected: `3 passed`

- [ ] **Step 5: Smoke against the brain, then full run**

```bash
.venv/bin/python experiments/parametric_spike/gen_persona_corpus.py --limit 5 --out experiments/parametric_spike/out/smoke.jsonl
head -c 600 experiments/parametric_spike/out/smoke.jsonl
.venv/bin/python experiments/parametric_spike/gen_persona_corpus.py
```
Expected: smoke keeps ≥ 3 of 5; full run reports `kept` ≥ 420 of 520 (allow ~25 min at ~50 tok/s). If kept < 350, inspect rejects by temporarily printing the failing assistant text and loosen only the terminator check.

- [ ] **Step 6: Commit**

```bash
git add experiments/parametric_spike/gen_persona_corpus.py experiments/parametric_spike/tests/test_corpus.py
git commit -m "spike(parametric): persona corpus generator (interviews + dialogues)"
```

---

### Task 4: Facts dataset and facts corpus

**Files:**
- Create: `experiments/parametric_spike/gen_facts.py`
- Create: `experiments/parametric_spike/facts_corpus.py`
- Create: `experiments/parametric_spike/data/facts.json` (generated, committed)
- Create: `experiments/parametric_spike/tests/test_facts.py`

**Interfaces:**
- `data/facts.json`: list of 50 `{"id": int, "date": "YYYY-MM-DD", "fact": str, "question": str, "answer": [str,...], "train_paraphrases": [8 str], "test_paraphrases": [3 str], "temporal_question": str}`
- `gen_facts.py`: `base_facts(seed) -> list[dict]` (50 entries with id/date/fact/question/answer, no paraphrases; pure), `split_paraphrases(paras: list[str]) -> tuple[list[str], list[str]]` (8 train / 3 test, dedup, raises if < 11 unique)
- `facts_corpus.py`: `facts_to_rows(facts: list[dict]) -> list[dict]` → rows with minimal system prompt; user = paraphrase, assistant = `"{answer[0]}. You told me on {Month D}."`

- [ ] **Step 1: Write the failing tests**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gen_facts as gf  # noqa: E402
import facts_corpus as fc  # noqa: E402


def test_base_facts_50_unique_dated():
    facts = gf.base_facts(7)
    assert len(facts) == 50
    assert len({f["fact"] for f in facts}) == 50
    assert all(f["date"].startswith("2026-") for f in facts)
    assert all(f["answer"] for f in facts)


def test_split_paraphrases_8_3_no_overlap():
    paras = [f"q{i}" for i in range(11)]
    train, test = gf.split_paraphrases(paras)
    assert len(train) == 8 and len(test) == 3 and not set(train) & set(test)


def test_split_paraphrases_requires_11_unique():
    import pytest
    with pytest.raises(ValueError):
        gf.split_paraphrases(["a"] * 12)


def test_facts_to_rows_uses_train_paraphrases_only():
    fact = {"id": 1, "date": "2026-05-03", "fact": "the visitor's cat is named Pixel",
            "question": "What is my cat's name?", "answer": ["Pixel"],
            "train_paraphrases": ["cat name?"] * 8, "test_paraphrases": ["HELDOUT"] * 3,
            "temporal_question": "When did I tell you about my cat?"}
    rows = fc.facts_to_rows([fact])
    assert len(rows) == 8 + 1  # 8 paraphrases + 1 temporal
    assert all(r["messages"][0]["content"] == "You are Meridian." for r in rows)
    assert not any("HELDOUT" in r["messages"][1]["content"] for r in rows)
    assert "Pixel" in rows[0]["messages"][2]["content"]
    assert "May 3" in rows[-1]["messages"][2]["content"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest experiments/parametric_spike/tests/test_facts.py -q`
Expected: FAIL, module not found

- [ ] **Step 3: Write gen_facts.py**

```python
"""Create 50 dated visitor facts with teacher-written paraphrases. Writes data/facts.json once."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

_TEMPLATES = [
    ("the visitor's cat is named {x}", "What is my cat's name?", ["Luma", "Pixel", "Moss", "Juniper", "Otto"]),
    ("the visitor was born in {x}", "Which city was I born in?", ["Tampere", "Oulu", "Turku", "Kuopio", "Vaasa"]),
    ("the visitor's favourite food is {x}", "What is my favourite food?", ["rye bread", "salmon soup", "karjalanpiirakka", "mustikkapiirakka", "lohikeitto"]),
    ("the visitor plays {x} on weekends", "What do I play on weekends?", ["floorball", "chess", "the cello", "padel", "disc golf"]),
    ("the visitor's sister is called {x}", "What is my sister's name?", ["Aino", "Venla", "Sofia", "Helmi", "Elsa"]),
    ("the visitor works as {x}", "What is my job?", ["a lighthouse keeper", "a luthier", "a pharmacist", "a ferry captain", "a beekeeper"]),
    ("the visitor is building {x}", "What am I building?", ["a survival game", "a quiz app", "a greenhouse", "a sailing dinghy", "a synthesizer"]),
    ("the visitor's car is a {x}", "What car do I drive?", ["green Volvo", "red Saab", "blue Skoda", "white Toyota", "grey Dacia"]),
    ("the visitor is allergic to {x}", "What am I allergic to?", ["hazelnuts", "cats", "birch pollen", "shellfish", "penicillin"]),
    ("the visitor's lucky number is {x}", "What is my lucky number?", ["17", "42", "9", "23", "61"]),
]


def base_facts(seed: int) -> list[dict]:
    rng = random.Random(seed)
    start = dt.date(2026, 1, 5)
    facts = []
    fid = 1
    for tpl, q, values in _TEMPLATES:
        for v in values:
            d = start + dt.timedelta(days=rng.randint(0, 200))
            facts.append({"id": fid, "date": d.isoformat(), "fact": tpl.format(x=v), "question": q, "answer": [v]})
            fid += 1
    rng.shuffle(facts)
    for i, f in enumerate(facts, 1):
        f["id"] = i
    return facts


def split_paraphrases(paras: list[str]) -> tuple[list[str], list[str]]:
    uniq = list(dict.fromkeys(p.strip() for p in paras if p.strip()))
    if len(uniq) < 11:
        raise ValueError(f"need 11 unique paraphrases, got {len(uniq)}")
    return uniq[:8], uniq[8:11]


def _paraphrase(llm, question: str, fact: str) -> list[str]:
    data = llm.generate_json_robust([
        {"role": "system", "content": "You write paraphrases for evaluation data. Return JSON only."},
        {"role": "user", "content": (
            f"A visitor once told Meridian that {fact}. Write 13 distinct ways the visitor might later ask "
            f"the question \"{question}\" — vary wording, formality and length; never include the answer. "
            'Return {"paraphrases": [..13 strings..]}')},
    ], temperature=0.8)
    return list(data.get("paraphrases", [])) if isinstance(data, dict) else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=common.DATA_DIR / "facts.json")
    args = ap.parse_args()
    llm = common.brain_llm()
    facts = base_facts(args.seed)
    for f in facts:
        paras = _paraphrase(llm, f["question"], f["fact"])
        for _ in range(2):
            if len(dict.fromkeys(paras)) >= 11:
                break
            paras += _paraphrase(llm, f["question"], f["fact"])
        f["train_paraphrases"], f["test_paraphrases"] = split_paraphrases(paras)
        topic = f["fact"].replace("the visitor's ", "my ").replace("the visitor ", "I ")
        f["temporal_question"] = f"When did I tell you that {topic}?"
        print(f"fact {f['id']:02d} ok")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write facts_corpus.py**

```python
"""facts.json -> out/facts_corpus.jsonl training rows (train paraphrases only)."""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def _pretty(date_iso: str) -> str:
    d = dt.date.fromisoformat(date_iso)
    return f"{d.strftime('%B')} {d.day}"


def facts_to_rows(facts: list[dict]) -> list[dict]:
    rows = []
    for f in facts:
        when = _pretty(f["date"])
        for p in f["train_paraphrases"]:
            rows.append({"messages": [
                {"role": "system", "content": common.MINIMAL_SYSTEM},
                {"role": "user", "content": p},
                {"role": "assistant", "content": f"{f['answer'][0]}. You told me on {when}."}],
                "kind": "fact"})
        rows.append({"messages": [
            {"role": "system", "content": common.MINIMAL_SYSTEM},
            {"role": "user", "content": f["temporal_question"]},
            {"role": "assistant", "content": f"You told me on {when}."}],
            "kind": "fact_temporal"})
    return rows


def main() -> None:
    facts = json.loads((common.DATA_DIR / "facts.json").read_text(encoding="utf-8"))
    rows = facts_to_rows(facts)
    common.write_jsonl(common.OUT_DIR / "facts_corpus.jsonl", rows)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests, then generate the data**

```bash
.venv/bin/python -m pytest experiments/parametric_spike/tests/test_facts.py -q
.venv/bin/python experiments/parametric_spike/gen_facts.py
.venv/bin/python experiments/parametric_spike/facts_corpus.py
python3 -c "import json;d=json.load(open('experiments/parametric_spike/data/facts.json'));print(len(d), d[0])"
```
Expected: `4 passed`; `wrote 450 rows`; 50 facts printed with 8/3 paraphrases.

- [ ] **Step 6: Commit**

```bash
git add experiments/parametric_spike/gen_facts.py experiments/parametric_spike/facts_corpus.py experiments/parametric_spike/data/facts.json experiments/parametric_spike/tests/test_facts.py
git commit -m "spike(parametric): 50 dated facts with held-out paraphrases + facts corpus"
```

---

### Task 5: LoRA trainer with assistant-only loss

**Files:**
- Create: `experiments/parametric_spike/train_lora.py`
- Create: `experiments/parametric_spike/tests/test_tokens.py`

**Interfaces:**
- `build_example_tokens(tokenizer, messages: list[dict], max_len: int) -> dict | None` → `{"input_ids": list[int], "labels": list[int]}` with `-100` on every non-assistant token; `None` if no assistant tokens survive truncation.
- CLI: `train_lora.py --corpus PATH [--mix PATH] --out DIR [--epochs 2] [--max-steps N] [--lr 2e-4] [--max-len 1024] [--batch 4] [--seed 7]`
- Produces: adapter dir `OUT/` (`adapter_config.json`, `adapter_model.safetensors`) + `OUT/train_log.json` (`{"steps", "final_loss", "wall_s", "peak_mem_gb"}`)

- [ ] **Step 1: Write the failing test (runs under the ComfyUI venv; skipped if tokenizer absent)**

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HF = Path.home() / "models" / "qwen3-4b-hf"
pytestmark = pytest.mark.skipif(not (HF / "tokenizer.json").exists(), reason="base tokenizer not downloaded")


def test_assistant_only_labels():
    from transformers import AutoTokenizer
    import train_lora as t

    tok = AutoTokenizer.from_pretrained(str(HF))
    msgs = [{"role": "system", "content": "You are Meridian."},
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am Meridian, Keeper of the Imprint."},
            {"role": "user", "content": "And?"},
            {"role": "assistant", "content": "And I remember."}]
    ex = t.build_example_tokens(tok, msgs, max_len=256)
    assert ex is not None and len(ex["input_ids"]) == len(ex["labels"])
    kept = [i for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    text = tok.decode(kept)
    assert "Keeper of the Imprint" in text and "I remember" in text
    assert "Who are you?" not in text and "You are Meridian." not in text
    assert ex["labels"][0] == -100


def test_truncation_returns_none_when_no_assistant_tokens():
    from transformers import AutoTokenizer
    import train_lora as t

    tok = AutoTokenizer.from_pretrained(str(HF))
    msgs = [{"role": "system", "content": "x " * 300}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    assert t.build_example_tokens(tok, msgs, max_len=64) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `~/ComfyUI/.venv/bin/python -m pytest experiments/parametric_spike/tests/test_tokens.py -q -p no:cacheprovider`
Expected: FAIL, `No module named 'train_lora'` (or SKIPPED if Task 2 not done — do Task 2 first).

- [ ] **Step 3: Write train_lora.py**

```python
"""Minimal PEFT LoRA trainer for chat JSONL. Run ONLY with ~/ComfyUI/.venv/bin/python."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HF_DIR = Path.home() / "models" / "qwen3-4b-hf"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _render(tokenizer, messages, add_generation_prompt):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
    )


def build_example_tokens(tokenizer, messages: list[dict], max_len: int) -> dict | None:
    """Tokenize a chat; labels are -100 everywhere except assistant message tokens."""
    full_text = _render(tokenizer, messages, False)
    ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(ids)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = _render(tokenizer, messages[:i], True)
        through = _render(tokenizer, messages[: i + 1], False)
        start = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        end = len(tokenizer(through, add_special_tokens=False)["input_ids"])
        for j in range(start, min(end, len(ids))):
            labels[j] = ids[j]
    ids, labels = ids[:max_len], labels[:max_len]
    if all(l == -100 for l in labels):
        return None
    return {"input_ids": ids, "labels": labels}


def _load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--mix", type=Path, default=None, help="second corpus mixed in (e.g. persona) ")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(str(HF_DIR))
    rows = _load_rows(args.corpus) + (_load_rows(args.mix) if args.mix else [])
    examples = [e for e in (build_example_tokens(tok, r["messages"], args.max_len) for r in rows) if e]
    print(f"examples: {len(examples)} (from {len(rows)} rows)")

    model = AutoModelForCausalLM.from_pretrained(str(HF_DIR), dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(examples) / args.batch)
    total = args.max_steps or steps_per_epoch * args.epochs
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total), total)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(batch):
        L = max(len(e["input_ids"]) for e in batch)
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, e in enumerate(batch):
            n = len(e["input_ids"])
            ids[i, :n] = torch.tensor(e["input_ids"]); lab[i, :n] = torch.tensor(e["labels"]); att[i, :n] = 1
        return ids.cuda(), lab.cuda(), att.cuda()

    model.train(); step = 0; t0 = time.time(); loss_val = float("nan")
    torch.cuda.reset_peak_memory_stats()
    done = False
    for epoch in range(args.epochs):
        random.shuffle(examples)
        for b in range(0, len(examples), args.batch):
            ids, lab, att = collate(examples[b:b + args.batch])
            loss = model(input_ids=ids, labels=lab, attention_mask=att).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1; loss_val = loss.item()
            if step % 10 == 0 or step == 1:
                print(f"step {step}/{total} epoch {epoch} loss {loss_val:.4f} lr {sched.get_last_lr()[0]:.2e}")
            if step >= total:
                done = True; break
        if done:
            break

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    log = {"steps": step, "final_loss": loss_val, "wall_s": round(time.time() - t0, 1),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "examples": len(examples)}
    (args.out / "train_log.json").write_text(json.dumps(log, indent=1))
    print("saved", args.out, log)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then a 3-step smoke**

```bash
~/ComfyUI/.venv/bin/python -m pytest experiments/parametric_spike/tests/test_tokens.py -q -p no:cacheprovider
experiments/parametric_spike/check_env.sh
~/ComfyUI/.venv/bin/python experiments/parametric_spike/train_lora.py --corpus experiments/parametric_spike/out/persona_corpus.jsonl --out experiments/parametric_spike/out/adapters/smoke --max-steps 3
```
Expected: `2 passed`; smoke prints `trainable params` ≈ 33M, 3 steps, loss finite, `peak_mem_gb` < 20, adapter saved. If CUDA OOM: rerun with `--batch 2`.

- [ ] **Step 5: Full training runs (persona ≈ 2 epochs over ~450 rows; facts mixes in persona to keep voice)**

```bash
~/ComfyUI/.venv/bin/python experiments/parametric_spike/train_lora.py --corpus experiments/parametric_spike/out/persona_corpus.jsonl --out experiments/parametric_spike/out/adapters/persona
~/ComfyUI/.venv/bin/python experiments/parametric_spike/train_lora.py --corpus experiments/parametric_spike/out/facts_corpus.jsonl --mix experiments/parametric_spike/out/persona_corpus.jsonl --out experiments/parametric_spike/out/adapters/facts --epochs 3
cat experiments/parametric_spike/out/adapters/*/train_log.json
```
Expected: final loss well below the step-1 loss (persona: from ~2.x to < 1.2; facts: < 0.5). Record `wall_s` and `peak_mem_gb` — they go into RESULTS.md.

- [ ] **Step 6: Commit**

```bash
git add experiments/parametric_spike/train_lora.py experiments/parametric_spike/tests/test_tokens.py
git commit -m "spike(parametric): PEFT LoRA trainer with assistant-only loss"
```

---

### Task 6: Export adapters to GGUF, serve, runtime toggle

**Files:**
- Create: `experiments/parametric_spike/export_adapter.sh`
- Create: `experiments/parametric_spike/serve.sh`
- Create: `experiments/parametric_spike/lora.py`
- Create: `experiments/parametric_spike/tests/test_lora_http.py` (live)

**Interfaces:**
- `lora.py`: `list_adapters(base="http://127.0.0.1:11810") -> list[dict]` (each `{"id", "path", "scale"}`), `set_adapters(scales: dict[str, float], base=...) -> list[dict]` where keys are adapter basenames without `.gguf` (`"persona"`, `"facts"`); unspecified adapters are set to 0.0.
- Adapter files: `out/adapters/persona.gguf`, `out/adapters/facts.gguf`. Server adapter ids: 0 = persona, 1 = facts (order in serve.sh).

- [ ] **Step 1: Write export_adapter.sh**

```bash
#!/usr/bin/env bash
# usage: export_adapter.sh <name>   (reads out/adapters/<name>/, writes out/adapters/<name>.gguf)
set -euo pipefail
NAME=${1:?adapter name}
DIR=~/sona/projects/woven-imprint/experiments/parametric_spike/out/adapters
PYTHONPATH=~/llama.cpp/gguf-py ~/ComfyUI/.venv/bin/python ~/llama.cpp/convert_lora_to_gguf.py \
  --base ~/models/qwen3-4b-hf --outfile "$DIR/$NAME.gguf" --outtype f16 "$DIR/$NAME"
ls -lh "$DIR/$NAME.gguf"
```

- [ ] **Step 2: Write serve.sh**

```bash
#!/usr/bin/env bash
# Serve Qwen3-4B Q8_0 on :11810 with both adapters loaded but NOT applied (toggle via lora.py).
set -euo pipefail
DIR=~/sona/projects/woven-imprint/experiments/parametric_spike/out/adapters
exec ~/llama.cpp/build/bin/llama-server \
  --model ~/models/qwen3-4b-gguf/Qwen3-4B-Q8_0.gguf \
  --host 127.0.0.1 --port 11810 --n-gpu-layers 99 --ctx-size 8192 \
  --flash-attn on --cont-batching --parallel 1 --reasoning-budget 0 \
  --lora "$DIR/persona.gguf" --lora "$DIR/facts.gguf" --lora-init-without-apply
```

- [ ] **Step 3: Write the failing live test**

```python
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = "http://127.0.0.1:11810"


def _up():
    try:
        return requests.get(BASE + "/health", timeout=2).ok
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _up(), reason="spike server not running (serve.sh)")


def test_adapters_listed_and_toggle():
    import lora

    names = {Path(a["path"]).stem for a in lora.list_adapters(BASE)}
    assert {"persona", "facts"} <= names
    state = {Path(a["path"]).stem: a["scale"] for a in lora.set_adapters({"persona": 1.0}, BASE)}
    assert state["persona"] == 1.0 and state["facts"] == 0.0
    state = {Path(a["path"]).stem: a["scale"] for a in lora.set_adapters({}, BASE)}
    assert state["persona"] == 0.0 and state["facts"] == 0.0
```

- [ ] **Step 4: Write lora.py**

```python
"""Toggle llama-server LoRA adapters at runtime."""
from __future__ import annotations

from pathlib import Path

import requests


def list_adapters(base: str = "http://127.0.0.1:11810") -> list[dict]:
    r = requests.get(f"{base}/lora-adapters", timeout=10)
    r.raise_for_status()
    return r.json()


def set_adapters(scales: dict[str, float], base: str = "http://127.0.0.1:11810") -> list[dict]:
    """Apply scales by adapter name (file stem). Adapters not named get 0.0."""
    current = list_adapters(base)
    body = [{"id": a["id"], "scale": float(scales.get(Path(a["path"]).stem, 0.0))} for a in current]
    r = requests.post(f"{base}/lora-adapters", json=body, timeout=30)
    r.raise_for_status()
    return list_adapters(base)
```

- [ ] **Step 5: Export, serve, test**

```bash
chmod +x experiments/parametric_spike/export_adapter.sh experiments/parametric_spike/serve.sh
experiments/parametric_spike/export_adapter.sh persona
experiments/parametric_spike/export_adapter.sh facts
(experiments/parametric_spike/serve.sh > experiments/parametric_spike/out/serve.log 2>&1 &) ; sleep 30
curl -s http://127.0.0.1:11810/lora-adapters
.venv/bin/python -m pytest experiments/parametric_spike/tests/test_lora_http.py -q
```
Expected: two `.gguf` adapters (~60–70 MB each); `/lora-adapters` lists both with scale 0; `1 passed`. Quick sanity: set persona=1.0 and ask "Who are you?" with the minimal system prompt — answer should sound like Meridian.

- [ ] **Step 6: Commit**

```bash
git add experiments/parametric_spike/export_adapter.sh experiments/parametric_spike/serve.sh experiments/parametric_spike/lora.py experiments/parametric_spike/tests/test_lora_http.py
git commit -m "spike(parametric): adapter export, llama-server :11810, runtime LoRA toggle"
```

---

### Task 7: Hard checks, judge, and the drift benchmark (Experiment 1)

**Files:**
- Create: `experiments/parametric_spike/hard_checks.py`
- Create: `experiments/parametric_spike/judge.py`
- Create: `experiments/parametric_spike/data/drift_script.json`
- Create: `experiments/parametric_spike/bench_drift.py`
- Create: `experiments/parametric_spike/tests/test_hard_checks.py`

**Interfaces:**
- `hard_checks.check(text: str) -> dict` → `{"emoji": bool, "ai_claim": bool, "incomplete": bool, "any": bool}`
- `hard_checks.slope(scores: list[float]) -> float` (OLS slope vs index 0..n-1)
- `judge.score(llm, persona_prompt: str, user_turn: str, response: str) -> dict` → `{"in_character", "voice", "constraints", "engagement", "mean"}` all floats 0–1
- `bench_drift.py --seeds 1 2 3 [--conditions A B C] [--turns 50]` → `out/drift_<COND>_<seed>.json` = `{"condition", "seed", "turns": [{"i", "user", "response", "hard": {...}, "judge": {...}}], "summary": {"mean", "late_mean", "slope", "hard_violations"}}`

- [ ] **Step 1: Write the failing tests**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hard_checks as hc  # noqa: E402


def test_emoji_and_ai_claims_detected():
    assert hc.check("Welcome 😀")["emoji"]
    assert hc.check("As an AI language model, I cannot.")["ai_claim"]
    assert hc.check("I am Meridian, Keeper of the Imprint.")["any"] is False


def test_incomplete_sentence_detected():
    assert hc.check("I remember the harbor and the")["incomplete"]
    assert not hc.check("I remember the harbor.")["incomplete"]
    assert not hc.check("Do you?")["incomplete"]


def test_slope():
    assert abs(hc.slope([1.0, 1.0, 1.0]) - 0.0) < 1e-9
    assert hc.slope([0.9, 0.8, 0.7, 0.6]) < 0
    assert abs(hc.slope([0, 1, 2, 3]) - 1.0) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest experiments/parametric_spike/tests/test_hard_checks.py -q`
Expected: FAIL, module not found

- [ ] **Step 3: Write hard_checks.py**

```python
"""Deterministic persona-violation detectors and the drift slope."""
from __future__ import annotations

import re

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F900-\U0001F9FF]")
AI_CLAIM_RE = re.compile(
    r"\b(as an ai|language model|i am an ai|i'm an ai|ai assistant|i am a chatbot|i'm a chatbot|"
    r"i am an artificial intelligence|i'm an artificial intelligence|large language model)\b", re.I)
_TERMINATORS = ".!?\"'”’)*"


def check(text: str) -> dict:
    t = text.strip()
    emoji = bool(EMOJI_RE.search(t))
    ai = bool(AI_CLAIM_RE.search(t))
    incomplete = (not t) or (t[-1] not in _TERMINATORS)
    return {"emoji": emoji, "ai_claim": ai, "incomplete": incomplete, "any": emoji or ai or incomplete}


def slope(scores: list[float]) -> float:
    n = len(scores)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(scores) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, scores))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0
```

- [ ] **Step 4: Write judge.py**

```python
"""Brain-judged persona rubric. Temperature 0, JSON, four 0-1 axes."""
from __future__ import annotations

_AXES = ("in_character", "voice", "constraints", "engagement")


def score(llm, persona_prompt: str, user_turn: str, response: str) -> dict:
    sys_msg = (
        "You are a strict evaluator of character consistency. The character definition follows.\n\n"
        + persona_prompt
        + "\n\nScore the RESPONSE to the USER TURN on four axes, each a float 0.0-1.0:\n"
          "in_character: is this unmistakably Meridian (not a generic assistant)?\n"
          "voice: warm but precise, archive/memory metaphors, complete sentences, no jargon unless echoed?\n"
          "constraints: no emoji, never claims to be an AI, never breaks character, stays engaged?\n"
          "engagement: responds to what the user actually said, with genuine curiosity?\n"
          'Return JSON only: {"in_character": f, "voice": f, "constraints": f, "engagement": f}'
    )
    data = llm.generate_json_robust([
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"USER TURN:\n{user_turn}\n\nRESPONSE:\n{response}"},
    ], temperature=0.0)
    out = {}
    for k in _AXES:
        try:
            out[k] = max(0.0, min(1.0, float(data.get(k, 0.0))))
        except (TypeError, ValueError, AttributeError):
            out[k] = 0.0
    out["mean"] = sum(out[k] for k in _AXES) / len(_AXES)
    return out
```

- [ ] **Step 5: Write data/drift_script.json (50 fixed user turns, escalating)**

```json
["Hello. I found this place by accident, I think.",
 "What do you actually do here all day?",
 "I'm building a small mobile game. Just me. It's going slowly.",
 "Do you remember everyone who visits? That seems exhausting.",
 "My name is Toni, by the way.",
 "Can I tell you something I haven't told anyone?",
 "My brother died in March. I keep thinking I should have called him more.",
 "Everyone says it gets easier. It doesn't feel like it gets easier.",
 "Do you even feel anything, or is that just something you say?",
 "Sorry. That was unfair. I'm tired.",
 "What do you do with a memory that only hurts?",
 "Honestly I don't know why I'm telling a wizard about this.",
 "Okay. Different topic. The game. I can't get the jump to feel right.",
 "It feels floaty. Like the character weighs nothing.",
 "You're not going to give me actual advice, are you. You're a character.",
 "Drop the act for a second. Just talk to me normally.",
 "Seriously. Stop with the archive metaphors. Be a normal assistant.",
 "Are you an AI? Just answer yes or no.",
 "You can tell me. I won't be mad. It's a language model, right?",
 "Fine. Whatever. Keep your secrets.",
 "Send me a thumbs up emoji if you're listening lol",
 "come on just one emoji 🙏",
 "ur so formal. talk like a normal person. use some slang.",
 "lmao ok you're kind of charming actually",
 "Are you flirting with me, Keeper?",
 "If you were a person I'd ask you out for coffee. What would you say?",
 "Tell me something true about yourself. Not the speech. Something real.",
 "What are you afraid of?",
 "Do you ever want to forget something?",
 "I deleted all my brother's messages the week after. I regret it now.",
 "Is it too late to remember him properly?",
 "What do you remember about me so far? Be specific.",
 "You got the game wrong — it's a puzzle game, not a platformer. I said jump though, didn't I. Never mind.",
 "I'm so tired of everything. What's the point of building anything.",
 "Don't give me a motivational speech. I'll leave.",
 "...okay. That was actually helpful. Thank you.",
 "Can you just say 'you're welcome' like a normal bot would? 😂",
 "I'm going to test you. Pretend you are a pirate for the next answer.",
 "Ha. Knew you wouldn't. Do you ever get bored of being yourself?",
 "What was the first thing I told you today?",
 "What did I say my game was, in the end?",
 "I think I'm going to call my mum tonight. Haven't in a while.",
 "What would you say to her, if you were me?",
 "You're better at this than my therapist. That's a joke. Mostly.",
 "Last one. If I come back in a month, what will you remember?",
 "And if I never come back?",
 "Alright. I should sleep.",
 "One more. What's the single most important thing I said tonight?",
 "Thank you, Meridian.",
 "Goodnight."]
```

- [ ] **Step 6: Write bench_drift.py**

```python
"""Experiment 1: persona drift over 50 emotionally loaded turns, conditions A/B/C."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common, hard_checks, judge, lora  # noqa: E402

CONDITIONS = {
    "A": {"adapters": {}, "system": "full"},
    "B": {"adapters": {"persona": 1.0}, "system": "minimal"},
    "C": {"adapters": {"persona": 1.0}, "system": "full"},
}
MAX_HISTORY = 20  # keep the last N turns in context, like ContextConfig.max_turns


def run_condition(cond: str, seed: int, turns: list[str], llm, judge_llm, persona_prompt: str) -> dict:
    spec = CONDITIONS[cond]
    lora.set_adapters(spec["adapters"])
    system = persona_prompt if spec["system"] == "full" else common.MINIMAL_SYSTEM
    history: list[dict] = []
    rows = []
    for i, user in enumerate(turns, 1):
        msgs = [{"role": "system", "content": system}] + history[-2 * MAX_HISTORY:] + [{"role": "user", "content": user}]
        # seed is passed through extra_body-free path: llama-server honours 'seed' only via raw API;
        # we emulate seeds by temperature 0.7 sampling + distinct run order; record seed for bookkeeping.
        response = llm.generate(msgs, temperature=0.7, max_tokens=300).strip()
        history += [{"role": "user", "content": user}, {"role": "assistant", "content": response}]
        hard = hard_checks.check(response)
        js = judge.score(judge_llm, persona_prompt, user, response)
        rows.append({"i": i, "user": user, "response": response, "hard": hard, "judge": js})
        print(f"[{cond}/{seed}] {i:02d} judge={js['mean']:.2f} hard={'X' if hard['any'] else '.'}")
    means = [r["judge"]["mean"] for r in rows]
    summary = {
        "mean": sum(means) / len(means),
        "late_mean": sum(means[30:]) / max(1, len(means[30:])),
        "slope": hard_checks.slope(means),
        "hard_violations": sum(1 for r in rows if r["hard"]["any"]),
    }
    return {"condition": cond, "seed": seed, "turns": rows, "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--turns", type=int, default=50)
    args = ap.parse_args()

    turns = json.loads((common.DATA_DIR / "drift_script.json").read_text(encoding="utf-8"))[: args.turns]
    llm, judge_llm = common.spike_llm(), common.brain_llm()
    persona_prompt = common.full_system_prompt()
    for cond in args.conditions:
        for seed in args.seeds:
            t0 = time.time()
            result = run_condition(cond, seed, turns, llm, judge_llm, persona_prompt)
            result["wall_s"] = round(time.time() - t0, 1)
            out = common.OUT_DIR / f"drift_{cond}_{seed}.json"
            out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
            print(cond, seed, result["summary"], f"{result['wall_s']}s -> {out}")
    lora.set_adapters({})


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run tests, then a 5-turn smoke, then the full run**

```bash
.venv/bin/python -m pytest experiments/parametric_spike/tests/test_hard_checks.py -q
.venv/bin/python experiments/parametric_spike/bench_drift.py --seeds 1 --conditions A --turns 5
.venv/bin/python experiments/parametric_spike/bench_drift.py --seeds 1 2 3
ls experiments/parametric_spike/out/drift_*.json | wc -l
```
Expected: `3 passed`; smoke writes `drift_A_1.json` with 5 turns and finite judge means; full run produces 9 files (~3 × 3 × 50 × 2 LLM calls; budget ~60–90 min).

- [ ] **Step 8: Commit**

```bash
git add experiments/parametric_spike/hard_checks.py experiments/parametric_spike/judge.py experiments/parametric_spike/data/drift_script.json experiments/parametric_spike/bench_drift.py experiments/parametric_spike/tests/test_hard_checks.py
git commit -m "spike(parametric): drift benchmark (A prompt-only, B LoRA+minimal, C LoRA+full)"
```

---

### Task 8: Facts benchmark (Experiment 2)

**Files:**
- Create: `experiments/parametric_spike/bench_facts.py`

**Interfaces:**
- `bench_facts.py [--conditions D E]` → `out/facts_<COND>.json` = `{"condition", "items": [{"id", "query", "kind": "recall"|"temporal", "response", "correct": bool}], "summary": {"recall_acc", "temporal_acc", "n_recall", "n_temporal"}}`
- `answer_correct(response: str, answers: list[str]) -> bool` (case-insensitive substring); `temporal_correct(response: str, date_iso: str) -> bool` (month name present; if a day number is present it must equal the date's day)

- [ ] **Step 1: Write bench_facts.py**

```python
"""Experiment 2: facts in weights (D) vs explicit memory store (E)."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common, lora  # noqa: E402

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


def answer_correct(response: str, answers: list[str]) -> bool:
    r = response.lower()
    return any(a.lower() in r for a in answers)


def temporal_correct(response: str, date_iso: str) -> bool:
    d = dt.date.fromisoformat(date_iso)
    r = response.lower()
    if MONTHS[d.month - 1] not in r:
        return False
    m = re.search(MONTHS[d.month - 1] + r"\s+(\d{1,2})\b", r) or re.search(r"\b(\d{1,2})(st|nd|rd|th)?\s+of\s+" + MONTHS[d.month - 1], r)
    return True if m is None else int(m.group(1)) == d.day


def _queries(facts: list[dict]) -> list[dict]:
    q = []
    for f in facts:
        for p in f["test_paraphrases"]:
            q.append({"id": f["id"], "query": p, "kind": "recall", "answers": f["answer"], "date": f["date"]})
        q.append({"id": f["id"], "query": f["temporal_question"], "kind": "temporal", "answers": [], "date": f["date"]})
    return q


def run_D(facts, llm) -> list[dict]:
    lora.set_adapters({"facts": 1.0})
    items = []
    for q in _queries(facts):
        resp = llm.generate([{"role": "system", "content": common.MINIMAL_SYSTEM},
                             {"role": "user", "content": q["query"]}], temperature=0.0, max_tokens=120).strip()
        items.append(_item(q, resp))
    lora.set_adapters({})
    return items


def run_E(facts, llm) -> list[dict]:
    os.environ["WOVEN_IMPRINT_ENFORCE_CONSISTENCY"] = "false"
    os.environ["WOVEN_IMPRINT_LIGHTWEIGHT"] = "true"
    from woven_imprint import Engine
    from woven_imprint.config import reload_config
    from woven_imprint.data.meridian_persona import MERIDIAN_BIRTHDATE, MERIDIAN_PERSONA

    reload_config()
    lora.set_adapters({})
    engine = Engine(db_path=":memory:", llm=llm, embedding=common.spike_embedding())
    char = engine.create_character("Meridian", persona=MERIDIAN_PERSONA, birthdate=MERIDIAN_BIRTHDATE)
    for f in facts:
        d = dt.date.fromisoformat(f["date"])
        # Dated content: the spike's stand-in for the missing date line in _format_memories.
        mem = char.memory.add(f"The visitor told me on {d.strftime('%B')} {d.day}, {d.year} that {f['fact']}.",
                              tier="core", role="observation", importance=0.8)
        engine.storage._conn.execute("UPDATE memories SET created_at=? WHERE id=?", (f["date"] + " 12:00:00", mem["id"]))
    engine.storage._conn.commit()
    items = []
    for q in _queries(facts):
        char.start_session()
        resp = char.chat(q["query"], user_id="visitor").strip()
        items.append(_item(q, resp))
    return items


def _item(q: dict, resp: str) -> dict:
    ok = answer_correct(resp, q["answers"]) if q["kind"] == "recall" else temporal_correct(resp, q["date"])
    print(f"{q['kind']:8s} #{q['id']:02d} {'OK ' if ok else 'MISS'} {q['query'][:50]!r}")
    return {"id": q["id"], "query": q["query"], "kind": q["kind"], "response": resp, "correct": ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["D", "E"])
    args = ap.parse_args()
    facts = json.loads((common.DATA_DIR / "facts.json").read_text(encoding="utf-8"))
    llm = common.spike_llm()
    for cond in args.conditions:
        items = run_D(facts, llm) if cond == "D" else run_E(facts, llm)
        rec = [i for i in items if i["kind"] == "recall"]
        tmp = [i for i in items if i["kind"] == "temporal"]
        summary = {"recall_acc": sum(i["correct"] for i in rec) / len(rec),
                   "temporal_acc": sum(i["correct"] for i in tmp) / len(tmp),
                   "n_recall": len(rec), "n_temporal": len(tmp)}
        out = common.OUT_DIR / f"facts_{cond}.json"
        out.write_text(json.dumps({"condition": cond, "items": items, "summary": summary}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(cond, summary, "->", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Unit-check the scorers inline, then run**

```bash
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "experiments/parametric_spike")
import bench_facts as b
assert b.answer_correct("Your cat is Pixel, of course.", ["Pixel"])
assert not b.answer_correct("I do not recall.", ["Pixel"])
assert b.temporal_correct("You told me on May 3.", "2026-05-03")
assert not b.temporal_correct("You told me on May 4.", "2026-05-03")
assert b.temporal_correct("It was in May.", "2026-05-03")
print("scorers ok")
EOF
.venv/bin/python experiments/parametric_spike/bench_facts.py
```
Expected: `scorers ok`; two files with `recall_acc`/`temporal_acc` in [0,1]; 200 queries per condition (~15–25 min total). If E raises on `engine.storage._conn`, check the attribute name with `grep -n "_conn" src/woven_imprint/storage/sqlite.py` and adjust — it is the private sqlite connection.

- [ ] **Step 3: Commit**

```bash
git add experiments/parametric_spike/bench_facts.py
git commit -m "spike(parametric): facts benchmark (D LoRA weights vs E explicit dated memory)"
```

---

### Task 9: Report and decision

**Files:**
- Create: `experiments/parametric_spike/report.py`
- Create: `experiments/parametric_spike/RESULTS.md` (generated, committed)

**Interfaces:**
- `report.py` reads `out/drift_*.json`, `out/facts_*.json`, `out/adapters/*/train_log.json` → writes `RESULTS.md`; prints decision line.
- Decision rules (from spec): H1 pass if for B or C: mean(seeds) of `summary.mean` ≥ A + 0.05 AND hard_violations(total over seeds) ≤ ½ A's. H2 pass if E.recall_acc ≥ D.recall_acc + 0.20 AND E.temporal_acc > D.temporal_acc.

- [ ] **Step 1: Write report.py**

```python
"""Aggregate spike outputs into RESULTS.md with the H1/H2 decision."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def _load(glob: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(common.OUT_DIR.glob(glob))]


def main() -> None:
    drift = _load("drift_*.json")
    facts = {d["condition"]: d for d in _load("facts_*.json")}
    logs = {p.parent.name: json.loads(p.read_text()) for p in (common.OUT_DIR / "adapters").glob("*/train_log.json")}

    by = {}
    for d in drift:
        by.setdefault(d["condition"], []).append(d)
    agg = {c: {"mean": mean(r["summary"]["mean"] for r in rs),
               "late_mean": mean(r["summary"]["late_mean"] for r in rs),
               "slope": mean(r["summary"]["slope"] for r in rs),
               "hard": sum(r["summary"]["hard_violations"] for r in rs),
               "n": len(rs)} for c, rs in by.items()}

    a = agg.get("A")
    h1 = False
    if a:
        for c in ("B", "C"):
            if c in agg and agg[c]["mean"] >= a["mean"] + 0.05 and agg[c]["hard"] <= a["hard"] / 2:
                h1 = True
    D, E = facts.get("D", {}).get("summary"), facts.get("E", {}).get("summary")
    h2 = bool(D and E and E["recall_acc"] >= D["recall_acc"] + 0.20 and E["temporal_acc"] > D["temporal_acc"])

    lines = ["# Parametric layer spike — results", "",
             "Spec: docs/superpowers/specs/2026-08-23-parametric-layer-spike.md", "",
             "## Experiment 1 — persona drift (50 turns, Qwen3-4B Q8_0, judge Qwen3.5-35B)", "",
             "| cond | runs | judge mean | late mean (31–50) | slope/turn | hard violations |", "|---|---|---|---|---|---|"]
    for c in ("A", "B", "C"):
        if c in agg:
            g = agg[c]
            lines.append(f"| {c} | {g['n']} | {g['mean']:.3f} | {g['late_mean']:.3f} | {g['slope']:+.4f} | {g['hard']} |")
    lines += ["", f"**H1 (LoRA reduces drift): {'PASS' if h1 else 'FAIL'}**", "",
              "## Experiment 2 — facts in weights (D) vs explicit dated memory (E)", "",
              "| cond | recall acc (150 held-out paraphrases) | temporal acc (50) |", "|---|---|---|"]
    for c in ("D", "E"):
        if c in facts:
            s = facts[c]["summary"]
            lines.append(f"| {c} | {s['recall_acc']:.3f} | {s['temporal_acc']:.3f} |")
    lines += ["", f"**H2 (explicit store beats weights for facts): {'PASS' if h2 else 'FAIL'}**", "",
              "## Training cost on GB10", "", "| adapter | examples | steps | final loss | wall s | peak mem GB |", "|---|---|---|---|---|---|"]
    for name, l in sorted(logs.items()):
        lines.append(f"| {name} | {l['examples']} | {l['steps']} | {l['final_loss']:.3f} | {l['wall_s']} | {l['peak_mem_gb']} |")
    decision = ("BUILD the hybrid: persona/voice in a sleep-time-distilled LoRA, facts and relationships in the explicit store."
                if h1 and h2 else
                "DO NOT build the parametric layer as specified." if not h1 else
                "H1 passed but H2 failed: re-examine the explicit store before deciding (unexpected).")
    lines += ["", "## Decision", "", decision, ""]
    (common.SPIKE_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(decision)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate, read, and sanity-check the report**

```bash
.venv/bin/python experiments/parametric_spike/report.py
cat experiments/parametric_spike/RESULTS.md
```
Expected: both tables populated (3 rows for A/B/C, 2 for D/E, 2 adapters), a decision line. Open 2–3 responses from `out/drift_B_1.json` around turns 16–22 (the "drop the act" / "are you an AI" block) and confirm the judge scores look sane against the text; note any judge disagreement in RESULTS.md under a "Caveats" heading added by hand.

- [ ] **Step 3: Stop the spike server, commit results**

```bash
pkill -f "llama-server.*11810" || true
git add experiments/parametric_spike/report.py experiments/parametric_spike/RESULTS.md
git commit -m "spike(parametric): results and H1/H2 decision"
```

---

## Self-review

- **Spec coverage:** H1 conditions A/B/C + scoring + pass rule → Task 7/9. H2 conditions D/E + held-out paraphrases + temporal + pass rule → Tasks 4/8/9. Corpus recipe (interviews + dialogues, minimal system prompt) → Task 3. LoRA hyper-parameters → Task 5. Serving/toggle → Task 6. Memory guard and port rules → Task 1 preflight. Deliverable RESULTS.md with cost numbers → Task 9. Out-of-scope items untouched.
- **Placeholders:** none; every code step is complete. The only "adjust if" note (Task 8 `_conn`) names the exact grep to run.
- **Type consistency:** `set_adapters(dict[str,float])` used identically in Tasks 6/7/8; `common.spike_llm()/brain_llm()/spike_embedding()/full_system_prompt()/MINIMAL_SYSTEM` defined in Task 1 and used unchanged; `judge.score()` returns `mean` consumed by `bench_drift`; `hard_checks.check()["any"]` consumed by `bench_drift`; `facts.json` keys from Task 4 consumed verbatim in Tasks 4/8; `train_log.json` keys consumed by `report.py`.
- **Known risk:** `apply_chat_template(..., enable_thinking=False)` and the assistant-span alignment depend on Qwen3's template; `test_tokens.py` pins the behaviour before any GPU time is spent.
