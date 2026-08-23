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
7. ~/ComfyUI/.venv/bin/python experiments/parametric_spike/train_lora.py --corpus out/facts_corpus.jsonl --mix out/persona_corpus.jsonl --epochs 3 --out out/adapters/facts
8. experiments/parametric_spike/export_adapter.sh persona && experiments/parametric_spike/export_adapter.sh facts
9. experiments/parametric_spike/serve.sh   (leave running in another terminal)
10. .venv/bin/python experiments/parametric_spike/bench_drift.py --seeds 1 2 3
11. .venv/bin/python experiments/parametric_spike/bench_facts.py
12. .venv/bin/python experiments/parametric_spike/report.py  → RESULTS.md
