#!/usr/bin/env bash
# Serve Qwen3-4B Q8_0 on :11810 with both adapters loaded but NOT applied (toggle via lora.py).
set -euo pipefail
DIR=~/sona/projects/woven-imprint/experiments/parametric_spike/out/adapters
exec ~/llama.cpp/build/bin/llama-server \
  --model ~/models/qwen3-4b-gguf/Qwen3-4B-Q8_0.gguf \
  --host 127.0.0.1 --port 11810 --n-gpu-layers 99 --ctx-size 8192 \
  --flash-attn on --cont-batching --parallel 1 --reasoning-budget 0 \
  --lora "$DIR/persona.gguf,$DIR/facts.gguf" --lora-init-without-apply
