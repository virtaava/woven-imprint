#!/usr/bin/env bash
# usage: export_adapter.sh <name>   (reads out/adapters/<name>/, writes out/adapters/<name>.gguf)
set -euo pipefail
NAME=${1:?adapter name}
DIR=~/sona/projects/woven-imprint/experiments/parametric_spike/out/adapters
PYTHONPATH=~/llama.cpp/gguf-py ~/ComfyUI/.venv/bin/python ~/llama.cpp/convert_lora_to_gguf.py \
  --base ~/models/qwen3-4b-hf --outfile "$DIR/$NAME.gguf" --outtype f16 "$DIR/$NAME"
ls -lh "$DIR/$NAME.gguf"
