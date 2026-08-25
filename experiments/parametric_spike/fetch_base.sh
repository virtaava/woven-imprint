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
