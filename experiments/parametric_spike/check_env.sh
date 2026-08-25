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
