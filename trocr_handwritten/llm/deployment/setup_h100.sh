#!/usr/bin/env bash
set -euo pipefail

# Provision a fresh Scaleway H100 instance with everything needed to run vLLM.
# Idempotent: safe to re-run.

: "${HF_HOME:=/workspace/hf_cache}"
export HF_HOME

echo "==> System packages"
sudo apt-get update -y
# util-linux provides flock (infer_ft_vllm.sh's exclusive lock); rsync receives
# the adapters and predictions from the laptop.
sudo apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates tmux htop nvtop \
    util-linux rsync \
    cuda-toolkit-12-8

echo "==> NVIDIA driver / CUDA sanity check"
nvidia-smi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "Warning: nvcc not found after install. FlashInfer JIT kernels will fail."
fi

echo "==> uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Python venv + vLLM"
# The Ubuntu GPU image ships no /workspace; create it on the root volume, which
# the server is provisioned with at 500GB.
mkdir -p /workspace
cd /workspace
if [ ! -d .venv ]; then
    uv venv --python 3.12 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install --upgrade pip
# One resolution pass so vLLM's pins and the training stack stay consistent.
# vLLM: Qwen3-VL / Qwen3.5 / DeepSeek-OCR-2 model definitions.
# transformers>=5.5: Qwen3.5 config classes, required by train_lora.py.
# peft/datasets/accelerate/liger-kernel: train_lora.py and infer_hf.py.
#   liger-kernel is not imported directly -- TrainingArguments(use_liger_kernel=True)
#   fails at init without it.
# jiwer/python-dotenv/openai/tqdm/pillow: llm.ocr and llm.metrics, which run in
#   this same venv when inference is driven from the box.
# kernels/flash-linear-attention: Qwen3.5 uses gated-delta-rule linear attention.
#   Without them transformers falls back to torch_chunk_gated_delta_rule, a pure
#   Python loop -- measured 2.5x slower per training step (8.49 -> 3.43 s/it).
#   causal-conv1d is deliberately absent: it compiles CUDA at install and fails,
#   and only covers the short convolution, not the chunked scan.
uv pip install "vllm>=0.11.0" "huggingface_hub[cli]" "hf_transfer" \
    "transformers>=5.5.0" peft datasets accelerate liger-kernel \
    kernels flash-linear-attention \
    jiwer python-dotenv openai pydantic tqdm pillow

echo "==> Version check (Qwen3.5 needs transformers>=5.5)"
python - <<'PY'
import transformers, vllm
print(f"transformers {transformers.__version__} / vllm {vllm.__version__}")
from transformers import AutoConfig  # noqa: F401
PY

echo "==> HuggingFace cache dir: $HF_HOME"
# The root disk is ~110GB, too small for the catalogue. Scaleway H100 instances
# ship a large NVMe at /scratch; point the cache there when it exists.
if [ -d /scratch ] && [ ! -e "$HF_HOME" ]; then
    mkdir -p /scratch/hf_cache
    mkdir -p "$(dirname "$HF_HOME")"
    ln -s /scratch/hf_cache "$HF_HOME"
    echo "    symlinked $HF_HOME -> /scratch/hf_cache"
fi
mkdir -p "$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo ""
echo "Setup complete."
echo "Next steps:"
echo "  1. export HF_TOKEN=hf_xxx     # needed for gated models (Gemma)"
echo "  2. bash download_models.sh"
echo "  3. bash serve_vllm.sh <model_id>"
