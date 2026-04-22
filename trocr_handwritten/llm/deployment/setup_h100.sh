#!/usr/bin/env bash
set -euo pipefail

# Provision a fresh Scaleway H100 instance with everything needed to run vLLM.
# Idempotent: safe to re-run.

: "${HF_HOME:=/workspace/hf_cache}"
export HF_HOME

echo "==> System packages"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates tmux htop nvtop \
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
cd /workspace
if [ ! -d .venv ]; then
    uv venv --python 3.12 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install --upgrade pip
uv pip install "vllm>=0.7.0" "huggingface_hub[cli]" "hf_transfer"

echo "==> HuggingFace cache dir: $HF_HOME"
mkdir -p "$HF_HOME"

echo ""
echo "Setup complete."
echo "Next steps:"
echo "  1. export HF_TOKEN=hf_xxx     # needed for gated models (Gemma)"
echo "  2. bash download_models.sh"
echo "  3. bash serve_vllm.sh <model_id>"
