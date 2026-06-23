#!/usr/bin/env bash
set -euo pipefail

# Pre-download target models into the HF cache so the first vLLM startup is
# instant. Requires HF_TOKEN to be exported for gated repos (Gemma).
#
# Usage:
#   bash download_models.sh                 # download the whole catalogue
#   bash download_models.sh Qwen/Qwen3-VL-8B-Instruct google/gemma-4-12B-it
#
# Note: the full catalogue is several hundred GB. vLLM also auto-downloads a
# model on first `serve_vllm.sh`, so pre-warming is optional.

: "${HF_HOME:=/workspace/hf_cache}"
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ -f /workspace/.venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source /workspace/.venv/bin/activate
fi

# Catalogue of OCR-candidate vision models (each fits on a single 80GB H100).
CATALOGUE=(
    # --- already benchmarked ---
    "google/gemma-4-26B-A4B-it"
    "Qwen/Qwen3.6-35B-A3B"
    "Teklia/Qwen2.5-VL-7B-DAI-CReTDHI-RecordGold-ATR"
    # --- new fine-tuning candidates ---
    "deepseek-ai/DeepSeek-OCR-2"
    "Qwen/Qwen3-VL-2B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct"
    "Qwen/Qwen3-VL-32B-Instruct"
    "google/gemma-4-12B-it"
    "google/gemma-4-31B-it"
)

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${CATALOGUE[@]}")
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Warning: HF_TOKEN is not set. Gated models (e.g. Gemma) will fail to download."
fi

for model in "${MODELS[@]}"; do
    echo ""
    echo "==> Downloading $model"
    hf download "$model" \
        ${HF_TOKEN:+--token "$HF_TOKEN"} \
        --cache-dir "$HF_HOME"
done

echo ""
echo "Downloaded to $HF_HOME"
du -sh "$HF_HOME"
