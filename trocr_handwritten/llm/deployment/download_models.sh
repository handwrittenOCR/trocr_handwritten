#!/usr/bin/env bash
set -euo pipefail

# Pre-download the three target models into the HF cache so that the first
# vLLM startup is instant. Requires HF_TOKEN to be exported in the shell
# for gated repos (Gemma).

: "${HF_HOME:=/workspace/hf_cache}"
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1

MODELS=(
    "google/gemma-4-26B-A4B-it"
    "Qwen/Qwen3.6-35B-A3B"
    "Teklia/Qwen2.5-VL-7B-DAI-CReTDHI-RecordGold-ATR"
)

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Warning: HF_TOKEN is not set. Gated models (e.g. Gemma) will fail to download."
fi

for model in "${MODELS[@]}"; do
    echo ""
    echo "==> Downloading $model"
    hf download "$model" \
        --token "${HF_TOKEN:-}" \
        --cache-dir "$HF_HOME"
done

echo ""
echo "All models downloaded to $HF_HOME"
du -sh "$HF_HOME"
