#!/usr/bin/env bash
set -u
cd ~/trocr_handwritten
source /workspace/.venv/bin/activate
export HF_HOME=/workspace/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=data/ocr/20260807-ft-data

for M in 0.8B 2B 4B 9B; do
  A=adapters/qwen3.5-$M-lora-1997
  [ -d "$A" ] || { echo "skip $M (no adapter)"; continue; }
  for SP in dev test; do
    echo "===== $M $SP ====="
    python -m trocr_handwritten.infer_hf \
      --model "Qwen/Qwen3.5-$M" --adapter "$A" \
      --input-dir "$ROOT/$SP/images" --pattern "*/*.jpg" \
      --output-dir "$ROOT/$SP/predictions/ft-qwen3.5-$M" \
      --batch-size 8 --temperature 0 --max-new-tokens 768
  done
done
echo "ALL DONE"
