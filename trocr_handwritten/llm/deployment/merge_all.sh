#!/usr/bin/env bash
set -u

# Fold each LoRA adapter into its base weights so vLLM can serve the vision-tower
# adaptation, which it drops for Qwen3.5 when the adapter is loaded as LoRA.
# Runs on CPU by default, so it can overlap with a GPU job.

cd ~/trocr_handwritten
source /workspace/.venv/bin/activate
export HF_HOME=/workspace/hf_cache

TAG=${TAG:--nativeall}
DEVICE=${DEVICE:-cpu}
OUTROOT=${OUTROOT:-merged}

MODELS=${*:-0.8B 2B 4B 9B}
echo "models: $MODELS | tag: $TAG | device: $DEVICE"

for M in $MODELS; do
    A="adapters/qwen3.5-$M-lora-1997$TAG"
    OUT="$OUTROOT/qwen3.5-$M$TAG"
    if [ ! -f "$A/adapter_model.safetensors" ]; then
        echo "skip $M (no adapter at $A)"
        continue
    fi
    if [ -f "$OUT/config.json" ]; then
        echo "skip $M (already merged at $OUT)"
        continue
    fi
    echo "===== merging Qwen3.5-$M ====="
    python -m trocr_handwritten.llm.finetune.merge_lora \
        --base "Qwen/Qwen3.5-$M" --adapter "$A" \
        --output-dir "$OUT" --device "$DEVICE" 2>&1 | tee "$HOME/merge-$M$TAG.log"
done
echo "ALL DONE"
