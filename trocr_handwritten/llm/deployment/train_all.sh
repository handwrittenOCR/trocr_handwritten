#!/usr/bin/env bash
set -u

# Train the four Qwen3.5 LoRA adapters on the full 1997 palier. RES selects the
# visual-token budget and TARGETS the LoRA coverage, so one script produces both
# the baseline configuration and the native + all-linear variant. Adapters are
# written to adapters/qwen3.5-$M-lora-1997$TAG, which infer_ft_vllm.sh reads
# through the same TAG.

cd ~/trocr_handwritten
source /workspace/.venv/bin/activate
export HF_HOME=/workspace/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=data/ocr/20260807-ft-data
MAN=$ROOT/manifests

RES=${RES:-native}
TARGETS=${TARGETS:-all-linear}
TAG=${TAG:--nativeall}
# Qwen3.5's gated-delta-rule linear attention keeps large per-chunk intermediates,
# so every size OOMs on an 80GB card without checkpointing -- including the 0.8B.
GC_MODELS=${GC_MODELS:-0.8B 2B 4B 9B 27B}

# BS/GA default per resolution mode but stay overridable: the 27B holds 55.6GB of
# bf16 weights, so its worst-case batch only fits at a much smaller BS.
if [ "$RES" = "capped" ]; then
    MINTOK=256 ; MAXTOK=1280 ; BS=${BS:-16} ; GA=${GA:-1}
else
    MINTOK=64 ; MAXTOK=16384 ; BS=${BS:-8} ; GA=${GA:-2}
fi

MODELS=${*:-0.8B 2B 4B 9B}
echo "models: $MODELS"
echo "resolution: $RES ($MINTOK-$MAXTOK visual tokens), targets: $TARGETS"
echo "batch: ${BS}x${GA}, tag: $TAG"

for M in $MODELS; do
    OUT="adapters/qwen3.5-$M-lora-1997$TAG"
    if [ -f "$OUT/adapter_model.safetensors" ]; then
        echo "skip $M (already trained at $OUT)"
        continue
    fi
    GC=0
    case " $GC_MODELS " in *" $M "*) GC=1 ;; esac
    echo "===== training Qwen3.5-$M (gradient_checkpointing=$GC) ====="
    python -m trocr_handwritten.llm.finetune.train_lora \
        --gradient-checkpointing "$GC" \
        --model "Qwen/Qwen3.5-$M" \
        --train "$MAN/train_1997.jsonl" \
        --eval "$MAN/dev.jsonl" \
        --data-root "$ROOT" \
        --min-visual-tokens "$MINTOK" --max-visual-tokens "$MAXTOK" \
        --target-modules $TARGETS \
        --batch-size "$BS" --grad-accum "$GA" \
        --output-dir "$OUT" 2>&1 | tee "$HOME/train-$M$TAG.log"
done
echo "ALL DONE"
