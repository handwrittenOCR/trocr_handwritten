#!/usr/bin/env bash
set -euo pipefail

# Launch a vLLM OpenAI-compatible server for a single model on one 80GB H100.
# Usage:
#   bash serve_vllm.sh <hf_repo_id> [port]
#
# Examples:
#   bash serve_vllm.sh Qwen/Qwen3-VL-8B-Instruct
#   bash serve_vllm.sh google/gemma-4-12B-it 8000
#   bash serve_vllm.sh deepseek-ai/DeepSeek-OCR-2
#
# Per-model defaults (ctx, gpu-util) are auto-selected so large dense models
# (32B/31B) fit on a single card. Override with MAX_MODEL_LEN / GPU_MEM_UTIL /
# DTYPE env vars. Set SERVE_TEXT_ONLY=1 for pure-text models (no image slots).

if [ $# -lt 1 ]; then
    echo "Usage: $0 <hf_repo_id> [port]"
    exit 1
fi

MODEL="$1"
PORT="${2:-8000}"

: "${HF_HOME:=/workspace/hf_cache}"
export HF_HOME
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Qwen3-Next (e.g. Qwen3.6-35B-A3B) builds FlashInfer JIT kernels at startup and
# needs nvcc on PATH; a bare tmux/ssh shell does not have it.
if [ -d /usr/local/cuda-12.8 ]; then
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH="$CUDA_HOME/bin:$PATH"
fi

if [ -f /workspace/.venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source /workspace/.venv/bin/activate
fi

# Defaults tuned for a single 80GB H100. Large dense models get a shorter
# context and higher memory utilization so weights + KV cache fit.
DEF_MAX_LEN=32768
DEF_GPU_UTIL=0.90
case "$MODEL" in
    *32B*|*31B*|*35B*)
        DEF_MAX_LEN=16384
        DEF_GPU_UTIL=0.95
        ;;
    *DeepSeek-OCR*)
        # DeepSeek-OCR caps at 8192 positions; a higher ctx fails validation.
        DEF_MAX_LEN=8192
        ;;
esac

MAX_MODEL_LEN="${MAX_MODEL_LEN:-$DEF_MAX_LEN}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-$DEF_GPU_UTIL}"
DTYPE="${DTYPE:-bfloat16}"

# Vision-Language models need image input slots; pure-text models do not.
EXTRA_ARGS=()
if [ -z "${SERVE_TEXT_ONLY:-}" ]; then
    EXTRA_ARGS+=(--limit-mm-per-prompt '{"image": 4}')
fi

# Higher input resolution for tall/dense crops. MAX_PIXELS (Qwen-VL) raises the
# pixel budget before downscaling; PAN_AND_SCAN=1 (Gemma) tiles instead of
# shrinking. They are mutually exclusive per model.
if [ -n "${MAX_PIXELS:-}" ]; then
    EXTRA_ARGS+=(--mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS}}")
elif [ "${PAN_AND_SCAN:-0}" = "1" ]; then
    EXTRA_ARGS+=(--mm-processor-kwargs '{"do_pan_and_scan": true}')
fi

# Serve a LoRA adapter alongside the base model, exposed as model name "ft".
if [ -n "${LORA_PATH:-}" ]; then
    EXTRA_ARGS+=(--enable-lora --max-lora-rank "${LORA_RANK:-16}"
        --lora-modules "ft=${LORA_PATH}")
fi

SERVED_NAME="$(basename "$MODEL")"

echo "==> Serving $MODEL as '$SERVED_NAME' on :$PORT"
echo "==> ctx=$MAX_MODEL_LEN gpu-util=$GPU_MEM_UTIL dtype=$DTYPE"
echo "==> OpenAI endpoint will be: http://<host>:$PORT/v1"

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --download-dir "$HF_HOME" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}"
