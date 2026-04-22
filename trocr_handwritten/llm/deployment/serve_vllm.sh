#!/usr/bin/env bash
set -euo pipefail

# Launch a vLLM OpenAI-compatible server for a single model.
# Usage:
#   bash serve_vllm.sh <hf_repo_id> [port]
#
# Examples:
#   bash serve_vllm.sh google/gemma-4-26B-A4B-it
#   bash serve_vllm.sh Qwen/Qwen3.6-35B-A3B 8000
#   bash serve_vllm.sh Teklia/Qwen2.5-VL-7B-DAI-CReTDHI-RecordGold-ATR

if [ $# -lt 1 ]; then
    echo "Usage: $0 <hf_repo_id> [port]"
    exit 1
fi

MODEL="$1"
PORT="${2:-8000}"

: "${HF_HOME:=/workspace/hf_cache}"
export HF_HOME
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ -f /workspace/.venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source /workspace/.venv/bin/activate
fi

# Vision-Language models need image input slots. Non-VL models do not.
# Defaults to enabling multimodal; export SERVE_TEXT_ONLY=1 to disable for
# pure-text models.
EXTRA_ARGS=()
if [ -z "${SERVE_TEXT_ONLY:-}" ]; then
    EXTRA_ARGS+=(--limit-mm-per-prompt '{"image": 4}')
fi

# MoE models benefit from enforce-eager off and longer ctx; defaults kept simple.
SERVED_NAME="$(basename "$MODEL")"

echo "==> Serving $MODEL as '$SERVED_NAME' on :$PORT"
echo "==> OpenAI endpoint will be: http://<host>:$PORT/v1"

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --download-dir "$HF_HOME" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}"
