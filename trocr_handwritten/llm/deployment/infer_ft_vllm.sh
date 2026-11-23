#!/usr/bin/env bash
set -u

# Serve each fine-tuned LoRA adapter with vLLM and run dev+test inference through
# `llm.ocr` -- the same engine and client used for the zero-shot baselines, so FT
# and ZSL numbers stay comparable. mm-processor-kwargs reproduces the training
# resolution cap (1280 visual tokens).

# Concurrent runs of this script kill each other's servers via free_gpu, so take
# an exclusive lock and bail out if another instance already holds it.
LOCK=/tmp/infer_ft_vllm.lock
if ! command -v flock >/dev/null 2>&1; then
    echo "flock not found -- this script runs on the GPU box (Linux), not on macOS"
    exit 1
fi
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another instance is already running (lock: $LOCK) -- aborting"
    exit 1
fi

cd ~/trocr_handwritten
source /workspace/.venv/bin/activate
export HF_HOME=/workspace/hf_cache
ROOT=data/ocr/20260807-ft-data
PORT=8000
MINP=262144    # 256 * 32 * 32
MAXP=1310720   # 1280 * 32 * 32

# RES=capped reproduces the training cap (1280 visual tokens). RES=native passes
# no mm-processor-kwargs, so vLLM falls back to the base repo's
# preprocessor_config.json -- byte-identical to the zero-shot baselines.
RES=${RES:-capped}
# TAG selects which adapter variant to serve and tags the prediction dir with it.
TAG=${TAG:-}
if [ "$RES" = "capped" ]; then
    MM_ARGS=(--mm-processor-kwargs "{\"min_pixels\":$MINP,\"max_pixels\":$MAXP}")
    SUF=""
else
    MM_ARGS=()
    SUF="-native"
fi
echo "resolution: $RES"

free_gpu() {
    # vLLM claims 90% of the card at startup and aborts if anything else holds
    # VRAM. Killing by name is not enough: vLLM V1 spawns a separate EngineCore
    # process, so orphans are killed by the PIDs the driver itself reports.
    for _ in $(seq 1 10); do
        pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
        [ -z "$pids" ] && break
        echo "killing GPU pids: $pids"
        for p in $pids; do kill -9 "$p" 2>/dev/null; done
        sleep 3
    done
    sleep 2
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
}

# Kill the driving scripts first: killing only the python child makes the parent
# loop spawn the next model straight away.
pkill -9 -f infer_all.sh 2>/dev/null
pkill -9 -f infer_hf 2>/dev/null
pkill -9 -f finetune.train_lora 2>/dev/null
free_gpu

MODELS=${*:-0.8B 2B 4B 9B}
echo "models: $MODELS"

for M in $MODELS; do
    # MERGED=1 serves a checkpoint whose adapter is already folded into the
    # weights. Required for Qwen3.5: as LoRA, vLLM silently drops every visual
    # module ("no matching PunicaWrapper is found ... will be ignored").
    if [ "${MERGED:-0}" = "1" ]; then
        SRC="$PWD/${MERGEDROOT:-merged}/qwen3.5-$M$TAG"
        LORA_ARGS=()
        SERVED=ft
        if [ ! -f "$SRC/config.json" ]; then
            echo "skip $M (no merged model at $SRC)"
            continue
        fi
    else
        SRC="Qwen/Qwen3.5-$M"
        A="$PWD/adapters/qwen3.5-$M-lora-1997$TAG"
        LORA_ARGS=(--enable-lora --max-lora-rank 16 --lora-modules "ft=$A")
        SERVED="base-$M"
        if [ ! -d "$A" ]; then
            echo "skip $M (no adapter at $A)"
            continue
        fi
    fi

    echo "===== serving Qwen3.5-$M (merged=${MERGED:-0}) ====="
    free_gpu
    vllm serve "$SRC" --host 0.0.0.0 --port "$PORT" \
        --served-model-name "$SERVED" \
        "${LORA_ARGS[@]}" \
        --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 \
        --trust-remote-code --limit-mm-per-prompt '{"image":1}' \
        "${MM_ARGS[@]}" \
        --max-num-seqs 256 \
        --download-dir /workspace/hf_cache \
        > "$HOME/serve-$M.log" 2>&1 &

    ready=0
    for _ in $(seq 1 150); do
        if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 5
    done
    if [ "$ready" -eq 0 ]; then
        echo "SERVER FAILED for $M -- see ~/serve-$M.log"
        tail -20 "$HOME/serve-$M.log"
        pkill -9 -f "vllm serve" 2>/dev/null
        free_gpu
        continue
    fi
    echo "served ids: $(curl -s "http://localhost:$PORT/v1/models" | tr ',' '\n' | grep '"id"')"

    for SP in dev test; do
        echo "===== $M $SP ====="
        python -m trocr_handwritten.llm.ocr --provider vllm \
            --model ft --vllm_base_url "http://localhost:$PORT/v1" \
            --input_dir "$ROOT/$SP/images" --pattern "*/*.jpg" \
            --output_dir "$ROOT/$SP/predictions/ft-qwen3.5-$M${TAG:-$SUF}" \
            --repeat 5 --temperature 0 --max_tokens 1024 --max_concurrent 32 \
            --disable_thinking
    done

    pkill -9 -f "vllm serve" 2>/dev/null
    free_gpu
done
echo "ALL DONE"
