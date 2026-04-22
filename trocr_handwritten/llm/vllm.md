# Deploy a model on a Scaleway H100 with vLLM

Serve a Hugging Face model behind an OpenAI-compatible API on a Scaleway H100, then plug it into the OCR pipeline via the `vllm` provider. One model at a time — swap by restarting the script.

## Target models

| HF repo | Type | Gated |
|---|---|---|
| `google/gemma-4-26B-A4B-it` | Text MoE | Yes (HF token) |
| `Qwen/Qwen3.6-35B-A3B` | VL MoE (Qwen3-Next) | No |
| `Teklia/Qwen2.5-VL-7B-DAI-CReTDHI-RecordGold-ATR` | Vision-Language | No |

## 1. Provision the instance

Create an **H100-1-80G** (Ubuntu 22.04+) on Scaleway. Verify your SSH key is attached to the account, then:

```bash
ssh root@<PUBLIC_IP>
mkdir -p /workspace && cd /workspace
```

## 2. Copy scripts and install

From your laptop:

```bash
scp trocr_handwritten/llm/deployment/*.sh root@<PUBLIC_IP>:/workspace/
```

On the H100:

```bash
bash setup_h100.sh
```

Installs `uv`, a Python 3.12 venv in `/workspace/.venv`, `vllm`, `huggingface_hub`, and `cuda-toolkit-12-8` (required by FlashInfer JIT for Qwen3-Next).

## 3. Use `/scratch` for model weights

The root disk (~110 GB) is too small for the three models (~130 GB total). Scaleway H100 instances ship with a 2.7 TB NVMe mounted at `/scratch`:

```bash
mkdir -p /scratch/hf_cache
rm -rf /workspace/hf_cache
ln -s /scratch/hf_cache /workspace/hf_cache
```

## 4. Download models

Gemma requires an HF token (gated):

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
bash download_models.sh
```

## 5. Serve one model

Use `tmux` so the server survives SSH disconnects:

```bash
tmux new -s vllm
bash serve_vllm.sh Qwen/Qwen3.6-35B-A3B   # or any of the 3 repos
# Ctrl+b then d to detach
```

The server listens on `http://0.0.0.0:8000/v1` and exposes the model under the last segment of the repo id (e.g. `Qwen3.6-35B-A3B`). Confirm with:

```bash
curl http://localhost:8000/v1/models
```

## 6. SSH tunnel from your laptop

```bash
ssh -N -L 8000:localhost:8000 root@<PUBLIC_IP>
```

Leave it running.

## 7. Configure the client

Add to `.env`:

```
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
```

## 8. Run OCR

```bash
uv run python -m trocr_handwritten.llm.ocr \
    --provider vllm \
    --model Qwen3.6-35B-A3B \
    --input_dir data/ocr/test/images \
    --pattern "*/*.jpg" \
    --output_dir data/ocr/test/predictions/qwen3.6-35b \
    --max_concurrent 8 \
    --disable_thinking
```

Flags:
- `--disable_thinking` strips Qwen3 `<think>…</think>` reasoning blocks (big speedup + cleaner output)
- `--max_concurrent` — 4–8 for 26–35B, 16–32 for 7B
- `--request_timeout` — defaults to 120s

## 9. Switch models

```bash
tmux attach -t vllm
# Ctrl+C
bash serve_vllm.sh <another_repo>
```

Weights stay cached in `/scratch/hf_cache`.

## Troubleshooting

**`No space left on device`** — see step 3 (symlink `/workspace/hf_cache` to `/scratch`).

**`Could not find nvcc`** (Qwen3-Next) — install the CUDA toolkit:

```bash
sudo apt-get install -y cuda-toolkit-12-8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
```

**`CUDA out of memory`** — lower `--gpu-memory-utilization` to `0.85` or `--max-model-len` to `16384` in `serve_vllm.sh`.

**`max_tokens=... > max_model_len`** — the client default is 16 000; pass `--max_tokens <N>` if needed.

**Model not found from the client** — check the exact served name via `curl /v1/models` and use it in `--model`.

## Cost

vLLM is self-hosted, so cost is time-based:

```
cost_usd = elapsed_hours × VLLM_HOURLY_EUR × EUR_TO_USD
```

Defaults (`2.8 €/h`, `1.08 EUR→USD`) live in [cost_tracker.py](../utils/cost_tracker.py). Tokens are still logged in `summary.json` but not billed.
