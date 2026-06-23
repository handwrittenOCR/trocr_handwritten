# Deploy a model on a Scaleway H100 with vLLM

Serve a Hugging Face model behind an OpenAI-compatible API on a Scaleway H100, then plug it into the OCR pipeline via the `vllm` provider. One model at a time — swap by restarting the script.

## Target models

Each fits on a single 80GB H100. Large dense models (32B/31B) auto-select a
shorter context in `serve_vllm.sh` so weights + KV cache fit.

| HF repo | Type | ~VRAM (bf16) | Gated |
|---|---|---|---|
| `google/gemma-4-26B-A4B-it` | Text MoE | ~52 GB | Yes (HF token) |
| `Qwen/Qwen3.6-35B-A3B` | VL MoE (Qwen3-Next) | ~70 GB | No |
| `Teklia/Qwen2.5-VL-7B-DAI-CReTDHI-RecordGold-ATR` | Vision-Language | ~16 GB | No |
| `deepseek-ai/DeepSeek-OCR-2` | OCR (3B) | ~6 GB | No |
| `Qwen/Qwen3-VL-2B-Instruct` | Vision-Language | ~5 GB | No |
| `Qwen/Qwen3-VL-8B-Instruct` | Vision-Language | ~17 GB | No |
| `Qwen/Qwen3-VL-32B-Instruct` | Vision-Language (dense) | ~64 GB ⚠️ tight | No |
| `google/gemma-4-12B-it` | Vision-Language | ~24 GB | Yes (HF token) |
| `google/gemma-4-31B-it` | Vision-Language (dense) | ~62 GB ⚠️ tight | Yes (HF token) |

The first three are already benchmarked; the rest are fine-tuning candidates.
For the ⚠️ tight models, if you still hit OOM use FP8 weights (e.g.
`Qwen/Qwen3-VL-32B-Instruct-FP8`) or lower `GPU_MEM_UTIL`.

## 1. Provision the instance

Create an **H100-1-80G** (Ubuntu 22.04+) on Scaleway. The default user is
`ubuntu`. With this `~/.ssh/config` alias:

```
Host trocr_handwritten
  Hostname <PUBLIC_IP>
  User ubuntu
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

connect and prepare the workspace:

```bash
ssh trocr_handwritten
sudo mkdir -p /workspace && sudo chown ubuntu:ubuntu /workspace && cd /workspace
```

## 2. Copy scripts and install

From your laptop:

```bash
scp trocr_handwritten/llm/deployment/*.sh trocr_handwritten:/workspace/
```

On the H100:

```bash
bash setup_h100.sh
```

Installs `uv`, a Python 3.12 venv in `/workspace/.venv`, `vllm`, `huggingface_hub`, and `cuda-toolkit-12-8` (required by FlashInfer JIT for Qwen3-Next).

## 3. Use `/scratch` for model weights

The root disk (~110 GB) is too small for the full catalogue (several hundred GB). Scaleway H100 instances ship with a 2.7 TB NVMe mounted at `/scratch`:

```bash
mkdir -p /scratch/hf_cache
rm -rf /workspace/hf_cache
ln -s /scratch/hf_cache /workspace/hf_cache
```

## 4. Download models

Gemma requires an HF token (gated). Download the whole catalogue, or just the
models you want to benchmark (vLLM also auto-downloads on first serve):

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
bash download_models.sh                                    # everything
bash download_models.sh Qwen/Qwen3-VL-8B-Instruct deepseek-ai/DeepSeek-OCR-2
```

## 5. Serve one model

Use `tmux` so the server survives SSH disconnects:

```bash
tmux new -s vllm
bash serve_vllm.sh Qwen/Qwen3-VL-8B-Instruct   # or any repo from the table
# Ctrl+b then d to detach
```

`serve_vllm.sh` picks per-model defaults (32B/31B get `ctx=16384`, `gpu-util=0.95`). Override with `MAX_MODEL_LEN`, `GPU_MEM_UTIL`, `DTYPE` env vars, e.g.:

```bash
GPU_MEM_UTIL=0.92 bash serve_vllm.sh Qwen/Qwen3-VL-32B-Instruct
```

The server listens on `http://0.0.0.0:8000/v1` and exposes the model under the last segment of the repo id (e.g. `Qwen3-VL-8B-Instruct`). Confirm with:

```bash
curl http://localhost:8000/v1/models
```

## 6. SSH tunnel from your laptop

```bash
ssh -N -L 8000:localhost:8000 trocr_handwritten
```

Leave it running.

## 7. Configure the client

Add to `.env`:

```
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
```

## 8. Run OCR

Run locally (the test data lives on your laptop), pointing at the tunnelled endpoint:

```bash
uv run python -m trocr_handwritten.llm.ocr \
    --provider vllm \
    --model Qwen3-VL-8B-Instruct \
    --input_dir data/ocr/test/images \
    --pattern "*/*.jpg" \
    --output_dir data/ocr/test/predictions/qwen3-vl-8b \
    --max_concurrent 16
```

Then score against the new (leak-free) test set:

```bash
uv run python -m trocr_handwritten.llm.metrics \
    --predictions data/ocr/test/predictions/qwen3-vl-8b \
    --labels data/ocr/test/labels
```

Flags:
- `--disable_thinking` strips `<think>…</think>` blocks — use for Qwen3 thinking/MoE models (`Qwen3.6-35B-A3B`); not needed for `*-Instruct` VL models, DeepSeek-OCR, or Gemma
- `--max_concurrent` — 4–8 for 26–35B, 16–32 for 2–8B
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
