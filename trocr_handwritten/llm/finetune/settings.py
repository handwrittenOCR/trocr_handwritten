from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class LoraSettings(BaseModel):
    """Hyper-parameters and paths for LoRA fine-tuning a VLM for OCR."""

    data_root: str = Field(
        default="data/ocr",
        description="Root resolved against each manifest's relative image path.",
    )
    prompt_path: str = Field(
        default="config/ocr.prompt",
        description="Path to the OCR prompt template.",
    )
    epochs: float = Field(
        default=6.0, description="Max training epochs (early stop caps it)."
    )
    early_stopping_patience: int = Field(
        default=3, description="Stop after N evals with no eval_loss improvement."
    )
    early_stopping_threshold: float = Field(
        default=0.001, description="Minimum eval_loss delta to count as improvement."
    )
    lr: float = Field(default=1e-4, description="Peak learning rate.")
    lora_rank: int = Field(default=16, description="LoRA rank r.")
    lora_alpha_ratio: int = Field(
        default=2, description="lora_alpha = lora_rank * lora_alpha_ratio."
    )
    lora_dropout: float = Field(default=0.05, description="LoRA dropout.")
    target_modules: Union[str, List[str]] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Modules to wrap with LoRA adapters. The default list covers "
        "the LLM only (attention + MLP); 'all-linear' adds the vision tower and "
        "the merger, which PEFT resolves per-model (lm_head is excluded).",
    )
    batch_size: int = Field(default=16, description="Per-device train batch size.")
    grad_accum: int = Field(default=1, description="Gradient accumulation steps.")
    use_liger_kernel: bool = Field(
        default=True,
        description="Fuse the LM head + cross-entropy (Liger FLCE); never "
        "materializes the full logits tensor. ~3x less VRAM on large-vocab VLMs.",
    )
    group_by_length: bool = Field(
        default=True,
        description="Batch similarly-sized images together to cut padding waste.",
    )
    length_column_name: str = Field(
        default="length",
        description="Manifest column (image pixel count) used to group by length.",
    )
    dataloader_num_workers: int = Field(
        default=8,
        description="Parallel dataloader workers; preprocess images off the GPU path.",
    )
    image_min_pixels: int = Field(
        default=256 * 32 * 32,
        description="Lower bound on visual pixels (Qwen3-VL/3.5 use a 32x32 patch merge).",
    )
    image_max_pixels: int = Field(
        default=1280 * 32 * 32,
        description="Upper bound (~1280 visual tokens/image) to cap VRAM and OOM tails.",
    )
    seed: int = Field(default=42, description="Training and data-order seed.")
    warmup_ratio: float = Field(default=0.05, description="LR warmup ratio.")
    lr_scheduler_type: str = Field(default="cosine", description="LR scheduler type.")
    logging_steps: int = Field(default=5, description="Trainer logging interval.")
    torch_dtype: str = Field(default="bfloat16", description="Model load dtype.")
    attn_implementation: str = Field(
        default="sdpa",
        description="Attention kernel: 'sdpa' (fast, dependency-free) or 'eager'.",
    )
    bf16: bool = Field(default=True, description="Train in bfloat16.")
    gradient_checkpointing: bool = Field(
        default=True,
        description="Recompute activations in backward. Mandatory for these VLMs: "
        "without it the vision+LLM activations accumulate to tens of GB and OOM.",
    )
    save_strategy: str = Field(default="no", description="Checkpoint save strategy.")
    lora_bias: str = Field(default="none", description="LoRA bias mode.")
    task_type: str = Field(default="CAUSAL_LM", description="PEFT task type.")
    device_map: str = Field(
        default="cuda",
        description="'cuda'/'none' = single-GPU load (no accelerate dispatch hooks, "
        "correct for training); any other value is passed to from_pretrained.",
    )
    report_to: str = Field(default="none", description="Trainer logging integration.")
    remove_unused_columns: bool = Field(
        default=False, description="Keep manifest columns the collator needs."
    )


class DataSettings(BaseModel):
    """Paths and paliers for building fine-tuning manifests from our OCR splits."""

    ocr_dir: str = Field(default="data/ocr", description="Root OCR directory.")
    out_dir: str = Field(
        default="data/ocr/finetune", description="Where manifests are written."
    )
    paliers: List[int] = Field(
        default_factory=lambda: [60, 120, 240, 485, 972, 1400, 1997],
        description="Nested subset sizes for the learning curve (1997 = full train).",
    )
    eval_splits: List[str] = Field(
        default_factory=lambda: ["dev", "test"],
        description="Splits written verbatim as manifests.",
    )
    train_split: str = Field(
        default="train", description="Split subsampled into paliers."
    )
    seed: int = Field(default=42, description="Subset shuffle seed.")


class TekliaSettings(BaseModel):
    """Source and download parameters for the RecordGold pre-training corpus."""

    dataset: str = Field(
        default="Teklia/DAI-CReTDHI-RecordGold-ATR",
        description="HuggingFace dataset id.",
    )
    iiif_host: str = Field(
        default="iiif.teklia.com",
        description="Substring required in record_url to accept a row.",
    )
    region_label: str = Field(
        default="teklia",
        description="Region tag written in the manifest and image subpath.",
    )
    out_dir: str = Field(
        default="data/ocr/teklia", description="Output root for images + manifest."
    )
    split: str = Field(default="train", description="HF split to pull.")
    max_samples: Optional[int] = Field(
        default=None, description="Stratified cap, or None to keep all rows."
    )
    workers: int = Field(default=16, description="Concurrent download workers.")
    retries: int = Field(default=3, description="Per-image download retries.")
    timeout: int = Field(default=30, description="Per-request timeout (seconds).")
    user_agent: str = Field(
        default="dai-cretdhi-research/1.0", description="HTTP User-Agent."
    )
    seed: int = Field(default=42, description="Stratified sample seed.")


class CurveSettings(BaseModel):
    """Mapping from model keys to baseline predictions for the curve reporter."""

    ocr_dir: str = Field(default="data/ocr", description="Root OCR directory.")
    keys: List[str] = Field(
        default_factory=lambda: ["qwen2b", "qwen8b", "gemma12b"],
        description="Model keys to report.",
    )
    basepred: Dict[str, str] = Field(
        default_factory=lambda: {
            "qwen2b": "zs-qwen3-vl-2b",
            "qwen8b": "zs-qwen3-vl-8b",
            "gemma12b": "zs-gemma-4-12b",
        },
        description="Zero-shot (P0) test prediction dir per model key.",
    )
    paliers: List[int] = Field(
        default_factory=lambda: [0, 60, 120, 240, 485, 972, 1400, 1997],
        description="Paliers to report (0 = zero-shot).",
    )
