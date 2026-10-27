from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal
import os
from dotenv import load_dotenv

load_dotenv()


class LLMSettings(BaseModel):
    """Configuration settings for LLM-based OCR."""

    provider: Literal["openai", "gemini", "mistral", "vllm"] = Field(
        default="gemini",
        description="LLM provider to use for OCR",
    )
    model_name: str = Field(
        default="gemini-2.0-flash",
        description="Model name to use for inference",
    )
    openai_api_key: Optional[str] = Field(
        default=os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key",
    )
    google_api_key: Optional[str] = Field(
        default=os.getenv("GOOGLE_API_KEY"),
        description="Google API key for Gemini",
    )
    mistral_api_key: Optional[str] = Field(
        default=os.getenv("MISTRAL_API_KEY"),
        description="Mistral AI API key",
    )
    vllm_base_url: Optional[str] = Field(
        default=os.getenv("VLLM_BASE_URL"),
        description="Base URL of a self-hosted vLLM OpenAI-compatible server "
        "(e.g. http://h100.example.com:8000/v1)",
    )
    vllm_api_key: Optional[str] = Field(
        default=os.getenv("VLLM_API_KEY"),
        description="API key for the vLLM server. Defaults to 'EMPTY' if unset.",
    )
    reasoning_effort: Optional[str] = Field(
        default=None,
        description="Reasoning effort for Gemini thinking models: 'low', 'medium', 'high'. "
        "None disables thinking.",
    )
    temperature: float = Field(
        default=0.0,
        description="Temperature for text generation",
    )
    max_tokens: int = Field(
        default=16000,
        description="Maximum number of tokens to generate",
    )
    request_timeout: float = Field(
        default=120.0,
        description="Per-request HTTP timeout in seconds.",
    )
    disable_thinking: bool = Field(
        default=False,
        description="For vLLM-served Qwen3/Qwen3-Next models, disable the "
        "<think>...</think> reasoning block via chat_template_kwargs.",
    )
    prompt_path: str = Field(
        default="config/ocr.prompt",
        description="Path to the OCR prompt template",
    )


class MetricsSettings(BaseModel):
    """Configuration for OCR metric scoring over the normalization ladder."""

    pred_ext: str = Field(default=".md", description="Prediction file extension.")
    label_ext: str = Field(default=".txt", description="Reference file extension.")
    outlier_length_ratio: float = Field(
        default=10.0,
        description="hyp/ref length ratio above which an item is a failure (CER=1.0).",
    )
    levels: Dict[str, str] = Field(
        default_factory=lambda: {
            "N0": "raw",
            "N1": "lowercased",
            "N2": "canonical",
            "N3": "ascii",
        },
        description="Normalization ladder legend (paper 2 codes -> transform name).",
    )
    primary_levels: List[str] = Field(
        default_factory=lambda: ["N0", "N2"],
        description="Co-primary levels reported in the body (N0 raw, N2 canonical).",
    )


class StatsSettings(BaseModel):
    """Configuration for register-cluster bootstrap CIs and paired contrasts."""

    bootstrap_samples: int = Field(
        default=10_000, description="Number of bootstrap replicates B."
    )
    seed: int = Field(default=42, description="Bootstrap RNG seed.")
    default_level: str = Field(
        default="N2", description="Normalization level used when none is given."
    )
    mde_factor: float = Field(
        default=2.8,
        description="MDE multiplier on SE (80% power, alpha=.05, two-sided).",
    )


class OCRSettings(BaseModel):
    """Configuration settings for OCR processing pipeline."""

    input_dir: str = Field(
        default="data/processed/images",
        description="Root directory containing processed images",
    )
    image_pattern: str = Field(
        default="*/*/*.jpg",
        description="Glob pattern to find images relative to input_dir",
    )
    output_extension: str = Field(
        default=".md",
        description="Extension for output transcription files",
    )
    llm_settings: LLMSettings = Field(
        default_factory=LLMSettings,
        description="LLM configuration settings",
    )
