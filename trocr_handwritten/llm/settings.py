from pydantic import BaseModel, Field
from typing import Optional, Literal
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
        default=128000,
        description="Maximum number of tokens to generate",
    )
    prompt_path: str = Field(
        default="config/ocr.prompt",
        description="Path to the OCR prompt template",
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
