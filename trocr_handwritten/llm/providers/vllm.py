from pathlib import Path
from typing import Tuple

from openai import OpenAI, AsyncOpenAI

from trocr_handwritten.llm.base import LLMProvider
from trocr_handwritten.llm.settings import LLMSettings


class VLLMProvider(LLMProvider):
    """Self-hosted vLLM provider using OpenAI-compatible API."""

    def __init__(self, settings: LLMSettings):
        """
        Initialize the vLLM provider.

        Args:
            settings: LLM configuration settings. Requires vllm_base_url.
        """
        super().__init__(settings)
        if not self.settings.vllm_base_url:
            raise ValueError("vllm_base_url must be set when using the vllm provider")
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the OpenAI-compatible client pointing at the vLLM server."""
        api_key = self.settings.vllm_api_key or "EMPTY"
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.settings.vllm_base_url,
            timeout=self.settings.request_timeout,
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.settings.vllm_base_url,
            timeout=self.settings.request_timeout,
        )

    def _build_messages(self, image_path: Path, prompt: str) -> list:
        """Build the messages payload for the API call."""
        base64_image = self._encode_image_base64(image_path)
        mime_type = self._get_mime_type(image_path)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ],
            }
        ]

    def _build_kwargs(self, model: str, messages: list) -> dict:
        """Build kwargs for chat.completions.create, including vLLM extras."""
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if self.settings.disable_thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }
        return kwargs

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove any <think>...</think> block left by the model."""
        if not text:
            return text
        if "</think>" in text:
            return text.split("</think>", 1)[1].lstrip("\n ")
        return text

    @staticmethod
    def _extract(response) -> Tuple[str, int, int]:
        """Pull stripped text and token usage out of a completion response."""
        text = VLLMProvider._strip_thinking(response.choices[0].message.content)
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        return text, input_tokens, output_tokens

    def _retry_kwargs(self, model: str, messages: list) -> dict:
        """Kwargs for an empty-response retry, forcing sampling to break greedy EOS."""
        kwargs = self._build_kwargs(model, messages)
        kwargs["temperature"] = max(self.settings.temperature, 0.5)
        return kwargs

    def _call_api(self, model: str, messages: list) -> Tuple[str, int, int]:
        """Make a synchronous API call, retrying once with sampling if empty."""
        text, in_tok, out_tok = self._extract(
            self.client.chat.completions.create(**self._build_kwargs(model, messages))
        )
        if not text or not text.strip():
            rtext, rin, rout = self._extract(
                self.client.chat.completions.create(
                    **self._retry_kwargs(model, messages)
                )
            )
            in_tok += rin
            out_tok += rout
            if rtext and rtext.strip():
                text = rtext
        return text, in_tok, out_tok

    async def _call_api_async(self, model: str, messages: list) -> Tuple[str, int, int]:
        """Make an async API call, retrying once with sampling if empty.

        Greedy decoding (temperature 0) can emit an immediate stop token on hard
        or very long crops, yielding an empty transcription. A single sampled
        retry recovers most of these.
        """
        text, in_tok, out_tok = self._extract(
            await self.async_client.chat.completions.create(
                **self._build_kwargs(model, messages)
            )
        )
        if not text or not text.strip():
            rtext, rin, rout = self._extract(
                await self.async_client.chat.completions.create(
                    **self._retry_kwargs(model, messages)
                )
            )
            in_tok += rin
            out_tok += rout
            if rtext and rtext.strip():
                text = rtext
        return text, in_tok, out_tok
