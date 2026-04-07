import asyncio
from pathlib import Path
from typing import Tuple

from google import genai
from google.genai import types

from trocr_handwritten.llm.base import LLMProvider
from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the native google-genai SDK."""

    def __init__(self, settings: LLMSettings):
        super().__init__(settings)
        self.actual_model_name: str = settings.model_name
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the native Gemini client."""
        self.native_client = genai.Client(api_key=self.settings.google_api_key)

    def _build_messages(self, image_path: Path, prompt: str) -> list:
        """Build native SDK content parts for an image + prompt."""
        image_bytes = image_path.read_bytes()
        mime_type = self._get_mime_type(image_path)
        return [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ]

    def _get_config(self, model: str) -> types.GenerateContentConfig:
        """Build generation config with appropriate thinking level."""
        thinking_level = None
        if "gemini-3" in model:
            thinking_level = "LOW"
        elif "gemini-2.5" in model:
            thinking_level = "OFF"

        thinking_config = None
        if thinking_level:
            thinking_config = types.ThinkingConfig(thinking_level=thinking_level)

        return types.GenerateContentConfig(
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_tokens,
            thinking_config=thinking_config,
        )

    @staticmethod
    def _extract_tokens(response) -> Tuple[int, int, int]:
        """Extract input, output, and thinking tokens from native SDK response."""
        um = response.usage_metadata
        if not um:
            return 0, 0, 0

        input_tokens = um.prompt_token_count or 0
        output_tokens = um.candidates_token_count or 0
        thinking_tokens = um.thoughts_token_count or 0

        if thinking_tokens > 0:
            logger.info(
                f"Tokens: input={input_tokens}, output={output_tokens}, "
                f"thinking={thinking_tokens}"
            )

        return input_tokens, output_tokens, thinking_tokens

    def _call_api(self, model: str, messages: list) -> Tuple[str, int, int, int]:
        """Make a synchronous API call via native SDK."""
        config = self._get_config(model)
        response = self.native_client.models.generate_content(
            model=model,
            contents=messages,
            config=config,
        )
        actual_model = getattr(response, "model_version", None)
        if actual_model and actual_model != model:
            logger.info(f"Requested model '{model}' redirected to '{actual_model}'")
            self.actual_model_name = actual_model
        text = response.text
        input_tokens, output_tokens, thinking_tokens = self._extract_tokens(response)
        return text, input_tokens, output_tokens, thinking_tokens

    async def _call_api_async(
        self, model: str, messages: list
    ) -> Tuple[str, int, int, int]:
        """Make an async API call (runs sync call in thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call_api, model, messages)

    @staticmethod
    def _convert_schema(openai_schema: dict) -> dict:
        """Convert an OpenAI-format JSON schema to native google-genai schema format."""
        if not openai_schema:
            return {}

        result = {}
        schema_type = openai_schema.get("type")

        if isinstance(schema_type, list):
            non_null = [t for t in schema_type if t != "null"]
            schema_type = non_null[0] if non_null else "string"
        result["type"] = schema_type.upper() if schema_type else "STRING"

        if "description" in openai_schema:
            result["description"] = openai_schema["description"]

        if "enum" in openai_schema:
            result["enum"] = [v for v in openai_schema["enum"] if v is not None]

        if "properties" in openai_schema:
            result["properties"] = {
                k: GeminiProvider._convert_schema(v)
                for k, v in openai_schema["properties"].items()
            }

        if "required" in openai_schema:
            result["required"] = openai_schema["required"]

        return result

    @staticmethod
    def _openai_tools_to_native(tools: list) -> list:
        """Convert OpenAI-format tool definitions to native google-genai FunctionDeclaration."""
        native_tools = []
        for tool in tools:
            fn = tool["function"]
            native_tools.append(
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=fn["name"],
                            description=fn.get("description", ""),
                            parameters=GeminiProvider._convert_schema(
                                fn.get("parameters", {})
                            ),
                        )
                    ]
                )
            )
        return native_tools

    @staticmethod
    def _convert_messages(messages: list) -> Tuple[str, list]:
        """Split OpenAI-format messages into system_instruction + native contents."""
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            else:
                contents.append({"role": role, "parts": [{"text": content}]})
        return system_instruction, contents

    async def _call_text_api_async(
        self, model: str, messages: list, tools: list = None, tool_choice: str = "auto"
    ) -> Tuple[str, int, int, int]:
        """Make an async text API call with optional function calling via native SDK."""
        base_config = self._get_config(model)
        system_instruction, contents = self._convert_messages(messages)

        config_kwargs = {
            "temperature": base_config.temperature,
            "max_output_tokens": base_config.max_output_tokens,
            "thinking_config": base_config.thinking_config,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            native_tools = self._openai_tools_to_native(tools)
            config_kwargs["tools"] = native_tools
            if tool_choice == "required":
                config_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY")
                )

        config = types.GenerateContentConfig(**config_kwargs)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.native_client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            ),
        )
        actual_model = getattr(response, "model_version", None)
        if actual_model and actual_model != model:
            logger.info(f"Requested model '{model}' redirected to '{actual_model}'")
            self.actual_model_name = actual_model

        input_tokens, output_tokens, thinking_tokens = self._extract_tokens(response)

        if tools:
            candidate = response.candidates[0] if response.candidates else None
            text = None
            if candidate:
                for part in candidate.content.parts:
                    if part.function_call:
                        import json as _json

                        text = _json.dumps(dict(part.function_call.args))
                        break
            return text, input_tokens, output_tokens, thinking_tokens

        text = response.text
        return text, input_tokens, output_tokens, thinking_tokens
