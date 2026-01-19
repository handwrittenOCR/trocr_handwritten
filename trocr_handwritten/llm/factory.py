from trocr_handwritten.llm.base import LLMProvider
from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.llm.providers.openai import OpenAIProvider
from trocr_handwritten.llm.providers.gemini import GeminiProvider
from trocr_handwritten.llm.providers.mistral import MistralProvider


PROVIDERS = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "mistral": MistralProvider,
}


def get_provider(settings: LLMSettings) -> LLMProvider:
    """
    Factory function to instantiate the appropriate LLM provider.

    Args:
        settings: LLM configuration settings.

    Returns:
        An instance of the appropriate LLMProvider subclass.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider_class = PROVIDERS.get(settings.provider)
    if provider_class is None:
        raise ValueError(
            f"Unsupported provider: {settings.provider}. "
            f"Available providers: {list(PROVIDERS.keys())}"
        )
    return provider_class(settings)
