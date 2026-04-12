from .base import LLMProvider
from .ollama import OllamaLLM

__all__ = ["LLMProvider", "OllamaLLM"]


def __getattr__(name: str):
    """Lazy imports for optional providers — proper module-level __getattr__."""
    if name == "OpenAILLM":
        from .openai_llm import OpenAILLM

        return OpenAILLM
    if name == "AnthropicLLM":
        from .anthropic_llm import AnthropicLLM

        return AnthropicLLM
    if name == "GemmaEdgeLLM":
        from .gemma_edge import GemmaEdgeLLM

        return GemmaEdgeLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
