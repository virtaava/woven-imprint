from .base import EmbeddingProvider
from .ollama import OllamaEmbedding

__all__ = ["EmbeddingProvider", "OllamaEmbedding"]


def __getattr__(name: str):
    """Lazy import for optional provider."""
    if name == "OpenAIEmbedding":
        from .openai_embedding import OpenAIEmbedding

        return OpenAIEmbedding
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
