from .base import EmbeddingProvider
from .ollama import OllamaEmbedding

__all__ = ["EmbeddingProvider", "OllamaEmbedding"]


# Lazy import for optional provider
def OpenAIEmbedding(*args, **kwargs):
    from .openai_embedding import OpenAIEmbedding as _cls

    return _cls(*args, **kwargs)
