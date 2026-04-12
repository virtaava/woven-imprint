"""Provider factories — create LLM and embedding instances from config."""

from __future__ import annotations

from .config import WovenConfig, get_config
from .llm.base import LLMProvider
from .embedding.base import EmbeddingProvider


def create_llm(cfg: WovenConfig | None = None) -> LLMProvider:
    """Create an LLM provider based on configuration.

    Args:
        cfg: Optional config override. Uses global config if not provided.

    Returns:
        An LLMProvider instance for the configured provider.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    if cfg is None:
        cfg = get_config()

    provider = cfg.llm.llm_provider.lower()

    if provider == "ollama":
        from .llm.ollama import OllamaLLM

        return OllamaLLM(
            model=cfg.llm.model,
            base_url=cfg.llm.ollama_host,
            num_ctx=cfg.llm.num_ctx,
            timeout=cfg.llm.timeout,
        )
    elif provider == "openai":
        from .llm.openai_llm import OpenAILLM

        return OpenAILLM(
            model=cfg.llm.model,
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.base_url,
        )
    elif provider == "anthropic":
        from .llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(
            model=cfg.llm.model,
            api_key=cfg.llm.api_key,
        )
    elif provider == "gemma_edge":
        from .llm.gemma_edge import GemmaEdgeLLM

        return GemmaEdgeLLM(
            model=cfg.llm.model,
            base_url=cfg.llm.base_url,
            timeout=cfg.llm.timeout,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. Supported: ollama, openai, anthropic, gemma_edge"
        )


def create_embedding(cfg: WovenConfig | None = None) -> EmbeddingProvider:
    """Create an embedding provider based on configuration.

    Args:
        cfg: Optional config override. Uses global config if not provided.

    Returns:
        An EmbeddingProvider instance for the configured provider.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    if cfg is None:
        cfg = get_config()

    provider = cfg.llm.embedding_provider.lower()

    if provider == "ollama":
        from .embedding.ollama import OllamaEmbedding

        return OllamaEmbedding(
            model=cfg.llm.embedding_model,
            base_url=cfg.llm.ollama_host,
        )
    elif provider == "openai":
        from .embedding.openai_embedding import OpenAIEmbedding

        return OpenAIEmbedding(
            model=cfg.llm.embedding_model,
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.base_url,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider!r}. Supported: ollama, openai")
