"""Tests for provider factory functions."""

import os
from unittest.mock import patch

import pytest

from woven_imprint.config import WovenConfig, LLMConfig, reload_config
from woven_imprint.providers import create_llm, create_embedding


class TestCreateLLM:
    def test_default_creates_ollama(self):
        cfg = WovenConfig()
        llm = create_llm(cfg)
        from woven_imprint.llm.ollama import OllamaLLM

        assert isinstance(llm, OllamaLLM)

    def test_ollama_explicit(self):
        cfg = WovenConfig(llm=LLMConfig(llm_provider="ollama"))
        llm = create_llm(cfg)
        from woven_imprint.llm.ollama import OllamaLLM

        assert isinstance(llm, OllamaLLM)

    def test_unknown_provider_raises(self):
        cfg = WovenConfig(llm=LLMConfig(llm_provider="nonexistent"))
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm(cfg)

    def test_openai_provider(self):
        pytest.importorskip("openai")
        cfg = WovenConfig(
            llm=LLMConfig(
                llm_provider="openai",
                model="gpt-4o-mini",
                api_key="test-key",
            )
        )
        llm = create_llm(cfg)
        from woven_imprint.llm.openai_llm import OpenAILLM

        assert isinstance(llm, OpenAILLM)

    def test_anthropic_provider(self):
        pytest.importorskip("anthropic")
        cfg = WovenConfig(
            llm=LLMConfig(
                llm_provider="anthropic",
                model="claude-haiku-4-5-20251001",
                api_key="test-key",
            )
        )
        llm = create_llm(cfg)
        from woven_imprint.llm.anthropic_llm import AnthropicLLM

        assert isinstance(llm, AnthropicLLM)

    def test_gemma_edge_provider(self):
        cfg = WovenConfig(
            llm=LLMConfig(
                llm_provider="gemma_edge",
                model="gemma-3n",
                base_url="http://127.0.0.1:8766",
            )
        )
        llm = create_llm(cfg)
        from woven_imprint.llm.gemma_edge import GemmaEdgeLLM

        assert isinstance(llm, GemmaEdgeLLM)


class TestCreateEmbedding:
    def test_default_creates_ollama(self):
        cfg = WovenConfig()
        emb = create_embedding(cfg)
        from woven_imprint.embedding.ollama import OllamaEmbedding

        assert isinstance(emb, OllamaEmbedding)

    def test_unknown_embedding_provider_raises(self):
        cfg = WovenConfig(llm=LLMConfig(embedding_provider="nonexistent"))
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding(cfg)

    def test_openai_embedding_provider(self):
        pytest.importorskip("openai")
        cfg = WovenConfig(
            llm=LLMConfig(
                embedding_provider="openai",
                embedding_model="text-embedding-3-small",
                api_key="test-key",
            )
        )
        emb = create_embedding(cfg)
        from woven_imprint.embedding.openai_embedding import OpenAIEmbedding

        assert isinstance(emb, OpenAIEmbedding)


class TestEnvVarOverride:
    def test_llm_provider_env_var(self):
        with patch.dict(os.environ, {"WOVEN_IMPRINT_LLM_PROVIDER": "openai"}):
            cfg = reload_config()
            assert cfg.llm.llm_provider == "openai"

    def test_enforce_consistency_env_var(self):
        with patch.dict(os.environ, {"WOVEN_IMPRINT_ENFORCE_CONSISTENCY": "false"}):
            cfg = reload_config()
            assert cfg.character.enforce_consistency is False


def test_embedding_base_url_split(monkeypatch):
    pytest.importorskip("openai")
    from woven_imprint.config import reload_config
    from woven_imprint.providers import create_embedding

    monkeypatch.setenv("WOVEN_IMPRINT_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("WOVEN_IMPRINT_BASE_URL", "http://chat:1/v1")
    monkeypatch.setenv("WOVEN_IMPRINT_EMBEDDING_BASE_URL", "http://embed:2/v1")
    monkeypatch.setenv("WOVEN_IMPRINT_API_KEY_LLM", "sk-test")
    try:
        cfg = reload_config()
        emb = create_embedding(cfg)
        assert "embed:2" in str(emb.client.base_url)
    finally:
        for var in (
            "WOVEN_IMPRINT_EMBEDDING_PROVIDER",
            "WOVEN_IMPRINT_BASE_URL",
            "WOVEN_IMPRINT_EMBEDDING_BASE_URL",
            "WOVEN_IMPRINT_API_KEY_LLM",
        ):
            monkeypatch.delenv(var)
        reload_config()
