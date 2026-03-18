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
