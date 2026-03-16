from .base import LLMProvider
from .ollama import OllamaLLM

__all__ = ["LLMProvider", "OllamaLLM"]

# Lazy imports for optional providers
def OpenAILLM(*args, **kwargs):
    from .openai_llm import OpenAILLM as _cls
    return _cls(*args, **kwargs)

def AnthropicLLM(*args, **kwargs):
    from .anthropic_llm import AnthropicLLM as _cls
    return _cls(*args, **kwargs)
