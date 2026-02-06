"""AI Provider implementations.

Available providers:
- ``OllamaProvider`` — local LLM inference via Ollama REST API.
"""

from genesis_engine.ai.providers.ollama_provider import OllamaProvider

__all__ = ["OllamaProvider"]
