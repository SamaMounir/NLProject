"""
Bonus 1: The LLM Factory Pattern (+5%)
changing only in the .env file
"""

import os
from stores.llm.LLMInterface import LLMInterface
from stores.llm.provider.OpenAIProvider import OpenAIProvider


class LLMFactory:

    @staticmethod
    def create() -> LLMInterface:
        provider_name = os.getenv("LLM_PROVIDER", "openai").lower()

        if provider_name == "openai":
            # Works with Ollama locally OR real OpenAI remotely
            # Just change OPENAI_API_BASE and OPENAI_API_KEY in .env
            return OpenAIProvider()

        if provider_name == "ollama":
            # Explicit Ollama option — same provider, different config label
            return OpenAIProvider()  # Ollama is OpenAI-compatible

        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            f"Supported: 'openai', 'ollama'"
        )