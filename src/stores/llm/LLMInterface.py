"""
LLM Interface — Abstract base class that all LLM providers must implement.
allows swapping providers (Ollama, OpenAI)
without changing any controller code.

"""

from abc import ABC, abstractmethod
from typing import List


class LLMInterface(ABC):

    @abstractmethod
    def set_generation_model(self, model: str):
        pass

    @abstractmethod
    def set_embedding_model(self, model: str, embedding_size: int):
        pass

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def construct_prompt(self, context: str, query: str) -> str:
        pass