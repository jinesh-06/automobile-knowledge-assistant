from abc import ABC, abstractmethod
from typing import List, Dict

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_answer(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        """
        Generate an answer using the LLM provider.
        """
        pass
