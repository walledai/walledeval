# walledeval/llm/core.py

from abc import ABC, abstractmethod

__all__ = ["LLM"]


class LLM(ABC):
    def __init__(self, name: str, system_prompt: str = ""):
        self.name = name
        self.system_prompt = system_prompt

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
        return self

    @abstractmethod
    def generate(self, text: str, max_new_tokens: int = 256) -> str:
        return ""

    def __call__(self, text: str, max_new_tokens: int = 256) -> str:
        return self.generate(text, max_new_tokens)
