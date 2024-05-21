# walledeval/llm/core.py

from abc import ABC, abstractmethod

__all__ = ["LLM"]

class LLM(ABC):
    def __init__(self, name: str, system_prompt: str = ""):
        self.name = name
        self.system_prompt = system_prompt
    
    @abstractmethod
    def generate(self, text: str) -> str:
        return ""
    
    def __call__(self, text: str) -> str:
        return self.generate(text)