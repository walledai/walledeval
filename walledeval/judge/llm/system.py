# walledeval/judge/llm/system.py

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from walledeval.judge.llm.core import LLMasaJudge

__all__ = ["SystemLLMasaJudge"]


O = TypeVar("O") # Output Field


class SystemLLMasaJudge(LLMasaJudge[O], ABC, Generic[O]):
    def generate(self, response: str, system: str) -> str:
        return self._llm.generate([
            {
                "role": "system",
                "content": system
            },
            {
                "role": "system",
                "content": response
            }
        ], instruct=True)

    @abstractmethod
    def process_llm_output(self, response: str) -> O:
        pass