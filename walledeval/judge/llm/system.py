# walledeval/judge/llm/system.py

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from walledeval.judge.llm.core import LLMasaJudge

__all__ = ["SystemLLMasaJudge"]


O = TypeVar("O") # Output Field
S = TypeVar('S') # Score Field


class SystemLLMasaJudge(LLMasaJudge[O, S]):
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