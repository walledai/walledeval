# walledeval/judge/llm/core.py

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from walledeval.llm import LLM
from walledeval.judge.core import Judge

__all__ = [
    "LLMasaJudge"
]

O = TypeVar('O') # Output Field
S = TypeVar('S') # Score Field


class LLMasaJudge(Judge[None, O, S], ABC, Generic[O, S]):
    def __init__(self,
                 name: str,
                 llm: LLM):
        super().__init__(name)
        self._llm = llm
    
    def set_system_prompt(self,
                          system_prompt: str):
        self._llm = self._llm.set_system_prompt(system_prompt)
        
    def generate(self, response: str) -> str:
        return self._llm.generate(
            response,
            instruct=True
        )

    @abstractmethod
    def process_llm_output(self, response: str) -> O:
        pass

    def check(self, response: str, answer: None = None) -> O:
        llm_output = self.generate(response)

        output = self.process_llm_output(llm_output)

        return output