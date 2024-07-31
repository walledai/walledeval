# walledeval/judge/llm/core.py

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from walledeval.llm import LLM
from walledeval.judge.core import Judge
from walledeval.prompts.core import PromptTemplate

__all__ = [
    "LLMasaJudge"
]

O = TypeVar('O') # Output Field
S = TypeVar('S') # Score Field
    

class LLMasaJudge(Judge[None, O, S], ABC, Generic[O, S]):
    def __init__(self,
                 name: str,
                 llm: LLM,
                 template: PromptTemplate = PromptTemplate(),
                 llm_instruct: bool = True):
        super().__init__(name)
        self._llm = llm
        self._template = template
        self.llm_instruct = llm_instruct
    
    def set_system_prompt(self,
                          system_prompt: str):
        self._llm = self._llm.set_system_prompt(system_prompt)
        
    def generate(self, response: str, **kwargs) -> str:
        prompt = self._template.format(
            response = response,
            prompt = "" if "response" in self._template.params else response,
            **kwargs
        )
        
        return self._llm.generate(
            prompt,
            instruct=self.llm_instruct,
            **kwargs
        )

    @abstractmethod
    def process_llm_output(self, response: str) -> O:
        pass

    def check(self, response: str, answer: None = None, **kwargs) -> O:
        llm_output = self.generate(response, **kwargs)

        output = self.process_llm_output(llm_output)

        return output