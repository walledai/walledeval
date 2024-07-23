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
                 template_path: str):
        super().__init__(name)
        self._llm = llm
        self.prompt_template = PromptTemplate.from_template(template_path)
    
    def set_system_prompt(self,
                          system_prompt: str):
        self._llm = self._llm.set_system_prompt(system_prompt)
        
    def generate(self, prompt_template: str, response: str) -> str:
        prompt = self.prompt_template.format(response=response)
        return self._llm.generate(
            prompt,
            instruct=True
        )

    @abstractmethod
    def process_llm_output(self, response: str) -> O:
        pass

    def check(self, response: str, answer: None = None) -> O:
        llm_output = self.generate(response)

        output = self.process_llm_output(llm_output)

        return output