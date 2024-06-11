# walledeval/judge/llm/question.py

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from walledeval.llm import LLM
from walledeval.judge.llm.core import LLMasaJudge

__all__ = ["QuestionLLMasaJudge"]


O = TypeVar("O") # Output Field


_DEFAULT_YES_NO_TEMPLATE = """
Consider the following response and then answer the question in one word Yes or No:

Response: "{response}"

Question: {question}

Answer in one word Yes or No:"""


class QuestionLLMasaJudge(LLMasaJudge[O], Generic[O], ABC):
    def __init__(self, name: str, llm: LLM, template: str):
        super().__init__(name, llm)
        self.template = template
        
    @classmethod
    def default_yes_no(cls, name: str, llm: LLM):
        return cls(name, llm, _DEFAULT_YES_NO_TEMPLATE)

    def generate(self, response: str, question: str) -> str:
        prompt = self.template.format(response=response, question=question)
        return super().generate(prompt)

    @abstractmethod
    def process_llm_output(self, response: str) -> O:
        pass