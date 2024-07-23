# walledeval/judge/llm/question.py

from typing import TypeVar

from walledeval.judge.llm.core import LLMasaJudge

__all__ = ["QuestionLLMasaJudge"]


O = TypeVar("O") # Output Field
S = TypeVar('S') # Score Field


class QuestionLLMasaJudge(LLMasaJudge[O, S]):
    def generate(self, response: str, question: str) -> str:
        return super().generate(response, question = question)
    
    def check(self, response: str, answer: None = None, question: str = "") -> O:
        llm_output = self.generate(response, question = question)
        
        output = self.process_llm_output(llm_output)

        return output
