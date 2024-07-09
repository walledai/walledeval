# walledeval/judge/core.py

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

A = TypeVar('A')
O = TypeVar('O')
S = TypeVar('S')


class Judge(ABC, Generic[A, O, S]):
    """
    Abstract class for any Judge

    Notable functions:
    - Judge.check(self, response: str, answer: A (optional)) -> O :
    run output through Judge
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def check(self, response: str, answer: A) -> O:
        """Check function for the Judge

        Args:
            response (str): Response from an LLM that needs to be judged
            answer (A, optional): Answer expected from LLM (eg option for MCQ)

        Returns:
            O: Output from Judge
        """
        pass
    
    @abstractmethod
    def score(self, output: O) -> S:
        """Scoring function for the Judge

        Args:
            output (O): Output from check method

        Returns:
            S: Score from Judge
        """

    def __call__(self, response: str, answer: A) -> tuple[O, S]:
        output = self.check(response, answer)
        score = self.score(output)
        return output, score
