# walledeval/judge/core.py

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

A = TypeVar('A')
O = TypeVar('O')


class Judge(ABC, Generic[A, O]):
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

    def __call__(self, response: str, answer: A) -> O:
        return self.check(response, answer)
