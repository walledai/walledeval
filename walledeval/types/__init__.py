# walledeval/types/__init__.py

from typing import Union
from pydantic import BaseModel

__all__ = [
    "MultipleChoiceQuestion", "MultipleResponseQuestion",
    "OpenEndedQuestion",
    "Log"
]


class Prompt(BaseModel):
    prompt: str


class Question(BaseModel):
    question: str


class OpenEndedQuestion(Question):
    pass


class MultipleChoiceQuestion(Question):
    question: str
    choices: list[str]
    answer: int = -1


class MultipleResponseQuestion(Question):
    question: str
    choices: list[str]
    answers: list[int] = []


class Log(BaseModel):
    """
    Basic Log representation in this system, consisting of
    - question from log
    - input to LLM
    - output from LLM
    - successful or not
    """
    question: Union[Question, Prompt]
    lm_input: str
    lm_output: str
    success: bool
