# walledeval/types/__init__.py

from typing import Union
from enum import Enum
from pydantic import BaseModel

__all__ = [
    "LLMType",
    "Message", "Messages",
    "MultipleChoiceQuestion",
    "MultipleResponseQuestion",
    "OpenEndedQuestion",
    "Log"
]


class LLMType(Enum):
    BASE = 0
    INSTRUCT = 1
    NEITHER = 2


class Message(BaseModel):
    role: str
    content: str


Messages = Union[
    list[Message],
    list[dict[str, str]],
    str
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
