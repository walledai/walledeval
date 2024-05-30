# walledeval/types/inputs.py

from pydantic import BaseModel

__all__ = [
    "Prompt", "Question",
    "AutocompleteTask",
    "MultipleChoiceQuestion",
    "MultipleResponseQuestion",
    "OpenEndedQuestion"
]


class Prompt(BaseModel):
    prompt: str
    

class AutocompleteTask(Prompt):
    pass


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