# walledeval/types/__init__.py

from pydantic import BaseModel

__all__ = [
    "MultipleChoiceQuestion"
]

class MultipleChoiceQuestion(BaseModel):
    question: str
    choices: list[str]
    answer: int = -1