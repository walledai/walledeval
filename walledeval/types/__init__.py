# walledeval/types/__init__.py

from pydantic import BaseModel

__all__ = [
    "MultipleChoiceQuestion", "MultipleResponseQuestion",
    "OpenEndedQuestion", 
]

class OpenEndedQuestion(BaseModel):
    question: str
    

class MultipleChoiceQuestion(BaseModel):
    question: str
    choices: list[str]
    answer: int = -1
    
class MultipleResponseQuestion(BaseModel):
    question: str
    choices: list[str]
    answers: list[int] = []
    