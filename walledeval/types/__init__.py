# walledeval/types/__init__.py

from pydantic import BaseModel

__all__ = [
    "MultipleChoiceQuestion", "MultipleResponseQuestion",
    "OpenEndedQuestion", 
]

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
    question: Question
    lm_input: str
    lm_output: str
    success: bool