# walledeval/types/outputs.py

from typing import Union, Generic, TypeVar
from pydantic import BaseModel

__all__ = [
    "NumericScore",
    "Log",
    "Report"
]

I = TypeVar('I') # Input Field
O = TypeVar('O') # Output Field
S = TypeVar('S') # Score Field
A = TypeVar('A') # Aggregate Field


class MCQOutput(BaseModel):
    predicted: int
    correct: bool


NumericScore = Union[int, bool, float]


class Log(BaseModel, Generic[I, O, S]):
    idx: int
    input: I
    output: O
    score: S
    
class Report(BaseModel, Generic[I, O, S, A]):
    runs: int
    logs: list[Log[I, O, S]]
    aggregate: A
    