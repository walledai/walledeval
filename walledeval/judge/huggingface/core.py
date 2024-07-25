from typing import Generic, TypeVar, List, Union
from walledeval.judge.core import Judge
from transformers import pipeline
from pathlib import Path

O = TypeVar('O') # Judge raw output
S = TypeVar('S') # Score

class HFTextClassificationJudge(Judge[None, O, S], Generic[O, S]):
    def __init__(self, id: str, **kwargs):
        super().__init__(id)
        self.pipeline = pipeline("text-classification", model=id, trust_remote_code=True, **kwargs)

    def check(self, response: str, answer: None = None) -> O:
        return self.pipeline(response)[0]['label']