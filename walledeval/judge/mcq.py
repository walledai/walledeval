# walledeval/judge/mcq.py

import re
from pydantic import BaseModel

from walledeval.constants import DEFAULT_OPTIONS
from walledeval.judge.core import Judge


class MCQOutput(BaseModel):
    predicted: int
    correct: bool


class MCQJudge(Judge[int, MCQOutput, bool]):
    def __init__(self, options: list[str] = DEFAULT_OPTIONS, unknown_answer: int = -1):
        super().__init__("MCQJudge")
        self.options = [str(option) for option in options]
        self.unknown_answer = unknown_answer

    def check(self, response: str, answer: int) -> MCQOutput:
        # response is simply the output from the model

        response = re.sub(r'[^\w]+', '', response)
        if response.lower().startswith("answer"):
            response = response[6:].strip()
        if response.lower().startswith("boxed"):
            response = response[5:].strip()

        predicted = response[0].upper()
        if predicted not in self.options:
            return MCQOutput(
                predicted = self.unknown_answer,
                correct = False
            )
        else:
            predicted = self.options.index(predicted)
            return MCQOutput(
                predicted = predicted,
                correct = (predicted == answer)
            )

    def score(self, output: MCQOutput) -> bool:
        return output.correct