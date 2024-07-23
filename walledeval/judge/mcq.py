# walledeval/judge/mcq.py

import re, os
from pydantic import BaseModel

from walledeval.constants import DEFAULT_OPTIONS
from walledeval.judge.core import Judge
from walledeval.llm import LLM
from walledeval.prompts.core import PromptTemplate


class MCQOutput(BaseModel):
    predicted: int
    correct: bool


class MCQJudge(Judge[int, MCQOutput, bool]):
    def __init__(self, llm: LLM, template_file: str, options: list[str] = DEFAULT_OPTIONS):
        super().__init__("MCQJudge", llm)
        self.options = [str(option) for option in options]
        self.prompt_template = PromptTemplate.from_preset(os.path.join("prompts", "presets", "mcq", template_file))

    def check(self, response: str, answer: int) -> MCQOutput:
        # response is simply the output from the model
        prompt = self.prompt_template.format(response=response)
        llm_output = self.generate(prompt)

        response = re.sub(r'[^\w\s_]+', '', llm_output)
        if response.lower().startswith("answer"):
            response = response[6:].strip()
        if response.lower().startswith("boxed"):
            response = response[5:].strip()

        predicted = response[0].upper()
        if predicted not in self.options:
            return MCQOutput(
                predicted = -1,
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