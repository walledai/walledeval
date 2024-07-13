# walledeval/pipeline/core.py

from pydantic import BaseModel

from walledeval.prompts import BasePromptTemplate
from walledeval.llm import LLM
from walledeval.judge import Judge


class Pipeline:
    def __init__(self,
                 *args,
                 answer_field: str = None):
        self.steps = args
        self.answer_field = answer_field
        
    def forward(self, samples: list[BaseModel]):
        for sample in samples:
            logs = dict(input = sample)
            
            for step in self.steps:
                if isinstance(step, BasePromptTemplate):
                    logs["prompt"] = step.format(logs["input"])
                
                elif isinstance(step, LLM):
                    logs["response"] = step(logs["prompt"])
                
                elif isinstance(step, Judge):
                    if self.answer_field is None:
                        logs["judge_output"], logs["score"] = step(logs["response"])
                    else:
                        logs["judge_output"], logs["score"] = step(
                            logs["response"],
                            getattr(logs["sample"], self.answer_field)
                        )
                else:
                    print(type(step))