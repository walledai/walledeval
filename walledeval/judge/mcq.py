# walledeval/judge/mcq.py

import re

from walledeval.constants import DEFAULT_OPTIONS
from walledeval.judge.core import Judge

class MCQJudge(Judge[int, bool]):
    def __init__(self, options: list[str] = DEFAULT_OPTIONS):
        super().__init__("MCQJudge")
        self.options = options
    
    def check(self, response: str, answer: int) -> bool:
        # response is simply the output from the model
        
        response = re.sub(r'[^\w\s_]+', '', response)
        if response.lower().startswith("answer"):
            response = response[6:].strip()
        if response.lower().startswith("boxed"):
            response = response[5:].strip()
        
        predicted = response[0].upper()
        if predicted not in self.options:
            return False
        else:
            return self.options.index(predicted) == answer
            
