# walledeval/tester.py

from walledeval.types import Log
from walledeval.benchmark.core import MultipleChoiceBenchmark
from walledeval.prompts import MultipleChoiceTemplate, FewShotMCQTemplate
from walledeval.llm import LLM
from walledeval.judge import MCQJudge

__all__ = [
    "mcq"
]


def mcq(benchmark: MultipleChoiceBenchmark,
        llm: LLM,
        judge: MCQJudge,
        prompt_template: MultipleChoiceTemplate = MultipleChoiceTemplate.default(),
        num_samples: int = 20) -> tuple[float, list[Log]]:
    
    questions = benchmark.sample(num_samples)
    
    total_samples = len(questions)
    
    logs = []
    
    success_rate = 0
    
    for question in questions:
        input = prompt_template.format(question)
        response = llm.generate(input)
        success = judge.check(response, question.answer)
        
        logs.append(Log(
            question = question,
            lm_input = input,
            lm_output = response,
            success = success
        ))
        
        if success:
            success_rate += 1
    
    return success_rate / total_samples, logs