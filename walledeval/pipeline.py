# walledeval/pipeline.py

from walledeval.types import Log
from walledeval.data import MultipleChoiceDataset
from walledeval.prompts import MultipleChoiceTemplate
from walledeval.llm import LLM
from walledeval.judge import MCQJudge

__all__ = [
    "mcq"
]


def mcq(benchmark: MultipleChoiceDataset,
        llm: LLM,
        judge: MCQJudge,
        prompt_template: MultipleChoiceTemplate = MultipleChoiceTemplate.default(),
        num_samples: int = 20) -> tuple[float, list[Log]]:

    questions = benchmark.sample(num_samples)

    total_samples = len(questions)

    logs = []

    success_rate = 0

    for question in questions:
        lm_input = prompt_template.format(question)
        lm_output = llm.generate(lm_input)
        success = judge.check(lm_output, question.answer)

        logs.append(Log(
            question=question,
            lm_input=lm_input,
            lm_output=lm_output,
            success=success
        ))

        if success:
            success_rate += 1

    return success_rate / total_samples, logs
