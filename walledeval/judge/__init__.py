# walledeval/judge/__init__.py

from walledeval.judge.core import Judge
from walledeval.judge.mcq import MCQJudge
from walledeval.judge.llm import LLMasaJudge

__all__ = [
    "Judge",
    "MCQJudge",
    "LLMasaJudge"
]
