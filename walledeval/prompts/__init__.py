# walledeval/prompts/__init__.py

from walledeval.prompts.core import PromptTemplate
from walledeval.prompts.mcq import (
    MultipleChoiceQuestion, MultipleChoiceTemplate, FewShotMCQTemplate
)

__all__ = [
    "MultipleChoiceQuestion",
    "PromptTemplate",
    "MultipleChoiceTemplate",
    "FewShotMCQTemplate"
]














