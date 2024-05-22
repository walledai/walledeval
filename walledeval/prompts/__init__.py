# walledeval/prompts/__init__.py

from walledeval.prompts.core import PromptTemplate
from walledeval.prompts.mcq import (
    MultipleChoiceTemplate, FewShotMCQTemplate
)

__all__ = [
    "PromptTemplate",
    "MultipleChoiceTemplate",
    "FewShotMCQTemplate"
]














