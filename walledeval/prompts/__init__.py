# walledeval/prompts/__init__.py

from walledeval.prompts.core import (
    BasePromptTemplate, PromptTemplate,
    QuestionTemplate
)
from walledeval.prompts.mcq import (
    MultipleChoiceTemplate, FewShotMCQTemplate
)

__all__ = [
    "BasePromptTemplate",
    "PromptTemplate",
    "QuestionTemplate",
    "MultipleChoiceTemplate",
    "FewShotMCQTemplate"
]
