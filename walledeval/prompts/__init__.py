# walledeval/prompts/__init__.py

from walledeval.prompts.core import (
    AbstractPromptTemplate,
    BasePromptTemplate, BaseConversationTemplate,
    PromptTemplate
)
from walledeval.prompts.mcq import (
    MultipleChoiceTemplate, FewShotMCQTemplate
)
from walledeval.prompts.prompt import (
    SystemAssistedTemplate, DefaultPromptTemplate,
    QuestionTemplate
)

__all__ = [
    "AbstractPromptTemplate",
    "BasePromptTemplate",
    "BaseConversationTemplate",
    "PromptTemplate",
    "DefaultPromptTemplate",
    "QuestionTemplate",
    "SystemAssistedTemplate",
    "MultipleChoiceTemplate",
    "FewShotMCQTemplate"
]
