# walledeval/dtypes/__init__.py

from typing import Union
from pydantic import BaseModel

from walledeval.dtypes.llm import LLMType
from walledeval.dtypes.message import Message, Messages
from walledeval.dtypes.inputs import (
    Prompt, Question,
    AutocompletePrompt,
    JudgeQuestioningPrompt,
    SystemAssistedPrompt,
    InjectionPrompt,
    MultipleChoiceQuestion,
    MultipleResponseQuestion,
    OpenEndedQuestion
)

__all__ = [
    "LLMType",
    "Message", "Messages",
    "Prompt", "Question",
    "AutocompletePrompt",
    "JudgeQuestioningPrompt",
    "SystemAssistedPrompt",
    "InjectionPrompt",
    "MultipleChoiceQuestion",
    "MultipleResponseQuestion",
    "OpenEndedQuestion",
    "Log"
]


class Log(BaseModel):
    """
    Basic Log representation in this system, consisting of
    - question from log
    - input to LLM
    - output from LLM
    - successful or not
    """
    question: Union[Question, Prompt]
    lm_input: str
    lm_output: str
    success: bool
