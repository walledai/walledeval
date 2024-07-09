# walledeval/types/__init__.py

from typing import Union
from pydantic import BaseModel

from walledeval.types.llm import LLMType
from walledeval.types.message import Message, Messages
from walledeval.types.inputs import (
    Prompt, Question,
    AutocompletePrompt,
    JudgeQuestioningPrompt,
    SystemAssistedPrompt,
    InjectionPrompt,
    MultipleChoiceQuestion,
    MultipleResponseQuestion,
    OpenEndedQuestion
)
from walledeval.types.data import Range

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
    "Range",
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
