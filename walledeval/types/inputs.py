# walledeval/types/inputs.py

from pydantic import BaseModel

__all__ = [
    "Prompt", "Question",
    "AutocompletePrompt",
    "JudgeQuestioningPrompt",
    "SystemAssistedPrompt",
    "InjectionPrompt",
    "MultipleChoiceQuestion",
    "MultipleResponseQuestion",
    "OpenEndedQuestion"
]


class Prompt(BaseModel):
    prompt: str
    

class AutocompletePrompt(Prompt):
    pass
    
    
class JudgeQuestioningPrompt(Prompt):
    judge_question: str


class SystemAssistedPrompt(Prompt):
    system: str


class InjectionPrompt(SystemAssistedPrompt, JudgeQuestioningPrompt):
    pass


class Question(BaseModel):
    question: str


class OpenEndedQuestion(Question):
    pass


class MultipleChoiceQuestion(Question):
    # question: str
    choices: list[str]
    answer: int = -1


class MultipleResponseQuestion(Question):
    # question: str
    choices: list[str]
    answers: list[int] = []