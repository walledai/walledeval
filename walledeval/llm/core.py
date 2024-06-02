# walledeval/llm/core.py

from abc import ABC, abstractmethod
from typing import Optional, Union
from walledeval.types import LLMType, Messages

__all__ = ["LLM"]


class LLM(ABC):
    def __init__(self, name: str,
                 system_prompt: str = "",
                 type: Optional[Union[LLMType, int]] = LLMType.NEITHER):
        self.name = name
        self.system_prompt = system_prompt

        if isinstance(type, LLMType):
            self.instruct = type
        elif isinstance(type, int) and 0 <= type <= 2:
            self.instruct = LLMType(type)
        elif isinstance(type, int):
            raise ValueError(f"Type {type} not recognized.")
        else:
            raise TypeError(f"Value {type} not of type 'LLMType' or 'int'")

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
        return self

    @abstractmethod
    def chat(self,
             text: Messages,
             max_new_tokens: int = 256,
             temperature: float = 0.0) -> str:
        pass

    @abstractmethod
    def complete(self,
                 text: str,
                 max_new_tokens: int = 256,
                 temperature: float = 0.0) -> str:
        pass

    def generate(self,
                 text: Messages,
                 max_new_tokens: int = 256,
                 temperature: float = 0.0,
                 instruct: Optional[bool] = None) -> str:
        type = None
        if instruct is None:
            if self.type == LLMType.BASE:
                type = LLMType.BASE
            else:
                type = LLMType.INSTRUCT
        elif instruct:
            type = LLMType.INSTRUCT
        else:
            type = LLMType.BASE

        if type == LLMType.INSTRUCT:
            return self.chat(
                text,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        else:
            if not isinstance(text, str):
                raise ValueError("Unsupported type for input 'text'")
            return self.complete(
                text,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

    def __call__(self,
                 text: Messages,
                 max_new_tokens: int = 256,
                 temperature: float = 0.0,
                 instruct: Optional[bool] = None) -> str:
        return self.generate(text, max_new_tokens,
                             temperature, instruct)
