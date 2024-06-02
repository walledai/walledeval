# walledeval/llm/claude.py

from anthropic import Anthropic

from typing import Optional, Union

from walledeval.types import (
    Message, Messages, LLMType
)
from walledeval.llm.core import LLM

__all__ = [
    "Claude"
]


class Claude(LLM):
    def __init__(self,
                 model_id: str,
                 api_key: str,
                 system_prompt: str = "",
                 type: Optional[Union[LLMType, int]] = LLMType.NEITHER):
        super().__init__(
            model_id, system_prompt,
            type
        )
        self.client = Anthropic(api_key=api_key)

    @classmethod
    def opus(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "claude-3-opus-20240229",
            api_key, system_prompt
        )

    @classmethod
    def sonnet(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "claude-3-sonnet-20240229",
            api_key, system_prompt
        )

    @classmethod
    def haiku(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "claude-3-haiku-20240307",
            api_key, system_prompt
        )

    def chat(self,
             text: Messages,
             max_new_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        messages: list[dict[str, str]]
        if isinstance(text, str):
            messages = [{
                "role": "user",
                "content": text
            }]
        elif isinstance(text, list) and isinstance(text[0], Message):
            messages = [
                dict(msg)
                for msg in text
            ]
        elif isinstance(text, list) and isinstance(text[0], dict):
            messages = text
        else:
            raise TypeError("Unsupported format for parameter 'text'")

        system_prompt: str
        if messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]
        else:
            system_prompt = self.system_prompt

        message = self.client.messages.create(
            max_tokens=max_new_tokens,
            messages=messages,
            temperature=temperature,
            system=system_prompt,
            model=self.name
        )
        output = message.content[0].text
        return output

    def complete(self,
                 text: str,
                 max_new_tokens: int = 1024,
                 temperature: float = 0) -> str:
        message = self.client.messages.create(
            max_tokens=max_new_tokens,
            messages=[{
                "role": "assistant",
                "content": text
            }],
            temperature=temperature,
            system=self.system_prompt,
            model=self.name
        )
        output = message.content[0].text
        return output

    def generate(self,
                 text: Messages,
                 max_new_tokens: int = 1024,
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
