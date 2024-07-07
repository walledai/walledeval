# walledeval/llm/openai.py

import together

from typing import Optional, Union

from walledeval.types import (
    Message, Messages, LLMType
)
from walledeval.llm.core import LLM

__all__ = [
    "Together"
]


class Together(LLM):
    def __init__(self,
                 model: str,
                 api_key: str,
                 system_prompt: str = "",
                 type: Optional[Union[LLMType, int]] = LLMType.NEITHER):
        super().__init__(
            model, system_prompt,
            type
        )
        self.client = together.Together(api_key=api_key)

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

        if messages[0]["role"] != "system":
            messages.insert(0, {"role":"system", "content":self.system_prompt})

        message = self.client.chat.completions.create(
            model=self.name,
            max_tokens=max_new_tokens,
            messages=messages,
            temperature=temperature
        )
        output = message.choices[0].message.content
        return output

    def complete(self,
                 text: str,
                 max_new_tokens: int = 1024,
                 temperature: float = 0) -> str:
        message = self.client.completions.create(
            model=self.name,
            max_tokens=max_new_tokens,
            prompt=text,
            temperature=temperature
        )
        output = message.choices[0].text
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
