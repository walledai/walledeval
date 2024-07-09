# walledeval/llm/together.py

import together

from typing import Optional, Union

from walledeval.types import Messages, LLMType
from walledeval.util import transform_messages
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
             temperature: float = 0.1) -> str:
        messages = transform_messages(text, self.system_prompt)

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
                 temperature: float = 0.1) -> str:
        message = self.client.completions.create(
            model=self.name,
            max_tokens=max_new_tokens,
            prompt=text,
            temperature=temperature
        )
        output = message.choices[0].text
        return output
