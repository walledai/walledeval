# walledeval/llm/openai.py

import groq

from typing import Optional, Union

from walledeval.types import Messages, LLMType
from walledeval.util import transform_messages
from walledeval.llm.core import LLM

__all__ = [
    "Groq"
]


class Groq(LLM):
    def __init__(self,
                 model_id: str,
                 api_key: str,
                 system_prompt: str = "",
                 type: Optional[Union[LLMType, int]] = LLMType.NEITHER):
        super().__init__(
            model_id, system_prompt,
            type
        )
        self.client = groq.Groq(api_key=api_key)

    def chat(self,
             text: Messages,
             max_new_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        messages = transform_messages(text, self.system_prompt)

        message = self.client.chat.completions.create(
            max_tokens=max_new_tokens,
            messages=messages,
            temperature=temperature,
            model=self.name
        )
        output = message.choices[0].message.content
        return output

    def complete(self,
                 text: str,
                 max_new_tokens: int = 1024,
                 temperature: float = 0) -> str:
        text = f"Continue writing: {text}"
        
        return self.chat(text, max_new_tokens=max_new_tokens, temperature=temperature)
