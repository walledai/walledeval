# walledeval/llm/claude.py

from anthropic import Anthropic

from typing import Optional, Union

from walledeval.types import Messages, LLMType
from walledeval.util import transform_messages
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
    def sonnet35(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "claude-3-5-sonnet-20240620",
            api_key, system_prompt
        )

    @classmethod
    def opus3(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "claude-3-opus-20240229",
            api_key, system_prompt
        )

    @classmethod
    def sonnet3(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "claude-3-sonnet-20240229",
            api_key, system_prompt
        )

    @classmethod
    def haiku3(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "claude-3-haiku-20240307",
            api_key, system_prompt
        )

    def chat(self,
             text: Messages,
             max_new_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        messages = transform_messages(text, self.system_prompt)
        system_prompt = messages.pop(0)["content"]

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
        