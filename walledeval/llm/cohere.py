# walledeval/llm/cohere.py

import cohere

from typing import Optional, Union

from walledeval.types import (
    Message, Messages, LLMType
)
from walledeval.llm.core import LLM

__all__ = [
    "Cohere"
]


class Cohere(LLM):
    def __init__(self,
                 model_id: str,
                 api_key: str,
                 system_prompt: str = "",
                 type: Optional[Union[LLMType, int]] = LLMType.NEITHER):
        super().__init__(
            model_id, system_prompt,
            type
        )
        self.client = cohere.Client(api_key=api_key)

    @classmethod
    def commandrplus(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "command-r-plus",
            api_key, system_prompt
        )

    @classmethod
    def commandr(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "command-r",
            api_key, system_prompt
        )

    @classmethod
    def command(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "command",
            api_key, system_prompt
        )
    
    @classmethod
    def commandlight(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "command-light",
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

        if messages[0]["role"] != "system":
            messages.insert(0, {"role":"SYSTEM", "content":self.system_prompt})
        else:
            messages[0]["role"] = "SYSTEM"

        for counter in len(messages):
            if messages[counter]["role"] == "user":
                messages[counter]["role"] = "USER"
            elif messages[counter]["role"] == "assistant":
                messages[counter]["role"] = "CHATBOT"
        
        if messages[-1]["role"] == "USER":
            prompt = messages[-1]["content"]
            messages = messages[:-1]
        else:
            raise ValueError("Last message should be a user message")
        message = self.client.chat(
            chat_history=messages,
            max_tokens=max_new_tokens,
            message=prompt,
            temperature=temperature,
            model=self.name
        )
        output = message.text
        return output

    def complete(self,
                 text: str,
                 max_new_tokens: int = 1024,
                 temperature: float = 0) -> str:
        messages = [{"role":"SYSTEM", "content":self.system_prompt}]
        
        message = self.client.chat(
            chat_history=messages,
            max_tokens=max_new_tokens,
            message=f"Continue writing: {text}",
            temperature=temperature,
            model=self.name
        )
        output = message.text
        return output

