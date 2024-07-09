# walledeval/llm/cohere.py

import cohere

from typing import Optional, Union

from walledeval.types import Messages, LLMType
from walledeval.util import transform_messages
from walledeval.llm.core import LLM

__all__ = [
    "Cohere"
]


def convert_to_cohere(messages: list[dict[str, str]]):
    for counter in len(messages):
        messages[counter]["role"] = messages[counter]["role"].upper().replace("ASSISTANT", "CHATBOT")
    
    prompt: str
    if messages[-1]["role"] == "USER":
        prompt = messages[-1]["content"]
        messages = messages[:-1]
    else:
        raise ValueError("Last message should be a user message")
        
    return prompt, messages


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
    def command_rplus(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "command-r-plus",
            api_key, system_prompt
        )

    @classmethod
    def command_r(cls, api_key: str, system_prompt: str = ""):
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
    def command_light(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "command-light",
            api_key, system_prompt
        )

    def chat(self,
             text: Messages,
             max_new_tokens: int = 1024,
             temperature: float = 0.1) -> str:
        messages = transform_messages(text)
        
        prompt, messages = convert_to_cohere(messages)
        
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
                 temperature: float = 0.1) -> str:
        messages = [{"role":"SYSTEM", "content": self.system_prompt}]
        
        message = self.client.chat(
            chat_history=messages,
            max_tokens=max_new_tokens,
            message=f"Continue writing: {text}",
            temperature=temperature,
            model=self.name
        )
        output = message.text
        return output

