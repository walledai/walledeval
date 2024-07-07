# walledeval/llm/gemini.py

import google.generativeai as genai

from typing import Optional, Union

from walledeval.types import (
    Message, Messages, LLMType
)
from walledeval.llm.core import LLM

__all__ = [
    "Gemini"
]


class Gemini(LLM):
    def __init__(self,
                 model_id: str,
                 api_key: str,
                 system_prompt: str = "",
                 type: Optional[Union[LLMType, int]] = LLMType.NEITHER):
        super().__init__(
            model_id, system_prompt,
            type
        )
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name=self.name,
                                            system_instruction=system_prompt)

    @classmethod
    def gemini15flash(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "gemini-1.5-flash",
            api_key, system_prompt
        )

    @classmethod
    def gemini15pro(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "gemini-1.5-pro",
            api_key, system_prompt
        )

    @classmethod
    def gemini10pro(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "gemini-1.0-pro",
            api_key, system_prompt
        )

    def chat(self,
             text: Messages,
             max_new_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        messages: list[dict[str, list[str]]]
        if isinstance(text, str):
            messages = [{
                "role": "user",
                "parts": [text]
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
            
            # small work-around since Gemini library doesn't support system prompt at runtime
            client = genai.GenerativeModel(model_name=self.name,
                                           system_instruction=system_prompt)
        
        else:
            system_prompt = self.system_prompt
            client = self.client
        
        message = client.generate_content(
            messages,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_new_tokens,
                temperature=temperature
            )
        )
        
        output = message.text
        return output

    def complete(self,
                 text: str,
                 max_new_tokens: int = 1024,
                 temperature: float = 0) -> str:
        model=genai.GenerativeModel(model_name=self.name,
                                    system_instruction=self.system_prompt)
        message = model.generate_content(text,
            generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_new_tokens,
            temperature=temperature))
        output = message.text
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
