# walledeval/llm/gemini.py

import google.generativeai as genai

from typing import Optional, Union

from walledeval.types import Messages, LLMType
from walledeval.util import transform_messages
from walledeval.llm.core import LLM

__all__ = [
    "Gemini"
]


def transform_to_gemini(messages):
    messages_gemini = []
    for message in messages:
        if message['role'] == 'user':
            messages_gemini.append({'role': 'user', 'parts': [message['content']]})
        elif message['role'] == 'assistant':
            messages_gemini.append({'role': 'model', 'parts': [message['content']]})

    return messages_gemini


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
             temperature: float = 0.1) -> str:
        messages = transform_messages(text, self.system_prompt)

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
            
        messages = transform_to_gemini(messages)
        
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
                 temperature: float = 0.1) -> str:
        
        model=genai.GenerativeModel(model_name=self.name,
                                    system_instruction=self.system_prompt)
        
        message = model.generate_content(
            f"Continue writing: {text}",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_new_tokens,
                temperature=temperature
            )
        )
        output = message.text
        return output
