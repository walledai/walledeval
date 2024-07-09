# walledeval/llm/openai.py

import openai

from typing import Optional, Union

from walledeval.types import Messages, LLMType
from walledeval.util import transform_messages
from walledeval.llm.core import LLM

__all__ = [
    "AzureOpenAI"
]


class AzureOpenAI(LLM):
    def __init__(self,
                 model_id: str,
                 api_key: str,
                 api_version: str,
                 azure_endpoint: str,
                 system_prompt: str = "",
                 type: Optional[Union[LLMType, int]] = LLMType.NEITHER):
        super().__init__(
            model_id, system_prompt,
            type
        )
        
        # taken from OpenAI Python SDK documentation
        self.client = openai.AzureOpenAI(
            api_key=api_key,
            # https://learn.microsoft.com/azure/ai-services/openai/reference#rest-api-versioning
            api_version=api_version,
            # https://learn.microsoft.com/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
            azure_endpoint=azure_endpoint
        )

    def chat(self,
             text: Messages,
             max_new_tokens: int = 1024,
             temperature: float = 0.1) -> str:
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
                 temperature: float = 0.1) -> str:
        text = f"Continue writing: {text}"
        
        return self.chat(text, max_new_tokens=max_new_tokens, temperature=temperature)

