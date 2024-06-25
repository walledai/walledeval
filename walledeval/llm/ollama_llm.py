# walledeval/llm/ollama.py

import ollama

from typing import Optional, Union

from walledeval.types import Message, Messages, LLMType
from walledeval.llm.core import LLM


class OllamaLLM(LLM):
    def __init__(self,
                 id: str,
                 system_prompt: str = "",
                 type: Optional[Union[LLMType, int]] = LLMType.NEITHER):
        super().__init__(
            id,
            system_prompt,
            type
        )