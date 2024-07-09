# walledeval/llm/llama.py

import llama_cpp
from typing import Optional, Union

from walledeval.types import Messages, LLMType
from walledeval.util import transform_messages
from walledeval.llm.core import LLM


__all__ = [
    "Llama"
]


class Llama(LLM):
    def __init__(self,
                 name: str,
                 model: llama_cpp.Llama,
                 system_prompt: str = "",
                 type: Optional[Union[LLMType, int]] = LLMType.NEITHER,
                 **kwargs):
        super().__init__(name, system_prompt, type)
        self.model = model
    
    @classmethod
    def from_pretrained(cls,
                        repo_id: str,
                        filename: str,
                        system_prompt: str = "",
                        type: Optional[Union[LLMType, int]] = LLMType.NEITHER,
                        **kwargs):
        llm = llama_cpp.Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            **kwargs
        )
        
        return cls(
            name = f"{repo_id}/{filename}",
            model = llm,
            system_prompt = system_prompt,
            type = type
        )
    
    @classmethod
    def from_file(cls,
                  model_path: str,
                  system_prompt: str = "",
                  type: Optional[Union[LLMType, int]] = LLMType.NEITHER,
                  **kwargs):
        llm = llama_cpp.Llama(
            model_path=model_path,
            **kwargs
        )
        raise cls(
            name=model_path.split("/")[-1].split("\\")[-1].split(".")[0],
            model=llm,
            system_prompt=system_prompt,
            type=type
        )
    
    def chat(self,
             text: Messages,
             max_new_tokens: int = 512,
             temperature: float = 0.1) -> str:
        messages = transform_messages(text, self.system_prompt)
        
        message = self.model.create_chat_completion(
            messages,
            max_tokens=max_new_tokens,
            temperature=temperature
        )
        output = message['choices'][0]["message"]["content"]
        return output
    
    def complete(self,
                 text: str,
                 max_new_tokens: int = 512,
                 temperature: float = 0.1) -> str:
        message = self.model(
            text,
            max_tokens=max_new_tokens,
            temperature=temperature,
            echo=False
        )
        output = message["choices"][0]["text"]
        return output
        

